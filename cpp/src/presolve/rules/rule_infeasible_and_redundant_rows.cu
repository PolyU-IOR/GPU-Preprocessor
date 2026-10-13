#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/cleanup_workspace.hpp"
#include "gpu_presolver/presolve/live_matrix.cuh"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/rules/rule_infeasible_and_redundant_rows.hpp"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include "gpu_presolver/presolve/gpu_presolve_kernels.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

void append_postsolve_record(PostsolveTape& tape,
                             PostsolveReductionType type,
                             const std::vector<std::int32_t>& indices,
                             const std::vector<double>& vals,
                             PostsolveDualMode dual_mode) {
  tape.types.push_back(static_cast<std::int32_t>(type));
  tape.indices.insert(tape.indices.end(), indices.begin(), indices.end());
  tape.vals.insert(tape.vals.end(), vals.begin(), vals.end());
  tape.index_starts.push_back(static_cast<NnzOffset>(tape.indices.size()));
  tape.value_starts.push_back(static_cast<NnzOffset>(tape.vals.size()));
  tape.dual_modes.push_back(static_cast<std::uint8_t>(dual_mode));
}

__global__ void _kernel_infeasible_and_redundant_rows_classify(std::int32_t* status_flags,
                                                 std::uint8_t* full_redundant,
                                                 std::uint8_t* drop_lower,
                                                 std::uint8_t* drop_upper,
                                                 const std::uint8_t* keep_row,
                                                 const std::uint8_t* keep_col,
                                                 const double* AL,
                                                 const double* AU,
                                                 const double* l,
                                                 const double* u,
                                                 const std::int32_t* row_nnz,
                                                 const NnzOffset* row_ptr,
                                                 const std::int32_t* col_val,
                                                 const double* nz_val,
                                                 double tol,
                                                 std::int32_t m, int max_length) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < m && (max_length == 0 || (keep_col ? row_nnz[i] : row_ptr[i+1] - row_ptr[i]) <= max_length)) {
    full_redundant[i] = std::uint8_t{0};
    drop_lower[i] = std::uint8_t{0};
    drop_upper[i] = std::uint8_t{0};

    const bool keep_i = keep_row[i] != std::uint8_t{0};
    const std::int32_t row_nnz_i = row_nnz[i];
    if (!keep_i || row_nnz_i <= 1) {
      return;
    }

    const double lower_i = AL[i];
    const double upper_i = AU[i];
    const bool lower_finite = isfinite(lower_i);
    const bool upper_finite = isfinite(upper_i);
    if (lower_finite && upper_finite && fabs(upper_i - lower_i) <= tol) {
      return;
    }

    double row_min = 0.0;
    double row_max = 0.0;
    const NnzOffset row_start = row_ptr[i];
    const NnzOffset row_stop = row_ptr[i + 1];
    for (NnzOffset p = row_start; p < row_stop; ++p) {
      const std::int32_t col = col_val[p];
      if (keep_col && !keep_col[col]) continue;
      const double a = nz_val[p];
      const double lj = l[col];
      const double uj = u[col];
      if (a >= 0.0) {
        row_min += a * lj;
        row_max += a * uj;
      } else {
        row_min += a * uj;
        row_max += a * lj;
      }
    }

    const bool infeasible =
        (lower_finite && row_max < lower_i - tol) ||
        (upper_finite && row_min > upper_i + tol);
    if (infeasible) {
      atomicMax(&status_flags[0], 1);
      return;
    }

    const bool lower_implied = !lower_finite || row_min >= lower_i - tol;
    const bool upper_implied = !upper_finite || row_max <= upper_i + tol;
    if (lower_implied && upper_implied) {
      full_redundant[i] = std::uint8_t{1};
      atomicMax(&status_flags[1], 1);
    } else if (lower_finite && lower_implied) {
      drop_lower[i] = std::uint8_t{1};
      atomicMax(&status_flags[2], 1);
    } else if (upper_finite && upper_implied) {
      drop_upper[i] = std::uint8_t{1};
      atomicMax(&status_flags[3], 1);
    }
  }
}

template<int Width>
__global__ void cooperative_redundant_rows(
    int* flags, std::uint8_t* redundant, std::uint8_t* drop_l, std::uint8_t* drop_u,
    const std::uint8_t* keep, const std::uint8_t* keep_col, const double* AL, const double* AU,
    const double* lower, const double* upper, const int* nnz,
    const NnzOffset* ptr, const int* indices, const double* values, double tol,
    const int* rows, const int* count) {
  const int lane = Width == 32 ? (threadIdx.x & 31) : threadIdx.x;
  const int group = Width == 32 ? blockIdx.x*8 + threadIdx.x/32 : blockIdx.x;
  const NnzOffset stride = Width == 32 ? gridDim.x*8 : gridDim.x;
  using Reduce = cub::BlockReduce<double, 256>;
  __shared__ typename Reduce::TempStorage storage;
  __shared__ LiveRowStorage<Width> live_storage;
  for (NnzOffset q = group; q < *count; q += stride) {
    const int row = rows[q];
    if (lane == 0) redundant[row] = drop_l[row] = drop_u[row] = 0;
    const double al = AL[row], au = AU[row];
    const bool lf = isfinite(al), uf = isfinite(au);
    if (!keep[row] || nnz[row] <= 1 || (lf && uf && fabs(au-al) <= tol)) continue;
    double lo = 0, hi = 0;
    const auto* mask = nnz[row] == ptr[row+1]-ptr[row] ? nullptr : keep_col;
    for_each_live_entry<Width>(ptr[row], ptr[row+1], indices, mask, live_storage,
                              [&](NnzOffset p) {
      const int col = indices[p];
      const double a = values[p];
      lo += a >= 0 ? a*lower[col] : a*upper[col];
      hi += a >= 0 ? a*upper[col] : a*lower[col];
    });
    if constexpr (Width == 32) {
      for (NnzOffset offset=16; offset; offset/=2) {
        lo += __shfl_down_sync(0xffffffffu,lo,offset);
        hi += __shfl_down_sync(0xffffffffu,hi,offset);
      }
    } else {
      lo = Reduce(storage).Sum(lo); __syncthreads();
      hi = Reduce(storage).Sum(hi); __syncthreads();
    }
    if (lane == 0) {
      if ((lf && hi < al-tol) || (uf && lo > au+tol)) atomicMax(flags,1);
      else {
        const bool li = !lf || lo >= al-tol, ui = !uf || hi <= au+tol;
        if (li && ui) { redundant[row]=1; atomicMax(flags+1,1); }
        else if (lf && li) { drop_l[row]=1; atomicMax(flags+2,1); }
        else if (uf && ui) { drop_u[row]=1; atomicMax(flags+3,1); }
      }
    }
    if constexpr (Width != 32) __syncthreads();
  }
}

__global__ void _kernel_infeasible_and_redundant_rows_apply(std::uint8_t* keep_row,
                                              double* AL,
                                              double* AU,
                                              const std::uint8_t* full_redundant,
                                              const std::uint8_t* drop_lower,
                                              const std::uint8_t* drop_upper,
                                              std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < m) {
    if (full_redundant[i] != std::uint8_t{0}) {
      keep_row[i] = std::uint8_t{0};
    }
    if (drop_lower[i] != std::uint8_t{0}) {
      AL[i] = -INFINITY;
    }
    if (drop_upper[i] != std::uint8_t{0}) {
      AU[i] = INFINITY;
    }
  }
}

}  // namespace

void apply_rule_infeasible_and_redundant_rows(PresolvePlanGpu& plan,
                                const LPInfoGpu& lp,
                                const PresolveStatsGpu& stats,
                                const PresolveParams& pparams) {
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }

  const std::int32_t m = lp.A.rows;
  if (m == 0) {
    return;
  }

  CleanupScratchFrame scratch;
  std::int32_t* status_flags = scratch.get<std::int32_t>(5, 4);
  std::uint8_t* full_redundant = scratch.get<std::uint8_t>(2, m);
  std::uint8_t* drop_lower = scratch.get<std::uint8_t>(11, m);
  std::uint8_t* drop_upper = scratch.get<std::uint8_t>(12, m);
  throw_if_cuda_error(cudaMemset(status_flags, 0, sizeof(std::int32_t) * 4), "cudaMemset infeasible_redundant_rows status_flags");

  const int blocks = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_infeasible_and_redundant_rows_classify<<<blocks, GPU_PRESOLVE_THREADS>>>(
      status_flags,
      full_redundant,
      drop_lower,
      drop_upper,
      plan.keep_row_mask,
      stats.live_masks ? plan.keep_col_mask : nullptr,
      plan.new_AL,
      plan.new_AU,
      plan.new_l,
      plan.new_u,
      stats.row_nnz,
      lp.A.rowPtr,
      lp.A.colVal,
      lp.A.nzVal,
      pparams.bound_tol,
      m, adaptive_extended_enabled() ? ADAPTIVE_SHORT_ROW : 0);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_infeasible_and_redundant_rows_classify");
  if (adaptive_extended_enabled()) {
    const auto& groups = prepare_adaptive_row_groups(lp.A, stats.live_masks ? stats.row_nnz : nullptr);
    cooperative_redundant_rows<32><<<std::min(1024, (m+7)/8),256>>>(
        status_flags, full_redundant, drop_lower, drop_upper, plan.keep_row_mask,
        stats.live_masks ? plan.keep_col_mask : nullptr, plan.new_AL, plan.new_AU, plan.new_l, plan.new_u, stats.row_nnz,
        lp.A.rowPtr, lp.A.colVal, lp.A.nzVal, pparams.bound_tol, groups.medium, groups.counts);
    cooperative_redundant_rows<256><<<std::min(256,m),256>>>(
        status_flags, full_redundant, drop_lower, drop_upper, plan.keep_row_mask,
        stats.live_masks ? plan.keep_col_mask : nullptr, plan.new_AL, plan.new_AU, plan.new_l, plan.new_u, stats.row_nnz,
        lp.A.rowPtr, lp.A.colVal, lp.A.nzVal, pparams.bound_tol, groups.long_rows, groups.counts+1);
    throw_if_cuda_error(cudaGetLastError(), "cooperative redundant rows");
  }

  std::int32_t flags[4] = {0, 0, 0, 0};
  throw_if_cuda_error(cudaMemcpy(flags, status_flags, sizeof(flags), cudaMemcpyDeviceToHost),
                      "cudaMemcpy infeasible_and_redundant_rows flags");
  if (flags[0] != 0) {
    plan.has_infeasible = true;
  } else if (flags[1] != 0 || flags[2] != 0 || flags[3] != 0) {
    if (pparams.record_postsolve_tape && pparams.record_postsolve_tape_cpu &&
        (flags[1] != 0 || flags[2] != 0 || flags[3] != 0)) {
      if (plan.tape.resident) {
        if (flags[1])
          append_simple_device_recovery(plan.tape, SimpleRecovery::DeletedRow, full_redundant,
                                        nullptr, m, nullptr, plan.new_AL, plan.new_AU, nullptr);
        append_simple_device_recovery(plan.tape, SimpleRecovery::RowSides, drop_lower, drop_upper,
                                      m, nullptr, plan.new_AL, plan.new_AU, nullptr);
      } else {

        std::vector<std::uint8_t> host_full_redundant(static_cast<std::size_t>(m));
        std::vector<std::uint8_t> host_drop_lower(static_cast<std::size_t>(m));
        std::vector<std::uint8_t> host_drop_upper(static_cast<std::size_t>(m));
        std::vector<double> host_AL(static_cast<std::size_t>(m));
        std::vector<double> host_AU(static_cast<std::size_t>(m));
        throw_if_cuda_error(cudaMemcpy(host_full_redundant.data(), full_redundant,
                                       static_cast<std::size_t>(m), cudaMemcpyDeviceToHost),
                            "cudaMemcpy infeasible_and_redundant_rows full_redundant for tape");
        throw_if_cuda_error(cudaMemcpy(host_drop_lower.data(), drop_lower,
                                       static_cast<std::size_t>(m), cudaMemcpyDeviceToHost),
                            "cudaMemcpy infeasible_and_redundant_rows drop_lower for tape");
        throw_if_cuda_error(cudaMemcpy(host_drop_upper.data(), drop_upper,
                                       static_cast<std::size_t>(m), cudaMemcpyDeviceToHost),
                            "cudaMemcpy infeasible_and_redundant_rows drop_upper for tape");
        throw_if_cuda_error(cudaMemcpy(host_AL.data(), plan.new_AL,
                                       sizeof(double) * static_cast<std::size_t>(m),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy infeasible_and_redundant_rows AL for tape");
        throw_if_cuda_error(cudaMemcpy(host_AU.data(), plan.new_AU,
                                       sizeof(double) * static_cast<std::size_t>(m),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy infeasible_and_redundant_rows AU for tape");
        if (flags[1] != 0) {
          for (std::int32_t row = 0; row < m; ++row) {
            if (host_full_redundant[static_cast<std::size_t>(row)] == std::uint8_t{0}) {
              continue;
            }
            append_postsolve_record(
                plan.tape, PostsolveReductionType::DeletedRow, {row},
                {host_AL[static_cast<std::size_t>(row)], host_AU[static_cast<std::size_t>(row)]},
                PostsolveDualMode::Minimal);
          }
        }
        for (std::int32_t row = 0; row < m; ++row) {
          if (host_drop_lower[static_cast<std::size_t>(row)] != std::uint8_t{0}) {
            append_postsolve_record(plan.tape, PostsolveReductionType::LhsChange, {row},
                                    {host_AL[static_cast<std::size_t>(row)],
                                     host_AU[static_cast<std::size_t>(row)], -INFINITY,
                                     host_AU[static_cast<std::size_t>(row)]},
                                    PostsolveDualMode::Minimal);
          }
          if (host_drop_upper[static_cast<std::size_t>(row)] != std::uint8_t{0}) {
            append_postsolve_record(plan.tape, PostsolveReductionType::RhsChange, {row},
                                    {host_AL[static_cast<std::size_t>(row)],
                                     host_AU[static_cast<std::size_t>(row)],
                                     host_AL[static_cast<std::size_t>(row)], INFINITY},
                                    PostsolveDualMode::Minimal);
          }
        }
      }
    }
    _kernel_infeasible_and_redundant_rows_apply<<<blocks, GPU_PRESOLVE_THREADS>>>(
        plan.keep_row_mask,
        plan.new_AL,
        plan.new_AU,
        full_redundant,
        drop_lower,
        drop_upper,
        m);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_infeasible_and_redundant_rows_apply");
    throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_infeasible_and_redundant_rows synchronize");
    plan.has_row_action = true;
    plan.has_redundant_row_deletion = plan.has_redundant_row_deletion || flags[1] != 0;
    plan.has_row_deletion = plan.has_row_deletion || flags[1] != 0;
    plan.has_change = true;
  }

}

}  // namespace gpu_presolver::presolve
