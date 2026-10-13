#include "gpu_presolver/presolve/cleanup_support.cuh"
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/rules/rule_dual_fix.hpp"

#include "gpu_presolver/presolve/fixed_col_tape.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

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


double sum_device_double(const double* values, std::int32_t n, const char* context) {
  if (n <= 0) {
    return 0.0;
  }
  CleanupScratchFrame reduction;
  void* temp_storage = nullptr;
  std::size_t temp_bytes = 0;
  double* result_device = nullptr;
  double result = 0.0;
  result_device = reduction.get<double>(7, 1);
  throw_if_cuda_error(cub::DeviceReduce::Sum(temp_storage, temp_bytes, values, result_device, n),
                      context);
  temp_storage = reduction.get<unsigned char>(6, temp_bytes);
  throw_if_cuda_error(cub::DeviceReduce::Sum(temp_storage, temp_bytes, values, result_device, n),
                      context);
  throw_if_cuda_error(cudaMemcpy(&result, result_device, sizeof(double), cudaMemcpyDeviceToHost),
                      context);
  return result;
}


void append_infinite_fixed_col_tape_from_device(PostsolveTape& tape,
                                                const std::uint8_t* fixed_mask,
                                                const std::uint8_t* infinite_fix_mask,
                                                const double* fixed_val,
                                                const std::uint8_t* keep_row,
                                                const double* l,
                                                const double* u,
                                                const double* AL,
                                                const double* AU,
                                                const DeviceCsrMatrix& A,
                                                const DeviceCsrMatrix& AT,
                                                std::int32_t m,
                                                std::int32_t n,
                                                const char* context,
                                                const std::uint8_t* entry_cols) {
  if (tape.resident) {
    append_infinite_device_recovery(tape, fixed_mask, infinite_fix_mask, fixed_val, keep_row, l, u,
                                    AL, AU, A, AT, n, entry_cols);
    return;
  }

  std::vector<std::uint8_t> host_fixed_mask(static_cast<std::size_t>(n));
  std::vector<std::uint8_t> host_infinite_fix_mask(static_cast<std::size_t>(n));
  std::vector<std::uint8_t> host_keep_row(static_cast<std::size_t>(m));

  throw_if_cuda_error(cudaMemcpy(host_fixed_mask.data(), fixed_mask, static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_infinite_fix_mask.data(), infinite_fix_mask,
                                 static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      context);
  bool has_infinite_fixed_col = false;
  for (std::int32_t col = 0; col < n; ++col) {
    const auto col_idx = static_cast<std::size_t>(col);
    if (host_fixed_mask[col_idx] != std::uint8_t{0} &&
        host_infinite_fix_mask[col_idx] != std::uint8_t{0}) {
      has_infinite_fixed_col = true;
      break;
    }
  }
  if (!has_infinite_fixed_col) {
    return;
  }

  std::vector<double> host_fixed_val(static_cast<std::size_t>(n));
  std::vector<double> host_l(static_cast<std::size_t>(n));
  std::vector<double> host_u(static_cast<std::size_t>(n));
  std::vector<double> host_AL(static_cast<std::size_t>(m));
  std::vector<double> host_AU(static_cast<std::size_t>(m));
  std::vector<NnzOffset> host_a_row_ptr(static_cast<std::size_t>(m + 1));
  std::vector<std::int32_t> host_a_col_val(static_cast<std::size_t>(A.nnz));
  std::vector<double> host_a_nz_val(static_cast<std::size_t>(A.nnz));
  std::vector<NnzOffset> host_at_row_ptr(static_cast<std::size_t>(n + 1));
  std::vector<std::int32_t> host_at_col_val(static_cast<std::size_t>(AT.nnz));

  throw_if_cuda_error(cudaMemcpy(host_fixed_val.data(), fixed_val,
                                 sizeof(double) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_l.data(), l, sizeof(double) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_u.data(), u, sizeof(double) * static_cast<std::size_t>(n),
                                 cudaMemcpyDeviceToHost),
                      context);
  if (m > 0) {
    throw_if_cuda_error(cudaMemcpy(host_keep_row.data(), keep_row, static_cast<std::size_t>(m),
                                   cudaMemcpyDeviceToHost),
                        context);
    throw_if_cuda_error(cudaMemcpy(host_AL.data(), AL, sizeof(double) * static_cast<std::size_t>(m),
                                   cudaMemcpyDeviceToHost),
                        context);
    throw_if_cuda_error(cudaMemcpy(host_AU.data(), AU, sizeof(double) * static_cast<std::size_t>(m),
                                   cudaMemcpyDeviceToHost),
                        context);
    throw_if_cuda_error(cudaMemcpy(host_a_row_ptr.data(), A.rowPtr,
                                   sizeof(NnzOffset) * static_cast<std::size_t>(m + 1),
                                   cudaMemcpyDeviceToHost),
                        context);
  }
  if (A.nnz > 0) {
    throw_if_cuda_error(cudaMemcpy(host_a_col_val.data(), A.colVal,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(A.nnz),
                                   cudaMemcpyDeviceToHost),
                        context);
    throw_if_cuda_error(cudaMemcpy(host_a_nz_val.data(), A.nzVal,
                                   sizeof(double) * static_cast<std::size_t>(A.nnz),
                                   cudaMemcpyDeviceToHost),
                        context);
  }
  throw_if_cuda_error(cudaMemcpy(host_at_row_ptr.data(), AT.rowPtr,
                                 sizeof(NnzOffset) * static_cast<std::size_t>(n + 1),
                                 cudaMemcpyDeviceToHost),
                      context);
  if (AT.nnz > 0) {
    throw_if_cuda_error(cudaMemcpy(host_at_col_val.data(), AT.colVal,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(AT.nnz),
                                   cudaMemcpyDeviceToHost),
                        context);
  }

  for (std::int32_t col = 0; col < n; ++col) {
    const auto col_idx = static_cast<std::size_t>(col);
    if (host_fixed_mask[col_idx] == std::uint8_t{0} ||
        host_infinite_fix_mask[col_idx] == std::uint8_t{0}) {
      continue;
    }
    const bool fix_to_pos_inf = host_fixed_val[col_idx] > 0.0;
    std::vector<std::int32_t> indices{fix_to_pos_inf ? 1 : -1, col};
    std::vector<double> vals{0.0, fix_to_pos_inf ? host_l[col_idx] : host_u[col_idx]};
    std::int32_t active_rows = 0;
    for (NnzOffset p = host_at_row_ptr[col_idx]; p < host_at_row_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
      const std::int32_t row = host_at_col_val[static_cast<std::size_t>(p)];
      if (host_keep_row[static_cast<std::size_t>(row)] == std::uint8_t{0}) {
        continue;
      }
      ++active_rows;
      const NnzOffset row_start = host_a_row_ptr[static_cast<std::size_t>(row)];
      const NnzOffset row_stop = host_a_row_ptr[static_cast<std::size_t>(row + 1)];
      const NnzOffset row_len = row_stop - row_start;
      indices.push_back(row_len);
      vals.push_back(std::isfinite(host_AL[static_cast<std::size_t>(row)])
                         ? host_AL[static_cast<std::size_t>(row)]
                         : host_AU[static_cast<std::size_t>(row)]);
      for (NnzOffset q = row_start; q < row_stop; ++q) {
        indices.push_back(host_a_col_val[static_cast<std::size_t>(q)]);
        vals.push_back(host_a_nz_val[static_cast<std::size_t>(q)]);
      }
    }
    vals[0] = static_cast<double>(active_rows);
    append_postsolve_record(
        tape, PostsolveReductionType::FixedColInf, indices, vals, PostsolveDualMode::Exact);
  }
}

__global__ void _kernel_dual_fix_candidates(std::int32_t* status_flag,
                                            std::uint8_t* fixed_mask,
                                            std::uint8_t* infinite_fix_mask,
                                            double* fixed_val,
                                            double* obj_contrib,
                                            const std::uint8_t* keep_col,
                                            const std::uint8_t* keep_row,
                                            const std::uint8_t* candidate_col_mask,
                                            const double* c,
                                            const double* l,
                                            const double* u,
                                            const double* AL,
                                            const double* AU,
                                            const NnzOffset* at_row_ptr,
                                            const std::int32_t* at_col_val,
                                            const double* at_nz_val,
                                            double zero_tol,
                                            double bound_tol,
                                            std::int32_t max_nonzero_cost_col_degree,
                                            const std::int32_t* live_col_nnz,
                                            std::int32_t n, const std::int32_t* active_cols) {
  const std::int32_t logical = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (logical >= n) return;
  const std::int32_t j = active_cols ? active_cols[logical] : logical;
  {
    fixed_mask[j] = std::uint8_t{0};
    infinite_fix_mask[j] = std::uint8_t{0};
    fixed_val[j] = 0.0;
    obj_contrib[j] = 0.0;
    if (keep_col[j] == std::uint8_t{0} ||
        (candidate_col_mask != nullptr &&
         candidate_col_mask[j] == std::uint8_t{0})) {
      return;
    }

    const double lj = l[j];
    const double uj = u[j];
    if (lj > uj + bound_tol) {
      atomicMax(&status_flag[0], 1);
      return;
    }

    bool has_down_lock = false;
    bool has_up_lock = false;
    const NnzOffset p_start = at_row_ptr[j];
    const NnzOffset p_stop = at_row_ptr[j + 1];
    if (fabs(c[j]) > zero_tol &&
        (live_col_nnz ? live_col_nnz[j] : p_stop - p_start) > max_nonzero_cost_col_degree) {
      return;
    }
    for (NnzOffset p = p_start; p < p_stop; ++p) {
      const std::int32_t row = at_col_val[p];
      if (keep_row[row] == std::uint8_t{0}) {
        continue;
      }

      const double aij = at_nz_val[p];
      if (aij > zero_tol) {
        has_down_lock = has_down_lock || isfinite(AL[row]);
        has_up_lock = has_up_lock || isfinite(AU[row]);
      } else if (aij < -zero_tol) {
        has_down_lock = has_down_lock || isfinite(AU[row]);
        has_up_lock = has_up_lock || isfinite(AL[row]);
      }

      if (has_down_lock && has_up_lock) {
        break;
      }
    }

    const double cj = c[j];
    bool fixed = false;
    bool infinite_fix = false;
    double vj = 0.0;

    if (cj > zero_tol) {
      if (!has_down_lock) {
        if (isfinite(lj)) {
          vj = lj;
          fixed = true;
        } else {
          atomicMax(&status_flag[0], 2);
          return;
        }
      }
    } else if (cj < -zero_tol) {
      if (!has_up_lock) {
        if (isfinite(uj)) {
          vj = uj;
          fixed = true;
        } else {
          atomicMax(&status_flag[0], 2);
          return;
        }
      }
    } else if (fabs(cj) <= zero_tol) {
      if (!has_down_lock) {
        fixed = true;
        if (isfinite(lj)) {
          vj = lj;
        } else {
          vj = -INFINITY;
          infinite_fix = true;
        }
      } else if (!has_up_lock) {
        fixed = true;
        if (isfinite(uj)) {
          vj = uj;
        } else {
          vj = INFINITY;
          infinite_fix = true;
        }
      }
    }

    if (fixed) {
      fixed_mask[j] = std::uint8_t{1};
      infinite_fix_mask[j] = infinite_fix ? std::uint8_t{1} : std::uint8_t{0};
      fixed_val[j] = vj;
      obj_contrib[j] = infinite_fix ? 0.0 : (cj * vj);
      atomicMax(&status_flag[1], 1);
      if (infinite_fix) {
        atomicMax(&status_flag[2], 1);
      }
    }
  }
}

// Certify order-independent integer sums before changing column addresses.
// Each term must be an integer of magnitude <= 2^31, and the sum of absolute
// values <= 2^53. Then every possible partial sum is exactly representable.
// The int32 nnz limit bounds the uint64 absolute total by 2^62 (no overflow).
__global__ void _kernel_cleanup_shift_terms(unsigned int* counts,
    unsigned long long* absolute_sum, const std::uint8_t* fixed,
    const std::uint8_t* inf, const double* value, const std::uint8_t* keep_row,
    const NnzOffset* ptr, const int* rows, const double* coefficients,
    const int* active_cols, int n) {
  const int logical = int(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (logical >= n) return;
  const int col = active_cols[logical];
  if (!fixed[col] || inf[col]) return;
  for (auto p = ptr[col]; p < ptr[col + 1]; ++p) {
    const int row = rows[p];
    if (!keep_row[row]) continue;
    const double contribution = coefficients[p] * value[col];
    const double magnitude = fabs(contribution);
    atomicAdd(counts + row, 1u);
    if (magnitude <= 2147483648.0 && trunc(contribution) == contribution)
      atomicAdd(absolute_sum + row, static_cast<unsigned long long>(magnitude));
    else atomicOr(counts + row, 0x80000000u);
  }
}
__global__ void _kernel_cleanup_shift_conflicts(const unsigned int* counts,
    const unsigned long long* absolute_sum, int* conflict, int m) {
  const int row = int(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= m) return;
  const auto packed = counts[row];
  if ((packed & 0x7fffffffu) > 1 &&
      ((packed & 0x80000000u) || absolute_sum[row] > (1ull << 53)))
    atomicExch(conflict, 1);
}

__global__ void _kernel_dual_fix_row_shift(double* row_shift,
                                           const std::uint8_t* fixed_mask,
                                           const std::uint8_t* infinite_fix_mask,
                                           const double* fixed_val,
                                           const std::uint8_t* keep_row,
                                           const NnzOffset* at_row_ptr,
                                           const std::int32_t* at_col_val,
                                           const double* at_nz_val,
                                           std::int32_t n, const std::int32_t* active_cols) {
  const std::int32_t logical = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (logical >= n) return;
  const std::int32_t j = active_cols ? active_cols[logical] : logical;
  if (fixed_mask[j] != std::uint8_t{0} && infinite_fix_mask[j] == std::uint8_t{0}) {
    const double vj = fixed_val[j];
    const NnzOffset p_start = at_row_ptr[j];
    const NnzOffset p_stop = at_row_ptr[j + 1];
    for (NnzOffset p = p_start; p < p_stop; ++p) {
      const std::int32_t row = at_col_val[p];
      if (keep_row[row] != std::uint8_t{0}) {
        atomicAdd(&row_shift[row], at_nz_val[p] * vj);
      }
    }
  }
}

__global__ void _kernel_dual_fix_row_remove(std::uint8_t* row_remove,
                                            std::int32_t* row_action_flag,
                                            const std::uint8_t* fixed_mask,
                                            const std::uint8_t* infinite_fix_mask,
                                            const std::uint8_t* keep_row,
                                            const NnzOffset* at_row_ptr,
                                            const std::int32_t* at_col_val,
                                            std::int32_t n, const std::int32_t* active_cols) {
  const std::int32_t logical = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (logical >= n) return;
  const std::int32_t j = active_cols ? active_cols[logical] : logical;
  if (fixed_mask[j] != std::uint8_t{0} && infinite_fix_mask[j] != std::uint8_t{0}) {
    const NnzOffset p_start = at_row_ptr[j];
    const NnzOffset p_stop = at_row_ptr[j + 1];
    for (NnzOffset p = p_start; p < p_stop; ++p) {
      const std::int32_t row = at_col_val[p];
      if (keep_row[row] != std::uint8_t{0}) {
        row_remove[row] = std::uint8_t{1};
        atomicMax(row_action_flag, 1);
      }
    }
  }
}

__global__ void _kernel_dual_fix_keep_row_after(std::uint8_t* keep_row_after,
                                                const std::uint8_t* keep_row,
                                                const std::uint8_t* row_remove,
                                                std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row < m) {
    keep_row_after[row] =
        (keep_row[row] != std::uint8_t{0} && row_remove[row] == std::uint8_t{0})
            ? std::uint8_t{1}
            : std::uint8_t{0};
  }
}

__global__ void _kernel_dual_fix_apply(std::uint8_t* keep_col,
                                       std::uint8_t* keep_row,
                                       double* AL,
                                       double* AU,
                                       double* new_l,
                                       double* new_u,
                                       const std::uint8_t* fixed_mask,
                                       const std::uint8_t* row_remove,
                                       const double* row_shift,
                                       const double* fixed_val,
                                       std::int32_t m,
                                       std::int32_t n) {
  const std::int32_t k = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (k < n && fixed_mask[k] != std::uint8_t{0}) {
    new_l[k] = fixed_val[k];
    new_u[k] = fixed_val[k];
    keep_col[k] = std::uint8_t{0};
  }
  if (k < m) {
    AL[k] -= row_shift[k];
    AU[k] -= row_shift[k];
    if (row_remove[k] != std::uint8_t{0}) {
      keep_row[k] = std::uint8_t{0};
    }
  }
}

}  // namespace

void apply_rule_dual_fix(PresolvePlanGpu& plan,
                         const LPInfoGpu& lp,
                         const PresolveStatsGpu& stats,
                         const PresolveParams& pparams,
                         const std::uint8_t* candidate_col_mask) {
  (void)stats;
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }

  const std::int32_t n = lp.A.cols;
  const std::int32_t m = lp.A.rows;
  if (n == 0) {
    return;
  }

  CleanupScratchFrame scratch;
  std::int32_t* status_flag = scratch.get<std::int32_t>(5, 3);
  std::uint8_t* fixed_mask = scratch.get<std::uint8_t>(2, n);
  std::uint8_t* infinite_fix_mask = scratch.get<std::uint8_t>(11, n);
  double* fixed_val = scratch.get<double>(0, n);
  double* obj_contrib = scratch.get<double>(1, n);
  double* row_shift = scratch.get<double>(3, m);
  std::uint8_t* row_remove = scratch.get<std::uint8_t>(4, m);
  std::int32_t* row_action_flag = scratch.get<std::int32_t>(12, 1);
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t) * 3), "cudaMemset dual_fix status_flag");
  throw_if_cuda_error(cudaMemset(row_action_flag, 0, sizeof(std::int32_t)), "cudaMemset dual_fix row_action_flag");
  throw_if_cuda_error(cudaMemset(row_shift, 0, sizeof(double) * static_cast<std::size_t>(m)), "cudaMemset dual_fix row_shift");
  throw_if_cuda_error(cudaMemset(row_remove, 0, static_cast<std::size_t>(m)), "cudaMemset dual_fix row_remove");

  const int active_n = plan.cleanup_active_cols ? plan.cleanup_active_count : n;
  const int blocks_n = (active_n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  // An all-deleted prefix leaves no dual candidates (the later row cleanup
  // still needs to run). Avoid a zero-grid launch.
  if (active_n == 0) {
    return;
  }
  if (plan.cleanup_active_cols) {
    throw_if_cuda_error(cudaMemset(fixed_mask, 0, n), "dual active fixed mask");
    throw_if_cuda_error(cudaMemset(infinite_fix_mask, 0, n), "dual active infinite mask");
    throw_if_cuda_error(cudaMemset(fixed_val, 0, sizeof(double) * n), "dual active fixed values");
    throw_if_cuda_error(cudaMemset(obj_contrib, 0, sizeof(double) * n), "dual active contributions");
  }
  _kernel_dual_fix_candidates<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      status_flag,
      fixed_mask,
      infinite_fix_mask,
      fixed_val,
      obj_contrib,
      plan.keep_col_mask,
      plan.keep_row_mask,
      candidate_col_mask,
      plan.new_c,
      plan.new_l,
      plan.new_u,
      plan.new_AL,
      plan.new_AU,
      lp.AT.rowPtr,
      lp.AT.colVal,
      lp.AT.nzVal,
      pparams.zero_tol,
      pparams.bound_tol,
      pparams.dual_fix_max_nonzero_cost_col_degree,
      stats.live_masks ? stats.col_nnz : nullptr,
      active_n, plan.cleanup_active_cols);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_dual_fix_candidates");

  std::int32_t status[3] = {0, 0, 0};
  throw_if_cuda_error(cudaMemcpy(status, status_flag, sizeof(status), cudaMemcpyDeviceToHost),
                      "cudaMemcpy dual_fix status");
  if (status[0] == 1) {
    plan.has_infeasible = true;
  } else if (status[0] == 2) {
    plan.has_unbounded = true;
  } else if (status[1] != 0) {
    if (plan.cleanup_active_cols && m > 0) {
      unsigned int* counts = scratch.get<unsigned int>(13, m);
      throw_if_cuda_error(cudaMemset(counts, 0, sizeof(unsigned int) * m), "cleanup zero shift counts");
      _kernel_cleanup_shift_terms<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
          counts, reinterpret_cast<unsigned long long*>(row_shift), fixed_mask,
          infinite_fix_mask, fixed_val, plan.keep_row_mask, lp.AT.rowPtr, lp.AT.colVal,
          lp.AT.nzVal, plan.cleanup_active_cols, active_n);
      _kernel_cleanup_shift_conflicts<<<(m + 255) / 256, 256>>>(
          counts, reinterpret_cast<unsigned long long*>(row_shift), row_action_flag, m);
      throw_if_cuda_error(cudaGetLastError(), "cleanup shift conflict check");
      int conflict = 0;
      throw_if_cuda_error(cudaMemcpy(&conflict, row_action_flag, sizeof(int),
          cudaMemcpyDeviceToHost), "cleanup shift conflict result");
      if (conflict) {
        // No candidates, tape, bounds, or masks have been committed yet.
        plan.cleanup_requires_materialization = true;
        return;
      }
      throw_if_cuda_error(cudaMemset(row_shift, 0, sizeof(double) * m), "cleanup reset shifts");
    }
    _kernel_dual_fix_row_shift<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
        row_shift,
        fixed_mask,
        infinite_fix_mask,
        fixed_val,
        plan.keep_row_mask,
        lp.AT.rowPtr,
        lp.AT.colVal,
        lp.AT.nzVal,
        active_n, plan.cleanup_active_cols);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_dual_fix_row_shift");
    _kernel_dual_fix_row_remove<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
        row_remove,
        row_action_flag,
        fixed_mask,
        infinite_fix_mask,
        plan.keep_row_mask,
        lp.AT.rowPtr,
        lp.AT.colVal,
        active_n, plan.cleanup_active_cols);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_dual_fix_row_remove");
    std::int32_t row_action = 0;
    throw_if_cuda_error(cudaMemcpy(&row_action, row_action_flag, sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy dual_fix row_action_flag");
    const bool row_changed = row_action != 0;
    if (pparams.record_postsolve_tape) {
      if (status[2] != 0) {
        append_infinite_fixed_col_tape_from_device(plan.tape,
                                                   fixed_mask,
                                                   infinite_fix_mask,
                                                   fixed_val,
                                                   plan.keep_row_mask,
                                                   plan.new_l,
                                                   plan.new_u,
                                                   plan.new_AL,
                                                   plan.new_AU,
                                                   lp.A,
                                                   lp.AT,
                                                   m,
                                                   n,
                                                   "cudaMemcpy dual_fix fixed-col-inf tape",
                                                   plan.rule_entry_keep_col ? plan.rule_entry_keep_col : plan.cleanup_prefix_keep_col);
      }
      std::uint8_t* keep_row_after_for_tape = plan.keep_row_mask;
      if (m > 0) {
        keep_row_after_for_tape = scratch.get<std::uint8_t>(14, m);
        const int blocks_m = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
        _kernel_dual_fix_keep_row_after<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
            keep_row_after_for_tape, plan.keep_row_mask, row_remove, m);
        throw_if_cuda_error(cudaGetLastError(), "_kernel_dual_fix_keep_row_after");
      }
      if (pparams.record_postsolve_tape_cpu) {
        append_compact_fixed_col_tape_from_device(&plan.tape,
                                                  fixed_mask,
                                                  infinite_fix_mask,
                                                  fixed_val,
                                                  keep_row_after_for_tape,
                                                  plan.new_c,
                                                  lp.AT,
                                                  n,
                                                  "dual_fix fixed-col compact tape", FixedColObjective::Skip);
      }
    }
    const std::int32_t max_mn = m > n ? m : n;
    const int blocks_mn = (max_mn + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    _kernel_dual_fix_apply<<<blocks_mn, GPU_PRESOLVE_THREADS>>>(
        plan.keep_col_mask,
        plan.keep_row_mask,
        plan.new_AL,
        plan.new_AU,
        plan.new_l,
        plan.new_u,
        fixed_mask,
        row_remove,
        row_shift,
        fixed_val,
        m,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_dual_fix_apply");
    throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_dual_fix synchronize");
    const double obj_delta = sum_cleanup_domain(obj_contrib, n, plan.rule_entry_keep_col ? plan.rule_entry_keep_col : plan.cleanup_prefix_keep_col,
        [](const double* values, int count) {
          return sum_device_double(values, count, "cudaMemcpy dual_fix objective delta");
        });
    plan.obj_constant_delta += obj_delta;
    plan.has_row_action = plan.has_row_action || row_changed;
    plan.has_col_action = true;
    plan.has_change = true;
  }

}

}  // namespace gpu_presolver::presolve
