#include "gpu_presolver/presolve/cleanup_workspace.hpp"
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/cleanup_support.cuh"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/fixed_col_tape.hpp"
#include "gpu_presolver/presolve/rules/rule_infeasible_and_fixed_variables.hpp"

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
  void* temp_storage = nullptr;
  std::size_t temp_bytes = 0;
  double* result_device = nullptr;
  double result = 0.0;
  CleanupScratchFrame reduction;
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

__global__ void _kernel_infeasible_and_fixed_variables_candidates(std::uint8_t* fixed_mask,
                                                double* fixed_val,
                                                double* obj_contrib,
                                                double* row_error,
                                                const std::uint8_t* keep_col,
                                                const std::uint8_t* keep_row,
                                                const double* l,
                                                const double* u,
                                                const double* c,
                                                const NnzOffset* at_row_ptr,
                                                const std::int32_t* at_col_val,
                                                const double* at_nz_val,
                                                double bound_tol,
                                                std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j < n) {
    fixed_mask[j] = std::uint8_t{0};
    fixed_val[j] = 0.0;
    obj_contrib[j] = 0.0;
    if (keep_col[j] == std::uint8_t{0}) {
      return;
    }
    const double lj = l[j];
    const double uj = u[j];
    if (isfinite(lj) && isfinite(uj) && lj <= uj && uj - lj <= bound_tol) {
      const double vj = lj + 0.5 * (uj - lj);
      const double radius = fmax(vj - lj, uj - vj);
      fixed_mask[j] = std::uint8_t{1};
      fixed_val[j] = vj;
      obj_contrib[j] = c[j] * vj;

      const NnzOffset p_start = at_row_ptr[j];
      const NnzOffset p_stop = at_row_ptr[j + 1];
      for (NnzOffset p = p_start; p < p_stop; ++p) {
        const std::int32_t row = at_col_val[p];
        if (keep_row[row] != std::uint8_t{0} && radius > 0.0) {
          // Sum absolute contributions before accepting any approximate
          // fixing. Opposite signs must not cancel the uncertainty budget.
          atomicAdd(&row_error[row], fabs(at_nz_val[p]) * radius);
        }
      }
    }
  }
}

__global__ void _kernel_filter_fixed_variables(std::uint8_t* fixed_mask,
                                               double* obj_contrib,
                                               double* row_shift,
                                               const double* fixed_val,
                                               const double* row_error,
                                               const std::uint8_t* keep_row,
                                               const double* l,
                                               const double* u,
                                               const NnzOffset* at_row_ptr,
                                               const std::int32_t* at_col_val,
                                               const double* at_nz_val,
                                               double row_error_budget,
                                               std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j >= n || fixed_mask[j] == std::uint8_t{0}) {
    return;
  }
  const NnzOffset start = at_row_ptr[j];
  const NnzOffset stop = at_row_ptr[j + 1];
  if (l[j] != u[j]) {
    for (NnzOffset p = start; p < stop; ++p) {
      const std::int32_t row = at_col_val[p];
      // Exact fixings do not consume uncertainty and remain eligible even
      // when approximate candidates in the same row exceed the budget.
      if (keep_row[row] != std::uint8_t{0} &&
          !(row_error[row] <= row_error_budget)) {
        fixed_mask[j] = std::uint8_t{0};
        obj_contrib[j] = 0.0;
        return;
      }
    }
  }
  for (NnzOffset p = start; p < stop; ++p) {
    const std::int32_t row = at_col_val[p];
    if (keep_row[row] != std::uint8_t{0}) {
      atomicAdd(&row_shift[row], at_nz_val[p] * fixed_val[j]);
    }
  }
}

__global__ void _kernel_infeasible_and_fixed_variables_apply(std::uint8_t* keep_col,
                                           double* new_AL,
                                           double* new_AU,
                                           double* new_l,
                                           double* new_u,
                                           const std::uint8_t* fixed_mask,
                                           const double* fixed_val,
                                           const double* row_shift,
                                           std::int32_t n,
                                           std::int32_t m) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j < n && fixed_mask[j] != std::uint8_t{0}) {
    // The generic postsolve recorder recognizes fixed columns from equal
    // bounds. Preserve the exact value used for the row/objective shifts.
    new_l[j] = fixed_val[j];
    new_u[j] = fixed_val[j];
    keep_col[j] = std::uint8_t{0};
  }
  if (j < m) {
    new_AL[j] -= row_shift[j];
    new_AU[j] -= row_shift[j];
  }
}

__global__ void _kernel_count_mask(const std::uint8_t* mask,
                                   std::int32_t n,
                                   std::int32_t* count) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j < n && mask[j] != std::uint8_t{0}) {
    atomicAdd(count, 1);
  }
}

}  // namespace

void apply_rule_infeasible_and_fixed_variables(PresolvePlanGpu& plan,
                             const LPInfoGpu& lp,
                             const PresolveParams& pparams) {
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }

  const std::int32_t n = lp.A.cols;
  const std::int32_t m = lp.A.rows;
  if (n == 0) {
    return;
  }

  CleanupScratchFrame scratch;
  std::uint8_t* fixed_mask = nullptr;
  double* fixed_val = nullptr;
  double* obj_contrib = nullptr;
  double* row_shift = nullptr;
  double* row_error = nullptr;
  std::int32_t* fixed_count_device = nullptr;

  fixed_mask = scratch.get<std::uint8_t>(2, n);
  fixed_val = scratch.get<double>(0, n);
  obj_contrib = scratch.get<double>(1, n);
  row_shift = scratch.get<double>(3, m);
  row_error = scratch.get<double>(4, m);
  fixed_count_device = scratch.get<std::int32_t>(5, 1);
  throw_if_cuda_error(cudaMemset(fixed_mask, 0, static_cast<std::size_t>(n)), "cudaMemset fixed_mask");
  throw_if_cuda_error(cudaMemset(row_shift, 0, sizeof(double) * static_cast<std::size_t>(m)), "cudaMemset row_shift");
  throw_if_cuda_error(cudaMemset(row_error, 0, sizeof(double) * static_cast<std::size_t>(m)), "cudaMemset fixed row_error");
  throw_if_cuda_error(cudaMemset(fixed_count_device, 0, sizeof(std::int32_t)), "cudaMemset fixed_count");

  const int blocks_n = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const std::int32_t max_mn = n > m ? n : m;
  const int blocks_mn = (max_mn + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;

  _kernel_infeasible_and_fixed_variables_candidates<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      fixed_mask,
      fixed_val,
      obj_contrib,
      row_error,
      plan.keep_col_mask,
      plan.keep_row_mask,
      plan.new_l,
      plan.new_u,
      plan.new_c,
      lp.AT.rowPtr,
      lp.AT.colVal,
      lp.AT.nzVal,
      pparams.bound_tol,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_infeasible_and_fixed_variables_candidates");

  // Reserve 90% of the row feasibility tolerance for other numerical errors.
  // This is a per-batch guard, not a bound on errors from all presolve rules.
  // Reject every approximate candidate touching an over-budget row; selecting
  // an arbitrary subset would make reductions depend on GPU execution order.
  _kernel_filter_fixed_variables<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      fixed_mask, obj_contrib, row_shift, fixed_val, row_error,
      plan.keep_row_mask, plan.new_l, plan.new_u,
      lp.AT.rowPtr, lp.AT.colVal, lp.AT.nzVal,
      0.1 * pparams.feasibility_tol, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_filter_fixed_variables");

  _kernel_count_mask<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      fixed_mask,
      n,
      fixed_count_device);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_count_mask infeasible_and_fixed_variables");

  std::int32_t fixed_count = 0;
  throw_if_cuda_error(cudaMemcpy(&fixed_count, fixed_count_device, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy fixed_count");
  if (fixed_count == 0) {
    return;
  }

  if (pparams.record_postsolve_tape && pparams.record_postsolve_tape_cpu) {
    if (plan.tape.resident) {
      (void)append_compact_fixed_col_tape_from_device(&plan.tape, fixed_mask, nullptr, fixed_val,
                                                      plan.keep_row_mask, plan.new_c, lp.AT, n,
                                                      "fixed variable device tape", FixedColObjective::Skip);
    } else {

      std::vector<std::uint8_t> host_fixed_mask(static_cast<std::size_t>(n));
      std::vector<std::uint8_t> host_keep_row(static_cast<std::size_t>(m));
      std::vector<double> host_fixed_val(static_cast<std::size_t>(n));
      std::vector<double> host_c(static_cast<std::size_t>(n));
      std::vector<NnzOffset> host_at_row_ptr(static_cast<std::size_t>(n + 1));
      std::vector<std::int32_t> host_at_col_val(static_cast<std::size_t>(lp.AT.nnz));
      std::vector<double> host_at_nz_val(static_cast<std::size_t>(lp.AT.nnz));
      throw_if_cuda_error(cudaMemcpy(host_fixed_mask.data(), fixed_mask,
                                     static_cast<std::size_t>(n), cudaMemcpyDeviceToHost),
                          "cudaMemcpy infeasible_and_fixed_variables fixed_mask for tape");
      throw_if_cuda_error(cudaMemcpy(host_keep_row.data(), plan.keep_row_mask,
                                     static_cast<std::size_t>(m), cudaMemcpyDeviceToHost),
                          "cudaMemcpy infeasible_and_fixed_variables keep_row for tape");
      throw_if_cuda_error(cudaMemcpy(host_fixed_val.data(), fixed_val,
                                     sizeof(double) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy infeasible_and_fixed_variables fixed_val for tape");
      throw_if_cuda_error(cudaMemcpy(host_c.data(), plan.new_c,
                                     sizeof(double) * static_cast<std::size_t>(n),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy infeasible_and_fixed_variables c for tape");
      throw_if_cuda_error(cudaMemcpy(host_at_row_ptr.data(), lp.AT.rowPtr,
                                     sizeof(NnzOffset) * static_cast<std::size_t>(n + 1),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy infeasible_and_fixed_variables AT rowPtr for tape");
      if (lp.AT.nnz > 0) {
        throw_if_cuda_error(cudaMemcpy(host_at_col_val.data(), lp.AT.colVal,
                                       sizeof(std::int32_t) * static_cast<std::size_t>(lp.AT.nnz),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy infeasible_and_fixed_variables AT colVal for tape");
        throw_if_cuda_error(cudaMemcpy(host_at_nz_val.data(), lp.AT.nzVal,
                                       sizeof(double) * static_cast<std::size_t>(lp.AT.nnz),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy infeasible_and_fixed_variables AT nzVal for tape");
      }
      for (std::int32_t col = 0; col < n; ++col) {
        if (host_fixed_mask[static_cast<std::size_t>(col)] == std::uint8_t{0}) {
          continue;
        }
        std::vector<std::int32_t> indices{col};
        std::vector<double> vals{host_fixed_val[static_cast<std::size_t>(col)],
                                 host_c[static_cast<std::size_t>(col)]};
        for (NnzOffset p = host_at_row_ptr[static_cast<std::size_t>(col)];
             p < host_at_row_ptr[static_cast<std::size_t>(col + 1)]; ++p) {
          const std::int32_t row = host_at_col_val[static_cast<std::size_t>(p)];
          if (host_keep_row[static_cast<std::size_t>(row)] == std::uint8_t{0}) {
            continue;
          }
          indices.push_back(row);
          vals.push_back(host_at_nz_val[static_cast<std::size_t>(p)]);
        }
        append_postsolve_record(plan.tape, PostsolveReductionType::FixedCol, indices, vals,
                                PostsolveDualMode::Minimal);
      }
    }
  }

  _kernel_infeasible_and_fixed_variables_apply<<<blocks_mn, GPU_PRESOLVE_THREADS>>>(
      plan.keep_col_mask,
      plan.new_AL,
      plan.new_AU,
      plan.new_l,
      plan.new_u,
      fixed_mask,
      fixed_val,
      row_shift,
      n,
      m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_infeasible_and_fixed_variables_apply");
  throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_infeasible_and_fixed_variables synchronize");

  const double obj_delta = sum_cleanup_domain(obj_contrib, n, plan.rule_entry_keep_col ? plan.rule_entry_keep_col : plan.cleanup_entry_keep_col,
      [](const double* values, int count) {
        return sum_device_double(values, count, "cudaMemcpy infeasible_and_fixed_variables obj_contrib");
      });
  plan.obj_constant_delta += obj_delta;
  plan.has_col_action = true;
  plan.has_change = true;

}

}  // namespace gpu_presolver::presolve
