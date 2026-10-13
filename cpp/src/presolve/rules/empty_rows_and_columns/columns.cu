#include "gpu_presolver/presolve/cleanup_workspace.hpp"
#include "gpu_presolver/presolve/cleanup_support.cuh"
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/rules/rule_empty_rows_and_columns.hpp"

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

__global__ void _kernel_empty_columns_classify(std::int32_t* flags,
                                            std::uint8_t* fixed_mask,
                                            double* fixed_val,
                                            double* obj_contrib,
                                            const std::uint8_t* keep_col,
                                            const std::uint8_t* empty_col_mask,
                                            const double* c,
                                            const double* l,
                                            const double* u,
                                            double bound_tol,
                                            double zero_tol,
                                            std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j < n) {
    fixed_mask[j] = std::uint8_t{0};
    fixed_val[j] = 0.0;
    obj_contrib[j] = 0.0;
    const bool active = keep_col[j] != std::uint8_t{0} && empty_col_mask[j] != std::uint8_t{0};
    if (!active) {
      return;
    }
    const double cj = c[j];
    const double lj = l[j];
    const double uj = u[j];
    if (lj > uj + bound_tol) {
      atomicMax(&flags[0], 1);
      return;
    }
    const bool unbounded_lower = cj > zero_tol && !isfinite(lj);
    const bool unbounded_upper = cj < -zero_tol && !isfinite(uj);
    if (unbounded_lower || unbounded_upper) {
      atomicMax(&flags[1], 1);
      return;
    }

    double value = 0.0;
    if (cj > zero_tol) {
      value = lj;
    } else if (cj < -zero_tol) {
      value = uj;
    } else if (isfinite(lj)) {
      value = lj;
    } else if (isfinite(uj)) {
      value = uj;
    }
    fixed_mask[j] = std::uint8_t{1};
    fixed_val[j] = value;
    obj_contrib[j] = cj * value;
    atomicMax(&flags[2], 1);
  }
}

__global__ void _kernel_empty_columns_apply(std::uint8_t* keep_col,
                                         double* new_l,
                                         double* new_u,
                                         const std::uint8_t* fixed_mask,
                                         const double* fixed_val,
                                         std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j < n && fixed_mask[j] != std::uint8_t{0}) {
    new_l[j] = fixed_val[j];
    new_u[j] = fixed_val[j];
    keep_col[j] = std::uint8_t{0};
  }
}

}  // namespace

std::size_t cleanup_reduction_temp_bytes(const double* input,
                                         double* output, int count) {
  if (count <= 0) return 0;
  void* storage = nullptr;
  std::size_t bytes = 0;
  throw_if_cuda_error(cub::DeviceReduce::Sum(storage, bytes, input, output, count),
                      "query cleanup reduction workspace");
  return bytes;
}

std::size_t cleanup_selection_temp_bytes(const double* input,
                                         const std::uint8_t* flags,
                                         double* output,
                                         int* selected_count,
                                         int count) {
  if (count <= 0) return 0;
  std::size_t bytes = 0;
  throw_if_cuda_error(cub::DeviceSelect::Flagged(
      nullptr, bytes, input, flags, output, selected_count, count),
      "query cleanup selection workspace");
  return bytes;
}

void apply_rule_empty_columns(PresolvePlanGpu& plan,
                           const LPInfoGpu& lp,
                           const PresolveStatsGpu& stats,
                           const PresolveParams& pparams) {
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }

  const std::int32_t n = lp.A.cols;
  if (n == 0) {
    return;
  }

  CleanupScratchFrame scratch;
  std::int32_t* flags_device = nullptr;
  std::uint8_t* fixed_mask = nullptr;
  double* fixed_val = nullptr;
  double* obj_contrib = nullptr;
  flags_device = scratch.get<std::int32_t>(5, 3);
  fixed_mask = scratch.get<std::uint8_t>(2, n);
  fixed_val = scratch.get<double>(0, n);
  obj_contrib = scratch.get<double>(1, n);
  throw_if_cuda_error(cudaMemset(flags_device, 0, sizeof(std::int32_t) * 3), "cudaMemset empty_columns flags");

  const int blocks = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_empty_columns_classify<<<blocks, GPU_PRESOLVE_THREADS>>>(
      flags_device,
      fixed_mask,
      fixed_val,
      obj_contrib,
      plan.keep_col_mask,
      stats.empty_col_mask,
      plan.new_c,
      plan.new_l,
      plan.new_u,
      pparams.bound_tol,
      pparams.zero_tol,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_empty_columns_classify");

  std::int32_t flags[3] = {0, 0, 0};
  throw_if_cuda_error(cudaMemcpy(flags, flags_device, sizeof(flags), cudaMemcpyDeviceToHost),
                      "cudaMemcpy empty_columns flags");
  if (flags[0] != 0) {
    plan.has_infeasible = true;
  } else if (flags[1] != 0) {
    plan.has_unbounded = true;
  } else if (flags[2] != 0) {
    if (pparams.record_postsolve_tape && pparams.record_postsolve_tape_cpu) {
      if (plan.tape.resident) {
        append_simple_device_recovery(plan.tape, SimpleRecovery::FixedCol, fixed_mask, nullptr, n,
                                      nullptr, fixed_val, plan.new_c, nullptr);
      } else {

        std::vector<std::uint8_t> host_fixed_mask(static_cast<std::size_t>(n));
        std::vector<double> host_fixed_val(static_cast<std::size_t>(n));
        std::vector<double> host_c(static_cast<std::size_t>(n));
        throw_if_cuda_error(cudaMemcpy(host_fixed_mask.data(), fixed_mask,
                                       static_cast<std::size_t>(n), cudaMemcpyDeviceToHost),
                            "cudaMemcpy empty_columns fixed_mask for tape");
        throw_if_cuda_error(cudaMemcpy(host_fixed_val.data(), fixed_val,
                                       sizeof(double) * static_cast<std::size_t>(n),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy empty_columns fixed_val for tape");
        throw_if_cuda_error(cudaMemcpy(host_c.data(), plan.new_c,
                                       sizeof(double) * static_cast<std::size_t>(n),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy empty_columns c for tape");
        for (std::int32_t col = 0; col < n; ++col) {
          if (host_fixed_mask[static_cast<std::size_t>(col)] == std::uint8_t{0}) {
            continue;
          }
          append_postsolve_record(plan.tape, PostsolveReductionType::FixedCol,
                                  std::vector<std::int32_t>{col},
                                  std::vector<double>{host_fixed_val[static_cast<std::size_t>(col)],
                                                      host_c[static_cast<std::size_t>(col)]},
                                  PostsolveDualMode::Minimal);
        }
      }
    }
    _kernel_empty_columns_apply<<<blocks, GPU_PRESOLVE_THREADS>>>(
        plan.keep_col_mask, plan.new_l, plan.new_u, fixed_mask, fixed_val, n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_empty_columns_apply");
    throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_empty_columns synchronize");
    const auto* domain = plan.cleanup_dual_keep_col ? plan.cleanup_dual_keep_col
                                                    : (plan.cleanup_prefix_keep_col ? plan.cleanup_prefix_keep_col : plan.cleanup_entry_keep_col);
    const double obj_delta = sum_cleanup_domain(obj_contrib, n, plan.rule_entry_keep_col ? plan.rule_entry_keep_col : domain,
        [](const double* values, int count) {
          return sum_device_double(values, count, "cudaMemcpy empty_columns obj_contrib");
        });
    plan.obj_constant_delta += obj_delta;
    plan.has_col_action = true;
    plan.has_change = true;
  }

}

}  // namespace gpu_presolver::presolve
