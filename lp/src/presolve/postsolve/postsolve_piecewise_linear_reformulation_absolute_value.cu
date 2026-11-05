// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "postsolve_piecewise_linear_reformulation_absolute_value.hpp"
#include "postsolve_common.cuh"

namespace gpu_preprocessor::lp::presolve::postsolve {
namespace {
__global__ void _kernel_structural_restore_splits(double* x_org,
                                                  const std::int32_t* t_cols,
                                                  const std::int32_t* e_cols,
                                                  const double* rhos,
                                                  std::int32_t k) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i >= k) {
    return;
  }
  const std::int32_t t_idx = t_cols[i];
  const std::int32_t e_idx = e_cols[i];
  const double rho = rhos[i];
  const double t_new = x_org[t_idx];
  const double e_new = x_org[e_idx];
  x_org[t_idx] = t_new + e_new;
  x_org[e_idx] = (t_new - e_new) / rho;
}

__global__ void _kernel_structural_restore_outer_pairs(double* x_org,
                                                       const std::int32_t* bound_cols,
                                                       const std::int32_t* free_cols,
                                                       std::int32_t k) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < k) {
    x_org[free_cols[i]] = x_org[bound_cols[i]];
  }
}

__global__ void _kernel_structural_restore_linked_slacks(double* x_org,
                                                         const std::int32_t* slack_cols,
                                                         const std::int32_t* t_cols,
                                                         const double* factors,
                                                         std::int32_t k) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < k) {
    x_org[slack_cols[i]] = factors[i] * x_org[t_cols[i]];
  }
}

__global__ void _kernel_structural_restore_max_slack(double* x_org,
                                                     std::int32_t slack_col,
                                                     const std::int32_t* t_cols,
                                                     const double* factors,
                                                     std::int32_t count,
                                                     double lower_bound) {
  const int tid = threadIdx.x;
  const int lane_count = blockDim.x;
  double partial = lower_bound;
  for (std::int32_t j = tid; j < count; j += lane_count) {
    partial = fmax(partial, factors[j] * x_org[t_cols[j]]);
  }

  __shared__ double scratch[GPU_PRESOLVE_THREADS];
  scratch[tid] = partial;
  __syncthreads();

  for (NnzOffset offset = lane_count >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      scratch[tid] = fmax(scratch[tid], scratch[tid + offset]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    x_org[slack_col] = scratch[0];
  }
}

__global__ void _kernel_device_restore_max_slack(double *x_org, const std::int32_t *slack_id,
                                                 const std::int32_t *t_cols, const double *factors,
                                                 std::int32_t count, const double *original_l) {
  const int slack_col = slack_id[0];
  double lower_bound = original_l ? original_l[slack_col] : 0.0;
  if (!isfinite(lower_bound))
    lower_bound = 0.0;
  const int tid = threadIdx.x;
  const int lane_count = blockDim.x;
  double partial = lower_bound;
  for (std::int32_t j = tid; j < count; j += lane_count) {
    partial = fmax(partial, factors[j] * x_org[t_cols[j]]);
  }

  __shared__ double scratch[GPU_PRESOLVE_THREADS];
  scratch[tid] = partial;
  __syncthreads();

  for (NnzOffset offset = lane_count >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      scratch[tid] = fmax(scratch[tid], scratch[tid + offset]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    x_org[slack_col] = scratch[0];
  }
}
}  // namespace

void restore_splits(double* x_org, const std::vector<PiecewiseLinearReformulationAbsoluteValueSplitRecovery>& splits) {
  if (splits.empty()) {
    return;
  }
  std::vector<std::int32_t> t_cols;
  std::vector<std::int32_t> e_cols;
  std::vector<double> rhos;
  t_cols.reserve(splits.size());
  e_cols.reserve(splits.size());
  rhos.reserve(splits.size());
  for (const PiecewiseLinearReformulationAbsoluteValueSplitRecovery& split : splits) {
    t_cols.push_back(split.t_col);
    e_cols.push_back(split.e_col);
    rhos.push_back(split.rho);
  }
  RecoveryArray<std::int32_t> t_d_storage;
  auto& t_d = t_d_storage.data;
  t_d = copy_vector_to_device(t_cols, "cudaMalloc/copy piecewise_linear_reformulation_absolute_value split t_cols");
  RecoveryArray<std::int32_t> e_d_storage;
  auto& e_d = e_d_storage.data;
  e_d = copy_vector_to_device(e_cols, "cudaMalloc/copy piecewise_linear_reformulation_absolute_value split e_cols");
  RecoveryArray<double> rho_d_storage;
  auto& rho_d = rho_d_storage.data;
  rho_d = copy_vector_to_device(rhos, "cudaMalloc/copy piecewise_linear_reformulation_absolute_value split rhos");
  const std::int32_t k = static_cast<std::int32_t>(splits.size());
  const int blocks = (k + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_structural_restore_splits<<<blocks, GPU_PRESOLVE_THREADS>>>(x_org, t_d, e_d, rho_d, k);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_restore_splits");
}

void restore_outer_pairs(double* x_org, const std::vector<AbsoluteValueOuterPairRecovery>& pairs) {
  if (pairs.empty()) {
    return;
  }
  std::vector<std::int32_t> bound_cols;
  std::vector<std::int32_t> free_cols;
  bound_cols.reserve(pairs.size());
  free_cols.reserve(pairs.size());
  for (const AbsoluteValueOuterPairRecovery& pair : pairs) {
    bound_cols.push_back(pair.bound_col);
    free_cols.push_back(pair.free_col);
  }
  RecoveryArray<std::int32_t> bound_d_storage;
  auto& bound_d = bound_d_storage.data;
  bound_d = copy_vector_to_device(bound_cols, "cudaMalloc/copy piecewise_linear_reformulation_absolute_value outer bound_cols");
  RecoveryArray<std::int32_t> free_d_storage;
  auto& free_d = free_d_storage.data;
  free_d = copy_vector_to_device(free_cols, "cudaMalloc/copy piecewise_linear_reformulation_absolute_value outer free_cols");
  const std::int32_t k = static_cast<std::int32_t>(pairs.size());
  const int blocks = (k + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_structural_restore_outer_pairs<<<blocks, GPU_PRESOLVE_THREADS>>>(x_org, bound_d, free_d, k);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_restore_outer_pairs");
}

void restore_linked_slacks(double* x_org, const std::vector<AbsoluteValueLinkedSlackRecovery>& slacks) {
  if (slacks.empty()) {
    return;
  }
  std::vector<std::int32_t> slack_cols;
  std::vector<std::int32_t> t_cols;
  std::vector<double> factors;
  slack_cols.reserve(slacks.size());
  t_cols.reserve(slacks.size());
  factors.reserve(slacks.size());
  for (const AbsoluteValueLinkedSlackRecovery& slack : slacks) {
    slack_cols.push_back(slack.slack_col);
    t_cols.push_back(slack.t_col);
    factors.push_back(slack.factor);
  }
  RecoveryArray<std::int32_t> slack_d_storage;
  auto& slack_d = slack_d_storage.data;
  slack_d = copy_vector_to_device(slack_cols, "cudaMalloc/copy piecewise_linear_reformulation_absolute_value linked slack_cols");
  RecoveryArray<std::int32_t> t_d_storage;
  auto& t_d = t_d_storage.data;
  t_d = copy_vector_to_device(t_cols, "cudaMalloc/copy piecewise_linear_reformulation_absolute_value linked t_cols");
  RecoveryArray<double> factor_d_storage;
  auto& factor_d = factor_d_storage.data;
  factor_d = copy_vector_to_device(factors, "cudaMalloc/copy piecewise_linear_reformulation_absolute_value linked factors");
  const std::int32_t k = static_cast<std::int32_t>(slacks.size());
  const int blocks = (k + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_structural_restore_linked_slacks<<<blocks, GPU_PRESOLVE_THREADS>>>(x_org, slack_d, t_d, factor_d, k);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_restore_linked_slacks");
}

void restore_max_slacks(double* x_org,
                        const std::vector<AbsoluteValueMaxSlackRecovery>& slacks,
                        const double* original_l) {
  for (const AbsoluteValueMaxSlackRecovery& slack : slacks) {
    if (slack.t_cols.empty()) {
      continue;
    }
    double lower_bound = 0.0;
    if (original_l != nullptr) {
      throw_if_cuda_error(cudaMemcpy(&lower_bound, original_l + slack.slack_col, sizeof(double),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy piecewise_linear_reformulation_absolute_value max slack lower bound");
      if (!isfinite(lower_bound)) {
        lower_bound = 0.0;
      }
    }
    RecoveryArray<std::int32_t> t_d_storage;
    auto& t_d = t_d_storage.data;
    t_d = copy_vector_to_device(slack.t_cols, "cudaMalloc/copy piecewise_linear_reformulation_absolute_value max t_cols");
    RecoveryArray<double> factor_d_storage;
    auto& factor_d = factor_d_storage.data;
    factor_d = copy_vector_to_device(slack.factors, "cudaMalloc/copy piecewise_linear_reformulation_absolute_value max factors");
    _kernel_structural_restore_max_slack<<<1, GPU_PRESOLVE_THREADS>>>(
        x_org, slack.slack_col, t_d, factor_d, static_cast<std::int32_t>(slack.t_cols.size()), lower_bound);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_restore_max_slack");
  }
}

void restore_piecewise_linear_reformulation_absolute_value_step(
    double* x_org,
    const PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep& recovery,
    const double* original_l) {
  restore_splits(x_org, recovery.splits);
  restore_outer_pairs(x_org, recovery.outer_pairs);
  restore_linked_slacks(x_org, recovery.linked_slacks);
  restore_max_slacks(x_org, recovery.max_slacks, original_l);
}

void restore_device_structural(double *x, const DevicePiecewiseLinearReformulationAbsoluteValueStep &s, const double *lower) {
  const auto &a = s.splits;
  int n = a.ids[0].count;
  if (n)
    _kernel_structural_restore_splits<<<(n + 255) / 256, 256>>>(x, a.ids[0].data, a.ids[1].data,
                                                                a.values[0].data, n);
  const auto &b = s.pairs;
  n = b.ids[0].count;
  if (n)
    _kernel_structural_restore_outer_pairs<<<(n + 255) / 256, 256>>>(x, b.ids[0].data,
                                                                     b.ids[1].data, n);
  const auto &c = s.linked;
  n = c.ids[0].count;
  if (n)
    _kernel_structural_restore_linked_slacks<<<(n + 255) / 256, 256>>>(
        x, c.ids[0].data, c.ids[1].data, c.values[0].data, n);
  for (const auto &g : s.max_slacks)
    if (g.ids[1].count)
      _kernel_device_restore_max_slack<<<1, 256>>>(x, g.ids[0].data, g.ids[1].data,
                                                   g.values[0].data, g.ids[1].count, lower);
  throw_if_cuda_error(cudaGetLastError(), "restore device piecewise_linear_reformulation_absolute_value payload");
}

void validate_piecewise_linear_reformulation_absolute_value_recovery(
    const PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep& recovery,
    std::int32_t n0) {
  for (const PiecewiseLinearReformulationAbsoluteValueSplitRecovery& split : recovery.splits) {
    validate_recovery_col(split.t_col, n0, "piecewise_linear_reformulation_absolute_value split recovery");
    validate_recovery_col(split.e_col, n0, "piecewise_linear_reformulation_absolute_value split recovery");
  }
  for (const AbsoluteValueOuterPairRecovery& pair : recovery.outer_pairs) {
    validate_recovery_col(pair.bound_col, n0,
                          "piecewise_linear_reformulation_absolute_value outer-pair recovery");
    validate_recovery_col(pair.free_col, n0,
                          "piecewise_linear_reformulation_absolute_value outer-pair recovery");
  }
  for (const AbsoluteValueLinkedSlackRecovery& slack :
       recovery.linked_slacks) {
    validate_recovery_col(slack.slack_col, n0,
                          "piecewise_linear_reformulation_absolute_value linked-slack recovery");
    validate_recovery_col(slack.t_col, n0,
                          "piecewise_linear_reformulation_absolute_value linked-slack recovery");
  }
  for (const AbsoluteValueMaxSlackRecovery& slack : recovery.max_slacks) {
    if (slack.t_cols.size() != slack.factors.size()) {
      throw std::runtime_error(
          "piecewise_linear_reformulation_absolute_value max-slack recovery has mismatched payloads");
    }
    validate_recovery_col(slack.slack_col, n0,
                          "piecewise_linear_reformulation_absolute_value max-slack recovery");
    for (const std::int32_t col : slack.t_cols) {
      validate_recovery_col(col, n0, "piecewise_linear_reformulation_absolute_value max-slack recovery");
    }
  }
}

}  // namespace gpu_preprocessor::lp::presolve::postsolve
