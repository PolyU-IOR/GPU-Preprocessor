// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "postsolve_piecewise_linear_reformulation_variable_pair.hpp"
#include "postsolve_common.cuh"

namespace gpu_preprocessor::lp::presolve::postsolve {
namespace {
__global__ void _kernel_cache_variable_pair_root_values(
    double* plus_cache,
    double* minus_cache,
    const double* x_org,
    const std::int32_t* root_plus_cols,
    const std::int32_t* root_minus_cols,
    std::int32_t count) {
  const std::int32_t i =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < count) {
    plus_cache[i] = x_org[root_plus_cols[i]];
    minus_cache[i] = x_org[root_minus_cols[i]];
  }
}

__global__ void _kernel_restore_piecewise_linear_reformulation_variable_pair_members(
    double* x_org,
    const std::int32_t* plus_cols,
    const std::int32_t* minus_cols,
    const double* plus_shift,
    const double* minus_shift,
    const double* plus_cache,
    const double* minus_cache,
    std::int32_t count) {
  const std::int32_t i =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < count) {
    x_org[plus_cols[i]] = plus_shift[i] + plus_cache[i];
    x_org[minus_cols[i]] = minus_shift[i] + minus_cache[i];
  }
}
}  // namespace

void restore_piecewise_linear_reformulation_variable_pair(
    double* x_org,
    const PiecewiseLinearReformulationVariablePairRecoveryStep& recovery) {
  const std::size_t size = recovery.plus_cols.size();
  if (size == 0) {
    return;
  }
  const std::int32_t count = static_cast<std::int32_t>(size);
  RecoveryArray<std::int32_t> plus_cols_storage;
  auto& plus_cols = plus_cols_storage.data;
  plus_cols = copy_vector_to_device(recovery.plus_cols,
                            "cudaMalloc/copy piecewise_linear_reformulation_variable_pair plus cols");
  RecoveryArray<std::int32_t> minus_cols_storage;
  auto& minus_cols = minus_cols_storage.data;
  minus_cols = copy_vector_to_device(recovery.minus_cols,
                            "cudaMalloc/copy piecewise_linear_reformulation_variable_pair minus cols");
  RecoveryArray<std::int32_t> root_plus_cols_storage;
  auto& root_plus_cols = root_plus_cols_storage.data;
  root_plus_cols = copy_vector_to_device(recovery.root_plus_cols,
                            "cudaMalloc/copy piecewise_linear_reformulation_variable_pair root plus cols");
  RecoveryArray<std::int32_t> root_minus_cols_storage;
  auto& root_minus_cols = root_minus_cols_storage.data;
  root_minus_cols = copy_vector_to_device(recovery.root_minus_cols,
                            "cudaMalloc/copy piecewise_linear_reformulation_variable_pair root minus cols");
  RecoveryArray<double> plus_shift_storage;
  auto& plus_shift = plus_shift_storage.data;
  plus_shift = copy_vector_to_device(recovery.plus_shift,
                            "cudaMalloc/copy piecewise_linear_reformulation_variable_pair plus lower");
  RecoveryArray<double> minus_shift_storage;
  auto& minus_shift = minus_shift_storage.data;
  minus_shift = copy_vector_to_device(recovery.minus_shift,
                            "cudaMalloc/copy piecewise_linear_reformulation_variable_pair minus lower");
  RecoveryArray<double> plus_cache_storage;
  auto& plus_cache = plus_cache_storage.data;
  RecoveryArray<double> minus_cache_storage;
  auto& minus_cache = minus_cache_storage.data;
  throw_if_cuda_error(cudaMalloc(&plus_cache, sizeof(double) * size),
                      "cudaMalloc piecewise_linear_reformulation_variable_pair plus cache");
  throw_if_cuda_error(cudaMalloc(&minus_cache, sizeof(double) * size),
                      "cudaMalloc piecewise_linear_reformulation_variable_pair minus cache");
  const int blocks =
      (count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_cache_variable_pair_root_values<<<blocks, GPU_PRESOLVE_THREADS>>>(
      plus_cache, minus_cache, x_org, root_plus_cols, root_minus_cols, count);
  throw_if_cuda_error(cudaGetLastError(),
                      "_kernel_cache_variable_pair_root_values");
  _kernel_restore_piecewise_linear_reformulation_variable_pair_members<<<blocks,
                                                GPU_PRESOLVE_THREADS>>>(
      x_org, plus_cols, minus_cols, plus_shift, minus_shift, plus_cache,
      minus_cache, count);
  throw_if_cuda_error(cudaGetLastError(),
                      "_kernel_restore_piecewise_linear_reformulation_variable_pair_members");
}

void restore_device_piecewise_linear_reformulation_variable_pair(double *x, const DeviceRecoveryGroup &g) {
  int n = g.ids[0].count;
  if (!n)
    return;
  RecoveryArray<double> plus_storage, minus_storage;
  auto& plus = plus_storage.data;
  auto& minus = minus_storage.data;
  throw_if_cuda_error(cudaMalloc(&plus, sizeof(double) * std::size_t(n)), "allocate plus cache");
  throw_if_cuda_error(cudaMalloc(&minus, sizeof(double) * std::size_t(n)), "allocate minus cache");
  _kernel_cache_variable_pair_root_values<<<(n + 255) / 256, 256>>>(plus, minus, x, g.ids[2].data,
                                                                g.ids[3].data, n);
  _kernel_restore_piecewise_linear_reformulation_variable_pair_members<<<(n + 255) / 256, 256>>>(
      x, g.ids[0].data, g.ids[1].data, g.values[0].data, g.values[1].data, plus, minus, n);
  throw_if_cuda_error(cudaGetLastError(), "restore device piecewise_linear_reformulation_variable_pair payload");
}

void validate_piecewise_linear_reformulation_variable_pair_recovery(
    const PiecewiseLinearReformulationVariablePairRecoveryStep& recovery,
    std::int32_t n0) {
  const std::size_t count = recovery.plus_cols.size();
  if (recovery.minus_cols.size() != count ||
      recovery.root_plus_cols.size() != count ||
      recovery.root_minus_cols.size() != count ||
      recovery.plus_shift.size() != count ||
      recovery.minus_shift.size() != count ||
      count > static_cast<std::size_t>(
                  std::numeric_limits<std::int32_t>::max())) {
    throw std::runtime_error("piecewise_linear_reformulation_variable_pair recovery has mismatched payloads");
  }
  for (std::size_t i = 0; i < count; ++i) {
    validate_recovery_col(recovery.plus_cols[i], n0,
                          "piecewise_linear_reformulation_variable_pair recovery");
    validate_recovery_col(recovery.minus_cols[i], n0,
                          "piecewise_linear_reformulation_variable_pair recovery");
    validate_recovery_col(recovery.root_plus_cols[i], n0,
                          "piecewise_linear_reformulation_variable_pair recovery");
    validate_recovery_col(recovery.root_minus_cols[i], n0,
                          "piecewise_linear_reformulation_variable_pair recovery");
  }
}

}  // namespace gpu_preprocessor::lp::presolve::postsolve
