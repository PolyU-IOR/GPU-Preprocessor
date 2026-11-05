// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_preprocessor::lp::presolve::postsolve {
inline constexpr int GPU_PRESOLVE_THREADS = 256;

inline void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

template <class T>
T* copy_vector_to_device(const std::vector<T>& values, const char* context) {
  if (values.empty()) {
    return nullptr;
  }
  RecoveryArray<T> device;
  throw_if_cuda_error(cudaMalloc(&device.data, sizeof(T) * values.size()), context);
  throw_if_cuda_error(cudaMemcpy(device.data, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice),
                      context);
  return std::exchange(device.data, nullptr);
}

__device__ __forceinline__ bool _postsolve_is_finite(double x) {
  return x > -INFINITY && x < INFINITY;
}

__device__ __forceinline__ bool _postsolve_at_lower(double x, double l, double tol) {
  return _postsolve_is_finite(l) && x <= l + tol;
}

__device__ __forceinline__ bool _postsolve_at_upper(double x, double u, double tol) {
  return _postsolve_is_finite(u) && x >= u - tol;
}

__device__ __forceinline__ double _clamp_with_inf(double x, double lo, double hi) {
  double y = x;
  if (_postsolve_is_finite(lo)) {
    y = fmax(y, lo);
  }
  if (_postsolve_is_finite(hi)) {
    y = fmin(y, hi);
  }
  return y;
}

inline void validate_recovery_col(std::int32_t col,
                           std::int32_t n0,
                           const char* context) {
  if (col < 0 || col >= n0) {
    throw std::runtime_error(std::string(context) +
                             " contains an invalid original column");
  }
}

__device__ __forceinline__ void replay_row_dual_shift(
    double* y_org,
    double* z_org,
    const std::int32_t* tape_idx,
    const double* tape_val,
    NnzOffset idx0,
    NnzOffset idx1,
    NnzOffset val0) {
  if (y_org != nullptr) {
    const std::int32_t row = tape_idx[idx0];
    if (idx1 >= idx0 + 1) {
      const int count = tape_idx[idx0 + 1];
      const double before = y_org[row];
      const double after = tape_val[val0] > 0.0 ? fmax(before, 0.0) : fmin(before, 0.0);
      const double delta = after - before;
      y_org[row] = after;
      if (z_org != nullptr && delta != 0.0) {
        for (int t = 0; t < count; ++t)
          z_org[tape_idx[idx0 + 2 + t]] -= tape_val[val0 + 1 + t] * delta;
      }
    } else {
      // Legacy records contain only an additive row update.
      y_org[row] += tape_val[val0];
    }
  }
}

}  // namespace gpu_preprocessor::lp::presolve::postsolve
