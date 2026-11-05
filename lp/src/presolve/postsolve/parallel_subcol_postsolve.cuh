// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gpu_preprocessor::lp::presolve::parallel_subcol_detail {

constexpr NnzOffset min_records = 1024;
inline void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("parallel SingletonColumnSubstitution replay: ") + cudaGetErrorString(e));
}

// Increasing rows give disjoint y updates. Atomic integer marks establish
// distinct x/z targets without requiring the target columns to be sorted.
__global__ void certify_targets(int* serial_only, int* targets,
    const int* types, const NnzOffset* is, const NnzOffset* vs,
    const int* ids, const double* values, const double* y,
    NnzOffset begin, NnzOffset end, int m, int n) {
  const auto stride = std::uint64_t(blockDim.x) * gridDim.x;
  for (auto k = std::uint64_t(begin) + std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       k < std::uint64_t(end); k += stride) {
    const auto ip = is[k], vp = vs[k];
    if (types[k] != int(PostsolveReductionType::SingletonColumnSubstitution) ||
        is[k + 1] - ip < 3 || vs[k + 1] - vp < 6) {
      atomicExch(serial_only, 1); continue;
    }
    const int col = ids[ip], row = ids[ip + 1], count = ids[ip + 2];
    if (col < 0 || col >= n || row < 0 || row >= m || count < 0 ||
        is[k + 1] - ip - 3 != count || vs[k + 1] - vp - 6 != count ||
        values[vp] == 0.0 || !isfinite(values[vp]) ||
        !isfinite(values[vp + 1]) || !isfinite(values[vp + 4]) ||
        (y && !isfinite(y[row]))) {
      atomicExch(serial_only, 1); continue;
    }
    if (k > std::uint64_t(begin) &&
        (types[k - 1] != int(PostsolveReductionType::SingletonColumnSubstitution) ||
         is[k] - is[k - 1] < 3 || ids[is[k - 1] + 1] >= row))
      atomicExch(serial_only, 1);
    if (atomicExch(targets + col, 1)) atomicExch(serial_only, 1);
  }
}

__global__ void certify_support(int* serial_only, const int* targets,
    const int* types, const NnzOffset* is, const NnzOffset* vs,
    const int* ids, const double* values, const double* x,
    NnzOffset begin, NnzOffset end, int n) {
  const auto stride = std::uint64_t(blockDim.x) * gridDim.x;
  for (auto k = std::uint64_t(begin) + std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       k < std::uint64_t(end); k += stride) {
    const auto ip = is[k], vp = vs[k];
    if (types[k] != int(PostsolveReductionType::SingletonColumnSubstitution) ||
        is[k + 1] - ip < 3 || vs[k + 1] - vp < 6) continue;
    const int count = ids[ip + 2];
    if (count < 0 || is[k + 1] - ip - 3 != count || vs[k + 1] - vp - 6 != count) continue;
    for (int q = 0; q < count; ++q) {
      const int col = ids[ip + 3 + q];
      if (col < 0 || col >= n || targets[col] ||
          !isfinite(values[vp + 6 + q]) || !isfinite(x[col]))
        atomicExch(serial_only, 1);
    }
  }
}

__global__ void replay(const int* serial_only, double* x, double* y, double* z,
    std::uint8_t* retrieved, const NnzOffset* is, const NnzOffset* vs,
    const int* ids, const double* values, NnzOffset begin, NnzOffset end) {
  if (*serial_only) return;
  const auto stride = std::uint64_t(blockDim.x) * gridDim.x;
  for (auto k = std::uint64_t(begin) + std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       k < std::uint64_t(end); k += stride) {
    const auto ip = is[k], vp = vs[k];
    const int col = ids[ip], row = ids[ip + 1], count = ids[ip + 2];
    const double pivot = values[vp], rhs = values[vp + 1], cost = values[vp + 4];
    const bool deleted = values[vp + 5] != 0.0;
    double partial = 0.0;
    for (int q = 0; q < count; ++q) {
      const int support_col = ids[ip + 3 + q];
      const double coefficient = values[vp + 6 + q];
      partial += coefficient * x[support_col];
    }
    // Preserve the scalar SingletonColumnSubstitution formula and operation order within a record.
    x[col] = (rhs - partial) / pivot;
    if (y != nullptr && row >= 0) {
      if (deleted) y[row] = cost / pivot;
      else y[row] += cost / pivot;
    }
    if (z != nullptr) {
      z[col] = deleted || y == nullptr ? 0.0 : cost - pivot * y[row];
      if (retrieved != nullptr) retrieved[col] = std::uint8_t{1};
    }
  }
}

inline bool try_replay(int* serial_only, double* x, double* y, double* z,
    std::uint8_t* retrieved, const int* types, const NnzOffset* is, const NnzOffset* vs,
    const int* ids, const double* values, NnzOffset begin, NnzOffset end, int m, int n) {
  if (!serial_only || !x || n <= 0 || end - begin < min_records) return false;
  int last = -1;
  check(cudaMemcpyAsync(&last, types + end - 1, sizeof(int), cudaMemcpyDeviceToHost));
  check(cudaStreamSynchronize(nullptr));
  if (last != int(PostsolveReductionType::SingletonColumnSubstitution)) return false;
  RecoveryArray<int> targets;
  const auto status = cudaMalloc(&targets.data, std::size_t(n) * sizeof(int));
  if (status == cudaErrorMemoryAllocation) {
    if (cudaPeekAtLastError() == cudaErrorMemoryAllocation) cudaGetLastError();
    return false;
  }
  check(status);
  check(cudaMemsetAsync(serial_only, 0, sizeof(int)));
  check(cudaMemsetAsync(targets.data, 0, std::size_t(n) * sizeof(int)));
  const int blocks = checked_launch_blocks(std::min(end - begin, NnzOffset(256) * 1024), 256);
  certify_targets<<<blocks, 256>>>(serial_only, targets.data, types, is, vs, ids, values, y, begin, end, m, n);
  certify_support<<<blocks, 256>>>(serial_only, targets.data, types, is, vs, ids, values, x, begin, end, n);
  replay<<<blocks, 256>>>(serial_only, x, y, z, retrieved, is, vs, ids, values, begin, end);
  check(cudaGetLastError());
  return true;
}

} // namespace gpu_preprocessor::lp::presolve::parallel_subcol_detail
