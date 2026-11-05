// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace gpu_preprocessor::lp::presolve::parallel_fixed_detail {

constexpr NnzOffset min_records = 1024;

// A homogeneous fixed-column range only reads y and writes distinct x/z slots.
// Strictly increasing target IDs certify uniqueness without a sort or host copy.
__global__ void certify(int* serial_only, const std::int32_t* types,
                        const NnzOffset* starts, const std::int32_t* indices,
                        NnzOffset begin, NnzOffset end) {
  const auto stride = std::uint64_t(blockDim.x) * gridDim.x;
  for (auto k = std::uint64_t(begin) + std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       k < std::uint64_t(end); k += stride) {
    if (types[k] != static_cast<std::int32_t>(PostsolveReductionType::FixedVariable) ||
        (k > std::uint64_t(begin) && indices[starts[k - 1]] >= indices[starts[k]])) {
      atomicExch(serial_only, 1);
      return;
    }
  }
}

__global__ void replay(const int* serial_only, double* x, const double* y,
                       double* z, std::uint8_t* retrieved,
                       const NnzOffset* index_starts, const NnzOffset* value_starts,
                       const std::int32_t* indices, const double* values,
                       NnzOffset begin, NnzOffset end, bool primal, bool dual) {
  if (*serial_only) return;
  const auto stride = std::uint64_t(blockDim.x) * gridDim.x;
  for (auto k = std::uint64_t(begin) + std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       k < std::uint64_t(end); k += stride) {
    const NnzOffset idx0 = index_starts[k], idx1 = index_starts[k + 1] - 1;
    const NnzOffset val0 = value_starts[k];
    const std::int32_t col = indices[idx0];
    if (primal) x[col] = values[val0];
    if (dual && z != nullptr) {
      // Match FixedVariable's scalar subtraction order; no cross-column reduction.
      double zj = values[val0 + 1];
      for (NnzOffset pos = idx0 + 1; pos <= idx1; ++pos) {
        const std::int32_t row = indices[pos];
        const double coeff = values[val0 + 2 + (pos - (idx0 + 1))];
        if (y != nullptr) zj -= coeff * y[row];
      }
      z[col] = zj;
      if (retrieved != nullptr) retrieved[col] = std::uint8_t{1};
    }
  }
}

} // namespace gpu_preprocessor::lp::presolve::parallel_fixed_detail
