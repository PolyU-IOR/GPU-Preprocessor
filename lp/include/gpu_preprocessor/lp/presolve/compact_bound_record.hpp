// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once
#include "gpu_preprocessor/lp/offset.hpp"
#include <cstdint>
#include <cstring>

namespace gpu_preprocessor::lp::presolve {
// Column, two source slots, and two batch-relative coefficient references.
// Wide builds retain both words of each reference; row/column IDs stay int32.
inline constexpr int compact_bound_indices = sizeof(NnzOffset) == 8 ? 7 : 5;
#if defined(__CUDACC__)
#define GPUP_BOUND_HD __host__ __device__
#else
#define GPUP_BOUND_HD
#endif
GPUP_BOUND_HD inline NnzOffset compact_bound_reference(const int* ids, int side) {
  if constexpr (sizeof(NnzOffset) == 4) return ids[3 + side];
  if (ids[3 + side] == -1 && ids[5 + side] == -1) return -1;
  const auto bits = std::uint64_t(std::uint32_t(ids[3 + side])) |
                    (std::uint64_t(std::uint32_t(ids[5 + side])) << 32);
  return static_cast<NnzOffset>(bits);
}
GPUP_BOUND_HD inline void set_compact_bound_reference(int* ids, int side, NnzOffset ref) {
  const auto low = std::uint32_t(ref);
  memcpy(ids + 3 + side, &low, sizeof(low));
  if constexpr (sizeof(NnzOffset) == 8) {
    const auto high = std::uint32_t(std::uint64_t(ref) >> 32);
    memcpy(ids + 5 + side, &high, sizeof(high));
  }
}
#undef GPUP_BOUND_HD
} // namespace gpu_preprocessor::lp::presolve
