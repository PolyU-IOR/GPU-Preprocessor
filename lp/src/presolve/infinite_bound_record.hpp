// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/offset.hpp"
#include <climits>
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define GPUPREPROCESSOR_INFINITE_HD __host__ __device__
#else
#define GPUPREPROCESSOR_INFINITE_HD
#endif

namespace gpu_preprocessor::lp::presolve::detail {
// Each supporting row contributes the same number of integer and value slots.
// Legacy headers have 2/2 slots; integer-count headers have 3/1. The span sizes
// therefore distinguish records individually, including mixed old/new segments.
GPUPREPROCESSOR_INFINITE_HD inline int infinite_bound_index_header(NnzOffset ni, NnzOffset nv) {
  if (ni >= 3 && nv >= 1 && ni - nv == 2) return 3;
  if (ni >= 2 && nv >= 2 && ni == nv) return 2;
  return 0;
}
struct InfiniteBoundHeader {
  std::int32_t rows = 0;
  int indices = 0;
  int values = 0;
};
GPUPREPROCESSOR_INFINITE_HD inline bool read_infinite_bound_header(
    const std::int32_t* ids, const double* vals, NnzOffset ni, NnzOffset nv,
    InfiniteBoundHeader& out) {
  out.indices = infinite_bound_index_header(ni, nv);
  if (!out.indices || !ids || !vals) return false;
  out.values = out.indices == 3 ? 1 : 2;
  if (out.indices == 3) out.rows = ids[2];
  else {
    const double count = vals[0];
    if (!(count >= 0 && count <= INT_MAX) || std::floor(count) != count) return false;
    out.rows = static_cast<std::int32_t>(count);
  }
  return out.rows >= 0 && NnzOffset(out.rows) <= ni - out.indices;
}
GPUPREPROCESSOR_INFINITE_HD inline bool valid_infinite_bound_record(
    const std::int32_t* ids, const double* vals, NnzOffset ni, NnzOffset nv) {
  InfiniteBoundHeader h;
  if (!read_infinite_bound_header(ids, vals, ni, nv, h)) return false;
  NnzOffset ip = h.indices, vp = h.values;
  for (int row = 0; row < h.rows; ++row) {
    if (ip >= ni || vp >= nv) return false;
    const NnzOffset length = ids[ip];
    if (length < 0 || length > ni - ip - 1 || length > nv - vp - 1) return false;
    ip += length + 1;
    vp += length + 1;
  }
  return ip == ni && vp == nv;
}
}  // namespace gpu_preprocessor::lp::presolve::detail

#undef GPUPREPROCESSOR_INFINITE_HD
