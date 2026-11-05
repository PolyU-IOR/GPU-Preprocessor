// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/offset.hpp"
#include <climits>
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define GPUPREPROCESSOR_PROJECTED_HD __host__ __device__
#else
#define GPUPREPROCESSOR_PROJECTED_HD
#endif

namespace gpu_preprocessor::lp::presolve::detail {
// New indices: K, column, rows[K], N, L, (orientation, length)[L], columns[N].
// New values: (constant, coefficients[length])[L]. Legacy records retain their
// expression metadata in values; the integer span distinguishes both layouts.
GPUPREPROCESSOR_PROJECTED_HD inline NnzOffset projected_column_support_begin(
    const std::int32_t* ids, NnzOffset ni) {
  if (!ids || ni < 3 || ids[0] < 0 || NnzOffset(ids[0]) > ni - 3) return -1;
  const NnzOffset header = NnzOffset(ids[0]) + 3;
  const int support = ids[header - 1];
  if (support < 0 || support > ni - header) return -1;
  if (ni - header == support) return header;
  const int expressions = ids[header];
  if (expressions < 0 || NnzOffset(expressions) > (ni - header - 1) / 2) return -1;
  const NnzOffset begin = header + 1 + NnzOffset{2} * expressions;
  return ni - begin == support ? begin : -1;
}
struct ProjectedColumnHeader {
  int expressions = 0;
  NnzOffset support_begin = 0;
  NnzOffset metadata_begin = 0;
  NnzOffset values_begin = 0;
  bool integer_metadata = false;
};
GPUPREPROCESSOR_PROJECTED_HD inline bool projected_column_count(double value, int& out) {
  if (!(value >= 0 && value <= INT_MAX) || std::floor(value) != value) return false;
  out = static_cast<int>(value);
  return true;
}
GPUPREPROCESSOR_PROJECTED_HD inline bool read_projected_column_header(
    const std::int32_t* ids, const double* vals, NnzOffset ni, NnzOffset nv,
    ProjectedColumnHeader& out) {
  out.support_begin = projected_column_support_begin(ids, ni);
  if (out.support_begin < 0 || !vals || nv < 0) return false;
  const NnzOffset header = NnzOffset(ids[0]) + 3;
  out.integer_metadata = out.support_begin != header;
  out.metadata_begin = header + 1;
  out.values_begin = out.integer_metadata ? 0 : 1;
  if (out.integer_metadata) out.expressions = ids[header];
  else if (nv < 1 || !projected_column_count(vals[0], out.expressions)) return false;
  const NnzOffset support = ids[header - 1];
  if (out.integer_metadata) {
    if (support > nv || NnzOffset(out.expressions) != nv - support) return false;
  } else if (support > nv - 1 || NnzOffset(out.expressions) > (nv - 1 - support) / 3 ||
             NnzOffset(out.expressions) * 3 != nv - 1 - support) return false;
  return true;
}
GPUPREPROCESSOR_PROJECTED_HD inline bool read_projected_column_expression(
    const std::int32_t* ids, const double* vals, NnzOffset ni, NnzOffset nv,
    const ProjectedColumnHeader& header, NnzOffset& metadata, NnzOffset& values,
    double& orientation, int& length) {
  if (header.integer_metadata) {
    if (metadata < 0 || metadata > header.support_begin - 2 || values >= nv) return false;
    orientation = ids[metadata++];
    length = ids[metadata++];
  } else {
    if (values < 0 || values > nv - 3) return false;
    orientation = vals[values++];
    if (!projected_column_count(vals[values++], length)) return false;
  }
  return length >= 0 && values < nv && length <= nv - values - 1 &&
         length <= ni - header.support_begin;
}
GPUPREPROCESSOR_PROJECTED_HD inline bool valid_projected_column_record(
    const std::int32_t* ids, const double* vals, NnzOffset ni, NnzOffset nv) {
  ProjectedColumnHeader header;
  if (!read_projected_column_header(ids, vals, ni, nv, header)) return false;
  NnzOffset metadata = header.metadata_begin, values = header.values_begin;
  NnzOffset support = header.support_begin;
  for (int e = 0; e < header.expressions; ++e) {
    double orientation; int length;
    if (!read_projected_column_expression(ids, vals, ni, nv, header, metadata, values,
                                         orientation, length) || length > ni - support) return false;
    support += length; values += NnzOffset(length) + 1;
  }
  return support == ni && values == nv;
}
} // namespace gpu_preprocessor::lp::presolve::detail

#undef GPUPREPROCESSOR_PROJECTED_HD
