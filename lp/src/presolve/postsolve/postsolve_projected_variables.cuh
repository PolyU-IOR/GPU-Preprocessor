// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "postsolve_common.cuh"
#include "../projected_column_record.hpp"

namespace gpu_preprocessor::lp::presolve::postsolve {

__device__ __forceinline__ void replay_projected_column(
    double* x_org,
    double* y_org,
    double* z_org,
    std::uint8_t* z_retrieved,
    const std::int32_t* tape_idx,
    const double* tape_val,
    std::int32_t replay_primal,
    std::int32_t replay_dual,
    NnzOffset idx0,
    NnzOffset val0,
    NnzOffset index_count,
    NnzOffset value_count) {
  const std::int32_t K = tape_idx[idx0];
  const std::int32_t elim_col = tape_idx[idx0 + 1];
  detail::ProjectedColumnHeader header;
  const auto* ids = tape_idx + idx0;
  const auto* vals = tape_val + val0;
  if (!detail::valid_projected_column_record(ids, vals, index_count, value_count) ||
      !detail::read_projected_column_header(ids, vals, index_count, value_count, header)) return;
  const std::int32_t L = header.expressions;
  NnzOffset col_cursor = idx0 + header.support_begin;
  NnzOffset metadata = header.metadata_begin;
  NnzOffset vpos = header.values_begin;
  double lo = -INFINITY;
  double hi = INFINITY;
  bool has_lo = false;
  bool has_hi = false;
  if (replay_primal != 0) {
    for (std::int32_t s = 0; s < L; ++s) {
      double orient; int len;
      if (!detail::read_projected_column_expression(ids, vals, index_count, value_count,
              header, metadata, vpos, orient, len)) return;
      double cand = vals[vpos];
      for (std::int32_t t = 0; t < len; ++t) {
        const std::int32_t row_col = tape_idx[col_cursor++];
        const double coeff = vals[vpos + 1 + t];
        cand += coeff * x_org[row_col];
      }
      if (orient > 0.0) {
        lo = fmax(lo, cand);
        has_lo = true;
      } else {
        hi = fmin(hi, cand);
        has_hi = true;
      }
      vpos += 1 + len;
    }
    x_org[elim_col] = has_lo ? lo : (has_hi ? hi : 0.0);
  }
  if (replay_dual != 0 && z_org != nullptr) {
    z_org[elim_col] = 0.0;
    if (z_retrieved != nullptr) {
      z_retrieved[elim_col] = std::uint8_t{1};
    }
  }
  if (replay_dual != 0 && y_org != nullptr) {
    for (std::int32_t t = 0; t < K; ++t) {
      y_org[tape_idx[idx0 + 2 + t]] = 0.0;
    }
  }
}

}  // namespace gpu_preprocessor::lp::presolve::postsolve
