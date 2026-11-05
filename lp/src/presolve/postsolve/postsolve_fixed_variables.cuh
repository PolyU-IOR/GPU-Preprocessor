// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "postsolve_common.cuh"
#include "../infinite_bound_record.hpp"

namespace gpu_preprocessor::lp::presolve::postsolve {

__device__ __forceinline__ void replay_fixed_variable(
    double* x_org,
    double* y_org,
    double* z_org,
    std::uint8_t* z_retrieved,
    const std::int32_t* tape_idx,
    const double* tape_val,
    std::int32_t replay_primal,
    std::int32_t replay_dual,
    NnzOffset idx0,
    NnzOffset idx1,
    NnzOffset val0) {
  const std::int32_t col = tape_idx[idx0];
  if (replay_primal != 0) {
    x_org[col] = tape_val[val0];
  }
  if (replay_dual != 0 && z_org != nullptr) {
    double zj = tape_val[val0 + 1];
    for (NnzOffset pos = idx0 + 1; pos <= idx1; ++pos) {
      const std::int32_t row = tape_idx[pos];
      const double coeff = tape_val[val0 + 2 + (pos - (idx0 + 1))];
      if (y_org != nullptr) {
        zj -= coeff * y_org[row];
      }
    }
    z_org[col] = zj;
    if (z_retrieved != nullptr) {
      z_retrieved[col] = std::uint8_t{1};
    }
  }
}

__device__ __forceinline__ void replay_infinite_bound_dual_fix(
    double* x_org,
    double* z_org,
    std::uint8_t* z_retrieved,
    const std::int32_t* tape_idx,
    const double* tape_val,
    std::int32_t replay_primal,
    std::int32_t replay_dual,
    NnzOffset idx0,
    NnzOffset idx1,
    NnzOffset val0,
    NnzOffset val1) {
  const bool fix_to_pos_inf = tape_idx[idx0] > 0;
  const std::int32_t col = tape_idx[idx0 + 1];
  detail::InfiniteBoundHeader header;
  if (!detail::read_infinite_bound_header(tape_idx + idx0, tape_val + val0,
          idx1 - idx0 + 1, val1 - val0 + 1, header)) return;
  const std::int32_t n_rows = header.rows;
  double extreme_val = tape_val[val0 + header.values - 1];
  NnzOffset idx_pos = idx0 + header.indices;
  NnzOffset val_pos = val0 + header.values;
  if (replay_primal != 0) {
    for (std::int32_t r = 0; r < n_rows; ++r) {
      const NnzOffset row_len = tape_idx[idx_pos];
      double side = tape_val[val_pos];
      double coeff = 0.0;
      for (std::int32_t j = 0; j < row_len; ++j) {
        const std::int32_t row_col = tape_idx[idx_pos + 1 + j];
        const double row_coeff = tape_val[val_pos + 1 + j];
        if (row_col == col) {
          coeff = row_coeff;
        } else {
          side -= row_coeff * x_org[row_col];
        }
      }
      const double candidate = side / coeff;
      extreme_val = fix_to_pos_inf ? fmax(extreme_val, candidate) : fmin(extreme_val, candidate);
      idx_pos += row_len + 1;
      val_pos += row_len + 1;
    }
    x_org[col] = extreme_val;
  }
  if (replay_dual != 0 && z_org != nullptr) {
    z_org[col] = 0.0;
    if (z_retrieved != nullptr) {
      z_retrieved[col] = std::uint8_t{1};
    }
  }
}

}  // namespace gpu_preprocessor::lp::presolve::postsolve
