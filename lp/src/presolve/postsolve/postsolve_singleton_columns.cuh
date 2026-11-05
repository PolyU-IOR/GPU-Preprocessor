// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "postsolve_common.cuh"

namespace gpu_preprocessor::lp::presolve::postsolve {

__device__ __forceinline__ void replay_singleton_column_substitution(
    double* x_org,
    double* y_org,
    double* z_org,
    std::uint8_t* z_retrieved,
    const std::int32_t* tape_idx,
    const double* tape_val,
    std::int32_t replay_primal,
    std::int32_t replay_dual,
    NnzOffset idx0,
    NnzOffset val0) {
  const std::int32_t elim_col = tape_idx[idx0];
  const std::int32_t row = tape_idx[idx0 + 1];
  const std::int32_t support_count = tape_idx[idx0 + 2];
  const double pivot_coeff = tape_val[val0];
  const double rhs = tape_val[val0 + 1];
  const double elim_obj = tape_val[val0 + 4];
  const bool row_deleted = tape_val[val0 + 5] != 0.0;
  double partial = 0.0;
  for (std::int32_t t = 0; t < support_count; ++t) {
    const std::int32_t col = tape_idx[idx0 + 3 + t];
    const double coeff = tape_val[val0 + 6 + t];
    partial += coeff * x_org[col];
  }
  if (replay_primal != 0) {
    x_org[elim_col] = (rhs - partial) / pivot_coeff;
  }
  if (replay_dual != 0 && y_org != nullptr && row >= 0) {
    if (row_deleted) {
      y_org[row] = elim_obj / pivot_coeff;
    } else {
      y_org[row] += elim_obj / pivot_coeff;
    }
  }
  if (replay_dual != 0 && z_org != nullptr) {
    z_org[elim_col] = row_deleted || y_org == nullptr ? 0.0 : elim_obj - pivot_coeff * y_org[row];
    if (z_retrieved != nullptr) {
      z_retrieved[elim_col] = std::uint8_t{1};
    }
  }
}

}  // namespace gpu_preprocessor::lp::presolve::postsolve
