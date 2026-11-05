// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "postsolve_common.cuh"

namespace gpu_preprocessor::lp::presolve::postsolve {

__device__ __forceinline__ void _doubleton_equations_recover_z_split(double x_keep,
                                              double z_keep_hat,
                                              double alpha,
                                              double old_keep_l,
                                              double old_keep_u,
                                              double new_keep_l,
                                              double new_keep_u,
                                              double lower_source,
                                              double upper_source,
                                              double tol,
                                              double* z_keep,
                                              double* z_elim) {
  constexpr double BOUND_UNKNOWN = 0.0;
  constexpr double BOUND_INHERITED = 1.0;
  constexpr double BOUND_INDUCED = 2.0;
  if (fabs(alpha) <= tol) {
    *z_keep = z_keep_hat;
    *z_elim = 0.0;
    return;
  }

  // The incoming signed multiplier selects its endpoint even if both new
  // bounds coincide. Final original coordinates need not equal this stage's x.
  (void)x_keep;
  const bool lower = z_keep_hat > 0.0;
  double source = lower ? lower_source : upper_source;
  if (source == BOUND_UNKNOWN) {
    const double old_bound = lower ? old_keep_l : old_keep_u;
    const double new_bound = lower ? new_keep_l : new_keep_u;
    source = old_bound == new_bound ? BOUND_INHERITED : BOUND_INDUCED;
  }

  if (source == BOUND_INDUCED) {
    *z_keep = 0.0;
    *z_elim = z_keep_hat / alpha;
  } else {
    *z_keep = z_keep_hat;
    *z_elim = 0.0;
  }
}

__device__ __forceinline__ double _doubleton_equations_deleted_row_dual_from_original(const double* y_org,
                                                               const NnzOffset* AT_rowPtr,
                                                               const std::int32_t* AT_colVal,
                                                               const double* AT_nzVal,
                                                               const double* c,
                                                               std::int32_t row,
                                                               std::int32_t elim_col,
                                                               double elim_coeff,
                                                               double z_elim) {
  double sum_other = 0.0;
  for (NnzOffset p = AT_rowPtr[elim_col]; p < AT_rowPtr[elim_col + 1]; ++p) {
    const std::int32_t prow = AT_colVal[p];
    if (prow != row) {
      sum_other += AT_nzVal[p] * y_org[prow];
    }
  }
  return (c[elim_col] - sum_other - z_elim) / elim_coeff;
}

__device__ __forceinline__ double _doubleton_equations_deleted_row_dual_from_tape(const double* y_org,
                                                           const std::int32_t* tape_idx,
                                                           const double* tape_val,
                                                           NnzOffset idx0,
                                                           NnzOffset val0,
                                                           double elim_coeff,
                                                           double z_elim) {
  const std::int32_t support_count = tape_idx[idx0 + 3];
  double sum_other = 0.0;
  for (std::int32_t t = 0; t < support_count; ++t) {
    const std::int32_t row = tape_idx[idx0 + 4 + t];
    const double coeff = tape_val[val0 + 12 + t];
    sum_other += coeff * y_org[row];
  }
  const double elim_obj = tape_val[val0 + 9];
  return (elim_obj - sum_other - z_elim) / elim_coeff;
}

__device__ __forceinline__ void replay_doubleton_equation(
    double* x_org,
    double* y_org,
    double* z_org,
    std::uint8_t* z_retrieved,
    const std::int32_t* tape_idx,
    const double* tape_val,
    const NnzOffset* AT_rowPtr,
    const std::int32_t* AT_colVal,
    const double* AT_nzVal,
    const double* c,
    std::int32_t has_original_model,
    std::int32_t replay_primal,
    std::int32_t replay_dual,
    double tol,
    NnzOffset idx0,
    NnzOffset idx1,
    NnzOffset val0,
    NnzOffset val1) {
  const std::int32_t elim_col = tape_idx[idx0];
  const std::int32_t keep_col = tape_idx[idx0 + 1];
  const std::int32_t row = tape_idx[idx0 + 2];
  const double elim_coeff = tape_val[val0];
  const double keep_coeff = tape_val[val0 + 1];
  const double rhs = tape_val[val0 + 2];
  const double elim_obj = tape_val[val0 + 9];
  const double alpha = -keep_coeff / elim_coeff;
  const double beta = rhs / elim_coeff;
  if (replay_primal != 0) {
    x_org[elim_col] = beta + alpha * x_org[keep_col];
  }
  const bool has_support_payload = idx1 >= idx0 + 3 && val1 >= val0 + 11;
  if (replay_dual != 0 && z_org != nullptr && y_org != nullptr &&
      (has_support_payload || has_original_model != 0)) {
    const double old_keep_l = tape_val[val0 + 5];
    const double old_keep_u = tape_val[val0 + 6];
    const double new_keep_l = tape_val[val0 + 7];
    const double new_keep_u = tape_val[val0 + 8];
    const double lower_source = val1 >= val0 + 10 ? tape_val[val0 + 10] : 0.0;
    const double upper_source = val1 >= val0 + 11 ? tape_val[val0 + 11] : 0.0;
    double z_keep = 0.0;
    double z_elim = 0.0;
    _doubleton_equations_recover_z_split(x_org[keep_col],
                                  z_org[keep_col],
                                  alpha,
                                  old_keep_l,
                                  old_keep_u,
                                  new_keep_l,
                                  new_keep_u,
                                  lower_source,
                                  upper_source,
                                  tol,
                                  &z_keep,
                                  &z_elim);
    y_org[row] = has_support_payload
                     ? _doubleton_equations_deleted_row_dual_from_tape(
                           y_org, tape_idx, tape_val, idx0, val0, elim_coeff, z_elim)
                     : _doubleton_equations_deleted_row_dual_from_original(
                           y_org, AT_rowPtr, AT_colVal, AT_nzVal, c, row, elim_col, elim_coeff, z_elim);
    z_org[keep_col] = z_keep;
    z_org[elim_col] = z_elim;
    if (z_retrieved != nullptr) {
      z_retrieved[keep_col] = std::uint8_t{1};
      z_retrieved[elim_col] = std::uint8_t{1};
    }
  } else if (replay_dual != 0) {
    if (z_org != nullptr) {
      z_org[elim_col] = 0.0;
      if (z_retrieved != nullptr) {
        z_retrieved[elim_col] = std::uint8_t{1};
      }
    }
    if (y_org != nullptr && row >= 0) {
      y_org[row] = elim_obj / elim_coeff;
    }
  }
}

}  // namespace gpu_preprocessor::lp::presolve::postsolve
