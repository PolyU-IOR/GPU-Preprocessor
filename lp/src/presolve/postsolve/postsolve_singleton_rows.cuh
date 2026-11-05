// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "postsolve_common.cuh"

namespace gpu_preprocessor::lp::presolve::postsolve {

__device__ __forceinline__ void replay_deleted_singleton_row(
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
    const double* l,
    const double* u,
    std::int32_t has_original_model,
    double tol,
    NnzOffset idx0,
    NnzOffset val0,
    NnzOffset val1) {
  const std::int32_t row = tape_idx[idx0];
  const std::int32_t col = tape_idx[idx0 + 1];
  const double old_AL = tape_val[val0];
  const double old_AU = tape_val[val0 + 1];
  const double aij = tape_val[val0 + 2];
  if (aij == 0.0 || !isfinite(aij)) {
    return;
  }
  if (val1 >= val0 + 8) {
    // A batch endpoint has one owner. Split the incoming stage multiplier;
    // the initial matrix and costs need not describe this reduction stage.
    const double incoming = z_org[col];
    const bool owns = (incoming > 0.0 && tape_val[val0 + 7] != 0.0) ||
                      (incoming < 0.0 && tape_val[val0 + 8] != 0.0);
    if (y_org != nullptr) y_org[row] = owns ? incoming / aij : 0.0;
    if (owns) z_org[col] = 0.0;
    if (z_retrieved != nullptr) z_retrieved[col] = std::uint8_t{1};
    return;
  }
  if (has_original_model == 0 || fabs(aij) <= tol) return;
  double gamma_j = c[col];
  for (NnzOffset p = AT_rowPtr[col]; p < AT_rowPtr[col + 1]; ++p) {
    const std::int32_t prow = AT_colVal[p];
    if (prow != row && y_org != nullptr) {
      gamma_j -= AT_nzVal[p] * y_org[prow];
    }
  }
  const double xj = x_org[col];
  const double zbar_j = z_org[col];
  const double row_activity = aij * xj;
  const bool row_at_lower = isfinite(old_AL) && row_activity <= old_AL + tol;
  const bool row_at_upper = isfinite(old_AU) && row_activity >= old_AU - tol;
  double iz_lo = -INFINITY;
  double iz_hi = INFINITY;
  if (!(row_at_lower && row_at_upper)) {
    if (row_at_lower) {
      if (aij > 0.0) {
        iz_hi = gamma_j;
      } else {
        iz_lo = gamma_j;
      }
    } else if (row_at_upper) {
      if (aij > 0.0) {
        iz_lo = gamma_j;
      } else {
        iz_hi = gamma_j;
      }
    } else {
      iz_lo = gamma_j;
      iz_hi = gamma_j;
    }
  }
  const bool at_lower = _postsolve_at_lower(xj, l[col], tol);
  const bool at_upper = _postsolve_at_upper(xj, u[col], tol);
  double z_lo = iz_lo;
  double z_hi = iz_hi;
  if (!at_lower && !at_upper) {
    if ((isfinite(iz_lo) && iz_lo > tol) || (isfinite(iz_hi) && iz_hi < -tol)) {
      z_lo = gamma_j;
      z_hi = gamma_j;
    } else {
      z_lo = 0.0;
      z_hi = 0.0;
    }
  } else if (at_lower && !at_upper) {
    z_lo = fmax(z_lo, 0.0);
  } else if (!at_lower && at_upper) {
    z_hi = fmin(z_hi, 0.0);
  }
  if (isfinite(z_lo) && isfinite(z_hi) && z_lo > z_hi + tol) {
    return;
  }
  const double z_new = _clamp_with_inf(zbar_j, z_lo, z_hi);
  z_org[col] = z_new;
  if (y_org != nullptr) {
    y_org[row] = (gamma_j - z_new) / aij;
  }
  if (z_retrieved != nullptr) {
    z_retrieved[col] = std::uint8_t{1};
  }
}

}  // namespace gpu_preprocessor::lp::presolve::postsolve
