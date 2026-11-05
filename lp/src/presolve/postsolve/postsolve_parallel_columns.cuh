// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "postsolve_common.cuh"

namespace gpu_preprocessor::lp::presolve::postsolve {

__device__ __forceinline__ double _choose_merged_source_value(double lo, double hi, double tol) {
  if (isfinite(lo) && isfinite(hi)) {
    return 0.5 * (lo + hi);
  }
  if (isfinite(lo)) {
    if (0.0 > lo + tol) {
      return 0.0;
    }
    return lo + fmax(1.0, fabs(lo) * fmax(tol, 1.0e-12));
  }
  if (isfinite(hi)) {
    if (0.0 < hi - tol) {
      return 0.0;
    }
    return hi - fmax(1.0, fabs(hi) * fmax(tol, 1.0e-12));
  }
  return 0.0;
}

__device__ __forceinline__ double _parallel_column_aggregate_lower_bound(double target_l,
                                                      double source_l,
                                                      double source_u,
                                                      double ratio) {
  if (ratio > 0.0) {
    if (isfinite(target_l) && isfinite(source_l)) {
      return target_l + ratio * source_l;
    }
  } else if (isfinite(target_l) && isfinite(source_u)) {
    return target_l + ratio * source_u;
  }
  return -INFINITY;
}

__device__ __forceinline__ double _parallel_column_aggregate_upper_bound(double target_u,
                                                      double source_l,
                                                      double source_u,
                                                      double ratio) {
  if (ratio > 0.0) {
    if (isfinite(target_u) && isfinite(source_u)) {
      return target_u + ratio * source_u;
    }
  } else if (isfinite(target_u) && isfinite(source_l)) {
    return target_u + ratio * source_l;
  }
  return INFINITY;
}

__device__ __forceinline__ void _recover_parallel_column_primal(double y_val,
                                             double z_to,
                                             double ratio,
                                             double source_l,
                                             double source_u,
                                             double target_l,
                                             double target_u,
                                             double tol,
                                             double* x_from,
                                             double* x_to) {
  double lo = source_l;
  double hi = source_u;
  if (isfinite(target_u)) {
    const double bound = (y_val - target_u) / ratio;
    if (ratio > 0.0) {
      lo = fmax(lo, bound);
    } else {
      hi = fmin(hi, bound);
    }
  }
  if (isfinite(target_l)) {
    const double bound = (y_val - target_l) / ratio;
    if (ratio > 0.0) {
      hi = fmin(hi, bound);
    } else {
      lo = fmax(lo, bound);
    }
  }

  const double lower = _parallel_column_aggregate_lower_bound(target_l, source_l, source_u, ratio);
  const double upper = _parallel_column_aggregate_upper_bound(target_u, source_l, source_u, ratio);
  if (z_to > tol && isfinite(lower) && fabs(y_val - lower) <= tol) {
    *x_from = ratio > 0.0 ? source_l : source_u;
  } else if (z_to < -tol && isfinite(upper) && fabs(y_val - upper) <= tol) {
    *x_from = ratio > 0.0 ? source_u : source_l;
  } else {
    *x_from = _choose_merged_source_value(lo, hi, tol);
  }
  *x_to = y_val - ratio * (*x_from);
}

__device__ __forceinline__ void replay_parallel_column(
    double* x_org,
    double* z_org,
    std::uint8_t* z_retrieved,
    const std::int32_t* tape_idx,
    const double* tape_val,
    std::int32_t replay_primal,
    std::int32_t replay_dual,
    double tol,
    NnzOffset idx0,
    NnzOffset val0) {
  const std::int32_t from_idx = tape_idx[idx0];
  const std::int32_t to_idx = tape_idx[idx0 + 1];
  const double ratio = tape_val[val0];
  const double lo = tape_val[val0 + 1];
  const double hi = tape_val[val0 + 2];
  const double to_l = tape_val[val0 + 3];
  const double to_u = tape_val[val0 + 4];
  if (replay_primal != 0) {
    const double y_val = x_org[to_idx];
    const double z_to = z_org != nullptr ? z_org[to_idx] : 0.0;
    double x_from = 0.0;
    double x_to = y_val;
    _recover_parallel_column_primal(y_val, z_to, ratio, lo, hi, to_l, to_u, tol, &x_from, &x_to);
    x_org[from_idx] = x_from;
    x_org[to_idx] = x_to;
  }
  if (replay_dual != 0 && z_org != nullptr) {
    z_org[from_idx] = ratio * z_org[to_idx];
    if (z_retrieved != nullptr) {
      z_retrieved[from_idx] = std::uint8_t{1};
    }
  }
}

}  // namespace gpu_preprocessor::lp::presolve::postsolve
