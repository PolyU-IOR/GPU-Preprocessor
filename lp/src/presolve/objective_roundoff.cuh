// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

namespace gpu_preprocessor::lp::presolve::detail {

// Relative tolerance for objective comparison and cancellation, independent of
// primal feasibility tolerances and without an absolute tolerance floor.
constexpr double objective_relative_tolerance = 1.0e-10;

__device__ inline bool objective_costs_proportional(double kept, double removed,
                                                   double ratio) {
  if (!isfinite(kept) || !isfinite(removed) || !isfinite(ratio) || ratio == 0.0) {
    return false;
  }
  const double expected = ratio * kept;
  // Underflow loses relative accuracy; conservatively retain these columns.
  if (!isfinite(expected) || (kept != 0.0 && fabs(expected) < DBL_MIN)) {
    return false;
  }
  const double scale = fmax(fabs(removed), fabs(expected));
  return fabs(removed - expected) <= objective_relative_tolerance * scale;
}

__device__ inline double substituted_objective_coefficient(double kept,
                                                           double removed,
                                                           double ratio) {
  const double update = removed * ratio;
  const double result = fma(removed, ratio, kept);
  // Only opposite-sign terms can cancel. There is no absolute tolerance floor:
  // a small cost without cancellation remains a real objective coefficient.
  if (isfinite(kept) && isfinite(update) && isfinite(result) &&
      ((kept > 0.0 && update < 0.0) || (kept < 0.0 && update > 0.0)) &&
      fabs(result) <= objective_relative_tolerance * fmax(fabs(kept), fabs(update))) {
    return 0.0;
  }
  return result;
}

}  // namespace gpu_preprocessor::lp::presolve::detail
