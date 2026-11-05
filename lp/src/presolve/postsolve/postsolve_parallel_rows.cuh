// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "postsolve_common.cuh"

namespace gpu_preprocessor::lp::presolve::postsolve {

__device__ __forceinline__ void _parallel_row_scaled_interval(double lower,
                                              double upper,
                                              double ratio,
                                              double* scaled_l,
                                              double* scaled_u) {
  const double raw_l = lower * ratio;
  const double raw_u = upper * ratio;
  if (ratio >= 0.0) {
    *scaled_l = raw_l;
    *scaled_u = raw_u;
  } else {
    *scaled_l = raw_u;
    *scaled_u = raw_l;
  }
}

__device__ __forceinline__ void replay_parallel_row(
    double* y_org,
    const std::int32_t* tape_idx,
    const double* tape_val,
    NnzOffset idx0,
    NnzOffset val0) {
  const std::int32_t kept_row = tape_idx[idx0];
  const std::int32_t deleted_row = tape_idx[idx0 + 1];
  const double ratio = tape_val[val0];
  const double kept_old_AL = tape_val[val0 + 1];
  const double kept_old_AU = tape_val[val0 + 2];
  const double deleted_old_AL = tape_val[val0 + 3];
  const double deleted_old_AU = tape_val[val0 + 4];
  if (ratio == 0.0 || !isfinite(ratio) || y_org == nullptr) {
    return;
  }
  double deleted_scaled_AL = 0.0;
  double deleted_scaled_AU = 0.0;
  _parallel_row_scaled_interval(deleted_old_AL, deleted_old_AU, ratio,
                                &deleted_scaled_AL, &deleted_scaled_AU);
  // The signed multiplier selects the endpoint of the interval intersection.
  // Inherited ties keep their multiplier; no final-space activity is used.
  const double incoming = y_org[kept_row];
  const bool transfer = incoming > 0.0 ? deleted_scaled_AL > kept_old_AL
                                      : incoming < 0.0 && deleted_scaled_AU < kept_old_AU;
  y_org[deleted_row] = transfer ? ratio * incoming : 0.0;
  y_org[kept_row] = transfer ? 0.0 : incoming;
}

}  // namespace gpu_preprocessor::lp::presolve::postsolve
