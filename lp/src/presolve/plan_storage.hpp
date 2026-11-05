// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once
#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"

namespace gpu_preprocessor::lp::presolve::detail {
struct PlanRequirements {
  bool objective = true;
  bool col_bounds = true;
  bool row_bounds = true;
};
void allocate_plan(PresolvePlanGpu& plan, const LPInfoGpu& lp,
                   const PlanRequirements& needs);
void free_plan(PresolvePlanGpu& plan);
}  // namespace gpu_preprocessor::lp::presolve::detail
