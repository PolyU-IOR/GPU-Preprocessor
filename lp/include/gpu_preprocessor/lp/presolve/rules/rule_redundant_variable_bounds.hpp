// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"

namespace gpu_preprocessor::lp::presolve {

enum class RedundantVariableBoundsMode {
  ZeroCostBoxes,
  IncludeOneSidedBounds
};

// Explicit direct-call mode. Neither overload adds this rule to Main.
void apply_rule_redundant_variable_bounds(
    PresolvePlanGpu& plan, const LPInfoGpu& lp, const PresolveStatsGpu& stats,
    const PresolveParams& params, RedundantVariableBoundsMode mode);

// Compatibility overload: reads params.enable_redundant_variable_bounds as
// a mode selector, retaining its historical false/true behavior.
void apply_rule_redundant_variable_bounds(PresolvePlanGpu& plan,
                                 const LPInfoGpu& lp,
                                 const PresolveStatsGpu& stats,
                                 const PresolveParams& pparams);

}  // namespace gpu_preprocessor::lp::presolve
