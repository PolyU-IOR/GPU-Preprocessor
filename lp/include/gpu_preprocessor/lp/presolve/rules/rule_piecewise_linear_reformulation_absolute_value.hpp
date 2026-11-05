// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"

namespace gpu_preprocessor::lp::presolve {

void apply_rule_piecewise_linear_reformulation_absolute_value(
    PresolvePlanGpu& plan, const LPInfoGpu& lp, const PresolveStatsGpu& stats,
    const PresolveParams& params);

bool piecewise_linear_reformulation_absolute_value_prefix_screen_passes(
    const LPInfoGpu& lp);

}  // namespace gpu_preprocessor::lp::presolve
