// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"

namespace gpu_preprocessor::lp::presolve {

void apply_rule_dual_fix(PresolvePlanGpu& plan,
                         const LPInfoGpu& lp,
                         const PresolveStatsGpu& stats,
                         const PresolveParams& pparams,
                         const std::uint8_t* candidate_col_mask = nullptr);

}  // namespace gpu_preprocessor::lp::presolve
