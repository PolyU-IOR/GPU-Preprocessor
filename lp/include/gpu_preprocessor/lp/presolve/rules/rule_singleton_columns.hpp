// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"

namespace gpu_preprocessor::lp::presolve {

// One batch handles equality elimination and dual inference on inequality rows.
// Newly tightened equalities are eligible for elimination in the next batch.
void apply_rule_singleton_columns(PresolvePlanGpu &plan, const LPInfoGpu &lp,
                                 const PresolveStatsGpu &stats,
                                 const PresolveParams &pparams,
                                 bool skip_precheck = false);

void release_singleton_columns_workspace() noexcept;

} // namespace gpu_preprocessor::lp::presolve
