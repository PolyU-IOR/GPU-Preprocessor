// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"

namespace gpu_preprocessor::lp::presolve {

std::size_t parallel_columns_scratch_bytes(std::size_t rows, std::size_t cols,
                                            bool parallel_verify);

void apply_rule_parallel_columns(PresolvePlanGpu &plan, const LPInfoGpu &lp,
                                  const PresolveStatsGpu &stats,
                                  const PresolveParams &pparams);

} // namespace gpu_preprocessor::lp::presolve
