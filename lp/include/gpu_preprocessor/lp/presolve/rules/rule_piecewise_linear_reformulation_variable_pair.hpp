// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once
#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"
#include <functional>

namespace gpu_preprocessor::lp::presolve {
enum class PiecewiseLinearReformulationVariablePairCase {
  None,
  Epigraph,
  ShiftedAntipodal
};

// Accepts an empty plan and allocates only after a detector accepts the model.
// Try the epigraph case first; fall through only if it produces no reduction
// or terminal status. The caller commits and releases the returned plan.
// stop preserves the scheduler's time-limit check between the two cases.
PiecewiseLinearReformulationVariablePairCase
build_piecewise_linear_reformulation_variable_pair_plan(
    PresolvePlanGpu& plan, const LPInfoGpu& lp, const PresolveParams& params,
    const std::function<bool()>& stop = {});
}  // namespace gpu_preprocessor::lp::presolve
