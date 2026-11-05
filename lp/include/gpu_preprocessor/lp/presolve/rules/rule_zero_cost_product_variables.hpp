// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_preprocessor::lp::presolve {

// Exact detector for a zero-cost unit-box auxiliary z whose complete current
// incidence is z >= x+y-1, z <= x, z <= y.  Conflicting gadgets and gadgets
// whose endpoint is itself selected are rejected conservatively.
struct ZeroCostProductVariablesAnalysisGpu {
  bool applicable = false;
  std::int32_t candidate_count = 0;
  std::uint8_t* selected_col = nullptr;
  std::int32_t* incident_rows = nullptr;
};

// Pending plans may remove rows/columns and update bounds, but must not rewrite
// coefficients. Selected incidence uses original matrix indices until APPLY.
ZeroCostProductVariablesAnalysisGpu analyze_zero_cost_product_variables(
    const LPInfoGpu& lp,
    const PresolveParams& params,
    const PresolvePlanGpu* pending = nullptr);

void build_zero_cost_product_variables_plan(
    PresolvePlanGpu& plan,
    const LPInfoGpu& lp,
    const ZeroCostProductVariablesAnalysisGpu& analysis,
    const PresolveParams& params);

void free_zero_cost_product_variables_analysis(
    ZeroCostProductVariablesAnalysisGpu& analysis);

}  // namespace gpu_preprocessor::lp::presolve
