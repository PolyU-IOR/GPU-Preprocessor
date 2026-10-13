#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_presolver::presolve {

// Exact detector for a zero-cost unit-box auxiliary z whose complete current
// incidence is z >= x+y-1, z <= x, z <= y.  Conflicting gadgets and gadgets
// whose endpoint is itself selected are rejected conservatively.
struct McCormickVariableEliminationAnalysisGpu {
  bool applicable = false;
  std::int32_t candidate_count = 0;
  std::uint8_t* selected_col = nullptr;
  std::int32_t* incident_rows = nullptr;
};

// Pending plans may remove rows/columns and update bounds, but must not rewrite
// coefficients. Selected incidence uses original matrix indices until APPLY.
McCormickVariableEliminationAnalysisGpu analyze_mccormick_variable_elimination(
    const LPInfoGpu& lp,
    const PresolveParams& params,
    const PresolvePlanGpu* pending = nullptr);

void build_mccormick_variable_elimination_plan(
    PresolvePlanGpu& plan,
    const LPInfoGpu& lp,
    const McCormickVariableEliminationAnalysisGpu& analysis,
    const PresolveParams& params);

void free_mccormick_variable_elimination_analysis(
    McCormickVariableEliminationAnalysisGpu& analysis);

}  // namespace gpu_presolver::presolve
