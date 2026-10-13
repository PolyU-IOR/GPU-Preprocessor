#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_presolver::presolve {

// A selected column is certified zero-cost, finite-bounded, degree two, and
// incident to two distinct short one-sided rows that give opposite affine
// bounds.  All three projected lower/upper compatibility conditions are
// strictly redundant over the current variable box.
struct TwoRowVariableEliminationAnalysisGpu {
  bool applicable = false;
  std::int32_t candidate_count = 0;
  std::uint8_t* selected_col = nullptr;
  std::int32_t* incident_rows = nullptr;
};

// Pending plans may remove rows/columns and update bounds, but must not rewrite
// coefficients. Selected incidence uses original matrix indices until APPLY.
TwoRowVariableEliminationAnalysisGpu analyze_two_row_variable_elimination(
    const LPInfoGpu& lp,
    const PresolveParams& params,
    const PresolvePlanGpu* pending = nullptr);

void build_two_row_variable_elimination_plan(
    PresolvePlanGpu& plan,
    const LPInfoGpu& lp,
    const TwoRowVariableEliminationAnalysisGpu& analysis,
    const PresolveParams& params);

void free_two_row_variable_elimination_analysis(
    TwoRowVariableEliminationAnalysisGpu& analysis);

}  // namespace gpu_presolver::presolve
