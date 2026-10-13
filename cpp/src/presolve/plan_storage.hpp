#pragma once
#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve::detail {
struct PlanRequirements {
  bool objective = true;
  bool col_bounds = true;
  bool row_bounds = true;
};
void allocate_plan(PresolvePlanGpu& plan, const LPInfoGpu& lp,
                   const PlanRequirements& needs);
void free_plan(PresolvePlanGpu& plan);
}  // namespace gpu_presolver::presolve::detail
