#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

// Fix finite equal bounds and narrow ordered intervals (width <= bound_tol).
// Approximate fixings are accepted only when every incident live row's summed
// candidate uncertainty is <= 0.1 * feasibility_tol. This is a per-batch guard;
// exact fixings remain eligible regardless of the approximate-candidate budget.
void apply_rule_infeasible_and_fixed_variables(PresolvePlanGpu &plan,
                                           const LPInfoGpu &lp,
                                           const PresolveParams &pparams);

} // namespace gpu_presolver::presolve
