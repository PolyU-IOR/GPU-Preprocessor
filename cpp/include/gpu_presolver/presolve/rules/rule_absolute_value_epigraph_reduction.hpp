#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

void apply_rule_absolute_value_epigraph_reduction(PresolvePlanGpu& plan,
                                           const LPInfoGpu& lp,
                                           const PresolveStatsGpu& stats,
                                           const PresolveParams& pparams);

bool absolute_value_epigraph_prefix_screen_passes(const LPInfoGpu& lp);

}  // namespace gpu_presolver::presolve
