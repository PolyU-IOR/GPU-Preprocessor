#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

// Empty rows and columns are one rule category. Keep the two entrypoints
// separate so the scheduler can retain its existing row/column order.
void apply_rule_empty_rows(PresolvePlanGpu& plan,
                           const LPInfoGpu& lp,
                           const PresolveStatsGpu& stats,
                           const PresolveParams& pparams);

void apply_rule_empty_columns(PresolvePlanGpu& plan,
                              const LPInfoGpu& lp,
                              const PresolveStatsGpu& stats,
                              const PresolveParams& pparams);

}  // namespace gpu_presolver::presolve
