#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

// One batch handles equality elimination and dual inference on inequality rows.
// Newly tightened equalities are eligible for elimination in the next batch.
void apply_rule_singleton_columns(PresolvePlanGpu &plan, const LPInfoGpu &lp,
                                 const PresolveStatsGpu &stats,
                                 const PresolveParams &pparams,
                                 bool skip_precheck = false);

void release_singleton_columns_workspace() noexcept;

} // namespace gpu_presolver::presolve
