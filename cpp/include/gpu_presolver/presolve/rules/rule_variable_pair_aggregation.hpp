#pragma once
#include "gpu_presolver/presolve/presolve_structs.hpp"
#include <functional>

namespace gpu_presolver::presolve {
enum class VariablePairAggregationCase { None, Epigraph, ShiftedAntipodal };

// Accepts an empty plan and allocates only after a detector accepts the model.
// Try the epigraph case first; fall through only if it produces no reduction
// or terminal status. The caller commits and releases the returned plan.
// stop preserves the scheduler's time-limit check between the two cases.
VariablePairAggregationCase build_variable_pair_aggregation_plan(
    PresolvePlanGpu& plan, const LPInfoGpu& lp, const PresolveParams& params,
    const std::function<bool()>& stop = {});
}  // namespace gpu_presolver::presolve
