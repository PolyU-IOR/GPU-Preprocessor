#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve::detail {

// Nonzero mask entries mean keep. No identity permutation is materialized.
bool plan_keeps_all_indices(const PresolvePlanGpu& plan, std::int32_t m, std::int32_t n);

// Caller owns lp's vectors and has already committed recovery using the old lp.
// Missing plan vectors leave their destinations untouched. This performs only
// bit-preserving assignments, never arithmetic or a change to either matrix.
void commit_identity_plan_vectors(const PresolvePlanGpu& plan, LPInfoGpu& lp);

}  // namespace gpu_presolver::presolve::detail
