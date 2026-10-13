#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

void apply_rule_bound_strengthening(PresolvePlanGpu &plan,
                                        const LPInfoGpu &lp,
                                        const PresolveStatsGpu &stats,
                                        const PresolveParams &pparams);

// Prepare the rule's reusable buffers before the caller starts its presolve
// timer. The inner presolve keeps them until this owner leaves scope.
void prepare_bound_strengthening_workspace(std::int32_t rows, std::int32_t cols);
bool bound_strengthening_workspace_prepared() noexcept;

// Releases thread-local CUDA scratch retained for reuse within one presolve.
void release_bound_strengthening_workspace() noexcept;

class PreparedBoundStrengtheningWorkspace {
 public:
  PreparedBoundStrengtheningWorkspace(std::int32_t rows, std::int32_t cols, bool enabled)
      : active_(enabled && rows > 0 && cols > 0) {
    if (active_) prepare_bound_strengthening_workspace(rows, cols);
  }
  ~PreparedBoundStrengtheningWorkspace() {
    if (active_) release_bound_strengthening_workspace();
  }
  PreparedBoundStrengtheningWorkspace(const PreparedBoundStrengtheningWorkspace&) = delete;
  PreparedBoundStrengtheningWorkspace& operator=(const PreparedBoundStrengtheningWorkspace&) = delete;

 private:
  bool active_ = false;
};

} // namespace gpu_presolver::presolve
