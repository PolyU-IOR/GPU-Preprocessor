#include "gpu_presolver/presolve/rules/rule_variable_pair_aggregation.hpp"
#include "variable_pair_aggregation/epigraph.hpp"
#include "variable_pair_aggregation/shifted_antipodal.hpp"
#include "../plan_storage.hpp"

namespace gpu_presolver::presolve {
VariablePairAggregationCase build_variable_pair_aggregation_plan(
    PresolvePlanGpu& plan, const LPInfoGpu& lp, const PresolveParams& params,
    const std::function<bool()>& stop) {
  if (!params.enable_structure_specific_rules || !params.enable_variable_pair_aggregation ||
      (stop && stop())) return VariablePairAggregationCase::None;

  auto epigraph = detail::analyze_variable_pair_epigraph(lp, params);
  if (epigraph.applicable) {
    detail::allocate_plan(plan, lp, detail::PlanRequirements{});
    detail::build_variable_pair_epigraph_plan(plan, lp, epigraph, params);
  }
  detail::free_variable_pair_epigraph_analysis(epigraph);
  if (plan.has_change || plan.has_infeasible || plan.has_unbounded)
    return VariablePairAggregationCase::Epigraph;

  detail::free_plan(plan);
  plan = {};
  if (stop && stop()) return VariablePairAggregationCase::None;
  auto shifted = detail::analyze_variable_pair_shifted_antipodal(lp, params);
  const bool applicable = shifted.applicable;
  if (applicable) {
    detail::allocate_plan(plan, lp, detail::PlanRequirements{});
    detail::build_variable_pair_shifted_antipodal_plan(plan, lp, shifted, params);
  }
  detail::free_variable_pair_shifted_antipodal_analysis(shifted);
  return applicable ? VariablePairAggregationCase::ShiftedAntipodal :
                      VariablePairAggregationCase::None;
}
}  // namespace gpu_presolver::presolve
