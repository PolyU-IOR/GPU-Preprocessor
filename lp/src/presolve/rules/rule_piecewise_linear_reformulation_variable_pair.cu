// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/rules/rule_piecewise_linear_reformulation_variable_pair.hpp"
#include "piecewise_linear_reformulation_variable_pair/epigraph.hpp"
#include "piecewise_linear_reformulation_variable_pair/shifted_antipodal.hpp"
#include "../plan_storage.hpp"

namespace gpu_preprocessor::lp::presolve {
PiecewiseLinearReformulationVariablePairCase
build_piecewise_linear_reformulation_variable_pair_plan(
    PresolvePlanGpu& plan, const LPInfoGpu& lp, const PresolveParams& params,
    const std::function<bool()>& stop) {
  if (!params.enable_structure_specific_rules || !params.enable_piecewise_linear_reformulation_variable_pair ||
      (stop && stop())) return PiecewiseLinearReformulationVariablePairCase::None;

  auto epigraph = detail::analyze_piecewise_linear_reformulation_variable_pair_epigraph(lp, params);
  if (epigraph.applicable) {
    detail::allocate_plan(plan, lp, detail::PlanRequirements{});
    detail::build_piecewise_linear_reformulation_variable_pair_epigraph_plan(plan, lp, epigraph, params);
  }
  detail::free_piecewise_linear_reformulation_variable_pair_epigraph_analysis(epigraph);
  if (plan.has_change || plan.has_infeasible || plan.has_unbounded)
    return PiecewiseLinearReformulationVariablePairCase::Epigraph;

  detail::free_plan(plan);
  plan = {};
  if (stop && stop()) return PiecewiseLinearReformulationVariablePairCase::None;
  auto shifted = detail::analyze_piecewise_linear_reformulation_variable_pair_shifted_antipodal(lp, params);
  const bool applicable = shifted.applicable;
  if (applicable) {
    detail::allocate_plan(plan, lp, detail::PlanRequirements{});
    detail::build_piecewise_linear_reformulation_variable_pair_shifted_antipodal_plan(plan, lp, shifted, params);
  }
  detail::free_piecewise_linear_reformulation_variable_pair_shifted_antipodal_analysis(shifted);
  return applicable ? PiecewiseLinearReformulationVariablePairCase::ShiftedAntipodal :
                      PiecewiseLinearReformulationVariablePairCase::None;
}
}  // namespace gpu_preprocessor::lp::presolve
