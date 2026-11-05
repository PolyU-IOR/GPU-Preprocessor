// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_preprocessor::lp::presolve::detail {

// Device-owned analysis retained between the full detector and the CSR plan
// builder.  Keeping it explicit avoids rerunning the union-find or copying a
// million-entry component map through the host.
struct PiecewiseLinearReformulationVariablePairShiftedAntipodalAnalysisGpu {
  bool applicable = false;
  std::int32_t pair_count = 0;
  std::int32_t edge_count = 0;
  std::int32_t component_count = 0;
  std::int32_t zero_rows = 0;

  unsigned long long* edge_by_row = nullptr;
  std::int32_t* parent = nullptr;
  double* pair_shift = nullptr;
  double* component_plus_cap = nullptr;
  double* component_minus_cap = nullptr;
  double* component_rho = nullptr;
  double objective_constant_delta = 0.0;
};

// Cheap sampled column-degree/pair checks plus a uniformly sampled merge-row
// density gate.  This is a rejection filter only: true means the full detector
// should run, never that the transformation is already proven.
bool quick_probe_piecewise_linear_reformulation_variable_pair_shifted_antipodal(const LPInfoGpu& lp,
                                      const PresolveParams& params);

PiecewiseLinearReformulationVariablePairShiftedAntipodalAnalysisGpu analyze_piecewise_linear_reformulation_variable_pair_shifted_antipodal(
    const LPInfoGpu& lp,
    const PresolveParams& params);

void build_piecewise_linear_reformulation_variable_pair_shifted_antipodal_plan(PresolvePlanGpu& plan,
                                    const LPInfoGpu& lp,
                                    PiecewiseLinearReformulationVariablePairShiftedAntipodalAnalysisGpu& analysis,
                                    const PresolveParams& params);

void free_piecewise_linear_reformulation_variable_pair_shifted_antipodal_analysis(PiecewiseLinearReformulationVariablePairShiftedAntipodalAnalysisGpu& analysis);

}  // namespace gpu_preprocessor::lp::presolve::detail
