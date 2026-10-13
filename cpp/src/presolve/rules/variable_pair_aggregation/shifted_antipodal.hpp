#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_presolver::presolve::detail {

// Device-owned analysis retained between the full detector and the CSR plan
// builder.  Keeping it explicit avoids rerunning the union-find or copying a
// million-entry component map through the host.
struct VariablePairShiftedAntipodalAnalysisGpu {
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
bool quick_probe_variable_pair_shifted_antipodal(const LPInfoGpu& lp,
                                      const PresolveParams& params);

VariablePairShiftedAntipodalAnalysisGpu analyze_variable_pair_shifted_antipodal(
    const LPInfoGpu& lp,
    const PresolveParams& params);

void build_variable_pair_shifted_antipodal_plan(PresolvePlanGpu& plan,
                                    const LPInfoGpu& lp,
                                    VariablePairShiftedAntipodalAnalysisGpu& analysis,
                                    const PresolveParams& params);

void free_variable_pair_shifted_antipodal_analysis(VariablePairShiftedAntipodalAnalysisGpu& analysis);

}  // namespace gpu_presolver::presolve::detail
