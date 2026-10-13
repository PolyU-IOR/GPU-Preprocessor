#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_presolver::presolve::detail {

// Device-owned state shared by the fail-closed detector and the one-pass CSR
// quotient builder.  variable-pair epigraph models are laid out as
//
//   p_0 ... p_{k-1}, n_0 ... n_{k-1}, T_0 ... T_{g-1},
//
// where the short objective suffix contains only the T columns.  Each pair
// has one three-entry epigraph row; all remaining rows use p_i/n_i with
// exactly opposite coefficients.
struct VariablePairEpigraphAnalysisGpu {
  bool applicable = false;
  std::int32_t pair_count = 0;
  std::int32_t tail_count = 0;
  std::int32_t edge_count = 0;
  std::int32_t component_count = 0;
  std::int32_t removed_rows = 0;

  // Per-row classification.  NO_EDGE / -1 denote an ordinary quotient row.
  unsigned long long* edge_by_row = nullptr;
  std::int32_t* g_pair_by_row = nullptr;

  // Per-pair epigraph metadata and shifted-difference offset.
  std::int32_t* pair_g_row = nullptr;
  std::int32_t* pair_tail_col = nullptr;
  double* pair_g_coefficient = nullptr;
  double* pair_g_lower = nullptr;
  double* pair_shift = nullptr;

  // DSU and root-only original-coordinate upper-bound intersections.
  std::int32_t* parent = nullptr;
  std::int32_t* component_size = nullptr;
  double* component_plus_upper = nullptr;
  double* component_minus_upper = nullptr;
};

// Tiny rejection filter: one objective entry is copied for the common reject
// path.  Only a plausible candidate inspects a suffix of at most nine entries,
// at most 256 pairs, and at most 1024 rows.  A true result only authorizes the
// full fail-closed detector; it never proves that the transform is safe.
bool quick_probe_variable_pair_epigraph(const LPInfoGpu& lp,
                                 const PresolveParams& params);

VariablePairEpigraphAnalysisGpu analyze_variable_pair_epigraph(
    const LPInfoGpu& lp,
    const PresolveParams& params);

void build_variable_pair_epigraph_plan(PresolvePlanGpu& plan,
                               const LPInfoGpu& lp,
                               VariablePairEpigraphAnalysisGpu& analysis,
                               const PresolveParams& params);

void free_variable_pair_epigraph_analysis(VariablePairEpigraphAnalysisGpu& analysis);

}  // namespace gpu_presolver::presolve::detail
