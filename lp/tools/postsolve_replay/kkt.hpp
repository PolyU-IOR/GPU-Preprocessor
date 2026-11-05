// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once
#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"

namespace gpu_preprocessor::lp::diagnostics {
struct EndpointMetrics {
  double complementarity_inf = 0, cone_inf = 0;
  int worst_complementarity_index = -1, worst_cone_index = -1;
};
enum class KktCheck { PrimalDualGap, Full };
const char* check_name(KktCheck check);
struct KktMetrics {
  EndpointMetrics row_lower, row_upper, col_lower, col_upper;
  bool available = false, finite = true, pdg_finite = true, dual_domain_valid = true;
  double primal = 0, dual = 0, gap = 0, cone = 0, complementarity = 0;
  double primal_inf = 0, stationarity_inf = 0, complementarity_inf = 0;
  double primal_objective = 0, dual_objective = 0;
  double stationarity_l2 = 0, objective_l2 = 0;
  double row_violation_l2 = 0, bound_violation_l2 = 0, rhs_l2 = 0;
  double complementarity_l1 = 0, objective_scale = 0;
  int worst_primal_row = -1, worst_primal_col = -1, worst_stationarity_col = -1;
  bool passed(double tolerance, KktCheck check = KktCheck::Full) const;
};
KktMetrics check_kkt(const presolve::LPInfoGpu*, const double*, const double*, const double*);
void print_kkt(const char* stage, const KktMetrics&, double tolerance, KktCheck check = KktCheck::Full);
}  // namespace gpu_preprocessor::lp::diagnostics
