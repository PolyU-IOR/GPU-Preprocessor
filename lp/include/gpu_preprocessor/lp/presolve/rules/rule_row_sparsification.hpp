// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once
#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"

namespace gpu_preprocessor::lp::presolve {

struct RowSparsificationReport {
  const char* reason = "off";
  std::int64_t old_nnz = 0, new_nnz = 0;
  int eligible_rows = 0, candidates = 0, changed_rows = 0, max_depth = 0;
  double seconds = 0;
};

// The remaining presolver operates in this immutable row basis. Its recovery
// must use this model before applying S^T to the recovered row duals.
struct StartupRowSparsification {
  LPInfoGpu transformed;
  int* parents = nullptr;
  int* alpha = nullptr;
  StartupRowSparsification() = default;
  StartupRowSparsification(const StartupRowSparsification&) = delete;
  StartupRowSparsification& operator=(const StartupRowSparsification&) = delete;
  ~StartupRowSparsification();
};

// No-hit leaves every input pointer/value unchanged and returns no recovery state.
std::shared_ptr<StartupRowSparsification> run_startup_row_sparsification(
    const LPInfoGpu& input, const PresolveParams& params, RowSparsificationReport& report);
// Implemented in postsolve/postsolve_row_sparsification.cu; applies y = S^T y'.
void restore_startup_row_duals(const StartupRowSparsification& state, double* y);
void print_startup_row_sparsification(const RowSparsificationReport& report);

}  // namespace gpu_preprocessor::lp::presolve
