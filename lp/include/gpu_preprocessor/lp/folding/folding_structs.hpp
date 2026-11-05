// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/offset.hpp"
#pragma once

#include "gpu_preprocessor/lp/presolve/gpu_presolve.hpp"
#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace gpu_preprocessor::lp::folding {

struct FoldingRefinementProfile {
  double total_seconds = 0.0;
  double init_color_seconds = 0.0;
  double spmv_at_seconds = 0.0;
  double refine_col_seconds = 0.0;
  double spmv_a_seconds = 0.0;
  double refine_row_seconds = 0.0;
  std::int32_t rounds = 0;
  bool refinement_ok = false;
  std::int32_t final_row_colors = 0;
  std::int32_t final_col_colors = 0;
};

struct FoldingReduceProfile {
  double total_seconds = 0.0;
  double count_color_seconds = 0.0;
  double emit_triplet_seconds = 0.0;
  double sort_reduce_seconds = 0.0;
  double copy_unique_to_host_seconds = 0.0;
  double host_filter_rebuild_seconds = 0.0;
  double host_csr_build_seconds = 0.0;
  double copy_reduced_matrix_to_device_seconds = 0.0;
  double transpose_seconds = 0.0;
  double sum_cost_seconds = 0.0;
  double row_bounds_seconds = 0.0;
  double col_bounds_seconds = 0.0;
  NnzOffset unique_entries = 0;
  NnzOffset kept_entries = 0;
};

struct FoldingMapProfile {
  double total_seconds = 0.0;
  double copy_color_ids_seconds = 0.0;
  double count_color_seconds = 0.0;
  double scale_seconds = 0.0;
};

struct FoldingProfile {
  double total_seconds = 0.0;
  FoldingRefinementProfile refinement;
  FoldingReduceProfile reduce;
  FoldingMapProfile map;
};

struct FoldingMapHost {
  std::int32_t original_rows = 0;
  std::int32_t original_cols = 0;
  std::int32_t reduced_rows = 0;
  std::int32_t reduced_cols = 0;
  std::vector<std::int32_t> row_color_id;
  std::vector<std::int32_t> col_color_id;
  std::vector<double> row_scale;
  std::vector<double> col_scale;
};

struct FoldingMapDevice {
  std::int32_t original_rows = 0;
  std::int32_t original_cols = 0;
  std::int32_t reduced_rows = 0;
  std::int32_t reduced_cols = 0;
  std::int32_t* row_color_id = nullptr;
  std::int32_t* col_color_id = nullptr;
  double* row_scale = nullptr;
  double* col_scale = nullptr;
  bool owns_buffers = false;
};

struct FoldingRunSummary {
  bool applied = false;
  std::int32_t original_rows = 0;
  std::int32_t original_cols = 0;
  std::int32_t folded_rows = 0;
  std::int32_t folded_cols = 0;
  FoldingProfile profile;
  FoldingMapDevice map;
  presolve::LPInfoGpu folded_lp;
  bool owns_folded_lp = false;
};

// Owns the immutable input basis used by Main recovery after unfolding.
// The row transform, when present, stays in this stage's recovery record.
struct StartupPresolveState {
  presolve::GpuPresolveSummary summary;
  StartupPresolveState() = default;
  StartupPresolveState(const StartupPresolveState&) = delete;
  StartupPresolveState& operator=(const StartupPresolveState&) = delete;
  ~StartupPresolveState();
};

struct FoldingPipelineSummary {
  // Wall time on an uploaded model, including all stage-owned workspaces.
  double preprocessing_seconds = 0.0;
  // Retained for source compatibility. Workspace lifetimes are now included
  // in the stage timers, so separate setup/release values remain zero.
  double workspace_setup_seconds = 0.0;
  double workspace_release_seconds = 0.0;
  bool presolve_enabled = true;
  bool folding_applied = false;
  bool startup_split = false;
  std::int32_t original_rows = 0;
  std::int32_t original_cols = 0;
  std::int32_t startup_rows = 0;
  std::int32_t startup_cols = 0;
  NnzOffset startup_nnz = 0;
  double startup_seconds = 0.0;
  std::int32_t folded_rows = 0;
  std::int32_t folded_cols = 0;
  FoldingRunSummary folding;
  // Null for an unchanged Startup: do not add an identity recovery pass.
  std::shared_ptr<StartupPresolveState> startup;
  // Main's coordinate system/record, with cumulative presolve time and
  // iteration count. Folding time is reported separately.
  presolve::GpuPresolveSummary presolve;
};

struct UnfoldedSolutionHost {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
};

}  // namespace gpu_preprocessor::lp::folding
