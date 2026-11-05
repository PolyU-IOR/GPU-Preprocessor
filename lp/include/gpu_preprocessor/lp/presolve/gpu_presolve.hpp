// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/offset.hpp"

#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_preprocessor::lp::presolve {

struct GpuPresolveSummary {
  std::int32_t original_rows = 0;
  std::int32_t original_cols = 0;
  std::int32_t reduced_rows = 0;
  std::int32_t reduced_cols = 0;
  NnzOffset reduced_nnz = 0;
  std::int32_t iterations = 0;
  bool stopped_at_iteration_limit = false;
  double elapsed_seconds = 0.0;
  double obj_constant_delta = 0.0;
  bool has_infeasible = false;
  bool has_unbounded = false;
  bool has_change = false;
  PresolveRecordGpu record;
  LPInfoGpu reduced_lp;
  bool owns_reduced_lp = false;
};

enum class PresolveStage { Full, Startup, Main };

// Shared by Startup and Main across folding. Folding has its own timer and
// does not consume this presolve-only time budget. Cleanup keeps its existing
// semantics: it can finish after the iteration budget has been consumed.
struct PresolveBudget {
  std::int32_t iterations = 0;
  double elapsed_seconds = 0.0;
};

// Startup never executes Cleanup; Main never executes Startup. The returned
// elapsed time is for this call; iterations is the cumulative budget count.
GpuPresolveSummary run_gpu_presolve_stage(const LPInfoGpu& lp,
                                          const PresolveParams& params,
                                          PresolveStage stage,
                                          PresolveBudget& budget,
                                          bool keep_reduced_lp = true);

GpuPresolveSummary run_gpu_presolve_with_record(const LPInfoGpu& lp,
                                                const PresolveParams& params);

GpuPresolveSummary run_gpu_presolve_with_reduced_lp(const LPInfoGpu& lp,
                                                    const PresolveParams& params);

// Queue owned output releases after its users on CUDA stream 0, then clear
// the host descriptors. Borrowed original/folded input is never freed.
void free_gpu_presolve_reduced_lp(GpuPresolveSummary& summary);

}  // namespace gpu_preprocessor::lp::presolve
