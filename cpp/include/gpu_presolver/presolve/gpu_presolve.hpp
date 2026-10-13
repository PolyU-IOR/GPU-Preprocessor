#pragma once

#include "gpu_presolver/offset.hpp"

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_presolver::presolve {

struct GpuPresolveSummary {
  std::int32_t original_rows = 0;
  std::int32_t original_cols = 0;
  std::int32_t reduced_rows = 0;
  std::int32_t reduced_cols = 0;
  NnzOffset reduced_nnz = 0;
  std::int32_t iterations = 0;
  bool stopped_at_iteration_limit = false;
  bool has_change = false;  // Any committed model update, including bounds only.
  double elapsed_seconds = 0.0;
  double obj_constant_delta = 0.0;
  bool has_infeasible = false;
  bool has_unbounded = false;
  PresolveRecordGpu record;
  LPInfoGpu reduced_lp;
  bool owns_reduced_lp = false;
};

GpuPresolveSummary run_gpu_presolve_with_record(const LPInfoGpu& lp,
                                                const PresolveParams& params);

GpuPresolveSummary run_gpu_presolve_with_reduced_lp(const LPInfoGpu& lp,
                                                    const PresolveParams& params);

void free_gpu_presolve_reduced_lp(GpuPresolveSummary& summary);

}  // namespace gpu_presolver::presolve
