// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"
#include <cstddef>

namespace gpu_preprocessor::lp::presolve {
// Historical helper symbols retained for existing external callers. Current
// presolve uses live-statistics refresh and active scratch-workspace helpers.
std::size_t cleanup_reduction_temp_bytes(const double*, double*, int);
void trace_cleanup_epoch(const PresolveRecordGpu&, const PresolvePlanGpu&,
                         const LPInfoGpu&, double);
void compute_col_nnz(std::int32_t*, const DeviceCsrMatrix&);
void compute_singleton_column_support(std::int32_t*, double*,
                                      const std::int32_t*, const DeviceCsrMatrix&);
std::size_t parallel_rows_scratch_bytes(std::size_t);
}  // namespace gpu_preprocessor::lp::presolve
