// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/cleanup_workspace.hpp"
#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_presolve_kernels.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_parallel_rows.hpp"
#include "gpu_preprocessor/lp/presolve/fixed_col_tape.hpp"

using gpu_preprocessor::lp::presolve::PostsolveReductionType;
static_assert(int(PostsolveReductionType::FixedVariable) == 0);
static_assert(PostsolveReductionType::FixedCol == PostsolveReductionType::FixedVariable);
static_assert(PostsolveReductionType::FixedColInf == PostsolveReductionType::InfiniteBoundDualFix);
static_assert(PostsolveReductionType::SubCol == PostsolveReductionType::SingletonColumnSubstitution);
static_assert(PostsolveReductionType::LhsChange == PostsolveReductionType::RowLowerBoundChange);
static_assert(PostsolveReductionType::RhsChange == PostsolveReductionType::RowUpperBoundChange);
static_assert(PostsolveReductionType::EqToIneq == PostsolveReductionType::RowDualShift);
static_assert(PostsolveReductionType::BoundChangeNoRow == PostsolveReductionType::VariableBoundChange);
static_assert(PostsolveReductionType::BoundChangeTheRow == PostsolveReductionType::RowImpliedBoundChange);
static_assert(int(PostsolveReductionType::AddedRow) == 6);
static_assert(int(PostsolveReductionType::AddedRows) == 7);
static_assert(int(PostsolveReductionType::BoundPackedBatch) == 19);

int main() {
    // Existing headers and out-of-line symbols remain usable by older callers.
    using namespace gpu_preprocessor::lp::presolve;
    auto volatile reduction = &cleanup_reduction_temp_bytes;
    auto volatile trace = &trace_cleanup_epoch;
    auto volatile degrees = &compute_col_nnz;
    auto volatile singleton = &compute_singleton_column_support;
    auto volatile parallel = &parallel_rows_scratch_bytes;
    auto volatile fixed_legacy = &append_compact_fixed_col_tape_from_device;
    auto volatile fixed_current = &append_compact_fixed_variable_tape_from_device;
    DeviceCsrMatrix empty;
    const double old_delta = fixed_legacy(nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, empty, 0, "compatibility", FixedColObjective::Skip);
    const double new_delta = fixed_current(nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, empty, 0, "compatibility", FixedVariableObjective::Skip);
    return reduction && trace && degrees && singleton && parallel &&
           fixed_legacy && fixed_current && old_delta == new_delta ? 0 : 1;
}
