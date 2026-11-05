// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/presolve/fixed_variable_tape.hpp"

namespace gpu_preprocessor::lp::presolve {
// Retain a distinct enum type so the earlier mangled function symbol is unchanged.
enum class FixedColObjective { Compute, Skip };
double append_compact_fixed_col_tape_from_device(PostsolveTape* tape,
    const std::uint8_t* fixed_mask, const std::uint8_t* excluded_mask,
    const double* fixed_val, const std::uint8_t* keep_row, const double* c,
    const DeviceCsrMatrix& AT, std::int32_t n, const char* context,
    FixedColObjective objective = FixedColObjective::Compute);
}  // namespace gpu_preprocessor::lp::presolve
