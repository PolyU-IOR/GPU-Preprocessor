// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_preprocessor::lp::presolve {

enum class FixedVariableObjective { Compute, Skip };

// Packs FixedVariable postsolve records on the GPU and transfers only selected
// columns and coefficients belonging to rows that remain in the model.
// `tape` may be null when only the stable fixed-objective delta is needed.
// Skip emits the same records but returns zero without objective accumulation.
// For a resident tape, Skip never downloads the packed payload to the CPU.
// `excluded_mask` may be null; otherwise nonzero entries are skipped.
double append_compact_fixed_variable_tape_from_device(PostsolveTape* tape,
                                               const std::uint8_t* fixed_mask,
                                               const std::uint8_t* excluded_mask,
                                               const double* fixed_val,
                                               const std::uint8_t* keep_row,
                                               const double* c,
                                               const DeviceCsrMatrix& AT,
                                               std::int32_t n,
                                               const char* context,
                                               FixedVariableObjective objective = FixedVariableObjective::Compute);

}  // namespace gpu_preprocessor::lp::presolve
