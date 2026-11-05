// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

// Compatibility symbol; the active implementation lives in fixed_variable_tape.cu.
#include "gpu_preprocessor/lp/presolve/fixed_col_tape.hpp"

namespace gpu_preprocessor::lp::presolve {
double append_compact_fixed_col_tape_from_device(PostsolveTape* tape,
    const std::uint8_t* fixed_mask, const std::uint8_t* excluded_mask,
    const double* fixed_val, const std::uint8_t* keep_row, const double* c,
    const DeviceCsrMatrix& AT, std::int32_t n, const char* context,
    FixedColObjective objective) {
  return append_compact_fixed_variable_tape_from_device(tape, fixed_mask,
      excluded_mask, fixed_val, keep_row, c, AT, n, context,
      static_cast<FixedVariableObjective>(static_cast<int>(objective)));
}
}  // namespace gpu_preprocessor::lp::presolve
