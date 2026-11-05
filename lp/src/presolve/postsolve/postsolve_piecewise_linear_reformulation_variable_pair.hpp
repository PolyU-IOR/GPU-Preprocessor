// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"

namespace gpu_preprocessor::lp::presolve::postsolve {
void restore_piecewise_linear_reformulation_variable_pair(
    double* x, const PiecewiseLinearReformulationVariablePairRecoveryStep& recovery);
void restore_device_piecewise_linear_reformulation_variable_pair(
    double* x, const DeviceRecoveryGroup& recovery);
void validate_piecewise_linear_reformulation_variable_pair_recovery(
    const PiecewiseLinearReformulationVariablePairRecoveryStep& recovery,
    std::int32_t original_columns);
}  // namespace gpu_preprocessor::lp::presolve::postsolve
