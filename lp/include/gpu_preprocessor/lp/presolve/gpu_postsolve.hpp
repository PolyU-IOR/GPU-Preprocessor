// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"

#include <vector>

namespace gpu_preprocessor::lp::presolve {

// Optional bounded primal projection for certified homogeneous antipodal pairs.
// Returns true only after accepting a candidate checked against the original LP.
// Unsupported structure or failed projection leaves x_org byte-for-byte intact.
bool polish_variable_pair_primal_gpu(double* x_org, const PresolveRecordGpu& record,
                                    const LPInfoGpu& original_model, double primal_tolerance);

// Absolute-value epigraph recovery helper.  Use postsolve_gpu for any record that
// may contain ordinary tape entries or interleaved primal-recovery checkpoints.
void postsolve_restore_piecewise_linear_reformulation_absolute_value_gpu(
    double* x_org,
    const std::vector<PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep>& recoveries,
    const double* original_l = nullptr);

GpuPostsolveResult postsolve_gpu(double* x_red,
                                 double* y_red,
                                 double* z_red,
                                 const PresolveRecordGpu& record,
                                 const double* original_l = nullptr);

GpuPostsolveResult postsolve_gpu(double* x_red,
                                 double* y_red,
                                 double* z_red,
                                 const PresolveRecordGpu& record,
                                 const LPInfoGpu* original_model,
                                 const double* original_l = nullptr);

}  // namespace gpu_preprocessor::lp::presolve
