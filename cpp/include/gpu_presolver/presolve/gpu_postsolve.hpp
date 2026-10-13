#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <vector>

namespace gpu_presolver::presolve {

// Absolute-value epigraph recovery helper.  Use postsolve_gpu for any record that
// may contain ordinary tape entries or interleaved primal-recovery checkpoints.
void postsolve_restore_absolute_value_epigraph_gpu(
    double* x_org,
    const std::vector<AbsoluteValueEpigraphPrimalRecoveryStep>& recoveries,
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

}  // namespace gpu_presolver::presolve
