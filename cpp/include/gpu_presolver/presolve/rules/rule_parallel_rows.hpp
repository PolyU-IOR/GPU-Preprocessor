#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

std::size_t parallel_rows_scratch_bytes(std::size_t rows);

// Preserves the original scalar/warp hash definition selected by global mean length.
void compute_parallel_row_hashes(unsigned long long* output, const DeviceCsrMatrix& matrix,
                                   const std::uint8_t* keep_row, double zero_tol);

void apply_rule_parallel_rows(PresolvePlanGpu &plan, const LPInfoGpu &lp,
                               const PresolveStatsGpu &stats,
                               const PresolveParams &pparams);

} // namespace gpu_presolver::presolve
