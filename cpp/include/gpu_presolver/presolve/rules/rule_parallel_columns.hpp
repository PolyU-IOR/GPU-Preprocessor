#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

std::size_t parallel_columns_scratch_bytes(std::size_t rows, std::size_t cols,
                                            bool parallel_verify);

void apply_rule_parallel_columns(PresolvePlanGpu &plan, const LPInfoGpu &lp,
                                  const PresolveStatsGpu &stats,
                                  const PresolveParams &pparams);

} // namespace gpu_presolver::presolve
