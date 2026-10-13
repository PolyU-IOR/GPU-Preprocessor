#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_presolver::presolve {

// Device-owned deletion certificate for the strict unit set-covering rule.
// `applicable` is set only after every active row, column, coefficient, and
// row witness has passed the full detector.
struct ColumnDominationAnalysisGpu {
  bool applicable = false;
  std::int32_t candidate_count = 0;
  std::uint8_t* delete_col = nullptr;
};

// Sampled rejection filter only.  A true result never authorizes a model
// change; analyze_column_domination still performs a full certificate.
bool quick_probe_column_domination(const LPInfoGpu& lp,
                                         const PresolveParams& params);

ColumnDominationAnalysisGpu analyze_column_domination(
    const LPInfoGpu& lp,
    const PresolveParams& params);

void build_column_domination_plan(
    PresolvePlanGpu& plan,
    const LPInfoGpu& lp,
    const ColumnDominationAnalysisGpu& analysis,
    const PresolveParams& params);

void free_column_domination_analysis(
    ColumnDominationAnalysisGpu& analysis);

}  // namespace gpu_presolver::presolve
