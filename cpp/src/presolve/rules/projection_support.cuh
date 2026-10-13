#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

// Projection batches only remove rows/columns; surviving coefficients stay intact.
inline LPInfoGpu projection_view(const LPInfoGpu& lp, const PresolvePlanGpu* plan) {
  LPInfoGpu view = lp;
  if (plan) {
    if (plan->new_c) view.c = plan->new_c;
    if (plan->new_l) view.l = plan->new_l;
    if (plan->new_u) view.u = plan->new_u;
    if (plan->new_AL) view.AL = plan->new_AL;
    if (plan->new_AU) view.AU = plan->new_AU;
  }
  return view;
}

// Capture the incidence before later kernels modify the shared masks.
template <int Count>
__device__ bool projection_rows(const NnzOffset* ptr, const std::int32_t* rows,
                                const std::uint8_t* keep_row, int col,
                                std::int32_t* incident_rows) {
  if (!keep_row && ptr[col + 1] - ptr[col] != Count) return false;
  int size = 0;
  for (auto p = ptr[col]; p < ptr[col + 1]; ++p) {
    if (keep_row && !keep_row[rows[p]]) continue;
    if (size == Count) return false;
    incident_rows[std::size_t(col) * Count + size++] = rows[p];
  }
  return size == Count;
}

}  // namespace gpu_presolver::presolve
