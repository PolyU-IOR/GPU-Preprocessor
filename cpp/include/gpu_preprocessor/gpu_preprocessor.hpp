#pragma once

#include "gpu_presolver/folding/folding.hpp"

namespace gpu_preprocessor {

using Parameters = gpu_presolver::presolve::PresolveParams;
// Device CSR matrices A and AT = A^T; device vectors c, AL, AU, l, and u.
// Indices are zero-based int32, sparse offsets use NnzOffset, and values are
// double. obj_constant is a host scalar.
using DeviceLp = gpu_presolver::presolve::LPInfoGpu;
using Summary = gpu_presolver::folding::FoldingPipelineSummary;
using Solution = gpu_presolver::folding::UnfoldedSolutionHost;

// Run folding, then presolve; both stages are enabled by default.
// The input remains caller-owned. Keep it and the returned summary alive for
// recovery. Do not copy owning summaries. The reduced model can borrow input
// storage; copy its arrays if the solver modifies them or takes ownership.
inline Summary preprocess(const DeviceLp& input, const Parameters& params = {}) {
  return gpu_presolver::folding::run_gpu_presolve_with_folding(input, params, true, true);
}

// x (primal), y (row duals), and z (reduced costs) are caller-owned device
// arrays in reduced-LP order, with reduced_cols/rows/cols entries. The dual
// convention is c - A^T y - z = 0. The returned x/y/z vectors are on the host
// in original-model order; validate their feasibility and KKT conditions.
inline Solution postprocess(double* x, double* y, double* z,
                            const Summary& summary, const DeviceLp& original) {
  return gpu_presolver::folding::postsolve_and_unfold_to_host(x, y, z, summary, &original);
}

inline void free_summary(Summary& summary) {
  gpu_presolver::presolve::free_gpu_presolve_reduced_lp(summary.presolve);
  gpu_presolver::folding::free_folded_lp(summary.folding);
  summary = Summary{};
}

}  // namespace gpu_preprocessor
