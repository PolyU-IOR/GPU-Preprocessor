// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/folding/folding.hpp"

namespace gpu_preprocessor::lp {

using Parameters = gpu_preprocessor::lp::presolve::PresolveParams;
// Device CSR matrices A and AT = A^T; device vectors c, AL, AU, l, and u.
// Indices are zero-based int32, sparse offsets use NnzOffset, and values are
// double. obj_constant is a host scalar.
using DeviceLp = gpu_preprocessor::lp::presolve::LPInfoGpu;
using Summary = gpu_preprocessor::lp::folding::FoldingPipelineSummary;
using Solution = gpu_preprocessor::lp::folding::UnfoldedSolutionHost;
using DeviceSolution = gpu_preprocessor::lp::presolve::GpuPostsolveResult;

// Default pipeline: Startup -> folding -> Main. Presolve and folding default on;
// set enable_folding = false to skip folding.
// The input remains caller-owned. Keep it and the returned summary alive for
// recovery. Do not copy owning summaries. The reduced model can borrow input
// storage; copy its arrays if the solver modifies them or takes ownership.
inline Summary preprocess(const DeviceLp& input, const Parameters& params = {}) {
  return gpu_preprocessor::lp::folding::run_gpu_presolve_with_folding(input, params, true, true);
}

// x (primal), y (row duals), and z (reduced costs) are caller-owned device
// arrays in reduced-LP order, with reduced_cols/rows/cols entries. The dual
// convention is c - A^T y - z = 0. The returned x/y/z vectors are on the host
// in original-model order.
inline Solution postprocess(double* x, double* y, double* z,
                            const Summary& summary, const DeviceLp& original,
                            double primal_tolerance = 0.0) {
  return gpu_preprocessor::lp::folding::postsolve_and_unfold_to_host(
      x, y, z, summary, &original, primal_tolerance);
}

// Same recovery with owning device output. Free x_org/y_org/z_org with cudaFree
// after their final readers. Reduced input vectors remain caller-owned.
inline DeviceSolution postprocess_device(double* x, double* y, double* z,
                                         const Summary& summary, const DeviceLp& original,
                                         double primal_tolerance = 0.0) {
  return gpu_preprocessor::lp::folding::postsolve_and_unfold_device(
      x, y, z, summary, &original, primal_tolerance);
}

inline void free_summary(Summary& summary) {
  gpu_preprocessor::lp::presolve::free_gpu_presolve_reduced_lp(summary.presolve);
  gpu_preprocessor::lp::folding::free_folded_lp(summary.folding);
  summary = Summary{};
}

}  // namespace gpu_preprocessor::lp
