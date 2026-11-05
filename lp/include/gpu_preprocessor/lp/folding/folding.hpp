// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/folding/folding_structs.hpp"

#include <vector>

namespace gpu_preprocessor::lp::folding {

std::vector<double> fold_initial_vector(const std::vector<double>* value,
                                        const std::vector<std::int32_t>& color_index,
                                        std::int32_t reduced_length);

FoldingRunSummary run_folding(const presolve::LPInfoGpu& original_model,
                              double tolerance,
                              bool verbose = false,
                              bool preserve_nonzero_coefficients = false);

void free_folded_lp(FoldingRunSummary& summary);

FoldingMapHost copy_map_to_host(const FoldingMapDevice& map);

void unfold_solution(const FoldingMapHost& map,
                     const std::vector<double>& x_red,
                     const std::vector<double>& y_red,
                     const std::vector<double>& z_red,
                     std::vector<double>* x,
                     std::vector<double>* y,
                     std::vector<double>* z);

UnfoldedSolutionHost unfold_solution_to_host(const FoldingMapDevice& map,
                                             const std::vector<double>& x_red,
                                             const std::vector<double>& y_red,
                                             const std::vector<double>& z_red);

// Unfold device vectors directly and copy only the final original-space vectors.
// The device overload transfers ownership of all returned vectors to caller.
presolve::GpuPostsolveResult unfold_solution_device(const FoldingMapDevice& map,
                                                    const double* x_red,
                                                    const double* y_red,
                                                    const double* z_red);

UnfoldedSolutionHost unfold_solution_device_to_host(const FoldingMapDevice& map,
                                                    const double* x_red,
                                                    const double* y_red,
                                                    const double* z_red);

// Run the current presolver, or return an identity record without invoking it.
presolve::GpuPresolveSummary run_optional_gpu_presolve(
    const presolve::LPInfoGpu& lp, const presolve::PresolveParams& params,
    bool keep_reduced_lp);

FoldingPipelineSummary run_gpu_presolve_with_folding(const presolve::LPInfoGpu& lp,
                                                     const presolve::PresolveParams& params,
                                                     bool keep_reduced_lp,
                                                     bool keep_folded_lp);

// Restore the same point as the host wrapper, retaining it on device. Inputs
// remain borrowed; caller owns all three returned vectors and frees them with
// cudaFree after their final readers finish. Recovery and polish order match
// the host API, including diagnostics at the existing stage boundaries.
presolve::GpuPostsolveResult postsolve_and_unfold_device(
    double* x_red, double* y_red, double* z_red,
    const FoldingPipelineSummary& summary,
    const presolve::LPInfoGpu* original_model = nullptr,
    double primal_tolerance = 0.0);

UnfoldedSolutionHost postsolve_and_unfold_to_host(double* x_red,
                                                  double* y_red,
                                                  double* z_red,
                                                  const FoldingPipelineSummary& summary,
                                                  const presolve::LPInfoGpu* original_model = nullptr,
                                                  double primal_tolerance = 0.0);

}  // namespace gpu_preprocessor::lp::folding
