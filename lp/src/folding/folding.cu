// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "folding_internal.cuh"
#include "gpu_preprocessor/lp/diagnostics/postsolve_observer.hpp"

#include "gpu_preprocessor/lp/folding/folding.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_postsolve.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_presolve.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpu_preprocessor::lp::folding {
namespace {

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

template <class T>
std::vector<T> copy_to_host(const T* device, std::size_t size, const char* context) {
  std::vector<T> values(size);
  if (size == 0) {
    return values;
  }
  throw_if_cuda_error(cudaMemcpy(values.data(), device, sizeof(T) * size, cudaMemcpyDeviceToHost), context);
  return values;
}

double seconds_since(std::chrono::steady_clock::time_point start,
                     std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double>(stop - start).count();
}

template <typename Cleanup>
struct ScopeExit {
  explicit ScopeExit(Cleanup cleanup_in) : cleanup(std::move(cleanup_in)) {}
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;
  ~ScopeExit() {
    if (active) {
      cleanup();
    }
  }
  void release() { active = false; }

  Cleanup cleanup;
  bool active = true;
};

template <typename Cleanup>
ScopeExit<Cleanup> make_scope_exit(Cleanup cleanup) {
  return ScopeExit<Cleanup>(std::move(cleanup));
}

}  // namespace

FoldingRunSummary run_folding(const presolve::LPInfoGpu& original_model,
                              double tolerance,
                              bool verbose,
                              bool preserve_nonzero_coefficients) {
  const auto total_start = std::chrono::steady_clock::now();
  FoldingRunSummary summary;
  summary.original_rows = original_model.A.rows;
  summary.original_cols = original_model.A.cols;
  summary.folded_rows = original_model.A.rows;
  summary.folded_cols = original_model.A.cols;

  if (!std::isfinite(tolerance) || tolerance < 0.0) {
    throw std::invalid_argument("folding tolerance must be finite and nonnegative");
  }
  // Empty-sided LPs are handled by the current presolver, without zero-block
  // folding kernel launches or zero-length cuSPARSE vectors.
  if (original_model.A.rows == 0 || original_model.A.cols == 0) {
    return summary;
  }

  FoldingWorkspace workspace;
  auto workspace_guard = make_scope_exit([&]() { free_workspace(workspace); });
  auto summary_guard = make_scope_exit([&]() { free_folded_lp(summary); });
  init_workspace(workspace, original_model.A.rows, original_model.A.cols);
  const bool refinement_ok =
      refine_color(original_model, workspace, tolerance, verbose, &summary.profile.refinement);
  if (!refinement_ok) {
    free_workspace(workspace);
    workspace_guard.release();
    throw_if_cuda_error(cudaDeviceSynchronize(), "folding completion");
    summary.profile.total_seconds =
        seconds_since(total_start, std::chrono::steady_clock::now());
    summary_guard.release();
    return summary;
  }

  if (workspace.num_row_color == original_model.A.rows &&
      workspace.num_col_color == original_model.A.cols) {
    free_workspace(workspace);
    workspace_guard.release();
    throw_if_cuda_error(cudaDeviceSynchronize(), "folding completion");
    summary.profile.total_seconds =
        seconds_since(total_start, std::chrono::steady_clock::now());
    summary_guard.release();
    return summary;
  }

  summary.folded_lp = reduce_size(workspace, original_model,
      preserve_nonzero_coefficients ? 0.0 : tolerance, &summary.profile.reduce);
  summary.owns_folded_lp = true;
  summary.map = build_device_map(workspace, &summary.profile.map);
  summary.applied = summary.folded_lp.A.rows != original_model.A.rows ||
                    summary.folded_lp.A.cols != original_model.A.cols;
  summary.folded_rows = summary.folded_lp.A.rows;
  summary.folded_cols = summary.folded_lp.A.cols;
  if (!summary.applied) {
    free_lp_device(summary.folded_lp);
    summary.owns_folded_lp = false;
  }
  free_workspace(workspace);
  workspace_guard.release();
  throw_if_cuda_error(cudaDeviceSynchronize(), "folding completion");
  summary.profile.total_seconds =
      seconds_since(total_start, std::chrono::steady_clock::now());
  summary_guard.release();
  return summary;
}

void free_folded_lp(FoldingRunSummary& summary) {
  free_folding_map_device(summary.map);
  if (!summary.owns_folded_lp) {
    summary.folded_lp = presolve::LPInfoGpu{};
    return;
  }
  free_lp_device(summary.folded_lp);
  summary.owns_folded_lp = false;
}

presolve::GpuPresolveSummary run_optional_gpu_presolve(
    const presolve::LPInfoGpu& lp, const presolve::PresolveParams& params,
    bool keep_reduced_lp) {
  if (params.enable_presolve) {
    return keep_reduced_lp ? presolve::run_gpu_presolve_with_reduced_lp(lp, params)
                           : presolve::run_gpu_presolve_with_record(lp, params);
  }
  presolve::GpuPresolveSummary summary;
  summary.original_rows = summary.reduced_rows = lp.A.rows;
  summary.original_cols = summary.reduced_cols = lp.A.cols;
  summary.reduced_nnz = lp.A.nnz;
  auto& record = summary.record;
  record.m0 = record.m1 = lp.A.rows;
  record.n0 = record.n1 = lp.A.cols;
  record.obj_constant_old = record.obj_constant_new = lp.obj_constant;
  record.row_red2org.resize(static_cast<std::size_t>(lp.A.rows));
  record.col_red2org.resize(static_cast<std::size_t>(lp.A.cols));
  std::iota(record.row_red2org.begin(), record.row_red2org.end(), 0);
  std::iota(record.col_red2org.begin(), record.col_red2org.end(), 0);
  record.row_org2red = record.row_red2org;
  record.col_org2red = record.col_red2org;
  if (keep_reduced_lp) {
    summary.reduced_lp = lp;  // Borrowed; the input owner retains this allocation.
  }
  return summary;
}

StartupPresolveState::~StartupPresolveState() {
  presolve::free_gpu_presolve_reduced_lp(summary);
}

static FoldingPipelineSummary run_gpu_presolve_with_folding_impl(const presolve::LPInfoGpu& lp,
                                                     const presolve::PresolveParams& params,
                                                     bool keep_reduced_lp,
                                                     bool keep_folded_lp) {
  FoldingPipelineSummary summary;
  summary.presolve_enabled = params.enable_presolve;
  auto summary_guard = make_scope_exit([&]() {
    presolve::free_gpu_presolve_reduced_lp(summary.presolve);
    free_folded_lp(summary.folding);
  });
  summary.original_rows = summary.startup_rows = summary.folded_rows = lp.A.rows;
  summary.original_cols = summary.startup_cols = summary.folded_cols = lp.A.cols;
  summary.startup_nnz = lp.A.nnz;

  if (!params.enable_folding) {
    // With no coordinate change between stages, retain the single-record path.
    summary.presolve = run_optional_gpu_presolve(lp, params, keep_reduced_lp);
    summary_guard.release();
    return summary;
  }

  presolve::PresolveBudget budget;
  presolve::LPInfoGpu prepared_input;
  const presolve::LPInfoGpu* folding_input = &lp;
  if (params.enable_presolve) {
    summary.startup_split = true;
    if (params.enable_startup && params.max_iters > 0) {
      auto startup = std::make_shared<StartupPresolveState>();
      startup->summary = presolve::run_gpu_presolve_stage(
          lp, params, presolve::PresolveStage::Startup, budget);
      summary.startup_seconds = startup->summary.elapsed_seconds;
      summary.startup_rows = summary.folded_rows = startup->summary.reduced_rows;
      summary.startup_cols = summary.folded_cols = startup->summary.reduced_cols;
      summary.startup_nnz = startup->summary.reduced_nnz;
      if (startup->summary.has_infeasible || startup->summary.has_unbounded) {
        // No later transform may run after a terminal Startup result.
        summary.presolve = std::move(startup->summary);
        // GpuPresolveSummary contains a raw ownership flag; moving its
        // descriptor does not clear that flag in the source.
        startup->summary.owns_reduced_lp = false;
        startup->summary.reduced_lp = {};
        if (!keep_reduced_lp) presolve::free_gpu_presolve_reduced_lp(summary.presolve);
        summary_guard.release();
        return summary;
      }
      if (startup->summary.has_change) {
        summary.startup = std::move(startup);
        folding_input = &summary.startup->summary.reduced_lp;
      }
      // No-hit discards the identity record and keeps the exact input pointers.
    }
  }

  // Startup may be disabled or have a zero iteration budget. Complete an
  // optional CSR-only input before folding in that case too.
  if (!folding_input->AT.rowPtr && params.prepare_startup_column_access) {
    prepared_input = *folding_input;
    params.prepare_startup_column_access(prepared_input, params.startup_column_access_context);
    folding_input = &prepared_input;
  }
  auto main_params = params;
  main_params.prepare_startup_column_access = nullptr;
  main_params.startup_column_access_context = nullptr;
  main_params.defer_startup_row_transpose = false;
  const bool poisson_response = summary.startup && summary.startup->summary.record.startup_poisson;
  if (poisson_response) {
    // Tiny Green responses can collide under ordinary folding signatures.
    // Keep the certified response and its inverse, and continue directly to
    // Main without discarding any nonzero response coefficients.
    main_params.zero_tol = 0.0;
    summary.folding.original_rows = summary.folding.folded_rows = folding_input->A.rows;
    summary.folding.original_cols = summary.folding.folded_cols = folding_input->A.cols;
    std::cout << "FOLDING skip=startup_poisson_applied\n";
  } else {
    summary.folding = run_folding(*folding_input, params.folding_tolerance, params.verbose);
  }
  summary.folding_applied = summary.folding.applied;
  summary.folded_rows = summary.folding.folded_rows;
  summary.folded_cols = summary.folding.folded_cols;
  if (summary.folding_applied && summary.startup) {
    // Main replay uses the independent folded basis. Unfolding needs only
    // its map, and Startup replay uses the record's retained row/response
    // basis (or caller input), never this intermediate working output.
    // Stream-0 release follows all folding reads; recovery records stay alive.
    presolve::free_gpu_presolve_reduced_lp(summary.startup->summary);
  }
  const auto& main_input = summary.folding_applied ? summary.folding.folded_lp : *folding_input;
  if (params.enable_presolve) {
    summary.presolve = presolve::run_gpu_presolve_stage(
        main_input, main_params, presolve::PresolveStage::Main, budget, keep_reduced_lp);
    summary.presolve.elapsed_seconds = budget.elapsed_seconds;
    if (summary.startup) {
      summary.presolve.has_change = true;
      summary.presolve.obj_constant_delta += summary.startup->summary.obj_constant_delta;
    }
  } else {
    summary.presolve = run_optional_gpu_presolve(main_input, params, keep_reduced_lp);
  }

  if (summary.folding_applied && !keep_folded_lp) {
    // Main borrows the folded input if no Apply materialized a replacement.
    // Transfer that allocation before dropping the folding owner. Startup's
    // recovery record keeps its own row/response basis for replay.
    if (keep_reduced_lp && !summary.presolve.owns_reduced_lp) {
      summary.presolve.owns_reduced_lp = summary.folding.owns_folded_lp;
      summary.folding.folded_lp = presolve::LPInfoGpu{};
    } else {
      free_lp_device(summary.folding.folded_lp);
    }
    summary.folding.owns_folded_lp = false;
  }
  summary_guard.release();
  return summary;
}

FoldingPipelineSummary run_gpu_presolve_with_folding(const presolve::LPInfoGpu& lp,
                                                     const presolve::PresolveParams& params,
                                                     bool keep_reduced_lp,
                                                     bool keep_folded_lp) {
  const auto start = std::chrono::steady_clock::now();
  auto summary = run_gpu_presolve_with_folding_impl(lp, params, keep_reduced_lp, keep_folded_lp);
  auto guard = make_scope_exit([&]() {
    presolve::free_gpu_presolve_reduced_lp(summary.presolve);
    free_folded_lp(summary.folding);
  });
  if (params.enable_presolve || params.enable_folding) {
    throw_if_cuda_error(cudaDeviceSynchronize(), "preprocessing completion");
    summary.preprocessing_seconds = seconds_since(start, std::chrono::steady_clock::now());
  }
  guard.release();
  return summary;
}

presolve::GpuPostsolveResult postsolve_and_unfold_device(
    double* x_red, double* y_red, double* z_red,
    const FoldingPipelineSummary& summary,
    const presolve::LPInfoGpu* original_model, double primal_tolerance) {
  if (!summary.presolve_enabled) {
    if (summary.folding_applied) {
      return unfold_solution_device(summary.folding.map, x_red, y_red, z_red);
    }
    presolve::GpuPostsolveResult post;
    post.n0 = summary.original_cols;
    post.m0 = summary.original_rows;
    auto guard = make_scope_exit([&]() {
      cudaFree(post.x_org); cudaFree(post.y_org); cudaFree(post.z_org);
    });
    auto copy = [](double*& out, const double* in, std::int32_t count) {
      if (!count) return;
      throw_if_cuda_error(cudaMalloc(&out, std::size_t(count) * sizeof(double)), "allocate identity point");
      throw_if_cuda_error(cudaMemcpy(out, in, std::size_t(count) * sizeof(double),
                                    cudaMemcpyDeviceToDevice), "copy identity point");
    };
    copy(post.x_org, x_red, post.n0);
    copy(post.y_org, y_red, post.m0);
    copy(post.z_org, z_red, post.n0);
    guard.release();
    return post;
  }
  if (summary.folding_applied && !summary.folding.owns_folded_lp) {
    throw std::invalid_argument("postsolve requires keep_folded_lp=true");
  }
  const auto* main_input = summary.folding_applied ? &summary.folding.folded_lp :
      (summary.startup ? &summary.startup->summary.reduced_lp : original_model);
  const auto& reduced = summary.presolve.reduced_lp;
  const bool has_reduced = reduced.A.rows == summary.presolve.reduced_rows &&
                           reduced.A.cols == summary.presolve.reduced_cols;
  diagnostics::observe_postsolve("reduced", has_reduced ? &reduced : nullptr, x_red, y_red, z_red);
  auto post = presolve::postsolve_gpu(x_red, y_red, z_red, summary.presolve.record, main_input);
  auto free_post = [](presolve::GpuPostsolveResult& p) {
    cudaFree(p.x_org); cudaFree(p.y_org); cudaFree(p.z_org);
    p = {};
  };
  auto post_guard = make_scope_exit([&]() { free_post(post); });
  diagnostics::observe_postsolve("main", main_input, post.x_org, post.y_org, post.z_org);
  if (summary.folding_applied) {
    auto unfolded = unfold_solution_device(summary.folding.map, post.x_org, post.y_org, post.z_org);
    free_post(post);
    post = unfolded;
    const auto* basis = summary.startup ? &summary.startup->summary.reduced_lp : original_model;
    if (basis && (basis->A.rows != post.m0 || basis->A.cols != post.n0)) basis = nullptr;
    diagnostics::observe_postsolve("unfold", basis, post.x_org, post.y_org, post.z_org);
  }
  if (summary.startup) {
    // Main replay -> unfold -> ordinary Startup replay -> S^T row inverse.
    // All stage handoffs remain on device. In particular, Main must never
    // replay a Startup record whose indices refer to pre-folding variables.
    auto restored = presolve::postsolve_gpu(post.x_org, post.y_org, post.z_org,
                                            summary.startup->summary.record, original_model);
    free_post(post);
    post = restored;
    diagnostics::observe_postsolve("startup", original_model, post.x_org, post.y_org, post.z_org);
  }
  if (original_model && primal_tolerance > 0.0 &&
      (summary.startup || !summary.folding_applied)) {
    // The pair map belongs to Startup's original coordinates. Only project
    // after Main replay, unfolding and Startup replay have all completed.
    const auto& original_record = summary.startup ? summary.startup->summary.record
                                                 : summary.presolve.record;
    presolve::polish_variable_pair_primal_gpu(post.x_org, original_record,
                                            *original_model, primal_tolerance);
  }
  diagnostics::observe_postsolve("final", original_model, post.x_org, post.y_org, post.z_org);
  post_guard.release();
  return post;
}

UnfoldedSolutionHost postsolve_and_unfold_to_host(double* x_red,
                                                  double* y_red,
                                                  double* z_red,
                                                  const FoldingPipelineSummary& summary,
                                                  const presolve::LPInfoGpu* original_model,
                                                  double primal_tolerance) {
  // Retain the allocation-free identity host path for callers with no stages.
  if (!summary.presolve_enabled && !summary.folding_applied) {
    return {copy_to_host(x_red, static_cast<std::size_t>(summary.original_cols), "copy identity x"),
            copy_to_host(y_red, static_cast<std::size_t>(summary.original_rows), "copy identity y"),
            copy_to_host(z_red, static_cast<std::size_t>(summary.original_cols), "copy identity z")};
  }
  auto post = postsolve_and_unfold_device(x_red, y_red, z_red, summary,
                                         original_model, primal_tolerance);
  auto guard = make_scope_exit([&]() {
    cudaFree(post.x_org); cudaFree(post.y_org); cudaFree(post.z_org);
  });
  UnfoldedSolutionHost result;
  result.x = copy_to_host(post.x_org, post.n0, "cudaMemcpy postsolve x");
  result.y = copy_to_host(post.y_org, post.m0, "cudaMemcpy postsolve y");
  result.z = copy_to_host(post.z_org, post.n0, "cudaMemcpy postsolve z");
  return result;
}

}  // namespace gpu_preprocessor::lp::folding
