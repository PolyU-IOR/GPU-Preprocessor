#include "gpu_presolver/folding/folding.hpp"
#include "../src/folding/folding_internal.cuh"
#include <cstring>
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

template <class T>
T* copy_to_device(const std::vector<T>& values) {
  if (values.empty()) {
    return nullptr;
  }
  T* device = nullptr;
  check(cudaMalloc(&device, sizeof(T) * values.size()), "cudaMalloc");
  check(cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy H2D");
  return device;
}

template <class T>
std::vector<T> copy_to_host(T* device, std::size_t size) {
  std::vector<T> values(size);
  if (size == 0) {
    return values;
  }
  check(cudaMemcpy(values.data(), device, sizeof(T) * size, cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H");
  return values;
}

void free_lp(gpu_presolver::presolve::LPInfoGpu& lp) {
  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(lp.c);
  cudaFree(lp.AL);
  cudaFree(lp.AU);
  cudaFree(lp.l);
  cudaFree(lp.u);
  lp = gpu_presolver::presolve::LPInfoGpu{};
}

gpu_presolver::presolve::LPInfoGpu make_symmetric_2x2_lp() {
  using gpu_presolver::presolve::LPInfoGpu;
  LPInfoGpu lp;
  lp.A = {2, 2, 4,
          copy_to_device<NnzOffset>({0, 2, 4}),
          copy_to_device<std::int32_t>({0, 1, 0, 1}),
          copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};
  lp.AT = {2, 2, 4,
           copy_to_device<NnzOffset>({0, 2, 4}),
           copy_to_device<std::int32_t>({0, 1, 0, 1}),
           copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};
  lp.c = copy_to_device<double>({2.0, 2.0});
  lp.AL = copy_to_device<double>({4.0, 4.0});
  lp.AU = copy_to_device<double>({4.0, 4.0});
  lp.l = copy_to_device<double>({0.0, 0.0});
  lp.u = copy_to_device<double>({10.0, 10.0});
  return lp;
}

void test_fold_initial_vector_uses_first_representative() {
  const std::vector<double> value{3.0, 7.0, 11.0, 13.0};
  const std::vector<std::int32_t> color_index{1, 2, 1, 2};
  const std::vector<double> folded =
      gpu_presolver::folding::fold_initial_vector(&value, color_index, 2);
  assert((folded == std::vector<double>{3.0, 7.0}));
}

void test_run_folding_reduces_identical_rows_and_cols() {
  gpu_presolver::presolve::LPInfoGpu lp = make_symmetric_2x2_lp();
  gpu_presolver::folding::FoldingRunSummary summary =
      gpu_presolver::folding::run_folding(lp, 1.0e-8, false);
  assert(summary.applied);
  assert(summary.original_rows == 2);
  assert(summary.original_cols == 2);
  assert(summary.folded_rows == 1);
  assert(summary.folded_cols == 1);
  const gpu_presolver::folding::FoldingMapHost map =
      gpu_presolver::folding::copy_map_to_host(summary.map);
  assert((map.row_color_id == std::vector<std::int32_t>{1, 1}));
  assert((map.col_color_id == std::vector<std::int32_t>{1, 1}));
  assert(map.row_scale.size() == 2);
  assert(map.col_scale.size() == 2);
  assert(std::fabs(map.row_scale[0] - 0.5) < 1.0e-12);
  assert(std::fabs(map.col_scale[0] - 0.5) < 1.0e-12);
  assert(summary.folded_lp.A.rows == 1);
  assert(summary.folded_lp.A.cols == 1);
  assert(summary.folded_lp.A.nnz == 1);
  assert(summary.profile.reduce.copy_unique_to_host_seconds == 0.0);
  assert(summary.profile.reduce.host_filter_rebuild_seconds == 0.0);
  assert(summary.profile.reduce.host_csr_build_seconds == 0.0);
  gpu_presolver::folding::free_folded_lp(summary);
  free_lp(lp);
}

void test_unfold_solution_scales_duals_and_slacks() {
  gpu_presolver::presolve::LPInfoGpu lp = make_symmetric_2x2_lp();
  gpu_presolver::folding::FoldingRunSummary summary =
      gpu_presolver::folding::run_folding(lp, 1.0e-8, false);
  assert(summary.applied);
  assert(summary.map.original_rows == 2);
  assert(summary.map.original_cols == 2);
  assert(summary.map.reduced_rows == 1);
  assert(summary.map.reduced_cols == 1);

  const gpu_presolver::folding::UnfoldedSolutionHost unfolded =
      gpu_presolver::folding::unfold_solution_to_host(summary.map, {4.0}, {6.0}, {8.0});

  const std::vector<double>& x = unfolded.x;
  const std::vector<double>& y = unfolded.y;
  const std::vector<double>& z = unfolded.z;
  assert((x == std::vector<double>{4.0, 4.0}));
  assert((y == std::vector<double>{3.0, 3.0}));
  assert((z == std::vector<double>{4.0, 4.0}));

  gpu_presolver::folding::free_folded_lp(summary);
  free_lp(lp);
}

void test_postsolve_and_unfold_restores_original_dimensions() {
  gpu_presolver::presolve::LPInfoGpu lp = make_symmetric_2x2_lp();

  gpu_presolver::presolve::PresolveParams params;
  params.max_iters = 1;
  params.enable_infeasible_and_fixed_variables = false;
  params.enable_empty_rows = false;
  params.enable_singleton_rows = false;
  params.enable_infeasible_and_redundant_rows = false;
  params.enable_bound_strengthening = false;
  params.enable_parallel_rows = false;
  params.enable_empty_columns = false;
  params.enable_singleton_columns = false;

  params.enable_doubleton_equations = false;
  params.enable_dual_fix = false;
  params.enable_parallel_columns = false;
  params.enable_absolute_value_epigraph_reduction = false;

  params.enable_folding = true;

  gpu_presolver::folding::FoldingPipelineSummary summary =
      gpu_presolver::folding::run_gpu_presolve_with_folding(lp, params, false, true);
  assert(summary.folding_applied);
  assert(summary.folded_rows == 1);
  assert(summary.folded_cols == 1);
  assert(summary.presolve.reduced_rows == 1);
  assert(summary.presolve.reduced_cols == 1);

  double* x_red = copy_to_device<double>({4.0});
  double* y_red = copy_to_device<double>({6.0});
  double* z_red = copy_to_device<double>({8.0});
  const gpu_presolver::folding::UnfoldedSolutionHost unfolded =
      gpu_presolver::folding::postsolve_and_unfold_to_host(x_red, y_red, z_red, summary);

  assert((unfolded.x == std::vector<double>{4.0, 4.0}));
  assert(unfolded.y.size() == 2);
  assert(unfolded.z.size() == 2);
  assert(std::isfinite(unfolded.y[0]));
  assert(std::isfinite(unfolded.y[1]));
  assert(std::isfinite(unfolded.z[0]));
  assert(std::isfinite(unfolded.z[1]));

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  gpu_presolver::folding::free_folded_lp(summary.folding);
  free_lp(lp);
}

// A true equitable partition with no parallel rows or columns.
void test_equitable_partition_and_original_kkt() {
  auto lp = make_symmetric_2x2_lp();
  const std::vector<double> values{1.0, 2.0, 2.0, 1.0};
  const std::vector<double> rhs{3.0, 3.0};
  check(cudaMemcpy(lp.A.nzVal, values.data(), 4 * sizeof(double), cudaMemcpyHostToDevice), "set A");
  check(cudaMemcpy(lp.AT.nzVal, values.data(), 4 * sizeof(double), cudaMemcpyHostToDevice), "set AT");
  check(cudaMemcpy(lp.AL, rhs.data(), 2 * sizeof(double), cudaMemcpyHostToDevice), "set AL");
  check(cudaMemcpy(lp.AU, rhs.data(), 2 * sizeof(double), cudaMemcpyHostToDevice), "set AU");
  lp.obj_constant = 7.0;
  gpu_presolver::presolve::PresolveParams params;
  assert(params.enable_folding && params.enable_presolve);
  // Check aggregation before any generic reductions.
  params.max_iters = 0;
  auto pipeline = gpu_presolver::folding::run_gpu_presolve_with_folding(lp, params, true, true);
  gpu_presolver::presolve::export_recovery_record(pipeline.presolve.record);
  assert(pipeline.folding_applied);
  assert(pipeline.presolve.original_rows == 1 && pipeline.presolve.original_cols == 1);
  assert(pipeline.original_rows == 2 && pipeline.original_cols == 2);
  assert(pipeline.presolve.reduced_lp.obj_constant == 7.0);
  assert(std::fabs(copy_to_host(pipeline.presolve.reduced_lp.A.nzVal, 1)[0] - 3.0) < 1e-12);
  assert(std::fabs(copy_to_host(pipeline.presolve.reduced_lp.c, 1)[0] - 4.0) < 1e-12);
  double* x = copy_to_device<double>({1.0});
  double* y = copy_to_device<double>({4.0 / 3.0});
  double* z = copy_to_device<double>({0.0});
  auto solution = gpu_presolver::folding::postsolve_and_unfold_to_host(x, y, z, pipeline, &lp);
  for (int j = 0; j < 2; ++j) {
    assert(std::fabs(solution.x[j] - 1.0) < 1e-12);
    const double aty = values[j] * solution.y[0] + values[2+j] * solution.y[1];
    assert(std::fabs(aty + solution.z[j] - 2.0) < 1e-12);
    assert(std::fabs(solution.z[j]) < 1e-12);
  }
  assert(std::fabs(2 * (solution.x[0] + solution.x[1]) + 7 - 11) < 1e-12);
  cudaFree(x); cudaFree(y); cudaFree(z);
  gpu_presolver::presolve::free_gpu_presolve_reduced_lp(pipeline.presolve);
  gpu_presolver::folding::free_folded_lp(pipeline.folding);

  // The same folding input followed by the current default presolve collapses
  // completely. Restore without any reduced vectors, including the row duals.
  params.max_iters = 10;
  pipeline = gpu_presolver::folding::run_gpu_presolve_with_folding(lp, params, true, true);
  assert(pipeline.folding_applied);
  assert(pipeline.presolve.reduced_rows == 0 && pipeline.presolve.reduced_cols == 0);
  assert(std::fabs(pipeline.presolve.record.obj_constant_new - 11.0) < 1e-12);
  solution = gpu_presolver::folding::postsolve_and_unfold_to_host(nullptr, nullptr, nullptr, pipeline, &lp);
  // Compare against the current presolver run directly on the folded LP.
  // Full elimination inherits its dual recovery behavior; unfolding must not
  // change it. Original KKT is checked above for a supplied optimal solution.
  auto direct = gpu_presolver::presolve::run_gpu_presolve_with_record(
      pipeline.folding.folded_lp, params);
  gpu_presolver::presolve::export_recovery_record(direct.record);
  auto direct_post = gpu_presolver::presolve::postsolve_gpu(
      nullptr, nullptr, nullptr, direct.record, &pipeline.folding.folded_lp);
  const auto direct_x = copy_to_host(direct_post.x_org, 1);
  const auto direct_y = copy_to_host(direct_post.y_org, 1);
  const auto direct_z = copy_to_host(direct_post.z_org, 1);
  std::cout << "full elimination direct folded postsolve x/y/z: "
            << direct_x[0] << "/" << direct_y[0] << "/" << direct_z[0] << "\n";
  for (int j = 0; j < 2; ++j) {
    assert(std::fabs(solution.x[j] - 1.0) < 1e-12);
    assert(solution.x[j] == direct_x[0]);
    assert(solution.y[j] == direct_y[0] / 2.0);
    assert(solution.z[j] == direct_z[0] / 2.0);
  }
  cudaFree(direct_post.x_org); cudaFree(direct_post.y_org); cudaFree(direct_post.z_org);
  gpu_presolver::presolve::free_gpu_presolve_reduced_lp(pipeline.presolve);
  gpu_presolver::folding::free_folded_lp(pipeline.folding);
  free_lp(lp);
}

void test_disabled_and_no_reduction_paths() {
  auto lp = make_symmetric_2x2_lp();
  gpu_presolver::presolve::PresolveParams params;
  params.enable_folding = false;
  auto raw = gpu_presolver::presolve::run_gpu_presolve_with_reduced_lp(lp, params);
  gpu_presolver::presolve::export_recovery_record(raw.record);
  auto pipeline = gpu_presolver::folding::run_gpu_presolve_with_folding(lp, params, true, true);
  gpu_presolver::presolve::export_recovery_record(pipeline.presolve.record);
  assert(!pipeline.folding_applied);
  assert(pipeline.presolve.reduced_rows == raw.reduced_rows);
  assert(pipeline.presolve.reduced_cols == raw.reduced_cols);
  assert(pipeline.presolve.record.obj_constant_new == raw.record.obj_constant_new);
  assert(pipeline.presolve.record.row_red2org == raw.record.row_red2org);
  assert(pipeline.presolve.record.col_red2org == raw.record.col_red2org);
  auto raw_solution = gpu_presolver::presolve::postsolve_gpu(nullptr, nullptr, nullptr, raw.record, &lp);
  const auto solution = gpu_presolver::folding::postsolve_and_unfold_to_host(nullptr, nullptr, nullptr, pipeline, &lp);
  assert(solution.x == copy_to_host(raw_solution.x_org, 2));
  assert(solution.y == copy_to_host(raw_solution.y_org, 2));
  assert(solution.z == copy_to_host(raw_solution.z_org, 2));
  cudaFree(raw_solution.x_org); cudaFree(raw_solution.y_org); cudaFree(raw_solution.z_org);
  gpu_presolver::presolve::free_gpu_presolve_reduced_lp(raw);
  gpu_presolver::presolve::free_gpu_presolve_reduced_lp(pipeline.presolve);
  gpu_presolver::folding::free_folded_lp(pipeline.folding);

  // Distinct metadata on both axes makes every color a singleton.
  const std::vector<double> costs{2.0, 3.0}, bounds{4.0, 5.0};
  check(cudaMemcpy(lp.c, costs.data(), 2 * sizeof(double), cudaMemcpyHostToDevice), "set costs");
  check(cudaMemcpy(lp.AL, bounds.data(), 2 * sizeof(double), cudaMemcpyHostToDevice), "set bounds");
  params.enable_folding = true;
  params.max_iters = 0;
  pipeline = gpu_presolver::folding::run_gpu_presolve_with_folding(lp, params, true, true);
  assert(!pipeline.folding_applied);
  assert(pipeline.folded_rows == 2 && pipeline.folded_cols == 2);
  assert(!pipeline.folding.owns_folded_lp && !pipeline.folding.map.owns_buffers);
  gpu_presolver::presolve::free_gpu_presolve_reduced_lp(pipeline.presolve);
  gpu_presolver::folding::free_folded_lp(pipeline.folding);
  free_lp(lp);
}

void test_unequal_color_sizes() {
  using namespace gpu_presolver;
  presolve::LPInfoGpu lp;
  lp.A = {2, 3, 6, copy_to_device<NnzOffset>({0,3,6}),
      copy_to_device<std::int32_t>({0,1,2,0,1,2}), copy_to_device<double>({1,1,1,1,1,1})};
  lp.AT = {3, 2, 6, copy_to_device<NnzOffset>({0,2,4,6}),
      copy_to_device<std::int32_t>({0,1,0,1,0,1}), copy_to_device<double>({1,1,1,1,1,1})};
  lp.c = copy_to_device<double>({2,2,2});
  lp.l = copy_to_device<double>({0,0,0});
  lp.u = copy_to_device<double>({10,10,10});
  lp.AL = copy_to_device<double>({6,6});
  lp.AU = copy_to_device<double>({6,6});
  auto folded = folding::run_folding(lp, 1e-8);
  assert(folded.applied && folded.folded_rows == 1 && folded.folded_cols == 1);
  const auto solution = folding::unfold_solution_to_host(folded.map, {2}, {2}, {0});
  for (double x : solution.x) assert(std::fabs(x - 2) < 1e-12);
  for (double y : solution.y) assert(std::fabs(y - 1) < 1e-12);
  for (int j = 0; j < 3; ++j) assert(std::fabs(solution.y[0] + solution.y[1] + solution.z[j] - 2) < 1e-12);
  folding::free_folded_lp(folded);
  free_lp(lp);
}

void test_independent_presolve_switch() {
  auto lp = make_symmetric_2x2_lp();
  gpu_presolver::presolve::PresolveParams params;
  params.enable_presolve = false;
  params.enable_folding = true;


  auto pipeline = gpu_presolver::folding::run_gpu_presolve_with_folding(lp, params, true, true);
  gpu_presolver::presolve::export_recovery_record(pipeline.presolve.record);
  assert(!pipeline.presolve_enabled && pipeline.folding_applied);
  assert(pipeline.presolve.reduced_rows == 1 && pipeline.presolve.reduced_cols == 1);
  assert(pipeline.presolve.iterations == 0 && pipeline.presolve.elapsed_seconds == 0.0);
  assert(pipeline.presolve.record.tape.types.empty());
  assert(pipeline.presolve.record.obj_constant_new == lp.obj_constant);
  double* x = copy_to_device<double>({2.0});
  double* y = copy_to_device<double>({2.0});
  double* z = copy_to_device<double>({0.0});
  auto solution = gpu_presolver::folding::postsolve_and_unfold_to_host(x, y, z, pipeline, &lp);
  assert(solution.x == std::vector<double>({2.0, 2.0}));
  assert(solution.y == std::vector<double>({1.0, 1.0}));
  assert(solution.z == std::vector<double>({0.0, 0.0}));
  cudaFree(x); cudaFree(y); cudaFree(z);
  gpu_presolver::presolve::free_gpu_presolve_reduced_lp(pipeline.presolve);
  gpu_presolver::folding::free_folded_lp(pipeline.folding);

  params.enable_folding = false;
  pipeline = gpu_presolver::folding::run_gpu_presolve_with_folding(lp, params, true, true);
  assert(!pipeline.presolve_enabled && !pipeline.folding_applied);
  assert(pipeline.presolve.reduced_lp.A.nzVal == lp.A.nzVal);
  assert(pipeline.presolve.reduced_rows == 2 && pipeline.presolve.reduced_cols == 2);
  x = copy_to_device<double>({2.0, 2.0});
  y = copy_to_device<double>({1.0, 1.0});
  z = copy_to_device<double>({0.0, 0.0});
  solution = gpu_presolver::folding::postsolve_and_unfold_to_host(x, y, z, pipeline, &lp);
  assert(solution.x == std::vector<double>({2.0, 2.0}));
  assert(solution.y == std::vector<double>({1.0, 1.0}));
  assert(solution.z == std::vector<double>({0.0, 0.0}));
  cudaFree(x); cudaFree(y); cudaFree(z);
  gpu_presolver::presolve::free_gpu_presolve_reduced_lp(pipeline.presolve);
  gpu_presolver::folding::free_folded_lp(pipeline.folding);
  free_lp(lp);
}

void test_export_retains_borrowed_reduced_lp() {
  auto lp = make_symmetric_2x2_lp();
  gpu_presolver::presolve::PresolveParams params;
  params.enable_folding = true;
  params.max_iters = 0;
  auto pipeline = gpu_presolver::folding::run_gpu_presolve_with_folding(lp, params, true, false);
  gpu_presolver::presolve::export_recovery_record(pipeline.presolve.record);
  assert(pipeline.folding_applied);
  assert(!pipeline.folding.owns_folded_lp);
  assert(pipeline.presolve.owns_reduced_lp);
  gpu_presolver::folding::free_folded_lp(pipeline.folding);
  // Export must still be able to read every reduced allocation.
  assert(copy_to_host(pipeline.presolve.reduced_lp.A.rowPtr, 2) == std::vector<NnzOffset>({0, 1}));
  assert(std::fabs(copy_to_host(pipeline.presolve.reduced_lp.A.nzVal, 1)[0] - 2.0) < 1e-12);
  assert(std::fabs(copy_to_host(pipeline.presolve.reduced_lp.c, 1)[0] - 4.0) < 1e-12);
  gpu_presolver::presolve::free_gpu_presolve_reduced_lp(pipeline.presolve);
  free_lp(lp);
}

void test_empty_and_discarded_folded_model() {
  gpu_presolver::presolve::LPInfoGpu empty;
  auto folded = gpu_presolver::folding::run_folding(empty, 1e-8);
  assert(!folded.applied);
  gpu_presolver::folding::free_folded_lp(folded);
  auto lp = make_symmetric_2x2_lp();
  gpu_presolver::presolve::PresolveParams params;
  params.enable_folding = true;
  auto pipeline = gpu_presolver::folding::run_gpu_presolve_with_folding(lp, params, false, false);
  gpu_presolver::presolve::export_recovery_record(pipeline.presolve.record);
  bool threw = false;
  try {
    gpu_presolver::folding::postsolve_and_unfold_to_host(nullptr, nullptr, nullptr, pipeline);
  } catch (const std::invalid_argument&) { threw = true; }
  assert(threw);
  gpu_presolver::folding::free_folded_lp(pipeline.folding);
  free_lp(lp);
}

void test_rejection_preserves_passthrough_model() {
  using gpu_presolver::presolve::LPInfoGpu;
  LPInfoGpu lp;
  const double inf=std::numeric_limits<double>::infinity();
  lp.A={5,5,5,copy_to_device<NnzOffset>({0,1,2,3,4,5}),
    copy_to_device<std::int32_t>({0,1,2,3,4}),copy_to_device<double>({1,2,3,4,5})};
  lp.AT={5,5,5,copy_to_device<NnzOffset>({0,1,2,3,4,5}),
    copy_to_device<std::int32_t>({0,1,2,3,4}),copy_to_device<double>({1,2,3,4,5})};
  lp.AL=copy_to_device<double>({0,1,2,3,4});lp.AU=copy_to_device<double>({inf,inf,inf,inf,inf});
  lp.c=copy_to_device<double>({0,0,0,0,0});lp.l=copy_to_device<double>({0,0,0,0,0});
  lp.u=copy_to_device<double>({inf,inf,inf,inf,inf});
  {
    auto result=gpu_presolver::folding::run_folding(lp,1e-8,false);
    assert(!result.applied && !result.owns_folded_lp);
    assert(result.folded_rows==5 && result.folded_cols==5);
    assert(!result.profile.refinement.refinement_ok);
    assert(result.profile.refinement.rounds==1);
    assert(result.profile.refinement.final_row_colors==5);
    assert(result.profile.refinement.final_col_colors==5);
    assert(copy_to_host(lp.A.nzVal,5)==std::vector<double>({1,2,3,4,5}));
    gpu_presolver::folding::free_folded_lp(result);
  }
  free_lp(lp);
}

// Exercise reduce_size independently of refinement: tolerance-grouped classes
// can contain unequal members, and even uniform decimal values may round during
// ordered accumulation. The certified kernels must match independent ordered sums bitwise.
void test_constant_reduction_matches_ordered_fallback() {
  using namespace gpu_presolver::folding;
  const double inf = std::numeric_limits<double>::infinity();
  const int n = 4097;
  const std::vector<double> constants = {0.0, -0.0, 1.25, 0.1, inf, -inf,
      std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::max()};
  for (double value : constants) for (bool varied : {false, true}) {
    gpu_presolver::presolve::LPInfoGpu lp;
    lp.A = {3, n, 0, copy_to_device<NnzOffset>({0,0,0,0}), nullptr, nullptr};
    std::vector<double> c(n,value), lower(n,value), upper(n,inf);
    if(varied) { c[n/2] = std::nextafter(value, inf); lower[n/3] = std::nextafter(value, -inf); }
    lp.c=copy_to_device(c);lp.l=copy_to_device(lower);lp.u=copy_to_device(upper);
    lp.AL=copy_to_device<double>({0.1,0.1,std::nextafter(0.1,inf)});
    lp.AU=copy_to_device<double>({inf,inf,inf});
    FoldingWorkspace w;init_workspace(w,3,n);
    w.num_row_color=1;w.num_col_color=1;
    std::vector<std::int32_t> row_ids(3,1),col_ids(n,1);
    check(cudaMemcpy(w.row_color_id,row_ids.data(),3*sizeof(std::int32_t),cudaMemcpyHostToDevice),"row IDs");
    check(cudaMemcpy(w.col_color_id,col_ids.data(),n*sizeof(std::int32_t),cudaMemcpyHostToDevice),"col IDs");
    auto optimized=reduce_size(w,lp,1e-8,nullptr);
    auto ordered_sum=[](const std::vector<double>& values) {
      double sum=0.0;
      for (double x:values) sum+=x;
      return sum;
    };
    const std::vector<double> expected={ordered_sum(c),ordered_sum(lower)/n,ordered_sum(upper)/n,
      ordered_sum({0.1,0.1,std::nextafter(0.1,inf)})/3,inf};
    const std::vector<double*> outputs={optimized.c,optimized.l,optimized.u,optimized.AL,optimized.AU};
    for(std::size_t i=0;i<outputs.size();++i) {
      const auto actual=copy_to_host(outputs[i],1);
      assert(std::memcmp(actual.data(),&expected[i],sizeof(double))==0);
    }
    free_lp(optimized);free_workspace(w);free_lp(lp);
  }
}

}  // namespace

int main() {
  test_constant_reduction_matches_ordered_fallback();
  test_rejection_preserves_passthrough_model();
  test_fold_initial_vector_uses_first_representative();
  test_run_folding_reduces_identical_rows_and_cols();
  test_unfold_solution_scales_duals_and_slacks();
  test_postsolve_and_unfold_restores_original_dimensions();
  test_equitable_partition_and_original_kkt();
  test_disabled_and_no_reduction_paths();
  test_unequal_color_sizes();
  test_empty_and_discarded_folded_model();
  test_export_retains_borrowed_reduced_lp();
  test_independent_presolve_switch();
  return 0;
}
