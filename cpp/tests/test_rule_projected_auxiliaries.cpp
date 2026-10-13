#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver/presolve/rules/rule_two_row_variable_elimination.hpp"
#include "gpu_presolver/presolve/rules/rule_mccormick_variable_elimination.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(status));
  }
}

template <class T>
T* copy_to_device(const std::vector<T>& values) {
  if (values.empty()) {
    return nullptr;
  }
  T* out = nullptr;
  check(cudaMalloc(&out, sizeof(T) * values.size()), "cudaMalloc test data");
  check(cudaMemcpy(out, values.data(), sizeof(T) * values.size(),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy test data");
  return out;
}

template <class T>
std::vector<T> copy_to_host(const T* values, std::size_t count) {
  std::vector<T> out(count);
  if (count > 0) {
    check(cudaMemcpy(out.data(), values, sizeof(T) * count,
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy test result");
  }
  return out;
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

struct HostCsr {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  std::vector<NnzOffset> row_ptr;
  std::vector<std::int32_t> col_val;
  std::vector<double> nz_val;
};

HostCsr transpose(const HostCsr& A) {
  HostCsr AT;
  AT.rows = A.cols;
  AT.cols = A.rows;
  AT.row_ptr.assign(static_cast<std::size_t>(AT.rows + 1), 0);
  for (const std::int32_t col : A.col_val) {
    ++AT.row_ptr[static_cast<std::size_t>(col + 1)];
  }
  for (std::int32_t i = 0; i < AT.rows; ++i) {
    AT.row_ptr[static_cast<std::size_t>(i + 1)] +=
        AT.row_ptr[static_cast<std::size_t>(i)];
  }
  AT.col_val.resize(A.col_val.size());
  AT.nz_val.resize(A.nz_val.size());
  std::vector<NnzOffset> cursor = AT.row_ptr;
  for (std::int32_t row = 0; row < A.rows; ++row) {
    for (NnzOffset p = A.row_ptr[static_cast<std::size_t>(row)];
         p < A.row_ptr[static_cast<std::size_t>(row + 1)]; ++p) {
      const std::int32_t col = A.col_val[static_cast<std::size_t>(p)];
      const std::int32_t dst = cursor[static_cast<std::size_t>(col)]++;
      AT.col_val[static_cast<std::size_t>(dst)] = row;
      AT.nz_val[static_cast<std::size_t>(dst)] =
          A.nz_val[static_cast<std::size_t>(p)];
    }
  }
  return AT;
}

gpu_presolver::presolve::LPInfoGpu upload_lp(
    const HostCsr& A,
    const std::vector<double>& c,
    const std::vector<double>& AL,
    const std::vector<double>& AU,
    const std::vector<double>& lower,
    const std::vector<double>& upper) {
  using gpu_presolver::presolve::LPInfoGpu;
  const HostCsr AT = transpose(A);
  LPInfoGpu lp;
  lp.A = {A.rows, A.cols, static_cast<std::int32_t>(A.nz_val.size()),
          copy_to_device(A.row_ptr), copy_to_device(A.col_val),
          copy_to_device(A.nz_val)};
  lp.AT = {AT.rows, AT.cols, static_cast<std::int32_t>(AT.nz_val.size()),
           copy_to_device(AT.row_ptr), copy_to_device(AT.col_val),
           copy_to_device(AT.nz_val)};
  lp.c = copy_to_device(c);
  lp.AL = copy_to_device(AL);
  lp.AU = copy_to_device(AU);
  lp.l = copy_to_device(lower);
  lp.u = copy_to_device(upper);
  return lp;
}

gpu_presolver::presolve::LPInfoGpu make_pair_block_lp(
    double second_schedule_rhs = 4.0,
    double recourse_cost = 0.0) {
  const double inf = std::numeric_limits<double>::infinity();
  HostCsr A;
  A.rows = 5;
  A.cols = 6;
  A.row_ptr = {0, 3, 5, 7, 11, 15};
  A.col_val = {0, 1, 4,
               0, 4,
               1, 4,
               2, 3, 4, 5,
               2, 3, 4, 5};
  A.nz_val = {-1.0, -1.0, 1.0,
              -1.0, 1.0,
              -1.0, 1.0,
              1.0, -1.0, 2.0, -2.0,
              -1.0, 1.0, 2.0, 2.0};
  return upload_lp(A,
                   {-1.0, -2.0, 0.0, 0.0, 0.0, recourse_cost},
                   {-1.0, -inf, -inf, -inf, -inf},
                   {inf, 0.0, 0.0, 2.0, second_schedule_rhs},
                   std::vector<double>(6, 0.0),
                   std::vector<double>(6, 1.0));
}

gpu_presolver::presolve::LPInfoGpu make_mccormick_lp(
    bool corrupt_upper_row,
    bool nonunit_endpoint) {
  const double inf = std::numeric_limits<double>::infinity();
  HostCsr A;
  A.rows = 3;
  A.cols = 3;
  A.row_ptr = {0, 3, 5, 7};
  A.col_val = {0, 1, 2, 0, 2, 1, 2};
  A.nz_val = {-1.0, -1.0, 1.0,
              corrupt_upper_row ? -0.5 : -1.0, 1.0,
              -1.0, 1.0};
  std::vector<double> upper{1.0, nonunit_endpoint ? 2.0 : 1.0, 1.0};
  return upload_lp(A,
                   {0.0, 0.0, 0.0},
                   {-1.0, -inf, -inf},
                   {inf, 0.0, 0.0},
                   {0.0, 0.0, 0.0},
                   upper);
}

gpu_presolver::presolve::PresolveParams projected_only_params() {
  using gpu_presolver::presolve::PresolveParams;
  PresolveParams params;
  params.max_iters = 1;
  params.enable_column_domination = false;
  params.enable_variable_pair_aggregation = false;
  params.enable_absolute_value_epigraph_reduction = false;
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


  params.two_row_variable_elimination_min_candidates = 1;
  params.two_row_variable_elimination_min_candidate_ratio = 0.0;
  params.mccormick_variable_elimination_min_candidates = 1;
  params.mccormick_variable_elimination_min_candidate_ratio = 0.0;
  return params;
}

void test_chained_projection_and_primal_recovery() {
  using namespace gpu_presolver::presolve;
  LPInfoGpu lp = make_pair_block_lp();
  PresolveParams params = projected_only_params();
  GpuPresolveSummary summary = run_gpu_presolve_with_reduced_lp(lp, params);
  gpu_presolver::presolve::export_recovery_record(summary.record);
  assert(!summary.has_infeasible);
  assert(!summary.has_unbounded);
  assert(summary.reduced_rows == 0);
  assert(summary.reduced_cols == 4);
  assert(summary.reduced_nnz == 0);
  assert(summary.record.has_primal_only_projected_auxiliary_reduction);
  assert(summary.record.tape.types.size() == 2);
  assert(summary.record.tape.types[0] ==
         static_cast<std::int32_t>(PostsolveReductionType::FmeCol));
  assert(summary.record.tape.types[1] ==
         static_cast<std::int32_t>(PostsolveReductionType::FmeCol));

  double* x_red = copy_to_device<double>({1.0, 1.0, 0.4, 0.2});
  double* z_red = copy_to_device<double>({0.0, 0.0, 0.0, 0.0});
  GpuPostsolveResult post =
      postsolve_gpu(x_red, nullptr, z_red, summary.record, &lp);
  const std::vector<double> x = copy_to_host(post.x_org, 6);
  assert(std::fabs(x[0] - 1.0) < 1.0e-12);
  assert(std::fabs(x[1] - 1.0) < 1.0e-12);
  assert(std::fabs(x[2] - 0.4) < 1.0e-12);
  assert(std::fabs(x[3] - 0.2) < 1.0e-12);
  assert(std::fabs(x[4] - 1.0) < 1.0e-12);
  assert(std::fabs(x[5] - 0.1) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(z_red);
  cudaFree(post.x_org);
  cudaFree(post.y_org);
  cudaFree(post.z_org);
  free_gpu_presolve_reduced_lp(summary);
  free_lp(lp);
}

void test_bounded_projection_fails_closed() {
  using namespace gpu_presolver::presolve;
  PresolveParams params = projected_only_params();

  LPInfoGpu boundary = make_pair_block_lp(3.0, 0.0);
  TwoRowVariableEliminationAnalysisGpu analysis =
      analyze_two_row_variable_elimination(boundary, params);
  assert(!analysis.applicable);
  assert(analysis.candidate_count == 0);
  free_two_row_variable_elimination_analysis(analysis);
  free_lp(boundary);

  LPInfoGpu nonzero_cost = make_pair_block_lp(4.0, 0.25);
  analysis = analyze_two_row_variable_elimination(nonzero_cost, params);
  assert(!analysis.applicable);
  assert(analysis.candidate_count == 0);
  free_two_row_variable_elimination_analysis(analysis);
  free_lp(nonzero_cost);
}

void test_mccormick_detector_is_exact_and_fail_closed() {
  using namespace gpu_presolver::presolve;
  PresolveParams params = projected_only_params();

  LPInfoGpu exact = make_mccormick_lp(false, false);
  McCormickVariableEliminationAnalysisGpu analysis =
      analyze_mccormick_variable_elimination(exact, params);
  assert(analysis.applicable);
  assert(analysis.candidate_count == 1);
  free_mccormick_variable_elimination_analysis(analysis);
  free_lp(exact);

  LPInfoGpu corrupted = make_mccormick_lp(true, false);
  analysis = analyze_mccormick_variable_elimination(corrupted, params);
  assert(!analysis.applicable);
  free_mccormick_variable_elimination_analysis(analysis);
  free_lp(corrupted);

  LPInfoGpu nonunit = make_mccormick_lp(false, true);
  analysis = analyze_mccormick_variable_elimination(nonunit, params);
  assert(!analysis.applicable);
  free_mccormick_variable_elimination_analysis(analysis);
  free_lp(nonunit);
}

void test_rules_are_startup_only() {
  using namespace gpu_presolver::presolve;
  LPInfoGpu lp = make_pair_block_lp();
  PresolveParams params = projected_only_params();
  params.enable_startup = false;
  GpuPresolveSummary summary = run_gpu_presolve_with_reduced_lp(lp, params);
  gpu_presolver::presolve::export_recovery_record(summary.record);
  assert(summary.reduced_rows == 5);
  assert(summary.reduced_cols == 6);
  assert(summary.reduced_nnz == 15);
  assert(summary.record.tape.types.empty());
  free_gpu_presolve_reduced_lp(summary);
  free_lp(lp);
}


// Compare the shared-plan chain with two physically materialized rule calls.
void test_batch_matches_separate_applies() {
  using namespace gpu_presolver::presolve;
  auto lp = make_pair_block_lp();
  auto params = projected_only_params();
  // The second rule sees 5 live columns: 1/5 passes, whereas 1/6 would fail.
  params.mccormick_variable_elimination_min_candidate_ratio = 0.19;
  auto batch = run_gpu_presolve_with_reduced_lp(lp, params);
  auto first_params = params;
  first_params.enable_mccormick_variable_elimination = false;
  auto first = run_gpu_presolve_with_reduced_lp(lp, first_params);
  auto second_params = params;
  second_params.enable_two_row_variable_elimination = false;
  auto second = run_gpu_presolve_with_reduced_lp(first.reduced_lp, second_params);
  assert(first.reduced_rows == 3 && first.reduced_cols == 5);
  assert(batch.reduced_rows == second.reduced_rows && batch.reduced_rows == 0);
  assert(batch.reduced_cols == second.reduced_cols && batch.reduced_cols == 4);
  assert(batch.reduced_nnz == second.reduced_nnz);
  for (auto v : {std::make_pair(batch.reduced_lp.c, second.reduced_lp.c),
                 std::make_pair(batch.reduced_lp.l, second.reduced_lp.l),
                 std::make_pair(batch.reduced_lp.u, second.reduced_lp.u)})
    assert(copy_to_host(v.first, 4) == copy_to_host(v.second, 4));
  double* x = copy_to_device<double>({1., 1., .4, .2});
  auto batched_post = postsolve_gpu(x, nullptr, nullptr, batch.record, &lp);
  auto second_post = postsolve_gpu(x, nullptr, nullptr, second.record, &first.reduced_lp);
  auto first_post = postsolve_gpu(second_post.x_org, nullptr, nullptr, first.record, &lp);
  assert(copy_to_host(batched_post.x_org, 6) == copy_to_host(first_post.x_org, 6));
  for (auto* result : {&batched_post, &second_post, &first_post}) {
    cudaFree(result->x_org); cudaFree(result->y_org); cudaFree(result->z_org);
  }
  cudaFree(x);
  free_gpu_presolve_reduced_lp(second);
  free_gpu_presolve_reduced_lp(first);
  free_gpu_presolve_reduced_lp(batch);
  free_lp(lp);
}

void test_pending_masks_and_gpu_tape() {
  using namespace gpu_presolver::presolve;
  const double inf = std::numeric_limits<double>::infinity();
  auto reference_lp = make_pair_block_lp();
  auto params = projected_only_params();
  auto reference = run_gpu_presolve_with_reduced_lp(reference_lp, params);
  export_recovery_record(reference.record);
  const auto rp = copy_to_host(reference_lp.A.rowPtr, 6);
  const auto ci = copy_to_host(reference_lp.A.colVal, 15);
  const auto av = copy_to_host(reference_lp.A.nzVal, 15);
  HostCsr a{6, 7, {0, 4}, {0, 4, 5, 6}, {1., 1., 1., 1.}};
  for (int row = 0; row < 5; ++row) {
    for (auto k = rp[row]; k < rp[row + 1]; ++k) {
      a.col_val.push_back(ci[k]); a.nz_val.push_back(av[k]);
    }
    a.col_val.push_back(6); a.nz_val.push_back(123.);
    a.row_ptr.push_back(a.col_val.size());
  }
  auto lp = upload_lp(a, {-1., -2., 0., 0., 0., 0., 3.},
      {-inf, -1., -inf, -inf, -inf, -inf}, {inf, inf, 0., 0., 2., 4.},
      std::vector<double>(7, 0.), {1., 1., 1., 1., 1., .25, 1.});
  // Run both recording backends against the same pending zero-column deletion.
  for (bool resident : {false, true}) {
    PresolvePlanGpu plan;
    plan.keep_row_mask = copy_to_device<std::uint8_t>({0, 1, 1, 1, 1, 1});
    plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1, 1, 1, 1, 1, 0});
    plan.new_l = copy_to_device<double>(std::vector<double>(7, 0.));
    plan.new_u = copy_to_device<double>({1., 1., 1., 1., 1., 1., 0.});
    plan.tape.resident = resident;
    auto two = analyze_two_row_variable_elimination(lp, params, &plan);
    assert(two.applicable && two.candidate_count == 1);
    build_two_row_variable_elimination_plan(plan, lp, two, params);
    free_two_row_variable_elimination_analysis(two);
    params.mccormick_variable_elimination_min_candidate_ratio = .19;
    auto mc = analyze_mccormick_variable_elimination(lp, params, &plan);
    assert(mc.applicable && mc.candidate_count == 1);
    build_mccormick_variable_elimination_plan(plan, lp, mc, params);
    free_mccormick_variable_elimination_analysis(mc);
    export_local_recovery_tape(plan.tape);
    // Only row IDs differ from the compact reference (extra deleted row 0).
    auto indices = plan.tape.indices;
    for (std::size_t k = 0; k < plan.tape.types.size(); ++k) {
      const auto begin = plan.tape.index_starts[k];
      for (int j = 0; j < indices[begin]; ++j) --indices[begin + 2 + j];
    }
    assert(indices == reference.record.tape.indices);
    assert(plan.tape.vals == reference.record.tape.vals);
    assert(plan.tape.types == reference.record.tape.types);
    assert(copy_to_host(plan.keep_row_mask, 6) == std::vector<std::uint8_t>(6, 0));
    assert(copy_to_host(plan.keep_col_mask, 7) == std::vector<std::uint8_t>({1,1,1,1,0,0,0}));
    cudaFree(plan.keep_row_mask); cudaFree(plan.keep_col_mask);
    cudaFree(plan.new_l); cudaFree(plan.new_u);
  }
  free_gpu_presolve_reduced_lp(reference);
  free_lp(reference_lp); free_lp(lp);
}

}  // namespace

int main() {
  test_chained_projection_and_primal_recovery();
  test_bounded_projection_fails_closed();
  test_mccormick_detector_is_exact_and_fail_closed();
  test_rules_are_startup_only();
  test_batch_matches_separate_applies();
  test_pending_masks_and_gpu_tape();
  std::cout << "test_rule_projected_auxiliaries passed\n";
  return 0;
}
