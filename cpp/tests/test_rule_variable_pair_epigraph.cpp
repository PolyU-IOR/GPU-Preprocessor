#include "gpu_presolver/presolve/rules/rule_variable_pair_aggregation.hpp"
#include "../src/presolve/plan_storage.hpp"
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "../src/presolve/rules/variable_pair_aggregation/epigraph.hpp"

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

using namespace gpu_presolver::presolve::detail;

namespace {

using Row = std::vector<std::pair<std::int32_t, double>>;

void check(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(status));
  }
}

template <class T>
T* copy_to_device(const std::vector<T>& values) {
  T* out = nullptr;
  check(cudaMalloc(&out, sizeof(T) * values.size()), "cudaMalloc test data");
  check(cudaMemcpy(out, values.data(), sizeof(T) * values.size(),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy test data");
  return out;
}

template <class T>
std::vector<T> copy_to_host(const T* values, std::size_t size) {
  std::vector<T> out(size);
  check(cudaMemcpy(out.data(), values, sizeof(T) * size,
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy test result");
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

gpu_presolver::presolve::LPInfoGpu make_variable_pair_epigraph_lp(bool wrong_tail,
                                                bool wrong_coefficient,
                                                bool normalization_row = false) {
  using gpu_presolver::presolve::LPInfoGpu;
  const double inf = std::numeric_limits<double>::infinity();
  const std::int32_t cols = wrong_tail ? 8 : 7;
  const std::int32_t second_tail = wrong_tail ? 7 : 6;
  const double second_coefficient = wrong_coefficient ? -3.0 : -2.0;

  // Columns are p0,p1,p2,n0,n1,n2,T0[,T1].  Rows 0--2 are the
  // three-entry G rows, row 3 merges pairs 0 and 1, and row 4 is a
  // retained quotient row whose finite lower and upper bounds both shift.
  std::vector<Row> rows = {
      {{0, -2.0}, {3, -2.0}, {6, 1.0}},
      {{1, second_coefficient},
       {4, second_coefficient},
       {second_tail, 1.0}},
      {{2, -3.0}, {5, -3.0}, {6, 1.0}},
      {{0, 1.0}, {1, -1.0}, {3, -1.0}, {4, 1.0}},
      {{0, 2.0}, {1, 3.0}, {2, -1.0},
       {3, -2.0}, {4, -3.0}, {5, 1.0}},
  };
  if (normalization_row) {
    rows.push_back({{0, 2.0}, {2, 4.0}, {3, -2.0}, {5, -4.0}});
  }

  std::vector<NnzOffset> row_ptr(rows.size() + 1, 0);
  std::vector<std::int32_t> col_val;
  std::vector<double> nz_val;
  for (std::size_t row = 0; row < rows.size(); ++row) {
    for (const auto& entry : rows[row]) {
      col_val.push_back(entry.first);
      nz_val.push_back(entry.second);
    }
    row_ptr[row + 1] = static_cast<std::int32_t>(col_val.size());
  }

  std::vector<NnzOffset> at_row_ptr(static_cast<std::size_t>(cols) + 1,
                                       0);
  for (std::int32_t col : col_val) {
    ++at_row_ptr[static_cast<std::size_t>(col) + 1];
  }
  for (std::int32_t col = 0; col < cols; ++col) {
    at_row_ptr[static_cast<std::size_t>(col) + 1] +=
        at_row_ptr[static_cast<std::size_t>(col)];
  }
  std::vector<std::int32_t> at_col_val(col_val.size());
  std::vector<double> at_nz_val(nz_val.size());
  std::vector<NnzOffset> next = at_row_ptr;
  for (std::int32_t row = 0;
       row < static_cast<std::int32_t>(rows.size()); ++row) {
    for (NnzOffset p = row_ptr[static_cast<std::size_t>(row)];
         p < row_ptr[static_cast<std::size_t>(row) + 1]; ++p) {
      const std::int32_t col = col_val[static_cast<std::size_t>(p)];
      const std::int32_t dest = next[static_cast<std::size_t>(col)]++;
      at_col_val[static_cast<std::size_t>(dest)] = row;
      at_nz_val[static_cast<std::size_t>(dest)] =
          nz_val[static_cast<std::size_t>(p)];
    }
  }

  LPInfoGpu lp;
  lp.A = {static_cast<std::int32_t>(rows.size()), cols,
          static_cast<std::int32_t>(nz_val.size()),
          copy_to_device(std::vector<NnzOffset>(row_ptr.begin(), row_ptr.end())), copy_to_device(col_val),
          copy_to_device(nz_val)};
  lp.AT = {cols, static_cast<std::int32_t>(rows.size()),
           static_cast<std::int32_t>(nz_val.size()),
           copy_to_device(std::vector<NnzOffset>(at_row_ptr.begin(), at_row_ptr.end())), copy_to_device(at_col_val),
           copy_to_device(at_nz_val)};

  std::vector<double> c(static_cast<std::size_t>(cols), 0.0);
  c[6] = 1.0;
  if (wrong_tail) {
    c[7] = 1.0;
  }
  lp.c = copy_to_device(c);
  std::vector<double> row_lower = {5.0, 5.0, 7.0, 0.0, 10.0};
  std::vector<double> row_upper = {inf, inf, inf, 0.0, 20.0};
  if (normalization_row) {
    row_lower.push_back(0.0);
    row_upper.push_back(0.0);
  }
  lp.AL = copy_to_device(row_lower);
  lp.AU = copy_to_device(row_upper);
  std::vector<double> lower = {2.0, 2.0, 4.0, 1.0, 1.0, 1.0, 0.0};
  std::vector<double> upper = {10.0, 7.0, 9.0, 6.0, 4.0, 8.0, inf};
  if (wrong_tail) {
    lower.push_back(0.0);
    upper.push_back(inf);
  }
  lp.l = copy_to_device(lower);
  lp.u = copy_to_device(upper);
  return lp;
}

gpu_presolver::presolve::PresolveParams test_params() {
  gpu_presolver::presolve::PresolveParams params;
  params.max_iters = 1;
  params.enable_startup = true;
  params.enable_column_domination = false;
  params.enable_two_row_variable_elimination = false;
  params.enable_mccormick_variable_elimination = false;
  params.enable_infeasible_and_fixed_variables = false;
  params.enable_empty_rows = false;
  params.enable_singleton_rows = false;
  params.enable_infeasible_and_redundant_rows = false;
  params.enable_bound_strengthening = false;
  params.enable_parallel_rows = false;
  params.enable_empty_columns = false;
  params.enable_singleton_columns = false;

  params.enable_doubleton_equations = false;
  params.enable_absolute_value_epigraph_reduction = false;
  params.enable_dual_fix = false;
  params.enable_parallel_columns = false;

  params.variable_pair_epigraph_min_pairs = 1;
  params.variable_pair_epigraph_probe_pairs = 3;
  params.variable_pair_epigraph_min_potential_gain = 0.20;
  params.variable_pair_epigraph_min_actual_gain = 0.20;
  params.variable_pair_epigraph_max_quotient_terms_per_row = 5;
  return params;
}

void assert_near(double actual, double expected) {
  assert(std::abs(actual - expected) < 1.0e-12);
}

void test_contraction_bounds_and_primal_recovery() {
  using namespace gpu_presolver::presolve;
  LPInfoGpu lp = make_variable_pair_epigraph_lp(false, false);
  PresolveParams params = test_params();
  GpuPresolveSummary summary = run_gpu_presolve_with_reduced_lp(lp, params);
  gpu_presolver::presolve::export_recovery_record(summary.record);

  assert(!summary.has_infeasible);
  assert(!summary.has_unbounded);
  assert(summary.reduced_rows == 3);
  assert(summary.reduced_cols == 5);
  assert(summary.reduced_nnz == 10);
  assert_near(summary.obj_constant_delta, 0.0);
  assert(summary.record.has_primal_only_variable_pair_aggregation_reduction);
  assert(summary.record.variable_pair_aggregation_recoveries.size() == 1);

  const LPInfoGpu& reduced = summary.reduced_lp;
  assert((copy_to_host(reduced.A.rowPtr, 4) ==
          std::vector<NnzOffset>{0, 3, 6, 10}));
  assert((copy_to_host(reduced.A.colVal, 10) ==
          std::vector<std::int32_t>{0, 2, 4, 1, 3, 4, 0, 1, 2, 3}));
  const std::vector<double> reduced_values =
      copy_to_host(reduced.A.nzVal, 10);
  const double sqrt_two = std::sqrt(2.0);
  const std::vector<double> expected_values =
      {-2.0 * sqrt_two, -2.0 * sqrt_two, sqrt_two,
       -3.0, -3.0, 1.0, 5.0, -1.0, -5.0, 1.0};
  for (std::size_t i = 0; i < expected_values.size(); ++i) {
    assert_near(reduced_values[i], expected_values[i]);
  }
  const std::vector<double> reduced_lower = copy_to_host(reduced.AL, 3);
  const std::vector<double> reduced_upper = copy_to_host(reduced.AU, 3);
  assert_near(reduced_lower[0], 5.0 * sqrt_two);
  assert_near(reduced_lower[1], 7.0);
  assert_near(reduced_lower[2], 10.0);
  assert(std::isinf(reduced_upper[0]));
  assert(std::isinf(reduced_upper[1]));
  assert_near(reduced_upper[2], 20.0);

  const std::vector<double> reduced_col_lower = copy_to_host(reduced.l, 5);
  const std::vector<double> reduced_col_upper = copy_to_host(reduced.u, 5);
  assert((reduced_col_lower ==
          std::vector<double>{2.0, 4.0, 1.0, 1.0, 0.0}));
  assert_near(reduced_col_upper[0], 7.0);
  assert_near(reduced_col_upper[1], 9.0);
  assert_near(reduced_col_upper[2], 4.0);
  assert_near(reduced_col_upper[3], 8.0);
  assert(std::isinf(reduced_col_upper[4]));

  // Compact order is P_root0, P_root2, N_root0, N_root2, T0.
  double* x_red =
      copy_to_device<double>({4.0, 4.5, 1.2, 1.75, 30.0});
  double* y_red = copy_to_device<double>({0.0, 0.0, 0.0});
  double* z_red = copy_to_device<double>({0.0, 0.0, 0.0, 0.0, 0.0});
  GpuPostsolveResult post =
      postsolve_gpu(x_red, y_red, z_red, summary.record);
  const std::vector<double> x = copy_to_host(post.x_org, 7);
  const std::vector<double> expected_x =
      {4.0, 4.0, 4.5, 1.2, 1.2, 1.75, 30.0};
  for (std::size_t i = 0; i < expected_x.size(); ++i) {
    assert_near(x[i], expected_x[i]);
  }
  assert_near((x[0] - x[3]) - (x[1] - x[4]), 0.0);
  assert_near(2.0 * (x[0] - x[3]) + 3.0 * (x[1] - x[4]) -
                  (x[2] - x[5]),
              11.25);

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  cudaFree(post.x_org);
  cudaFree(post.y_org);
  cudaFree(post.z_org);
  free_gpu_presolve_reduced_lp(summary);
  free_lp(lp);
}

void test_unified_entry() {
  using namespace gpu_presolver::presolve;
  LPInfoGpu lp = make_variable_pair_epigraph_lp(false, false);
  PresolveParams params = test_params();
  PresolvePlanGpu plan;
  const auto selected = build_variable_pair_aggregation_plan(plan, lp, params);
  assert(selected == VariablePairAggregationCase::Epigraph);
  assert(plan.has_change && !plan.has_infeasible && !plan.has_unbounded);
  assert(plan.has_variable_pair_aggregation_recovery);
  detail::free_plan(plan);
  plan = {};
  params.enable_variable_pair_aggregation = false;
  assert(build_variable_pair_aggregation_plan(plan, lp, params) == VariablePairAggregationCase::None);
  assert(!plan.has_change && !plan.contiguous_storage);
  params.enable_variable_pair_aggregation = true;
  params.enable_structure_specific_rules = false;
  assert(build_variable_pair_aggregation_plan(plan, lp, params) == VariablePairAggregationCase::None);
  assert(!plan.has_change && !plan.contiguous_storage);
  params.enable_structure_specific_rules = true;
  assert(build_variable_pair_aggregation_plan(plan, lp, params, [] { return true; }) ==
         VariablePairAggregationCase::None);
  assert(!plan.has_change && !plan.contiguous_storage);
  free_lp(lp);
}

void test_rule_is_owned_by_startup() {
  using namespace gpu_presolver::presolve;
  LPInfoGpu lp = make_variable_pair_epigraph_lp(false, false);
  PresolveParams params = test_params();
  params.enable_startup = false;
  GpuPresolveSummary summary = run_gpu_presolve_with_reduced_lp(lp, params);
  gpu_presolver::presolve::export_recovery_record(summary.record);
  assert(summary.reduced_rows == 5);
  assert(summary.reduced_cols == 7);
  assert(summary.reduced_nnz == 19);
  assert(!summary.record.has_primal_only_variable_pair_aggregation_reduction);
  free_gpu_presolve_reduced_lp(summary);
  free_lp(lp);
}

void test_retained_equality_is_canonically_normalized() {
  using namespace gpu_presolver::presolve;
  LPInfoGpu lp = make_variable_pair_epigraph_lp(false, false, true);
  PresolveParams params = test_params();
  GpuPresolveSummary summary = run_gpu_presolve_with_reduced_lp(lp, params);
  gpu_presolver::presolve::export_recovery_record(summary.record);
  assert(summary.reduced_rows == 4);
  assert(summary.reduced_cols == 5);
  assert(summary.reduced_nnz == 14);
  const std::vector<NnzOffset> row_ptr =
      copy_to_host(summary.reduced_lp.A.rowPtr, 5);
  assert((row_ptr == std::vector<NnzOffset>{0, 3, 6, 10, 14}));
  const std::vector<double> values =
      copy_to_host(summary.reduced_lp.A.nzVal, 14);
  const std::vector<double> expected_tail = {1.0, 2.0, -1.0, -2.0};
  for (std::size_t k = 0; k < expected_tail.size(); ++k) {
    assert_near(values[10 + k], expected_tail[k]);
  }
  free_gpu_presolve_reduced_lp(summary);
  free_lp(lp);
}

void test_component_metadata_mismatch_is_fail_closed(bool wrong_tail,
                                                     bool wrong_coefficient) {
  using namespace gpu_presolver::presolve;
  LPInfoGpu lp = make_variable_pair_epigraph_lp(wrong_tail, wrong_coefficient);
  PresolveParams params = test_params();
  assert(quick_probe_variable_pair_epigraph(lp, params));
  VariablePairEpigraphAnalysisGpu analysis = analyze_variable_pair_epigraph(lp, params);
  assert(!analysis.applicable);
  assert(analysis.parent == nullptr);
  assert(analysis.edge_by_row == nullptr);
  assert(analysis.pair_g_row == nullptr);
  free_variable_pair_epigraph_analysis(analysis);
  free_lp(lp);
}

}  // namespace

int main() {
  test_unified_entry();
  test_contraction_bounds_and_primal_recovery();
  test_rule_is_owned_by_startup();
  test_retained_equality_is_canonically_normalized();
  test_component_metadata_mismatch_is_fail_closed(true, false);
  test_component_metadata_mismatch_is_fail_closed(false, true);
  std::cout << "test_rule_variable_pair_epigraph passed\n";
  return 0;
}
