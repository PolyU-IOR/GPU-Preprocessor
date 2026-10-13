#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/rules/rule_doubleton_equations.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
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
  T* device = nullptr;
  check(cudaMalloc(&device, sizeof(T) * values.size()), "cudaMalloc");
  check(cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy H2D");
  return device;
}

template <class T>
std::vector<T> copy_to_host(T* device, std::size_t size) {
  std::vector<T> values(size);
  check(cudaMemcpy(values.data(), device, sizeof(T) * size, cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H");
  return values;
}

}  // namespace

void test_eliminates_doubleton_equation_row() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {1, 2, 2,
          copy_to_device<NnzOffset>({0, 2}),
          copy_to_device<std::int32_t>({0, 1}),
          copy_to_device<double>({1.0, 2.0})};
  lp.AT = {2, 1, 2,
           copy_to_device<NnzOffset>({0, 1, 2}),
           copy_to_device<std::int32_t>({0, 0}),
           copy_to_device<double>({1.0, 2.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({3.0, 5.0});
  plan.new_l = copy_to_device<double>({0.0, -inf});
  plan.new_u = copy_to_device<double>({10.0, inf});
  plan.new_AL = copy_to_device<double>({4.0});
  plan.new_AU = copy_to_device<double>({4.0});

  PresolveStatsGpu stats;
  stats.col_nnz = nullptr;
  PresolveParams params;
  params.doubleton_equations_scan = false;
  params.doubleton_equations_max_fill_in_proxy = 10;

  gpu_presolver::presolve::apply_rule_doubleton_equations(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep_row = copy_to_host(plan.keep_row_mask, 1);
  const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> c = copy_to_host(plan.new_c, 2);
  assert(keep_row[0] == 0);
  assert(keep_col[0] == 0);
  assert(keep_col[1] == 1);
  assert(std::fabs(c[1] - (-1.0)) < 1.0e-12);
  assert(std::fabs(plan.obj_constant_delta - 12.0) < 1.0e-12);
  assert(plan.has_row_action);
  assert(plan.has_col_action);
  assert(plan.has_change);

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
}

void test_batch_records_exact_doubleton_tape() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {1, 2, 2,
          copy_to_device<NnzOffset>({0, 2}),
          copy_to_device<std::int32_t>({0, 1}),
          copy_to_device<double>({1.0, 2.0})};
  lp.AT = {2, 1, 2,
           copy_to_device<NnzOffset>({0, 1, 2}),
           copy_to_device<std::int32_t>({0, 0}),
           copy_to_device<double>({1.0, 2.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({3.0, 5.0});
  plan.new_l = copy_to_device<double>({0.0, -inf});
  plan.new_u = copy_to_device<double>({10.0, inf});
  plan.new_AL = copy_to_device<double>({4.0});
  plan.new_AU = copy_to_device<double>({4.0});

  PresolveStatsGpu stats;
  PresolveParams params;
  params.doubleton_equations_scan = true;
  params.doubleton_equations_min_selected_per_batch = 1;
  params.doubleton_equations_min_selected_ratio = 0.0;
  params.doubleton_equations_max_batch_rounds = 1;

  gpu_presolver::presolve::apply_rule_doubleton_equations(plan, lp, stats, params);

  assert(plan.tape.types.size() == 1);
  assert(plan.tape.types[0] == static_cast<std::int32_t>(PostsolveReductionType::DoubletonEquation));
  assert((plan.tape.index_starts == std::vector<NnzOffset>{0, 4}));
  assert((plan.tape.value_starts == std::vector<NnzOffset>{0, 12}));
  assert((plan.tape.dual_modes ==
          std::vector<std::uint8_t>{static_cast<std::uint8_t>(PostsolveDualMode::Minimal)}));
  assert((plan.tape.indices == std::vector<std::int32_t>{0, 1, 0, 0}));
  assert(plan.tape.vals.size() == 12);
  const std::vector<double> expected{1.0, 2.0, 4.0, 0.0, 10.0, -inf,
                                     inf, -3.0, 2.0, 3.0, 0.0, 0.0};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (std::isinf(expected[i])) {
      assert(std::isinf(plan.tape.vals[i]));
      assert(std::signbit(plan.tape.vals[i]) == std::signbit(expected[i]));
    } else {
      assert(std::fabs(plan.tape.vals[i] - expected[i]) < 1.0e-12);
    }
  }

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  if (plan.has_new_A) {
    cudaFree(plan.new_A.rowPtr);
    cudaFree(plan.new_A.colVal);
    cudaFree(plan.new_A.nzVal);
  }
}

bool run_doubleton_fill_cap_case(int max_fill_in_proxy) {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();
  constexpr std::int32_t m = 26;
  constexpr std::int32_t n = 2;

  // Row 0 is x0 + x1 = 0.  Column x0 has 12 additional rows and x1
  // has 13, so x0 is eliminated and its computed fill proxy is 11.
  std::vector<NnzOffset> row_ptr(static_cast<std::size_t>(m + 1));
  std::vector<std::int32_t> col_val;
  col_val.reserve(27);
  row_ptr[0] = 0;
  col_val.push_back(0);
  col_val.push_back(1);
  row_ptr[1] = 2;
  for (std::int32_t row = 1; row <= 12; ++row) {
    col_val.push_back(0);
    row_ptr[static_cast<std::size_t>(row + 1)] =
        static_cast<std::int32_t>(col_val.size());
  }
  for (std::int32_t row = 13; row < m; ++row) {
    col_val.push_back(1);
    row_ptr[static_cast<std::size_t>(row + 1)] =
        static_cast<std::int32_t>(col_val.size());
  }

  std::vector<std::int32_t> at_col_val;
  at_col_val.reserve(27);
  for (std::int32_t row = 0; row <= 12; ++row) {
    at_col_val.push_back(row);
  }
  at_col_val.push_back(0);
  for (std::int32_t row = 13; row < m; ++row) {
    at_col_val.push_back(row);
  }

  LPInfoGpu lp;
  lp.A = {m, n, 27,
          copy_to_device<NnzOffset>(row_ptr),
          copy_to_device<std::int32_t>(col_val),
          copy_to_device<double>(std::vector<double>(27, 1.0))};
  lp.AT = {n, m, 27,
           copy_to_device<NnzOffset>({0, 13, 27}),
           copy_to_device<std::int32_t>(at_col_val),
           copy_to_device<double>(std::vector<double>(27, 1.0))};

  std::vector<double> AL(static_cast<std::size_t>(m), -inf);
  std::vector<double> AU(static_cast<std::size_t>(m), inf);
  AL[0] = 0.0;
  AU[0] = 0.0;
  PresolvePlanGpu plan;
  plan.keep_row_mask =
      copy_to_device<std::uint8_t>(std::vector<std::uint8_t>(m, 1));
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({0.0, 0.0});
  plan.new_l = copy_to_device<double>({-inf, -inf});
  plan.new_u = copy_to_device<double>({inf, inf});
  plan.new_AL = copy_to_device<double>(AL);
  plan.new_AU = copy_to_device<double>(AU);

  PresolveStatsGpu stats;
  PresolveParams params;
  params.doubleton_equations_scan = true;
  params.doubleton_equations_max_fill_in_proxy = max_fill_in_proxy;
  params.doubleton_equations_min_selected_per_batch = 1;
  params.doubleton_equations_min_selected_ratio = 0.0;
  params.doubleton_equations_max_batch_rounds = 1;
  gpu_presolver::presolve::apply_rule_doubleton_equations(plan, lp, stats, params);

  const bool changed = plan.has_change;
  if (changed) {
    const std::vector<std::uint8_t> keep_row = copy_to_host(plan.keep_row_mask, m);
    const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, n);
    assert(keep_row[0] == 0);
    assert(keep_col[0] == 0);
    assert(keep_col[1] == 1);
  }

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  if (plan.has_new_A) {
    cudaFree(plan.new_A.rowPtr);
    cudaFree(plan.new_A.colVal);
    cudaFree(plan.new_A.nzVal);
  }
  return changed;
}

void test_default_fill_cap_accepts_more_than_ten() {
  const gpu_presolver::presolve::PresolveParams defaults;
  assert(defaults.doubleton_equations_max_fill_in_proxy == 96);
  assert(!run_doubleton_fill_cap_case(10));
  assert(run_doubleton_fill_cap_case(defaults.doubleton_equations_max_fill_in_proxy));
}

void test_one_batch_leaves_conflicting_equation_for_later() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  // x0 + x1 = 1 and x1 + x2 = 2 share x1. Only the first
  // equation may be selected in this batch. The old internal loop would
  // eliminate the second equation in its next batch in the same invocation.
  LPInfoGpu lp;
  lp.A = {2, 3, 4,
          copy_to_device<NnzOffset>({0, 2, 4}),
          copy_to_device<std::int32_t>({0, 1, 1, 2}),
          copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};
  lp.AT = {3, 2, 4,
           copy_to_device<NnzOffset>({0, 1, 3, 4}),
           copy_to_device<std::int32_t>({0, 0, 1, 1}),
           copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};
  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1, 1});
  plan.new_c = copy_to_device<double>({3.0, 5.0, 7.0});
  plan.new_l = copy_to_device<double>({-inf, -inf, -inf});
  plan.new_u = copy_to_device<double>({inf, inf, inf});
  plan.new_AL = copy_to_device<double>({1.0, 2.0});
  plan.new_AU = copy_to_device<double>({1.0, 2.0});
  PresolveStatsGpu stats;
  PresolveParams params;
  params.doubleton_equations_min_selected_per_batch = 1;
  params.doubleton_equations_min_selected_ratio = 0.0;

  gpu_presolver::presolve::apply_rule_doubleton_equations(plan, lp, stats, params);

  assert(plan.has_change && !plan.has_infeasible && !plan.has_unbounded);
  assert((copy_to_host(plan.keep_row_mask, 2) == std::vector<std::uint8_t>{0, 1}));
  assert((copy_to_host(plan.keep_col_mask, 3) == std::vector<std::uint8_t>{0, 1, 1}));
  const auto c = copy_to_host(plan.new_c, 3);
  assert(std::fabs(c[1] - 2.0) < 1.0e-12);
  assert(std::fabs(c[2] - 7.0) < 1.0e-12);
  assert(std::fabs(plan.obj_constant_delta - 3.0) < 1.0e-12);
  assert(plan.tape.types.size() == 1);

  cudaFree(lp.A.rowPtr); cudaFree(lp.A.colVal); cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr); cudaFree(lp.AT.colVal); cudaFree(lp.AT.nzVal);
  cudaFree(plan.keep_row_mask); cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c); cudaFree(plan.new_l); cudaFree(plan.new_u);
  cudaFree(plan.new_AL); cudaFree(plan.new_AU);
  if (plan.has_new_A) {
    cudaFree(plan.new_A.rowPtr); cudaFree(plan.new_A.colVal); cudaFree(plan.new_A.nzVal);
  }
}

int main() {
  test_eliminates_doubleton_equation_row();
  test_batch_records_exact_doubleton_tape();
  test_default_fill_cap_accepts_more_than_ten();
  test_one_batch_leaves_conflicting_equation_for_later();
  std::cout << "test_rule_doubleton_equations passed\n";
  return 0;
}
