#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/rules/rule_bound_strengthening.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace runtime = gpu_presolver::runtime;

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

void free_lp(gpu_presolver::presolve::LPInfoGpu& lp) {
  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
}

}  // namespace

void test_tightens_bounds_from_row_activity() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {2, 3, 4,
          copy_to_device<NnzOffset>({0, 2, 4}),
          copy_to_device<std::int32_t>({0, 1, 0, 2}),
          copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};
  lp.AT = {3, 2, 4,
           copy_to_device<NnzOffset>({0, 2, 3, 4}),
           copy_to_device<std::int32_t>({0, 1, 0, 1}),
           copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1, 1});
  plan.new_c = copy_to_device<double>({0.0, 0.0, 0.0});
  plan.new_AL = copy_to_device<double>({5.0, -inf});
  plan.new_AU = copy_to_device<double>({inf, 6.0});
  plan.new_l = copy_to_device<double>({0.0, 2.0, 0.0});
  plan.new_u = copy_to_device<double>({10.0, 2.0, 0.0});

  PresolveStatsGpu stats;
  stats.row_nnz = copy_to_device<std::int32_t>({2, 2});

  PresolveParams params;
  params.feasibility_tol = 1.0e-9;
  params.zero_tol = 1.0e-12;

  gpu_presolver::presolve::apply_rule_bound_strengthening(plan, lp, stats, params);
  const std::vector<double> l = copy_to_host(plan.new_l, 3);
  const std::vector<double> u = copy_to_host(plan.new_u, 3);
  assert(std::fabs(l[0] - 3.0) < 1.0e-12);
  assert(std::fabs(u[0] - 6.0) < 1.0e-12);
  assert(!plan.has_row_action);
  assert(!plan.has_col_action);
  assert(plan.has_change);
  assert(!plan.has_infeasible);
  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::BoundChangeTheRow),
                                 static_cast<std::int32_t>(PostsolveReductionType::BoundChangeTheRow)}));
  assert((plan.tape.index_starts == std::vector<NnzOffset>{0, 2, 4}));
  assert((plan.tape.value_starts == std::vector<NnzOffset>{0, 4, 8}));
  assert((plan.tape.dual_modes == std::vector<std::uint8_t>{
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal),
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal)}));
  assert((plan.tape.indices == std::vector<std::int32_t>{0, 0, 0, 1}));
  assert(std::fabs(plan.tape.vals[0] - 0.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[1] - 10.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[2] - 3.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[3] - 10.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[4] - 3.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[5] - 10.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[6] - 3.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[7] - 6.0) < 1.0e-12);

  free_lp(lp);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.row_nnz);
}

void test_records_fixed_column_from_row_activity() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {1, 2, 2,
          copy_to_device<NnzOffset>({0, 2}),
          copy_to_device<std::int32_t>({0, 1}),
          copy_to_device<double>({1.0, 1.0})};
  lp.AT = {2, 1, 2,
           copy_to_device<NnzOffset>({0, 1, 2}),
           copy_to_device<std::int32_t>({0, 0}),
           copy_to_device<double>({1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({4.0, 0.0});
  plan.new_AL = copy_to_device<double>({6.0});
  plan.new_AU = copy_to_device<double>({inf});
  plan.new_l = copy_to_device<double>({0.0, 5.0});
  plan.new_u = copy_to_device<double>({1.0, 5.0});

  PresolveStatsGpu stats;
  stats.row_nnz = copy_to_device<std::int32_t>({2});
  PresolveParams params;
  params.feasibility_tol = 1.0e-9;
  params.zero_tol = 1.0e-12;

  gpu_presolver::presolve::apply_rule_bound_strengthening(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> l = copy_to_host(plan.new_l, 2);
  const std::vector<double> u = copy_to_host(plan.new_u, 2);
  assert(keep_col[0] == 0);
  assert(std::fabs(l[0] - 1.0) < 1.0e-12);
  assert(std::fabs(u[0] - 1.0) < 1.0e-12);
  assert(std::fabs(plan.obj_constant_delta - 4.0) < 1.0e-12);
  assert(plan.tape.types.size() == 1);
  assert(plan.tape.types[0] == static_cast<std::int32_t>(PostsolveReductionType::FixedCol));
  assert((plan.tape.indices == std::vector<std::int32_t>{0, 0}));
  assert(std::fabs(plan.tape.vals[0] - 1.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[1] - 4.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[2] - 1.0) < 1.0e-12);

  free_lp(lp);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.row_nnz);
}

void test_detects_infeasible_tightening() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {1, 2, 2,
          copy_to_device<NnzOffset>({0, 2}),
          copy_to_device<std::int32_t>({0, 1}),
          copy_to_device<double>({1.0, 1.0})};
  lp.AT = {2, 1, 2,
           copy_to_device<NnzOffset>({0, 1, 2}),
           copy_to_device<std::int32_t>({0, 0}),
           copy_to_device<double>({1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({0.0, 0.0});
  plan.new_AL = copy_to_device<double>({10.0});
  plan.new_AU = copy_to_device<double>({inf});
  plan.new_l = copy_to_device<double>({0.0, 0.0});
  plan.new_u = copy_to_device<double>({4.0, 0.0});

  PresolveStatsGpu stats;
  stats.row_nnz = copy_to_device<std::int32_t>({2});
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_bound_strengthening(plan, lp, stats, params);
  assert(plan.has_infeasible);

  free_lp(lp);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.row_nnz);
}

void test_preserves_wide_box_on_free_zero_cost_column() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {2, 2, 4,
          copy_to_device<NnzOffset>({0, 2, 4}),
          copy_to_device<std::int32_t>({0, 1, 0, 1}),
          copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};
  lp.AT = {2, 2, 4,
           copy_to_device<NnzOffset>({0, 2, 4}),
           copy_to_device<std::int32_t>({0, 1, 0, 1}),
           copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({0.0, 0.0});
  plan.new_AL = copy_to_device<double>({-5.0, -inf});
  plan.new_AU = copy_to_device<double>({inf, 5.0});
  plan.new_l = copy_to_device<double>({-inf, 0.0});
  plan.new_u = copy_to_device<double>({inf, 0.0});

  PresolveStatsGpu stats;
  stats.row_nnz = copy_to_device<std::int32_t>({2, 2});
  PresolveParams params;
  params.feasibility_tol = 1.0e-9;
  params.zero_tol = 1.0e-12;

  gpu_presolver::presolve::apply_rule_bound_strengthening(plan, lp, stats, params);
  std::vector<double> l = copy_to_host(plan.new_l, 2);
  std::vector<double> u = copy_to_host(plan.new_u, 2);
  assert(std::isinf(l[0]) && l[0] < 0.0);
  assert(std::isinf(u[0]) && u[0] > 0.0);
  assert(!plan.has_change);
  assert(plan.tape.types.empty());

  params.bound_strengthening_preserve_free_zero_cost = false;
  gpu_presolver::presolve::apply_rule_bound_strengthening(plan, lp, stats, params);
  l = copy_to_host(plan.new_l, 2);
  u = copy_to_host(plan.new_u, 2);
  assert(std::fabs(l[0] + 5.0) < 1.0e-12);
  assert(std::fabs(u[0] - 5.0) < 1.0e-12);
  assert(plan.has_change);

  free_lp(lp);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.row_nnz);
}

void test_preserves_free_zero_cost_state_across_propagation_rounds() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  // Round one derives x >= 0 from 0 <= x-z and z <= 1 from z+w <= 1.
  // Only round two can derive x <= 11 from x-z <= 10.  x was originally
  // free and has zero cost, so the second side must not materialize a box.
  LPInfoGpu lp;
  lp.A = {2, 3, 4,
          copy_to_device<NnzOffset>({0, 2, 4}),
          copy_to_device<std::int32_t>({0, 1, 1, 2}),
          copy_to_device<double>({1.0, -1.0, 1.0, 1.0})};
  lp.AT = {3, 2, 4,
           copy_to_device<NnzOffset>({0, 1, 3, 4}),
           copy_to_device<std::int32_t>({0, 0, 1, 1}),
           copy_to_device<double>({1.0, -1.0, 1.0, 1.0})};
  lp.l = copy_to_device<double>({-inf, 0.0, 0.0});
  lp.u = copy_to_device<double>({inf, inf, 0.0});

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1, 1});
  plan.new_c = copy_to_device<double>({0.0, 0.0, 0.0});
  plan.new_AL = copy_to_device<double>({0.0, -inf});
  plan.new_AU = copy_to_device<double>({10.0, 1.0});
  plan.new_l = copy_to_device<double>({-inf, 0.0, 0.0});
  plan.new_u = copy_to_device<double>({inf, inf, 0.0});

  PresolveStatsGpu stats;
  stats.row_nnz = copy_to_device<std::int32_t>({2, 2});
  PresolveParams params;
  params.feasibility_tol = 1.0e-9;
  params.zero_tol = 1.0e-12;

  gpu_presolver::presolve::apply_rule_bound_strengthening(plan, lp, stats, params);
  std::vector<double> l = copy_to_host(plan.new_l, 3);
  std::vector<double> u = copy_to_host(plan.new_u, 3);
  assert(std::fabs(l[0]) < 1.0e-12);
  assert(std::isinf(u[0]) && u[0] > 0.0);
  assert(std::fabs(u[1] - 1.0) < 1.0e-12);

  plan.has_change = false;
  gpu_presolver::presolve::apply_rule_bound_strengthening(plan, lp, stats, params);
  l = copy_to_host(plan.new_l, 3);
  u = copy_to_host(plan.new_u, 3);
  assert(std::fabs(l[0]) < 1.0e-12);
  assert(std::isinf(u[0]) && u[0] > 0.0);
  assert(!plan.has_change);

  free_lp(lp);
  cudaFree(lp.l);
  cudaFree(lp.u);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.row_nnz);
}

void test_long_column_support_minimum() {
  using namespace gpu_presolver::presolve;
  const auto saved = runtime::options().gather_live_rows;
  for(int m:{9,1025}) for (bool flag : {false,true}) {
    runtime::options().gather_live_rows = flag;
    const int n=m+1;
    std::vector<int> ptr{0},idx,atptr{0},atidx;
    std::vector<double> al(m),lower(n,0),upper(n,0);upper[0]=10;
    for(int row=0;row<m;++row) {
      idx.push_back(0);idx.push_back(row+1);ptr.push_back(idx.size());
      atidx.push_back(row);al[row]=row%7?1:4;
    }
    atptr.push_back(m);
    for(int row=0;row<m;++row) {atidx.push_back(row);atptr.push_back(atidx.size());}
    LPInfoGpu lp;
    lp.A={m,n,2*m,copy_to_device(std::vector<NnzOffset>(ptr.begin(),ptr.end())),copy_to_device(idx),copy_to_device(std::vector<double>(2*m,1))};
    lp.AT={n,m,2*m,copy_to_device(std::vector<NnzOffset>(atptr.begin(),atptr.end())),copy_to_device(atidx),copy_to_device(std::vector<double>(2*m,1))};
    PresolvePlanGpu plan;
    plan.keep_row_mask=copy_to_device(std::vector<std::uint8_t>(m,1));plan.keep_col_mask=copy_to_device(std::vector<std::uint8_t>(n,1));
    plan.new_c=copy_to_device(std::vector<double>(n,0));plan.new_l=copy_to_device(lower);plan.new_u=copy_to_device(upper);
    plan.new_AL=copy_to_device(al);plan.new_AU=copy_to_device(std::vector<double>(m,INFINITY));
    PresolveStatsGpu stats;stats.row_nnz=copy_to_device(std::vector<int>(m,2));
    PresolveParams params;
    apply_rule_bound_strengthening(plan,lp,stats,params);
    assert(!plan.has_infeasible && copy_to_host(plan.new_l,n)[0]==4);
    assert((plan.tape.indices==std::vector<int>{0,0}));
    assert((plan.tape.types==std::vector<int>{static_cast<int>(PostsolveReductionType::BoundChangeTheRow)}));
    free_lp(lp);cudaFree(plan.keep_row_mask);cudaFree(plan.keep_col_mask);cudaFree(plan.new_c);
    cudaFree(plan.new_l);cudaFree(plan.new_u);cudaFree(plan.new_AL);cudaFree(plan.new_AU);cudaFree(stats.row_nnz);
  }
  runtime::options().gather_live_rows = saved;
}

// Increasing the upper margin must weaken only the upper bound; the lower
// margin and integral-upper shortcut retain the existing policy.
void test_upper_margin_preserves_lower_and_integral_bounds() {
  using namespace gpu_presolver::presolve;
  for (double upper_target : {2.499999, -2.499999, 2.0}) {
    for (double factor : {0.5, 1.0, 2.0}) {
      const double lower_target = upper_target - 1.500001;
      LPInfoGpu lp;
      lp.A = {1, 2, 2, copy_to_device<NnzOffset>({0, 2}),
              copy_to_device<std::int32_t>({0, 1}), copy_to_device<double>({1, 1})};
      lp.AT = {2, 1, 2, copy_to_device<NnzOffset>({0, 1, 2}),
               copy_to_device<std::int32_t>({0, 0}), copy_to_device<double>({1, 1})};
      PresolvePlanGpu plan;
      plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
      plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
      plan.new_c = copy_to_device<double>({1, 0});
      plan.new_AL = copy_to_device<double>({lower_target});
      plan.new_AU = copy_to_device<double>({upper_target});
      plan.new_l = copy_to_device<double>({-10, 0});
      plan.new_u = copy_to_device<double>({10, 0});
      PresolveStatsGpu stats;
      stats.row_nnz = copy_to_device<std::int32_t>({2});
      PresolveParams params;
      assert(params.feasibility_tol == 1e-6);
      params.bound_strengthening_upper_relaxation_factor = factor;
      apply_rule_bound_strengthening(plan, lp, stats, params);
      const auto lower = copy_to_host(plan.new_l, 2);
      const auto upper = copy_to_host(plan.new_u, 2);
      assert(!plan.has_infeasible && !plan.has_col_action && plan.has_change);
      assert(lower[0] <= lower_target && upper[0] >= upper_target);
      assert(lower[0] > -10 && upper[0] < 10);
      const double expected_lower = lower_target == std::nearbyint(lower_target)
          ? lower_target : lower_target - .5e-6 * std::fabs(lower_target);
      assert(std::fabs(lower[0] - expected_lower) < 1e-12);
      if (upper_target == std::nearbyint(upper_target)) {
        assert(upper[0] == upper_target);
      } else {
        assert(std::fabs(upper[0] - upper_target - factor * 1e-6 * std::fabs(upper_target)) < 1e-12);
      }
      assert(lower[1] == 0 && upper[1] == 0);
      free_lp(lp);
      cudaFree(plan.keep_row_mask); cudaFree(plan.keep_col_mask);
      cudaFree(plan.new_c); cudaFree(plan.new_AL); cudaFree(plan.new_AU);
      cudaFree(plan.new_l); cudaFree(plan.new_u); cudaFree(stats.row_nnz);
    }
  }
}

int main() {
  test_upper_margin_preserves_lower_and_integral_bounds();
  test_long_column_support_minimum();
  test_tightens_bounds_from_row_activity();
  test_records_fixed_column_from_row_activity();
  test_detects_infeasible_tightening();
  test_preserves_wide_box_on_free_zero_cost_column();
  test_preserves_free_zero_cost_state_across_propagation_rounds();
  std::cout << "test_rule_bound_strengthening passed\n";
  return 0;
}
