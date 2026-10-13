#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/rules/rule_singleton_columns.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
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

}  // namespace

void test_eliminates_eq_singleton_column() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

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
  plan.new_c = copy_to_device<double>({4.0, 1.0});
  plan.new_l = copy_to_device<double>({0.0, 2.0});
  plan.new_u = copy_to_device<double>({10.0, 2.0});
  plan.new_AL = copy_to_device<double>({5.0});
  plan.new_AU = copy_to_device<double>({5.0});

  PresolveStatsGpu stats;
  stats.singleton_column_mask = copy_to_device<std::uint8_t>({1, 0});
  stats.singleton_column_row = copy_to_device<std::int32_t>({0, -1});
  stats.singleton_column_val = copy_to_device<double>({1.0, 0.0});
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_singleton_columns(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep_row = copy_to_host(plan.keep_row_mask, 1);
  const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> c = copy_to_host(plan.new_c, 2);
  assert(keep_row[0] == 0);
  assert(keep_col[0] == 0);
  assert(keep_col[1] == 1);
  assert(std::fabs(c[1] + 3.0) < 1.0e-12);
  assert(std::fabs(plan.obj_constant_delta - 20.0) < 1.0e-12);
  assert(plan.has_row_action);
  assert(plan.has_col_action);
  assert(plan.has_change);
  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::SubCol)}));
  assert((plan.tape.indices == std::vector<std::int32_t>{0, 0, 1, 1}));
  assert(plan.tape.vals.size() == 7);
  assert(std::fabs(plan.tape.vals[0] - 1.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[1] - 5.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[2] - 0.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[3] - 10.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[4] - 4.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[5] - 1.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[6] - 1.0) < 1.0e-12);

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
  cudaFree(stats.singleton_column_mask);
  cudaFree(stats.singleton_column_row);
  cudaFree(stats.singleton_column_val);
}

void test_eq_objective_updates_are_bitwise_deterministic() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  std::uint64_t reference_bits = 0;
  for (int repetition = 0; repetition < 32; ++repetition) {
    LPInfoGpu lp;
    lp.A = {2, 3, 4,
            copy_to_device<NnzOffset>({0, 2, 4}),
            copy_to_device<std::int32_t>({0, 2, 1, 2}),
            copy_to_device<double>({3.0, 0.1, 7.0, 0.2})};
    lp.AT = {3, 2, 4,
             copy_to_device<NnzOffset>({0, 1, 2, 4}),
             copy_to_device<std::int32_t>({0, 1, 0, 1}),
             copy_to_device<double>({3.0, 7.0, 0.1, 0.2})};

    PresolvePlanGpu plan;
    plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
    plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1, 1});
    plan.new_c = copy_to_device<double>({100000000.1, -99999999.9, 0.3});
    plan.new_l = copy_to_device<double>({-inf, -inf, 0.0});
    plan.new_u = copy_to_device<double>({inf, inf, 0.0});
    plan.new_AL = copy_to_device<double>({5.0, 6.0});
    plan.new_AU = copy_to_device<double>({5.0, 6.0});

    PresolveStatsGpu stats;
    stats.singleton_column_mask = copy_to_device<std::uint8_t>({1, 1, 0});
    stats.singleton_column_row = copy_to_device<std::int32_t>({0, 1, -1});
    stats.singleton_column_val = copy_to_device<double>({3.0, 7.0, 0.0});
    PresolveParams params;

    gpu_presolver::presolve::apply_rule_singleton_columns(plan, lp, stats, params);
    const std::vector<double> c = copy_to_host(plan.new_c, 3);
    std::uint64_t current_bits = 0;
    std::memcpy(&current_bits, &c[2], sizeof(current_bits));
    if (repetition == 0) {
      reference_bits = current_bits;
    } else {
      assert(current_bits == reference_bits);
    }

    double expected = 0.3;
    expected += -(100000000.1 * 0.1 / 3.0);
    expected += -(-99999999.9 * 0.2 / 7.0);
    std::uint64_t expected_bits = 0;
    std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
    assert(current_bits == expected_bits);

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
    cudaFree(stats.singleton_column_mask);
    cudaFree(stats.singleton_column_row);
    cudaFree(stats.singleton_column_val);
  }
}

void test_dual_infer_marks_direct_unbounded() {
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
  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({1.0, 0.0});
  plan.new_l = copy_to_device<double>({-inf, 0.0});
  plan.new_u = copy_to_device<double>({10.0, 1.0});
  plan.new_AL = copy_to_device<double>({-inf});
  plan.new_AU = copy_to_device<double>({5.0});
  PresolveStatsGpu stats;
  stats.singleton_column_mask = copy_to_device<std::uint8_t>({1, 0});
  stats.singleton_column_row = copy_to_device<std::int32_t>({0, -1});
  stats.singleton_column_val = copy_to_device<double>({1.0, 0.0});
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_singleton_columns(plan, lp, stats, params);
  assert(plan.has_unbounded);

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(stats.singleton_column_mask);
  cudaFree(stats.singleton_column_row);
  cudaFree(stats.singleton_column_val);
}

void test_dual_infer_records_eq_to_ineq_tape() {
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
          copy_to_device<double>({1.0, 1.0})};
  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({1.0, 0.0});
  plan.new_l = copy_to_device<double>({-inf, 0.0});
  plan.new_u = copy_to_device<double>({inf, 0.0});
  plan.new_AL = copy_to_device<double>({1.0});
  plan.new_AU = copy_to_device<double>({5.0});
  PresolveStatsGpu stats;
  stats.singleton_column_mask = copy_to_device<std::uint8_t>({1, 0});
  stats.singleton_column_row = copy_to_device<std::int32_t>({0, -1});
  stats.singleton_column_val = copy_to_device<double>({1.0, 0.0});
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_singleton_columns(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> AL = copy_to_host(plan.new_AL, 1);
  const std::vector<double> AU = copy_to_host(plan.new_AU, 1);
  assert(keep_col[0] == 1);
  assert(std::fabs(AL[0] - 1.0) < 1.0e-12);
  assert(std::fabs(AU[0] - 1.0) < 1.0e-12);
  assert(plan.has_row_action);
  assert(!plan.has_col_action);
  assert(plan.has_change);
  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::EqToIneq)}));
  assert((plan.tape.index_starts == std::vector<NnzOffset>{0, 1}));
  assert((plan.tape.value_starts == std::vector<NnzOffset>{0, 1}));
  assert((plan.tape.indices == std::vector<std::int32_t>{0}));
  assert(plan.tape.vals.size() == 1);
  assert(std::fabs(plan.tape.vals[0]) < 1.0e-12);
  assert((plan.tape.dual_modes == std::vector<std::uint8_t>{
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal)}));

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(stats.singleton_column_mask);
  cudaFree(stats.singleton_column_row);
  cudaFree(stats.singleton_column_val);
}

struct OneSidedSingletonCase {
  double objective;
  double coefficient;
  double lower;
  double upper;
  double lhs;
  double rhs;
  double expected_side;
};

void check_dual_infer_tightens_one_sided_implied_free_row(
    const OneSidedSingletonCase& test_case) {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  LPInfoGpu lp;
  lp.A = {1, 2, 2,
          copy_to_device<NnzOffset>({0, 2}),
          copy_to_device<std::int32_t>({0, 1}),
          copy_to_device<double>({test_case.coefficient, 1.0})};
  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({test_case.objective, 0.0});
  plan.new_l = copy_to_device<double>({test_case.lower, 0.0});
  plan.new_u = copy_to_device<double>({test_case.upper, 0.0});
  plan.new_AL = copy_to_device<double>({test_case.lhs});
  plan.new_AU = copy_to_device<double>({test_case.rhs});
  PresolveStatsGpu stats;
  stats.singleton_column_mask = copy_to_device<std::uint8_t>({1, 0});
  stats.singleton_column_row = copy_to_device<std::int32_t>({0, -1});
  stats.singleton_column_val = copy_to_device<double>({test_case.coefficient, 0.0});
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_singleton_columns(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> AL = copy_to_host(plan.new_AL, 1);
  const std::vector<double> AU = copy_to_host(plan.new_AU, 1);
  assert(keep_col[0] == 1);
  assert(std::fabs(AL[0] - test_case.expected_side) < 1.0e-12);
  assert(std::fabs(AU[0] - test_case.expected_side) < 1.0e-12);
  assert(plan.has_row_action);
  assert(!plan.has_col_action);
  assert(plan.has_change);
  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::EqToIneq)}));
  assert((plan.tape.index_starts == std::vector<NnzOffset>{0, 1}));
  assert((plan.tape.value_starts == std::vector<NnzOffset>{0, 1}));
  assert((plan.tape.indices == std::vector<std::int32_t>{0}));
  assert(plan.tape.vals.size() == 1);
  assert(std::fabs(plan.tape.vals[0]) < 1.0e-12);
  assert((plan.tape.dual_modes == std::vector<std::uint8_t>{
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal)}));

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(stats.singleton_column_mask);
  cudaFree(stats.singleton_column_row);
  cudaFree(stats.singleton_column_val);
}

void test_dual_infer_tightens_one_sided_implied_free_row() {
  // c < 0, a > 0, implied-free from above: the upper row side is active.
  check_dual_infer_tightens_one_sided_implied_free_row(
      OneSidedSingletonCase{-1.0, 1.0, 10.0, 30.0, 5.0, 20.0, 20.0});
  // c > 0, a < 0, implied-free from below: the upper row side is active.
  check_dual_infer_tightens_one_sided_implied_free_row(
      OneSidedSingletonCase{1.0, -1.0, 0.0, 10.0, -20.0, -5.0, -5.0});
  // c > 0, a > 0, implied-free from below: the lower row side is active.
  check_dual_infer_tightens_one_sided_implied_free_row(
      OneSidedSingletonCase{1.0, 1.0, 0.0, 10.0, 5.0, 20.0, 5.0});
  // c < 0, a < 0, implied-free from above: the lower row side is active.
  check_dual_infer_tightens_one_sided_implied_free_row(
      OneSidedSingletonCase{-1.0, -1.0, 10.0, 30.0, -20.0, -5.0, -20.0});
}

void test_long_column_objective_order() {
  using namespace gpu_presolver::presolve;
  const double inf=std::numeric_limits<double>::infinity();
  const auto saved = runtime::options().gather_live_rows;
  for(int m:{9,1025}) for (bool flag : {false,true}) {
    runtime::options().gather_live_rows = flag;
    const int n=m+1;
    std::vector<int> ptr{0},idx,atptr{0},atidx,supportrow(n,-1);
    std::vector<double> val,atval,cost(n),supportval(n),rhs(m,5);
    std::vector<std::uint8_t> single(n,1);single[m]=0;cost[m]=0.3;
    double expected=cost[m];
    for(int row=0;row<m;++row) {
      const double pivot=row%2?7.0:3.0,coefficient=row%2?0.2:0.1;
      idx.push_back(row);idx.push_back(m);val.push_back(pivot);val.push_back(coefficient);ptr.push_back(idx.size());
      atidx.push_back(row);atval.push_back(pivot);atptr.push_back(atidx.size());
      cost[row]=row%2?-99999999.9:100000000.1;
      expected+=-(cost[row]*coefficient/pivot);
      supportrow[row]=row;supportval[row]=pivot;
    }
    for(int row=0;row<m;++row) {atidx.push_back(row);atval.push_back(row%2?0.2:0.1);}atptr.push_back(atidx.size());
    LPInfoGpu lp;
    lp.A={m,n,2*m,copy_to_device(std::vector<NnzOffset>(ptr.begin(),ptr.end())),copy_to_device(idx),copy_to_device(val)};
    lp.AT={n,m,2*m,copy_to_device(std::vector<NnzOffset>(atptr.begin(),atptr.end())),copy_to_device(atidx),copy_to_device(atval)};
    std::vector<double> lower(n,-inf),upper(n,inf);lower[m]=upper[m]=0;
    PresolvePlanGpu plan;
    plan.keep_row_mask=copy_to_device(std::vector<std::uint8_t>(m,1));
    plan.keep_col_mask=copy_to_device(std::vector<std::uint8_t>(n,1));
    plan.new_c=copy_to_device(cost);plan.new_l=copy_to_device(lower);plan.new_u=copy_to_device(upper);
    plan.new_AL=copy_to_device(rhs);plan.new_AU=copy_to_device(rhs);
    PresolveStatsGpu stats;
    stats.singleton_column_mask=copy_to_device(single);stats.singleton_column_row=copy_to_device(supportrow);stats.singleton_column_val=copy_to_device(supportval);
    PresolveParams params;
    apply_rule_singleton_columns(plan,lp,stats,params);
    const auto output=copy_to_host(plan.new_c,n);
    assert(!plan.has_infeasible && !plan.has_unbounded);
    assert(std::memcmp(&output[m],&expected,sizeof(double))==0);
    cudaFree(lp.A.rowPtr);cudaFree(lp.A.colVal);cudaFree(lp.A.nzVal);
    cudaFree(lp.AT.rowPtr);cudaFree(lp.AT.colVal);cudaFree(lp.AT.nzVal);
    cudaFree(plan.keep_row_mask);cudaFree(plan.keep_col_mask);cudaFree(plan.new_c);
    cudaFree(plan.new_l);cudaFree(plan.new_u);cudaFree(plan.new_AL);cudaFree(plan.new_AU);
    cudaFree(stats.singleton_column_mask);cudaFree(stats.singleton_column_row);cudaFree(stats.singleton_column_val);
  }
  runtime::options().gather_live_rows = saved;
}

// The CPU oracle describes the chronological recovery format independently of
// the GPU prefix sums. Include both sides of the long-row threshold, mask holes,
// signed zero coefficients, negative pivots, and append into a nonempty tape.
void test_batched_subcol_tape_matches_cpu_oracle() {
  using namespace gpu_presolver::presolve;
  const double inf = std::numeric_limits<double>::infinity();
  const std::vector<int> lengths{1, 2, 1024, 1025, 2051, 3};
  const int m = static_cast<int>(lengths.size());
  std::vector<NnzOffset> ptr{0}, atptr{0};
  std::vector<int> idx, atidx, owner, single_row;
  std::vector<double> coeff, atval, cost, lower, upper, single_val, rhs;
  std::vector<std::uint8_t> keep, single;
  for (int row = 0; row < m; ++row) {
    const int first = static_cast<int>(idx.size());
    const int pivot_col = first + (row % 2 ? lengths[row] - 1 : 0);
    const double pivot = row % 2 ? -2.0 : 2.0;
    owner.push_back(pivot_col);
    rhs.push_back(row + 1.0);
    for (int j = 0; j < lengths[row]; ++j) {
      const int col = first + j;
      const bool is_pivot = col == pivot_col;
      const double a = is_pivot ? pivot : (j % 5 == 0 ? -0.0 : (j % 3 - 1) * 0.25);
      idx.push_back(col); coeff.push_back(a);
      atidx.push_back(row); atval.push_back(a); atptr.push_back(atidx.size());
      cost.push_back(is_pivot ? row + 0.5 : 0.125);
      lower.push_back(is_pivot ? -inf : 0.0);
      upper.push_back(is_pivot ? inf : 0.0);
      keep.push_back(is_pivot || j % 7 != 0);
      single.push_back(is_pivot);
      single_row.push_back(is_pivot ? row : -1);
      single_val.push_back(is_pivot ? pivot : 0.0);
    }
    ptr.push_back(idx.size());
  }
  const int n = static_cast<int>(idx.size());
  std::vector<void*> allocations;
  const auto upload = [&](const auto& values) {
    auto* result = copy_to_device(values);
    allocations.push_back(result);
    return result;
  };
  LPInfoGpu lp;
  lp.A = {m, n, static_cast<NnzOffset>(idx.size()), upload(ptr), upload(idx), upload(coeff)};
  lp.AT = {n, m, static_cast<NnzOffset>(idx.size()), upload(atptr), upload(atidx), upload(atval)};
  PresolvePlanGpu plan;
  plan.keep_row_mask = upload(std::vector<std::uint8_t>(m, 1));
  plan.keep_col_mask = upload(keep);
  plan.new_c = upload(cost); plan.new_l = upload(lower); plan.new_u = upload(upper);
  plan.new_AL = upload(rhs); plan.new_AU = upload(rhs);
  PresolveStatsGpu stats;
  stats.singleton_column_mask = upload(single);
  stats.singleton_column_row = upload(single_row);
  stats.singleton_column_val = upload(single_val);
  PresolveParams params;
  PostsolveTape expected;
  expected.types = {static_cast<int>(PostsolveReductionType::EqToIneq)};
  expected.dual_modes = {static_cast<std::uint8_t>(PostsolveDualMode::Minimal)};
  expected.indices = {0}; expected.vals = {-0.0};
  expected.index_starts = {0, 1}; expected.value_starts = {0, 1};
  plan.tape = expected;
  const auto equal_bits = [](const auto& actual, const auto& reference) {
    assert(actual.size() == reference.size());
    assert(actual.empty() || std::memcmp(actual.data(), reference.data(),
                                        actual.size() * sizeof(actual[0])) == 0);
  };
  for (int batch = 0; batch < 2; ++batch) {
    for (int row = 0; row < m; ++row) {
      const int col = owner[row];
      int count = 0;
      for (NnzOffset p = ptr[row]; p < ptr[row + 1]; ++p)
        if (idx[p] != col && keep[idx[p]]) ++count;
      // Rows with only the pivot left are handled by the singleton-row rule.
      if (count == 0) continue;
      expected.types.push_back(static_cast<int>(PostsolveReductionType::SubCol));
      expected.dual_modes.push_back(static_cast<std::uint8_t>(PostsolveDualMode::Minimal));
      expected.indices.insert(expected.indices.end(), {col, row, count});
      expected.vals.insert(expected.vals.end(),
                           {single_val[col], rhs[row], lower[col], upper[col], cost[col], 1.0});
      for (NnzOffset p = ptr[row]; p < ptr[row + 1]; ++p) {
        if (idx[p] != col && keep[idx[p]]) {
          expected.indices.push_back(idx[p]); expected.vals.push_back(coeff[p]);
        }
      }
      expected.index_starts.push_back(expected.indices.size());
      expected.value_starts.push_back(expected.vals.size());
    }
    apply_rule_singleton_columns(plan, lp, stats, params);
    assert(!plan.has_infeasible && !plan.has_unbounded);
    equal_bits(plan.tape.types, expected.types);
    equal_bits(plan.tape.dual_modes, expected.dual_modes);
    equal_bits(plan.tape.index_starts, expected.index_starts);
    equal_bits(plan.tape.value_starts, expected.value_starts);
    equal_bits(plan.tape.indices, expected.indices);
    equal_bits(plan.tape.vals, expected.vals);
    const auto output_c = copy_to_host(plan.new_c, n);
    const auto output_keep = copy_to_host(plan.keep_col_mask, n);
    for (int row = 0; row < m; ++row) {
      int support_count = 0;
      for (NnzOffset p = ptr[row]; p < ptr[row + 1]; ++p)
        if (idx[p] != owner[row] && keep[idx[p]]) ++support_count;
      assert(output_keep[owner[row]] == (support_count == 0 ? 1 : 0));
      for (NnzOffset p = ptr[row]; p < ptr[row + 1]; ++p) {
        const int col = idx[p];
        if (col == owner[row] || !keep[col]) continue;
        double expected_c = cost[col];
        expected_c += -(cost[owner[row]] * coeff[p] / single_val[owner[row]]);
        assert(std::memcmp(&output_c[col], &expected_c, sizeof(double)) == 0);
      }
    }
    // Reset the LP state to append another batch with nonzero record/payload bases.
    check(cudaMemset(plan.keep_row_mask, 1, m), "reset rows");
    check(cudaMemcpy(plan.keep_col_mask, keep.data(), n, cudaMemcpyHostToDevice), "reset cols");
    check(cudaMemcpy(plan.new_c, cost.data(), n * sizeof(double), cudaMemcpyHostToDevice), "reset cost");
  }
  for (void* pointer : allocations) check(cudaFree(pointer), "free batched fixture");
}

// The first singleton is ineligible, the next two compete for one row.
// This also checks that selective commit preserves inactive entries bitwise.
void test_action_owner_and_selective_commit(bool resident) {
  using namespace gpu_presolver::presolve;
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<void*> allocations;
  auto upload = [&](const auto& values) {
    auto* ptr = copy_to_device(values);
    allocations.push_back(ptr);
    return ptr;
  };
  LPInfoGpu lp;
  lp.A = {2, 6, 5, upload(std::vector<NnzOffset>{0, 3, 5}),
          upload(std::vector<int>{0, 1, 2, 4, 5}), upload(std::vector<double>(5, 1.0))};
  lp.AT = {6, 2, 5, upload(std::vector<NnzOffset>{0, 1, 2, 3, 3, 4, 5}),
           upload(std::vector<int>{0, 0, 0, 1, 1}), upload(std::vector<double>(5, 1.0))};
  PresolvePlanGpu plan;
  plan.tape.resident = resident;
  plan.keep_row_mask = upload(std::vector<std::uint8_t>{1, 0});
  plan.keep_col_mask = upload(std::vector<std::uint8_t>{1, 1, 1, 0, 1, 1});
  plan.new_c = upload(std::vector<double>{-0.0, 3.0, 1.0, -0.0, -0.0, 7.0});
  plan.new_l = upload(std::vector<double>{0.0, -inf, -inf, 0.0, 0.0, 0.0});
  plan.new_u = upload(std::vector<double>{1.0, inf, inf, 1.0, 1.0, 1.0});
  plan.new_AL = upload(std::vector<double>{5.0, -0.0});
  plan.new_AU = upload(std::vector<double>{5.0, 0.0});
  PresolveStatsGpu stats;
  stats.singleton_column_mask = upload(std::vector<std::uint8_t>{1, 1, 1, 0, 0, 0});
  stats.singleton_column_row = upload(std::vector<int>{0, 0, 0, -1, -1, -1});
  stats.singleton_column_val = upload(std::vector<double>{1, 1, 1, 0, 0, 0});
  PresolveParams params;
  apply_rule_singleton_columns(plan, lp, stats, params);
  assert(!plan.has_unbounded && plan.has_change);
  assert((copy_to_host(plan.keep_col_mask, 6) == std::vector<std::uint8_t>{1, 0, 1, 0, 1, 1}));
  const std::vector<double> expected{-3.0, 3.0, -2.0, -0.0, -0.0, 7.0};
  const auto cost = copy_to_host(plan.new_c, 6);
  assert(std::memcmp(cost.data(), expected.data(), expected.size() * sizeof(double)) == 0);
  const auto lhs = copy_to_host(plan.new_AL, 2);
  assert(lhs[0] == 5.0 && lhs[1] == 0.0 && std::signbit(lhs[1]));
  assert(plan.obj_constant_delta == 15.0);
  if (resident) {
    assert(plan.tape.types.empty());
    export_local_recovery_tape(plan.tape);
  }
  assert(plan.tape.types.size() == 1);
  assert((plan.tape.indices == std::vector<int>{1, 0, 2, 0, 2}));
  for (void* ptr : allocations) check(cudaFree(ptr), "free action-owner fixture");
}

// A valid action on one row must not commit if any other candidate is unbounded.
void test_unbounded_proposal_suppresses_batch() {
  using namespace gpu_presolver::presolve;
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<void*> allocations;
  auto upload = [&](const auto& values) {
    auto* ptr = copy_to_device(values); allocations.push_back(ptr); return ptr;
  };
  LPInfoGpu lp;
  lp.A = {2, 4, 4, upload(std::vector<NnzOffset>{0, 2, 4}),
          upload(std::vector<int>{0, 1, 2, 3}), upload(std::vector<double>(4, 1.0))};
  PresolvePlanGpu plan;
  plan.keep_row_mask = upload(std::vector<std::uint8_t>{1, 1});
  plan.keep_col_mask = upload(std::vector<std::uint8_t>{1, 1, 1, 1});
  plan.new_c = upload(std::vector<double>{1.0, 0.0, -1.0, -0.0});
  plan.new_l = upload(std::vector<double>{-inf, 0.0, 10.0, 0.0});
  plan.new_u = upload(std::vector<double>{10.0, 1.0, 30.0, 0.0});
  plan.new_AL = upload(std::vector<double>{-inf, 5.0});
  plan.new_AU = upload(std::vector<double>{5.0, 20.0});
  PresolveStatsGpu stats;
  stats.singleton_column_mask = upload(std::vector<std::uint8_t>{1, 0, 1, 0});
  stats.singleton_column_row = upload(std::vector<int>{0, -1, 1, -1});
  stats.singleton_column_val = upload(std::vector<double>{1.0, 0.0, 1.0, 0.0});
  PresolveParams params;
  apply_rule_singleton_columns(plan, lp, stats, params);
  assert(plan.has_unbounded && !plan.has_change);
  assert(!plan.has_row_action && !plan.has_col_action);
  assert(plan.obj_constant_delta == 0.0 && plan.tape.types.empty());
  assert((copy_to_host(plan.new_AL, 2) == std::vector<double>{-inf, 5.0}));
  assert((copy_to_host(plan.new_AU, 2) == std::vector<double>{5.0, 20.0}));
  assert((copy_to_host(plan.keep_row_mask, 2) == std::vector<std::uint8_t>{1, 1}));
  assert((copy_to_host(plan.keep_col_mask, 4) == std::vector<std::uint8_t>{1, 1, 1, 1}));
  for (void* ptr : allocations) check(cudaFree(ptr), "free unbounded-batch fixture");
}

// Mix both action families in one batch, including shared surviving support.
// Check feature switches, the closure's skip-precheck path, and both tape backends.
void test_unified_mixed_batch(bool resident, bool skip_precheck, bool enabled,
                              bool unbounded = false) {
  using namespace gpu_presolver::presolve;
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<void*> allocations;
  auto upload = [&](const auto& values) {
    auto* ptr = copy_to_device(values); allocations.push_back(ptr); return ptr;
  };
  LPInfoGpu lp;
  lp.A = {2, 3, 4, upload(std::vector<NnzOffset>{0, 2, 4}),
          upload(std::vector<int>{0, 2, 1, 2}), upload(std::vector<double>(4, 1.0))};
  lp.AT = {3, 2, 4, upload(std::vector<NnzOffset>{0, 1, 2, 4}),
           upload(std::vector<int>{0, 1, 0, 1}), upload(std::vector<double>(4, 1.0))};
  PresolvePlanGpu plan;
  plan.tape.resident = resident;
  plan.keep_row_mask = upload(std::vector<std::uint8_t>{1, 1});
  plan.keep_col_mask = upload(std::vector<std::uint8_t>{1, 1, 1});
  plan.new_c = upload(std::vector<double>{3.0, 2.0, 10.0});
  plan.new_l = upload(std::vector<double>{-inf, -inf, 0.0});
  plan.new_u = upload(std::vector<double>{inf, inf, 2.0});
  plan.new_AL = upload(std::vector<double>{5.0, unbounded ? -inf : 1.0});
  plan.new_AU = upload(std::vector<double>{5.0, 7.0});
  PresolveStatsGpu stats;
  stats.singleton_column_mask = upload(std::vector<std::uint8_t>{1, 1, 0});
  stats.singleton_column_row = upload(std::vector<int>{0, 1, -1});
  stats.singleton_column_val = upload(std::vector<double>{1.0, 1.0, 0.0});
  PresolveParams params;
  params.enable_singleton_columns = enabled;
  apply_rule_singleton_columns(plan, lp, stats, params, skip_precheck);
  const bool eq = !unbounded && enabled;
  const bool infer = !unbounded && enabled;
  assert(plan.has_unbounded == unbounded && !plan.has_infeasible);
  assert(plan.has_change == (eq || infer));
  assert(plan.has_col_action == eq);
  assert((copy_to_host(plan.keep_row_mask, 2) ==
          std::vector<std::uint8_t>{std::uint8_t(!eq), 1}));
  assert((copy_to_host(plan.keep_col_mask, 3) ==
          std::vector<std::uint8_t>{std::uint8_t(!eq), 1, 1}));
  assert((copy_to_host(plan.new_c, 3) ==
          std::vector<double>{3.0, 2.0, eq ? 7.0 : 10.0}));
  assert((copy_to_host(plan.new_AL, 2) ==
          std::vector<double>{5.0, unbounded ? -inf : 1.0}));
  assert((copy_to_host(plan.new_AU, 2) == std::vector<double>{5.0, infer ? 1.0 : 7.0}));
  assert(plan.obj_constant_delta == (eq ? 15.0 : 0.0));
  if (enabled && !unbounded) {
    // The inferred equality is eliminated by the next unified batch.
    plan.has_change = false;
    apply_rule_singleton_columns(plan, lp, stats, params, skip_precheck);
    assert(plan.has_change);
    assert((copy_to_host(plan.keep_col_mask, 3) == std::vector<std::uint8_t>{0, 0, 1}));
    assert(copy_to_host(plan.new_c, 3)[2] == 5.0);
    assert(plan.obj_constant_delta == 17.0);
    plan.has_change = false;
    apply_rule_singleton_columns(plan, lp, stats, params, skip_precheck);
    assert(!plan.has_change);
  }
  if (resident) {
    assert(plan.tape.types.empty());
    export_local_recovery_tape(plan.tape);
  }
  std::vector<int> types, indices;
  std::vector<double> vals;
  if (eq) {
    types.push_back(static_cast<int>(PostsolveReductionType::SubCol));
    indices.insert(indices.end(), {0, 0, 1, 2});
    vals.insert(vals.end(), {1.0, 5.0, -inf, inf, 3.0, 1.0, 1.0});
  }
  if (infer) {
    types.push_back(static_cast<int>(PostsolveReductionType::EqToIneq));
    indices.push_back(1);
    vals.push_back(0.0);
  }
  if (eq && infer) {
    types.push_back(static_cast<int>(PostsolveReductionType::SubCol));
    indices.insert(indices.end(), {1, 1, 1, 2});
    vals.insert(vals.end(), {1.0, 1.0, -inf, inf, 2.0, 1.0, 1.0});
  }
  assert(plan.tape.types == types);
  assert(plan.tape.indices == indices);
  assert(plan.tape.vals == vals);
  for (void* ptr : allocations) check(cudaFree(ptr), "free unified mixed fixture");
}

int main() {
  for (bool resident : {false, true}) {
    for (bool skip_precheck : {false, true}) {
      for (bool enabled : {false, true})
        test_unified_mixed_batch(resident, skip_precheck, enabled);
      test_unified_mixed_batch(resident, skip_precheck, true, true);
    }
  }
  test_action_owner_and_selective_commit(false);
  test_action_owner_and_selective_commit(true);
  test_unbounded_proposal_suppresses_batch();
  test_batched_subcol_tape_matches_cpu_oracle();
  test_long_column_objective_order();
  test_eliminates_eq_singleton_column();
  test_eq_objective_updates_are_bitwise_deterministic();
  test_dual_infer_marks_direct_unbounded();
  test_dual_infer_records_eq_to_ineq_tape();
  test_dual_infer_tightens_one_sided_implied_free_row();
  std::cout << "test_rule_singleton_columns passed\n";
  return 0;
}
