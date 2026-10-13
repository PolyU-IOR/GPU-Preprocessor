#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/duplicate_rule_workspace.hpp"
#include "gpu_presolver/presolve/rules/rule_parallel_columns.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
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

void test_merges_objective_compatible_parallel_columnumns() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  LPInfoGpu lp;
  lp.A.rows = 2;
  lp.A.cols = 2;
  lp.AT = {2, 2, 4,
           copy_to_device<NnzOffset>({0, 2, 4}),
           copy_to_device<std::int32_t>({0, 1, 0, 1}),
           copy_to_device<double>({1.0, 2.0, 2.0, 4.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({1.0, 2.0});
  plan.new_l = copy_to_device<double>({0.0, 1.0});
  plan.new_u = copy_to_device<double>({10.0, 3.0});
  plan.new_AL = copy_to_device<double>({0.0, 0.0});
  plan.new_AU = copy_to_device<double>({10.0, 10.0});
  PresolveStatsGpu stats;
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_parallel_columns(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> l = copy_to_host(plan.new_l, 2);
  const std::vector<double> u = copy_to_host(plan.new_u, 2);
  assert(keep[0] == 1);
  assert(keep[1] == 0);
  assert(std::fabs(l[0] - 2.0) < 1.0e-12);
  assert(std::fabs(u[0] - 16.0) < 1.0e-12);
  assert(plan.has_col_action);
  assert(plan.has_change);
  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::ParallelColumn)}));
  assert((plan.tape.index_starts == std::vector<NnzOffset>{0, 2}));
  assert((plan.tape.value_starts == std::vector<NnzOffset>{0, 5}));
  assert((plan.tape.dual_modes == std::vector<std::uint8_t>{
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal)}));
  assert((plan.tape.indices == std::vector<std::int32_t>{1, 0}));
  assert(std::fabs(plan.tape.vals[0] - 2.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[1] - 1.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[2] - 3.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[3] - 0.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[4] - 10.0) < 1.0e-12);

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

void test_fixed_parallel_columnumn_records_tape() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A.rows = 2;
  lp.A.cols = 2;
  lp.AT = {2, 2, 4,
           copy_to_device<NnzOffset>({0, 2, 4}),
           copy_to_device<std::int32_t>({0, 1, 0, 1}),
           copy_to_device<double>({1.0, 2.0, 1.0, 2.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({1.0, 2.0});
  plan.new_l = copy_to_device<double>({0.0, 3.0});
  plan.new_u = copy_to_device<double>({inf, 5.0});
  plan.new_AL = copy_to_device<double>({0.0, 0.0});
  plan.new_AU = copy_to_device<double>({10.0, 20.0});
  PresolveStatsGpu stats;
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_parallel_columns(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> l = copy_to_host(plan.new_l, 2);
  const std::vector<double> u = copy_to_host(plan.new_u, 2);
  const std::vector<double> AL = copy_to_host(plan.new_AL, 2);
  const std::vector<double> AU = copy_to_host(plan.new_AU, 2);
  assert(keep[0] == 1);
  assert(keep[1] == 0);
  assert(std::fabs(l[1] - 3.0) < 1.0e-12);
  assert(std::fabs(u[1] - 3.0) < 1.0e-12);
  assert(std::fabs(AL[0] - -3.0) < 1.0e-12);
  assert(std::fabs(AU[0] - 7.0) < 1.0e-12);
  assert(std::fabs(AL[1] - -6.0) < 1.0e-12);
  assert(std::fabs(AU[1] - 14.0) < 1.0e-12);
  assert(std::fabs(plan.obj_constant_delta - 6.0) < 1.0e-12);
  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::FixedCol)}));
  assert((plan.tape.index_starts == std::vector<NnzOffset>{0, 3}));
  assert((plan.tape.value_starts == std::vector<NnzOffset>{0, 4}));
  assert((plan.tape.dual_modes == std::vector<std::uint8_t>{
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal)}));
  assert((plan.tape.indices == std::vector<std::int32_t>{1, 0, 1}));
  assert(std::fabs(plan.tape.vals[0] - 3.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[1] - 2.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[2] - 1.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[3] - 2.0) < 1.0e-12);

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

void test_ratio_guard_keeps_scaled_parallel_columnumns() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  LPInfoGpu lp;
  lp.A.rows = 2;
  lp.A.cols = 2;
  lp.AT = {2, 2, 4,
           copy_to_device<NnzOffset>({0, 2, 4}),
           copy_to_device<std::int32_t>({0, 1, 0, 1}),
           copy_to_device<double>({1.0, 2.0, 2.0, 4.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({1.0, 2.0});
  plan.new_l = copy_to_device<double>({0.0, 1.0});
  plan.new_u = copy_to_device<double>({10.0, 3.0});
  plan.new_AL = copy_to_device<double>({0.0, 0.0});
  plan.new_AU = copy_to_device<double>({10.0, 10.0});
  PresolveStatsGpu stats;
  PresolveParams params;
  params.parallel_columns_max_abs_ratio = 1.0;
  params.parallel_columns_scale_guard_min_groups = 1;

  gpu_presolver::presolve::apply_rule_parallel_columns(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep = copy_to_host(plan.keep_col_mask, 2);
  assert(keep[0] == 1);
  assert(keep[1] == 1);
  assert(!plan.has_col_action);
  assert(!plan.has_change);
  assert(plan.tape.types.empty());

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

void test_bounded_objective_prefilter() {
  using namespace gpu_presolver::presolve;
  constexpr int m=4097;
  PresolveParams params;
  const double tol=std::fmax(params.zero_tol,params.bound_tol);
  for(double ratio:{2.0,-2.0}) for(double gap:{0.0,0.25*tol,2.0*tol}) {
    std::vector<int> ri(2*m);std::vector<double> av(2*m);
    for(int i=0;i<m;++i){ri[i]=ri[m+i]=i;av[i]=i==0?0.0:1.0+(i%3);av[m+i]=ratio*av[i];}
    LPInfoGpu lp;lp.A.rows=m;lp.A.cols=2;
    lp.AT={2,m,2*m,copy_to_device(std::vector<NnzOffset>{0,m,2*m}),copy_to_device(ri),copy_to_device(av)};
    PresolvePlanGpu plan;
    std::vector<std::uint8_t> keep_rows(m,1);keep_rows[0]=0;
    plan.keep_row_mask=copy_to_device(keep_rows);plan.keep_col_mask=copy_to_device(std::vector<std::uint8_t>{1,1});
    plan.new_c=copy_to_device(std::vector<double>{2.0,ratio*2.0+gap});
    plan.new_l=copy_to_device(std::vector<double>{-1.0,2.0});plan.new_u=copy_to_device(std::vector<double>{3.0,4.0});
    plan.new_AL=copy_to_device(std::vector<double>(m,0.0));plan.new_AU=copy_to_device(std::vector<double>(m,10.0));
    PresolveStatsGpu stats;apply_rule_parallel_columns(plan,lp,stats,params);
    auto keep=copy_to_host(plan.keep_col_mask,2);
    if(gap>tol){assert(keep[0]==1 && keep[1]==1);assert(!plan.has_change && plan.tape.types.empty());}
    else {
      assert(keep[0]==1 && keep[1]==0);assert(plan.has_change && plan.tape.types.size()==1);
      assert((plan.tape.indices==std::vector<int>{1,0}));
      assert(plan.tape.vals[0]==ratio);
      auto lower=copy_to_host(plan.new_l,2);auto upper=copy_to_host(plan.new_u,2);
      assert(lower[0]==-1.0+ratio*(ratio>0?2.0:4.0));assert(upper[0]==3.0+ratio*(ratio>0?4.0:2.0));
    }
    cudaFree(lp.AT.rowPtr);cudaFree(lp.AT.colVal);cudaFree(lp.AT.nzVal);
    cudaFree(plan.keep_row_mask);cudaFree(plan.keep_col_mask);cudaFree(plan.new_c);cudaFree(plan.new_l);cudaFree(plan.new_u);cudaFree(plan.new_AL);cudaFree(plan.new_AU);
  }
}

void test_parallel_large_group_matches_serial(bool masked, bool device_tape) {
  using namespace gpu_presolver::presolve;
  constexpr int m = 2048;
  constexpr int n = 10;
  std::vector<NnzOffset> ptr(n + 1);
  std::vector<std::int32_t> rows(n * m);
  std::vector<double> values(n * m);
  for (int j = 0; j < n; ++j) {
    ptr[j] = static_cast<NnzOffset>(j) * m;
    for (int r = 0; r < m; ++r) {
      rows[j * m + r] = r;
      values[j * m + r] = (j < 5 ? 1.0 : 2.0) * (1.0 + (r % 4));
    }
  }
  ptr[n] = static_cast<NnzOffset>(n) * m;
  std::vector<std::uint8_t> keep_rows(m, 1), keep_cols(n, 1);
  if (masked) {
    keep_rows[0] = 0;
    keep_cols[n - 1] = 0;
    for (int j = 0; j < n; ++j) values[j * m] = 17.0 + j;
  }

  auto run = [&](bool enabled) {
    runtime::options().parallel_column_preverify = enabled;
    LPInfoGpu lp;
    lp.A.rows = m;
    lp.A.cols = n;
    lp.AT = {n, m, static_cast<NnzOffset>(n) * m,
             copy_to_device(ptr), copy_to_device(rows), copy_to_device(values)};
    PresolvePlanGpu plan;
    plan.tape.resident = device_tape;
    plan.keep_row_mask = copy_to_device(keep_rows);
    plan.keep_col_mask = copy_to_device(keep_cols);
    plan.new_c = copy_to_device(std::vector<double>(n, 1.0));
    plan.new_l = copy_to_device(std::vector<double>(n, 0.0));
    plan.new_u = copy_to_device(std::vector<double>(n, 10.0));
    plan.new_AL = copy_to_device(std::vector<double>(m, 0.0));
    plan.new_AU = copy_to_device(std::vector<double>(m, 10.0));
    PresolveStatsGpu stats;
    PresolveParams params;
    apply_rule_parallel_columns(plan, lp, stats, params);
    if (device_tape) export_local_recovery_tape(plan.tape);
    auto result = std::make_tuple(copy_to_host(plan.keep_col_mask, n),
                                  copy_to_host(plan.new_l, n),
                                  copy_to_host(plan.new_u, n),
                                  plan.tape.types, plan.tape.indices, plan.tape.vals,
                                  plan.tape.index_starts, plan.tape.value_starts,
                                  plan.tape.dual_modes, plan.has_change,
                                  plan.has_infeasible, plan.has_unbounded);
    cudaFree(lp.AT.rowPtr); cudaFree(lp.AT.colVal); cudaFree(lp.AT.nzVal);
    cudaFree(plan.keep_row_mask); cudaFree(plan.keep_col_mask);
    cudaFree(plan.new_c); cudaFree(plan.new_l); cudaFree(plan.new_u);
    cudaFree(plan.new_AL); cudaFree(plan.new_AU);
    return result;
  };

  const auto saved = runtime::options().parallel_column_preverify;
  const auto serial = run(false);
  const auto parallel = run(true);
  runtime::options().parallel_column_preverify = saved;
  assert(serial == parallel);
  const auto& keep = std::get<0>(parallel);
  assert(keep[0] == 1 && keep[5] == 1);
  for (int j = 1; j < n; ++j) if (j != 5) assert(keep[j] == 0);
  assert(std::get<2>(parallel)[0] == 50.0);
  assert(std::get<2>(parallel)[5] == (masked ? 40.0 : 50.0));
  assert(std::get<3>(parallel).size() == n - 2 - int(masked));
}

// Compare all numerical plan fields and recovery bytes, including short,
// warp-sized and block-sized columns, zero-tolerance skipping and sign collisions.
void test_signature_preserves_plan_and_recovery(bool masked, bool device_tape,
                                               bool preverify, bool scale_gate) {
  using namespace gpu_presolver::presolve;
  constexpr int m = 1300, n = 18;
  std::vector<NnzOffset> ptr{0};
  std::vector<int> rows;
  std::vector<double> values;
  std::vector<double> costs(n, 1.0);
  auto append = [&](int length, int start, double scale, int pattern) {
    for (int r = 0; r < length; ++r) {
      rows.push_back(start+r);
      values.push_back(scale * (pattern == 0 ? 1.0+(r%5) :
                              1.0+(r%7)*0.25));
    }
    ptr.push_back(static_cast<NnzOffset>(rows.size()));
  };
  append(1100, 0, 1.0, 0); append(1100, 0, 4.0, 0); costs[1]=4.0;
  append(32, 0, 1.0, 1); append(32, 0, -4.0, 1); costs[3]=-4.0;
  append(3, 20, 1.0, 0); append(3, 20, 2.0, 0); costs[5]=2.0;
  append(32, 0, 1.0, 0); append(1100, 0, 1.0, 0);
  append(1100, 0, 1.0, 1); append(0, 0, 1.0, 0);
  append(3, 0, 1e-12, 0);
  for (int j : {11,12}) {
    rows.insert(rows.end(), {100,101,102});
    values.insert(values.end(), {1e-5,1e300,1e300});
    ptr.push_back(static_cast<NnzOffset>(rows.size()));
  }
  for (int j : {13,14,15}) {
    rows.insert(rows.end(), {200,201,202});
    values.insert(values.end(), {j==15 ? 2.0 : 1.0,
                                j==13 ? 1.0 : (j==15 ? -2.0 : -1.0),
                                j==13 ? 1.0 : (j==15 ? -2.0 : -1.0)});
    costs[j] = j==15 ? 2.0 : 1.0;
    ptr.push_back(static_cast<NnzOffset>(rows.size()));
  }
  rows.insert(rows.end(), {300,301});
  values.insert(values.end(), {std::numeric_limits<double>::infinity(),2.0});
  ptr.push_back(static_cast<NnzOffset>(rows.size()));
  append(64, 400, 1.0, 1);
  assert(ptr.size()==n+1);
  // Stored zero/tiny entries must be ignored exactly as in the original hash.
  values[ptr[2]+2]=values[ptr[3]+2]=0.0;
  std::vector<std::uint8_t> keep_rows(m,1), keep_cols(n,1);
  keep_cols[7]=0;
  if (masked) {
    keep_rows[0]=0; keep_rows[8]=0; keep_rows[1000]=0;
    for (int j : {0,1,2,3,6,7,8}) values[ptr[j]]=19.0+j;
  }
  runtime::options().parallel_column_preverify = preverify;
  auto run = [&](bool enabled) {
    runtime::options().column_signatures = (enabled) ? runtime::SignatureMode::Enabled : runtime::SignatureMode::Disabled;
    LPInfoGpu lp;
    lp.A.rows=m; lp.A.cols=n;
    lp.AT={n,m,static_cast<NnzOffset>(rows.size()),copy_to_device(ptr),
           copy_to_device(rows),copy_to_device(values)};
    PresolvePlanGpu plan;
    plan.tape.resident=device_tape;
    plan.keep_row_mask=copy_to_device(keep_rows);
    plan.keep_col_mask=copy_to_device(keep_cols);
    plan.new_c=copy_to_device(costs);
    plan.new_l=copy_to_device(std::vector<double>(n,-1.0));
    plan.new_u=copy_to_device(std::vector<double>(n,10.0));
    plan.new_AL=copy_to_device(std::vector<double>(m,-2.0));
    plan.new_AU=copy_to_device(std::vector<double>(m,20.0));
    PresolveParams params;
    if (scale_gate) {
      params.parallel_columns_scale_guard_min_groups=2;
      params.parallel_columns_scale_guard_max_group_size=2;
    }
    PresolveStatsGpu stats;
    {
      PreparedDuplicateRuleWorkspace prepared(parallel_columns_scratch_bytes(m,n,preverify));
      DuplicateRuleWorkspaceScope active;
      apply_rule_parallel_columns(plan,lp,stats,params);
    }
    if (device_tape) export_local_recovery_tape(plan.tape);
    std::vector<std::uint8_t> bytes;
    auto add = [&](const auto& v) {
      const auto* first=reinterpret_cast<const std::uint8_t*>(v.data());
      if (!v.empty()) bytes.insert(bytes.end(),first,first+v.size()*sizeof(v[0]));
    };
    add(copy_to_host(plan.keep_col_mask,n)); add(copy_to_host(plan.keep_row_mask,m));
    add(copy_to_host(plan.new_l,n)); add(copy_to_host(plan.new_u,n));
    add(copy_to_host(plan.new_c,n));
    add(copy_to_host(plan.new_AL,m)); add(copy_to_host(plan.new_AU,m));
    add(std::vector<double>{plan.obj_constant_delta});
    add(std::vector<std::uint8_t>{std::uint8_t(plan.has_change),
        std::uint8_t(plan.has_col_action),std::uint8_t(plan.has_row_action),
        std::uint8_t(plan.has_infeasible),std::uint8_t(plan.has_unbounded)});
    add(plan.tape.types); add(plan.tape.indices); add(plan.tape.vals);
    add(plan.tape.index_starts); add(plan.tape.value_starts); add(plan.tape.dual_modes);
    assert(plan.has_change && !plan.tape.types.empty());
    auto keep=copy_to_host(plan.keep_col_mask,n);
    assert(keep[7]==0 && keep[9]==1 && keep[10]==1 && keep[12]==0 && keep[15]==0);
    assert(keep[1]==std::uint8_t(scale_gate) && keep[3]==std::uint8_t(scale_gate));
    cudaFree(lp.AT.rowPtr); cudaFree(lp.AT.colVal); cudaFree(lp.AT.nzVal);
    cudaFree(plan.keep_row_mask); cudaFree(plan.keep_col_mask);
    cudaFree(plan.new_c); cudaFree(plan.new_l); cudaFree(plan.new_u);
    cudaFree(plan.new_AL); cudaFree(plan.new_AU);
    return bytes;
  };
  const auto original=run(false);
  const auto filtered=run(true);
  assert(original==filtered);
}

void test_signature_no_candidates_and_unbounded(bool terminal) {
  using namespace gpu_presolver::presolve;
  for (bool mode : {false,true}) {
    runtime::options().column_signatures = (mode) ? runtime::SignatureMode::Enabled : runtime::SignatureMode::Disabled;
    LPInfoGpu lp;
    lp.A.rows=3; lp.A.cols=3;
    lp.AT={3,3,3,copy_to_device<NnzOffset>({0,1,2,3}),
           copy_to_device<int>(terminal ? std::vector<int>{0,0,2} : std::vector<int>{0,1,2}),
           copy_to_device<double>({1.0,1.0,1.0})};
    const double inf=std::numeric_limits<double>::infinity();
    PresolvePlanGpu plan;
    plan.keep_row_mask=copy_to_device<std::uint8_t>({1,1,1});
    plan.keep_col_mask=copy_to_device<std::uint8_t>({1,1,1});
    plan.new_c=copy_to_device<double>({1.0,2.0,3.0});
    plan.new_l=copy_to_device<double>({0.0,terminal ? -inf : 0.0,0.0});
    plan.new_u=copy_to_device<double>({terminal ? inf : 10.0,10.0,10.0});
    plan.new_AL=copy_to_device<double>({0.0,0.0,0.0});
    plan.new_AU=copy_to_device<double>({10.0,10.0,10.0});
    PresolveStatsGpu stats;
    PresolveParams params;
    apply_rule_parallel_columns(plan,lp,stats,params);
    assert(plan.has_unbounded==terminal && !plan.has_change);
    assert((copy_to_host(plan.keep_col_mask,3)==std::vector<std::uint8_t>{1,1,1}));
    assert(plan.tape.types.empty() && plan.obj_constant_delta==0.0);
    cudaFree(lp.AT.rowPtr); cudaFree(lp.AT.colVal); cudaFree(lp.AT.nzVal);
    cudaFree(plan.keep_row_mask); cudaFree(plan.keep_col_mask);
    cudaFree(plan.new_c); cudaFree(plan.new_l); cudaFree(plan.new_u);
    cudaFree(plan.new_AL); cudaFree(plan.new_AU);
  }
}

// ns168703 produces (-1,-eps,-eps,-eps,-eps) and
// (1,-eps,-eps,-eps,-eps). Their normalized hashes collide, but they
// are not antiparallel. With -x+y=1e8 and z>=eps*(x+y), x,y,z>=0,
// falsely merging them changes min(z) from 1e8*eps to zero.
void test_small_coefficients_do_not_hide_nonproportional_columns() {
  using namespace gpu_presolver::presolve;
  const auto saved = runtime::options().parallel_column_preverify;
  const double eps = 1.9877659e-8;
  const double inf = std::numeric_limits<double>::infinity();
  for (bool verify : {false, true}) {
    runtime::options().parallel_column_preverify = verify;
    for (int m : {5, 2049}) {
      for (bool proportional : {false, true}) {
        std::vector<int> rows(2*m);
        std::vector<double> values(2*m);
        for (int i=0; i<m; ++i) {
          rows[i] = rows[m+i] = i;
          values[i] = i==0 ? -1.0 : -eps;
          values[m+i] = i==0 ? 1.0 : (proportional ? eps : -eps);
        }
        LPInfoGpu lp;
        lp.A.rows=m; lp.A.cols=2;
        lp.AT={2,m,2*m,copy_to_device(std::vector<NnzOffset>{0,m,2*m}),
               copy_to_device(rows),copy_to_device(values)};
        PresolvePlanGpu plan;
        plan.keep_row_mask=copy_to_device(std::vector<std::uint8_t>(m,1));
        plan.keep_col_mask=copy_to_device(std::vector<std::uint8_t>{1,1});
        plan.new_c=copy_to_device(std::vector<double>{0,0});
        plan.new_l=copy_to_device(std::vector<double>{0,0});
        plan.new_u=copy_to_device(std::vector<double>{inf,inf});
        plan.new_AL=copy_to_device(std::vector<double>(m,0));
        plan.new_AU=copy_to_device(std::vector<double>(m,inf));
        PresolveStatsGpu stats; PresolveParams params;
        assert(2*eps < params.bound_tol);
        apply_rule_parallel_columns(plan,lp,stats,params);
        const auto keep=copy_to_host(plan.keep_col_mask,2);
        assert(keep[0]==1 && keep[1]==(proportional ? 0 : 1));
        assert(plan.has_change==proportional);
        assert(plan.tape.types.size()==(proportional ? 1 : 0));
        if (proportional) assert(copy_to_host(plan.new_l,2)[0]==-inf);
        cudaFree(lp.AT.rowPtr); cudaFree(lp.AT.colVal); cudaFree(lp.AT.nzVal);
        cudaFree(plan.keep_row_mask); cudaFree(plan.keep_col_mask);
        cudaFree(plan.new_c); cudaFree(plan.new_l); cudaFree(plan.new_u);
        cudaFree(plan.new_AL); cudaFree(plan.new_AU);
      }
    }
  }
  runtime::options().parallel_column_preverify = saved;
}

int main() {
  const auto saved = runtime::options().parallel_column_prefilter;
  const auto signature_saved = runtime::options().column_signatures;
  const auto audit_saved = runtime::options().audit_column_signatures;
  const auto verify_saved = runtime::options().parallel_column_preverify;
  runtime::options().audit_column_signatures = true;
  for (bool flag : {false,true}) for (bool signature : {false,true}) {
    runtime::options().column_signatures = (signature) ? runtime::SignatureMode::Enabled : runtime::SignatureMode::Disabled;
    runtime::options().parallel_column_prefilter = flag;
    test_merges_objective_compatible_parallel_columnumns();
    test_fixed_parallel_columnumn_records_tape();
    test_ratio_guard_keeps_scaled_parallel_columnumns();
    test_bounded_objective_prefilter();
    test_small_coefficients_do_not_hide_nonproportional_columns();
    for (bool masked : {false, true})
      for (bool device_tape : {false, true})
        test_parallel_large_group_matches_serial(masked, device_tape);
  }
  for (bool masked : {false,true}) for (bool device_tape : {false,true})
    for (bool preverify : {false,true}) for (bool gate : {false,true})
      test_signature_preserves_plan_and_recovery(masked,device_tape,preverify,gate);
  test_signature_no_candidates_and_unbounded(false);
  test_signature_no_candidates_and_unbounded(true);
  runtime::options().parallel_column_prefilter = saved;
  runtime::options().column_signatures = signature_saved;
  runtime::options().audit_column_signatures = audit_saved;
  runtime::options().parallel_column_preverify = verify_saved;
  std::cout << "test_rule_parallel_columns passed\n";
  return 0;
}
