#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver/presolve/stable_mapping.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
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

void test_parallel_nonidentity_mapping() {
  using gpu_presolver::presolve::detail::compact_mapping_stable;
  constexpr int n=2100003;
  std::vector<std::int32_t> old(n),expected,dropped{-7},actual,removed{-7};
  std::vector<std::uint8_t> keep(n);
  for(int i=0;i<n;++i){old[i]=3*i+17;keep[i]=(i%7==0 || i%13==0)?0:2;
    (keep[i]?expected:dropped).push_back(old[i]);}
  compact_mapping_stable(old,keep,expected.size(),actual,&removed);
  assert(actual==expected && removed==dropped);
  old=actual;keep.resize(old.size());expected.clear();
  for(std::size_t i=0;i<old.size();++i){keep[i]=i%3!=0;if(keep[i])expected.push_back(old[i]);}
  compact_mapping_stable(old,keep,expected.size(),actual,nullptr);
  assert(actual==expected);
  bool rejected=false;
  try{compact_mapping_stable(old,keep,expected.size()+1,actual,nullptr);}catch(const std::runtime_error&){rejected=true;}
  assert(rejected);
}

// Independent stable-filter oracle, including long rows in a small sparse
// matrix (the former 10M gate), zero coefficients and duplicate coordinates.
void test_compact_long_rows_oracle() {
  using namespace gpu_presolver::presolve;
  constexpr int m = 32768, n = 9000;
  const int lengths[] = {0, 1, 8, 9, 31, 32, 33, 1024, 1025, 4097, 8193};
  std::vector<int> rp{0}, ci, expected_rp{0}, expected_ci, row_map, col_map(n, -1);
  std::vector<double> av, expected_av;
  std::vector<int> expected_cols;
  for (int j = 0; j < n; ++j) if (j % 7) {
    col_map[j] = static_cast<int>(expected_cols.size()); expected_cols.push_back(j);
  }
  for (int i = 0; i < m; ++i) {
    // Fill almost every queue slot, so front/back queue separation is tested.
    NnzOffset len = i < 11 ? lengths[i] : 9;
    auto before = expected_ci.size();
    for (int k = 0; k < len; ++k) {
      int j = i < 11 ? k / 2 : (i * 3 + k) % n;
      double v = k % 17 == 0 ? -0.0 : (k % 3 == 0 ? -0.1 : 0.3);
      ci.push_back(j); av.push_back(v);
      if (col_map[j] >= 0) { expected_ci.push_back(col_map[j]); expected_av.push_back(v); }
    }
    rp.push_back(static_cast<int>(ci.size()));
    // Empty-row removal counts stored entries (including numerical zeros).
    if (expected_ci.size() != before) {
      row_map.push_back(i); expected_rp.push_back(static_cast<int>(expected_ci.size()));
    }
  }
  std::vector<int> trp(n+1,0), tci(ci.size()); std::vector<double> tav(av.size());
  for (int j : ci) ++trp[j+1];
  for (int j=0;j<n;++j) trp[j+1]+=trp[j];
  auto cursor=trp;
  for (int i=0;i<m;++i) for(int k=rp[i];k<rp[i+1];++k) {
    NnzOffset q=cursor[ci[k]]++;tci[q]=i;tav[q]=av[k];
  }
  LPInfoGpu lp;
  lp.A={m,n,static_cast<int>(ci.size()),copy_to_device(std::vector<NnzOffset>(rp.begin(), rp.end())),copy_to_device(ci),copy_to_device(av)};
  lp.AT={n,m,static_cast<int>(ci.size()),copy_to_device(std::vector<NnzOffset>(trp.begin(), trp.end())),copy_to_device(tci),copy_to_device(tav)};
  std::vector<double> upper(n,10.0);for(int j=0;j<n;j+=7)upper[j]=0.0;
  lp.c=copy_to_device(std::vector<double>(n,0.0));lp.l=copy_to_device(std::vector<double>(n,0.0));lp.u=copy_to_device(upper);
  lp.AL=copy_to_device(std::vector<double>(m,-1e9));lp.AU=copy_to_device(std::vector<double>(m,1e9));
  PresolveParams params;
  params.max_iters=3;params.record_postsolve_tape=true;
  params.enable_startup=false;params.enable_structure_specific_rules=false;
  params.enable_singleton_rows=false;params.enable_infeasible_and_redundant_rows=false;
  params.enable_bound_strengthening=false;params.enable_parallel_rows=false;
  params.enable_empty_columns=false;params.enable_singleton_columns=false;
  params.enable_doubleton_equations=false;
  params.enable_dual_fix=false;params.enable_parallel_columns=false;
  params.enable_absolute_value_epigraph_reduction=false;
  const auto old = runtime::options().long_row_compaction;
  const auto saved = old;
  const auto saved_warp = runtime::options().warp_compaction;
  for (bool warp : {false,true}) for (bool flag : {false,true}) {
    runtime::options().warp_compaction = warp;
    runtime::options().long_row_compaction = flag;
    auto result=run_gpu_presolve_with_reduced_lp(lp,params);
    gpu_presolver::presolve::export_recovery_record(result.record);
    assert(!result.has_infeasible && !result.has_unbounded);
    assert(result.record.row_red2org==row_map && result.record.col_red2org==expected_cols);
    const auto& A=result.reduced_lp.A;
    assert(A.rows==static_cast<int>(row_map.size()) && A.cols==static_cast<int>(expected_cols.size()));
    assert(A.nnz==static_cast<int>(expected_ci.size()));
    assert(copy_to_host(A.rowPtr,expected_rp.size())==std::vector<NnzOffset>(expected_rp.begin(),expected_rp.end()));
    assert(copy_to_host(A.colVal,expected_ci.size())==expected_ci);
    auto actual=copy_to_host(A.nzVal,expected_av.size());
    assert(std::memcmp(actual.data(),expected_av.data(),actual.size()*sizeof(double))==0);
    free_gpu_presolve_reduced_lp(result);
  }
  runtime::options().long_row_compaction = saved;
  runtime::options().warp_compaction = saved_warp;
  cudaFree(lp.A.rowPtr);cudaFree(lp.A.colVal);cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);cudaFree(lp.AT.colVal);cudaFree(lp.AT.nzVal);
  cudaFree(lp.c);cudaFree(lp.l);cudaFree(lp.u);cudaFree(lp.AL);cudaFree(lp.AU);
}

// Exercise the parallel remapping threshold with real singleton records,
// comparing the entire chronological tape byte for byte against serial apply.
// Two large deletion epochs exercise appended fixed records and nonidentity
// column maps. Compare the complete fixed record sequence against legacy apply.
void test_fixed_record_direct_equivalence() {
  using namespace gpu_presolver::presolve;
  constexpr int m=1100000, n=2*m;
  std::vector<int> rp(m+1),ci(n),trp(n+1),tci(n);
  std::vector<double> upper(n,10.0);
  for(int i=0;i<m;++i) {
    rp[i]=2*i;
    for(int j=2*i;j<2*i+2;++j){ci[j]=j;trp[j]=j;tci[j]=i;}
    upper[2*i]=0.0;
  }
  rp[m]=n;trp[n]=n;
  LPInfoGpu lp;
  lp.A={m,n,n,copy_to_device(std::vector<NnzOffset>(rp.begin(), rp.end())),copy_to_device(ci),copy_to_device(std::vector<double>(n,1.0))};
  lp.AT={n,m,n,copy_to_device(std::vector<NnzOffset>(trp.begin(), trp.end())),copy_to_device(tci),copy_to_device(std::vector<double>(n,1.0))};
  lp.c=copy_to_device(std::vector<double>(n,0.0));lp.l=copy_to_device(std::vector<double>(n,0.0));lp.u=copy_to_device(upper);
  lp.AL=copy_to_device(std::vector<double>(m,0.0));lp.AU=copy_to_device(std::vector<double>(m,0.0));
  PresolveParams params;
  params.max_iters=3;params.record_postsolve_tape=true;
  params.enable_startup=false;params.enable_structure_specific_rules=false;
  params.enable_infeasible_and_redundant_rows=false;params.enable_bound_strengthening=false;
  params.enable_parallel_rows=false;params.enable_empty_columns=false;
  params.enable_singleton_columns=false;
  params.enable_doubleton_equations=false;params.enable_dual_fix=false;params.enable_parallel_columns=false;
  params.enable_absolute_value_epigraph_reduction=false;
  const auto saved = runtime::options().direct_fixed_values;
  std::vector<int> indices;std::vector<double> values;
  for (bool flag : {false,true}) {
    runtime::options().direct_fixed_values = flag;
    auto result=run_gpu_presolve_with_reduced_lp(lp,params);
    gpu_presolver::presolve::export_recovery_record(result.record);
    assert(!result.has_infeasible && !result.has_unbounded);
    assert(result.record.fixed_idx.size()==n);
    if(!flag){indices=result.record.fixed_idx;values=result.record.fixed_val;}
    else {
      assert(result.record.fixed_idx==indices);
      assert(result.record.fixed_val.size()==values.size());
      assert(std::memcmp(result.record.fixed_val.data(),values.data(),values.size()*sizeof(double))==0);
    }
    free_gpu_presolve_reduced_lp(result);
  }
  runtime::options().direct_fixed_values = saved;
  cudaFree(lp.A.rowPtr);cudaFree(lp.A.colVal);cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);cudaFree(lp.AT.colVal);cudaFree(lp.AT.nzVal);
  cudaFree(lp.c);cudaFree(lp.l);cudaFree(lp.u);cudaFree(lp.AL);cudaFree(lp.AU);
}

void test_large_apply_record_equivalence() {
  using namespace gpu_presolver::presolve;
  constexpr int m = 350000, n = 2 * m;
  std::vector<std::int32_t> rp(m + 1), ci(n), trp(n + 1), tci(n);
  std::vector<double> av(n), cost(n), lower(n), upper(n, 10.0);
  for (int i = 0; i < m; ++i) {
    rp[i] = 2 * i;
    for (int j = 2 * i; j < 2 * i + 2; ++j) {
      ci[j] = j; trp[j] = j; tci[j] = i;
      av[j] = j % 2 ? 3.0 : 2.0;
      cost[j] = j % 2 ? 0.2 : 0.1;
      lower[j] = j % 2 ? 0.0 : -std::numeric_limits<double>::infinity();
      if (!(j % 2)) upper[j] = std::numeric_limits<double>::infinity();
    }
  }
  rp[m] = n; trp[n] = n;
  LPInfoGpu lp;
  lp.A = {m, n, n, copy_to_device(std::vector<NnzOffset>(rp.begin(), rp.end())), copy_to_device(ci), copy_to_device(av)};
  lp.AT = {n, m, n, copy_to_device(std::vector<NnzOffset>(trp.begin(), trp.end())), copy_to_device(tci), copy_to_device(av)};
  lp.c = copy_to_device(cost); lp.l = copy_to_device(lower); lp.u = copy_to_device(upper);
  lp.AL = copy_to_device(std::vector<double>(m, 1.0));
  lp.AU = copy_to_device(std::vector<double>(m, 1.0));
  PresolveParams params;
  params.max_iters = 1;
  params.enable_startup = false;
  params.record_postsolve_tape = true;
  params.enable_infeasible_and_fixed_variables = false;
  params.enable_empty_rows = false;
  params.enable_singleton_rows = false;
  params.enable_infeasible_and_redundant_rows = false;
  params.enable_bound_strengthening = false;
  params.enable_parallel_rows = false;
  params.enable_empty_columns = false;

  params.enable_doubleton_equations = false;
  params.enable_dual_fix = false;
  params.enable_parallel_columns = false;
  params.enable_absolute_value_epigraph_reduction = false;

  const auto saved_chunks = runtime::options().chunked_tape;
  const auto saved_headroom = runtime::options().reserve_tape_headroom;
  const auto saved_parallel = runtime::options().parallel_tape_build;
  runtime::options().parallel_tape_build = false;
  runtime::options().reserve_tape_headroom = false;
  runtime::options().chunked_tape = false;
  auto serial = run_gpu_presolve_with_record(lp, params);
  gpu_presolver::presolve::export_recovery_record(serial.record);
  runtime::options().parallel_tape_build = true;
  runtime::options().reserve_tape_headroom = true;
  runtime::options().chunked_tape = true;
  auto parallel = run_gpu_presolve_with_record(lp, params);
  gpu_presolver::presolve::export_recovery_record(parallel.record);
  assert(!serial.has_infeasible && !parallel.has_infeasible);
  assert(!serial.has_unbounded && !parallel.has_unbounded);
  assert(serial.record.tape.indices.size() >= (1U << 20));
  assert(serial.reduced_rows == parallel.reduced_rows);
  assert(serial.reduced_cols == parallel.reduced_cols);
  assert(serial.reduced_nnz == parallel.reduced_nnz);
  assert(serial.record.row_red2org == parallel.record.row_red2org);
  assert(serial.record.col_red2org == parallel.record.col_red2org);
  const auto& a = serial.record.tape;
  const auto& b = parallel.record.tape;
  assert(a.types == b.types && a.dual_modes == b.dual_modes);
  assert(a.index_starts == b.index_starts && a.value_starts == b.value_starts);
  assert(a.indices == b.indices && a.vals.size() == b.vals.size());
  assert(std::memcmp(a.vals.data(), b.vals.data(), a.vals.size() * sizeof(double)) == 0);
  runtime::options().chunked_tape = saved_chunks;
  runtime::options().reserve_tape_headroom = saved_headroom;
  runtime::options().parallel_tape_build = saved_parallel;
  cudaFree(lp.A.rowPtr); cudaFree(lp.A.colVal); cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr); cudaFree(lp.AT.colVal); cudaFree(lp.AT.nzVal);
  cudaFree(lp.c); cudaFree(lp.l); cudaFree(lp.u); cudaFree(lp.AL); cudaFree(lp.AU);
}

void test_post_propagation_activity_closure(bool record_activity_tape) {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::GpuPresolveSummary;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;

  const double inf = std::numeric_limits<double>::infinity();
  LPInfoGpu lp;
  lp.A = {2, 3, 5,
          copy_to_device<NnzOffset>({0, 2, 5}),
          copy_to_device<std::int32_t>({0, 1, 0, 1, 2}),
          copy_to_device<double>({1.0, 1.0, 1.0, 1.0, -1.0})};
  lp.AT = {3, 2, 5,
           copy_to_device<NnzOffset>({0, 2, 4, 5}),
           copy_to_device<std::int32_t>({0, 1, 0, 1, 1}),
           copy_to_device<double>({1.0, 1.0, 1.0, 1.0, -1.0})};
  lp.c = copy_to_device<double>({0.0, 0.0, 0.0});
  lp.l = copy_to_device<double>({0.0, 0.0, 0.0});
  lp.u = copy_to_device<double>({10.0, 1.0, inf});
  lp.AL = copy_to_device<double>({-inf, -inf});
  lp.AU = copy_to_device<double>({2.0, 3.0});

  PresolveParams params;
  params.max_iters = 2;
  params.enable_startup = false;
  params.record_postsolve_tape = true;
  params.record_postsolve_tape_cpu = record_activity_tape;
  params.enable_infeasible_and_fixed_variables = false;
  params.enable_empty_rows = false;
  params.enable_singleton_rows = false;
  params.enable_parallel_rows = false;
  params.enable_empty_columns = false;
  params.enable_singleton_columns = false;

  params.enable_doubleton_equations = false;
  params.enable_dual_fix = false;
  params.enable_parallel_columns = false;
  params.enable_absolute_value_epigraph_reduction = false;

  GpuPresolveSummary summary = gpu_presolver::presolve::run_gpu_presolve_with_record(lp, params);
  gpu_presolver::presolve::export_recovery_record(summary.record);
  assert(!summary.has_infeasible);
  assert(!summary.has_unbounded);
  // Bound strengthening makes the second row redundant within this round.
  // The interleaved detector removes it before the shared plan is applied.
  assert(summary.reduced_rows == 1);
  assert(summary.reduced_cols == 3);
  assert(summary.reduced_nnz == 2);
  assert(summary.record.row_red2org == (std::vector<std::int32_t>{0}));
  assert(summary.record.row_org2red == (std::vector<std::int32_t>{0, -1}));

  assert(summary.record.tape.types.size() == (record_activity_tape ? 2 : 1));
  assert(summary.record.tape_gpu.record_count == 0);
  assert(summary.record.tape_gpu.index_count == 0);
  assert(summary.record.tape_gpu.value_count == 0);
  const std::int32_t bound_type = summary.record.tape.types[0];
  assert(bound_type == static_cast<std::int32_t>(PostsolveReductionType::BoundChangeTheRow) ||
         bound_type == static_cast<std::int32_t>(PostsolveReductionType::BoundChangeNoRow));
  if (record_activity_tape) {
    assert(summary.record.tape.types[1] == static_cast<std::int32_t>(PostsolveReductionType::DeletedRow));
  }

  double* x_red = copy_to_device<double>({1.0, 1.0, 0.0});
  double* y_red = copy_to_device<double>(std::vector<double>(summary.reduced_rows, 0.0));
  double* z_red = copy_to_device<double>({0.0, 0.0, 0.0});
  GpuPostsolveResult post = gpu_presolver::presolve::postsolve_gpu(
      x_red, y_red, z_red, summary.record, &lp);
  const std::vector<double> x = copy_to_host(post.x_org, 3);
  const std::vector<double> y = copy_to_host(post.y_org, 2);
  const std::vector<double> z = copy_to_host(post.z_org, 3);
  assert(std::fabs(x[0] - 1.0) < 1.0e-12);
  assert(std::fabs(x[1] - 1.0) < 1.0e-12);
  assert(std::fabs(x[2]) < 1.0e-12);
  assert(std::fabs(y[0]) < 1.0e-12);
  assert(std::fabs(y[1]) < 1.0e-12);
  assert(std::fabs(z[0]) < 1.0e-12);
  assert(std::fabs(z[1]) < 1.0e-12);
  assert(std::fabs(z[2]) < 1.0e-12);
  assert(x[0] + x[1] <= 2.0 + 1.0e-12);
  assert(x[0] + x[1] - x[2] <= 3.0 + 1.0e-12);

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  cudaFree(post.x_org);
  cudaFree(post.y_org);
  cudaFree(post.z_org);
  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(lp.c);
  cudaFree(lp.l);
  cudaFree(lp.u);
  cudaFree(lp.AL);
  cudaFree(lp.AU);
}

void test_bound_strengthening_defers_dependency_chain() {
  using gpu_presolver::presolve::GpuPresolveSummary;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;

  const double inf = std::numeric_limits<double>::infinity();
  LPInfoGpu lp;
  lp.A = {2, 4, 5,
          copy_to_device<NnzOffset>({0, 2, 5}),
          copy_to_device<std::int32_t>({0, 1, 1, 2, 3}),
          copy_to_device<double>({1.0, 1.0, -1.0, 1.0, 1.0})};
  lp.AT = {4, 2, 5,
           copy_to_device<NnzOffset>({0, 1, 3, 4, 5}),
           copy_to_device<std::int32_t>({0, 0, 1, 1, 1}),
           copy_to_device<double>({1.0, 1.0, -1.0, 1.0, 1.0})};
  lp.c = copy_to_device<double>({0.0, 0.0, 0.0, 0.0});
  lp.l = copy_to_device<double>({1.0, 0.0, 0.0, 0.0});
  lp.u = copy_to_device<double>({1.0, inf, inf, 0.0});
  lp.AL = copy_to_device<double>({-inf, -inf});
  lp.AU = copy_to_device<double>({1.0, 0.0});

  PresolveParams params;
  params.max_iters = 1;
  params.enable_startup = false;
  params.enable_infeasible_and_fixed_variables = false;
  params.enable_empty_rows = false;
  params.enable_singleton_rows = false;
  params.enable_infeasible_and_redundant_rows = false;
  params.enable_parallel_rows = false;
  params.enable_empty_columns = false;
  params.enable_singleton_columns = false;

  params.enable_doubleton_equations = false;
  params.enable_dual_fix = false;
  params.enable_parallel_columns = false;
  params.enable_absolute_value_epigraph_reduction = false;

  // One phase fixes x1, leaving the dependent x2 for a later outer phase.
  params.bound_strengthening_bound_only_nnz_round_budget = 1;

  GpuPresolveSummary summary = gpu_presolver::presolve::run_gpu_presolve_with_record(lp, params);
  gpu_presolver::presolve::export_recovery_record(summary.record);
  assert(!summary.has_infeasible);
  assert(!summary.has_unbounded);
  assert(summary.reduced_rows == 2);
  assert(summary.reduced_cols == 3);
  assert(summary.reduced_nnz == 3);

  params.max_iters = 2;
  auto repeated = gpu_presolver::presolve::run_gpu_presolve_with_record(lp, params);
  gpu_presolver::presolve::export_recovery_record(repeated.record);
  assert(!repeated.has_infeasible && !repeated.has_unbounded);
  assert(repeated.reduced_rows == 2 && repeated.reduced_cols == 2);
  assert(repeated.reduced_nnz == 2);

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(lp.c);
  cudaFree(lp.l);
  cudaFree(lp.u);
  cudaFree(lp.AL);
  cudaFree(lp.AU);
}

std::vector<double> run_pure_bound_dependency_chain(std::int64_t nnz_round_budget) {
  using gpu_presolver::presolve::GpuPresolveSummary;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;

  const double inf = std::numeric_limits<double>::infinity();
  LPInfoGpu lp;
  // x0 + x1 <= 2 and -x1 + x2 <= 0 with x0 fixed at one.  Synchronous
  // propagation discovers u1=1 in the first round and u2=1 in the second;
  // neither tightening is structural.
  lp.A = {2, 3, 4,
          copy_to_device<NnzOffset>({0, 2, 4}),
          copy_to_device<std::int32_t>({0, 1, 1, 2}),
          copy_to_device<double>({1.0, 1.0, -1.0, 1.0})};
  lp.AT = {3, 2, 4,
           copy_to_device<NnzOffset>({0, 1, 3, 4}),
           copy_to_device<std::int32_t>({0, 0, 1, 1}),
           copy_to_device<double>({1.0, 1.0, -1.0, 1.0})};
  lp.c = copy_to_device<double>({0.0, 0.0, 0.0});
  lp.l = copy_to_device<double>({1.0, 0.0, 0.0});
  lp.u = copy_to_device<double>({1.0, inf, inf});
  lp.AL = copy_to_device<double>({-inf, -inf});
  lp.AU = copy_to_device<double>({2.0, 0.0});

  PresolveParams params;
  params.max_iters = 1;
  params.enable_startup = false;
  params.record_postsolve_tape = false;
  params.enable_infeasible_and_fixed_variables = false;
  params.enable_empty_rows = false;
  params.enable_singleton_rows = false;
  params.enable_infeasible_and_redundant_rows = false;
  params.enable_parallel_rows = false;
  params.enable_empty_columns = false;
  params.enable_singleton_columns = false;

  params.enable_doubleton_equations = false;
  params.enable_dual_fix = false;
  params.enable_parallel_columns = false;
  params.enable_absolute_value_epigraph_reduction = false;

  params.bound_strengthening_bound_only_nnz_round_budget = nnz_round_budget;

  GpuPresolveSummary summary =
      gpu_presolver::presolve::run_gpu_presolve_with_reduced_lp(lp, params);
  gpu_presolver::presolve::export_recovery_record(summary.record);
  assert(!summary.has_infeasible);
  assert(!summary.has_unbounded);
  assert(summary.reduced_rows == 2);
  assert(summary.reduced_cols == 3);
  assert(summary.reduced_nnz == 4);
  const std::vector<double> upper = copy_to_host(summary.reduced_lp.u, 3);
  gpu_presolver::presolve::free_gpu_presolve_reduced_lp(summary);

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(lp.c);
  cudaFree(lp.l);
  cudaFree(lp.u);
  cudaFree(lp.AL);
  cudaFree(lp.AU);
  return upper;
}

void test_bound_strengthening_single_pass_ignores_legacy_budget() {
  // Four nnz and a four-nnz-round budget allow exactly one bound-only round.
  const std::vector<double> capped = run_pure_bound_dependency_chain(4);
  assert(std::fabs(capped[0] - 1.0) < 1.0e-12);
  assert(std::fabs(capped[1] - 1.0) < 1.0e-12);
  assert(std::isinf(capped[2]));

  // Removing the legacy budget does not add another pass inside the phase.
  const std::vector<double> uncapped = run_pure_bound_dependency_chain(0);
  assert(std::fabs(uncapped[0] - 1.0) < 1.0e-12);
  assert(std::fabs(uncapped[1] - 1.0) < 1.0e-12);
  assert(std::isinf(uncapped[2]));
}

}  // namespace

int main() {
  // Exercise both activity-recording modes on the production schedule.
  test_post_propagation_activity_closure(false);
  test_post_propagation_activity_closure(true);
  test_bound_strengthening_defers_dependency_chain();
  test_bound_strengthening_single_pass_ignores_legacy_budget();

  const double inf = std::numeric_limits<double>::infinity();
  gpu_presolver::presolve::LPInfoGpu sched_lp;
  sched_lp.A = {3, 3, 2,
                copy_to_device<NnzOffset>({0, 1, 2, 2}),
                copy_to_device<std::int32_t>({0, 1}),
                copy_to_device<double>({1.0, 1.0})};
  sched_lp.AT = {3, 3, 2,
                 copy_to_device<NnzOffset>({0, 1, 2, 2}),
                 copy_to_device<std::int32_t>({0, 1}),
                 copy_to_device<double>({1.0, 1.0})};
  sched_lp.c = copy_to_device<double>({0.0, 0.0, 2.0});
  sched_lp.l = copy_to_device<double>({5.0, 0.0, 1.0});
  sched_lp.u = copy_to_device<double>({5.0, 10.0, 3.0});
  sched_lp.AL = copy_to_device<double>({5.0, -inf, -1.0});
  sched_lp.AU = copy_to_device<double>({5.0, inf, 1.0});

  gpu_presolver::presolve::PresolveParams params;
  params.max_iters = 4;
  gpu_presolver::presolve::GpuPresolveSummary summary =
      gpu_presolver::presolve::run_gpu_presolve_with_record(sched_lp, params);
  gpu_presolver::presolve::export_recovery_record(summary.record);
  assert(!summary.has_infeasible);
  assert(!summary.has_unbounded);
  assert(summary.reduced_rows == 0);
  assert(summary.reduced_cols == 0);

  cudaFree(sched_lp.A.rowPtr);
  cudaFree(sched_lp.A.colVal);
  cudaFree(sched_lp.A.nzVal);
  cudaFree(sched_lp.AT.rowPtr);
  cudaFree(sched_lp.AT.colVal);
  cudaFree(sched_lp.AT.nzVal);
  cudaFree(sched_lp.c);
  cudaFree(sched_lp.l);
  cudaFree(sched_lp.u);
  cudaFree(sched_lp.AL);
  cudaFree(sched_lp.AU);

  constexpr std::int32_t doubleton_rows = 300;
  constexpr std::int32_t doubleton_cols = doubleton_rows * 2;
  std::vector<NnzOffset> doubleton_row_ptr(static_cast<std::size_t>(doubleton_rows + 1));
  std::vector<std::int32_t> doubleton_col_val(static_cast<std::size_t>(doubleton_rows * 2));
  std::vector<double> doubleton_a(static_cast<std::size_t>(doubleton_rows * 2), 1.0);
  for (std::int32_t row = 0; row < doubleton_rows; ++row) {
    doubleton_row_ptr[static_cast<std::size_t>(row)] = 2 * row;
    doubleton_col_val[static_cast<std::size_t>(2 * row)] = 2 * row;
    doubleton_col_val[static_cast<std::size_t>(2 * row + 1)] = 2 * row + 1;
  }
  doubleton_row_ptr[static_cast<std::size_t>(doubleton_rows)] = doubleton_rows * 2;
  std::vector<NnzOffset> doubleton_at_row_ptr(static_cast<std::size_t>(doubleton_cols + 1));
  std::vector<std::int32_t> doubleton_at_col_val(static_cast<std::size_t>(doubleton_rows * 2));
  for (std::int32_t col = 0; col < doubleton_cols; ++col) {
    doubleton_at_row_ptr[static_cast<std::size_t>(col)] = col;
    doubleton_at_col_val[static_cast<std::size_t>(col)] = col / 2;
  }
  doubleton_at_row_ptr[static_cast<std::size_t>(doubleton_cols)] = doubleton_cols;

  gpu_presolver::presolve::LPInfoGpu doubleton_only_lp;
  doubleton_only_lp.A = {doubleton_rows, doubleton_cols, doubleton_rows * 2,
                         copy_to_device<NnzOffset>(doubleton_row_ptr),
                         copy_to_device<std::int32_t>(doubleton_col_val),
                         copy_to_device<double>(doubleton_a)};
  doubleton_only_lp.AT = {doubleton_cols, doubleton_rows, doubleton_rows * 2,
                          copy_to_device<NnzOffset>(doubleton_at_row_ptr),
                          copy_to_device<std::int32_t>(doubleton_at_col_val),
                          copy_to_device<double>(doubleton_a)};
  doubleton_only_lp.c = copy_to_device<double>(std::vector<double>(static_cast<std::size_t>(doubleton_cols), 0.0));
  doubleton_only_lp.l = copy_to_device<double>(std::vector<double>(static_cast<std::size_t>(doubleton_cols), 0.0));
  doubleton_only_lp.u = copy_to_device<double>(std::vector<double>(static_cast<std::size_t>(doubleton_cols), 10.0));
  doubleton_only_lp.AL = copy_to_device<double>(std::vector<double>(static_cast<std::size_t>(doubleton_rows), 1.0));
  doubleton_only_lp.AU = copy_to_device<double>(std::vector<double>(static_cast<std::size_t>(doubleton_rows), 1.0));

  gpu_presolver::presolve::PresolveParams doubleton_only_params;
  doubleton_only_params.max_iters = 4;
  doubleton_only_params.enable_infeasible_and_fixed_variables = false;
  doubleton_only_params.enable_empty_rows = false;
  doubleton_only_params.enable_singleton_rows = false;
  doubleton_only_params.enable_infeasible_and_redundant_rows = false;
  doubleton_only_params.enable_bound_strengthening = false;
  doubleton_only_params.enable_parallel_rows = false;
  doubleton_only_params.enable_empty_columns = false;
  doubleton_only_params.enable_singleton_columns = false;

  doubleton_only_params.enable_dual_fix = false;
  doubleton_only_params.enable_parallel_columns = false;
  doubleton_only_params.enable_absolute_value_epigraph_reduction = false;

  gpu_presolver::presolve::GpuPresolveSummary doubleton_only_summary =
      gpu_presolver::presolve::run_gpu_presolve_with_record(doubleton_only_lp,
                                                            doubleton_only_params);
  gpu_presolver::presolve::export_recovery_record(doubleton_only_summary.record);
  assert(!doubleton_only_summary.has_infeasible);
  assert(!doubleton_only_summary.has_unbounded);
  assert(doubleton_only_summary.reduced_rows == 0);
  assert(doubleton_only_summary.reduced_cols == doubleton_rows);
  assert(doubleton_only_summary.reduced_nnz == 0);
  assert(doubleton_only_summary.iterations > 1);

  cudaFree(doubleton_only_lp.A.rowPtr);
  cudaFree(doubleton_only_lp.A.colVal);
  cudaFree(doubleton_only_lp.A.nzVal);
  cudaFree(doubleton_only_lp.AT.rowPtr);
  cudaFree(doubleton_only_lp.AT.colVal);
  cudaFree(doubleton_only_lp.AT.nzVal);
  cudaFree(doubleton_only_lp.c);
  cudaFree(doubleton_only_lp.l);
  cudaFree(doubleton_only_lp.u);
  cudaFree(doubleton_only_lp.AL);
  cudaFree(doubleton_only_lp.AU);

  test_parallel_nonidentity_mapping();
  test_compact_long_rows_oracle();
  test_fixed_record_direct_equivalence();
  test_large_apply_record_equivalence();

  std::cout << "test_gpu_direct_port passed\n";
  return 0;
}
