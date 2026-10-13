#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/gpu_presolve_kernels.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include "gpu_presolver/presolve/rules/rule_parallel_rows.hpp"
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

void test_adaptive_activity_mixed_lengths() {
  using namespace gpu_presolver::presolve;
  const int lengths[] = {0, 1, 8, 9, 31, 32, 33, 1024, 1025, 4097};
  constexpr int rows = 2100, cols = 4097;
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<int> ptr(1, 0), indices;
  std::vector<double> values, lower(cols), upper(cols), expected_lo(rows), expected_hi(rows);
  std::vector<int> expected_ni(rows), expected_pi(rows);
  for (int j = 0; j < cols; ++j) {
    lower[j] = j % 113 == 0 ? -inf : -(j % 7 + 1) * 0.5;
    upper[j] = j % 127 == 0 ? inf : (j % 11 + 1) * 0.5;
  }
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < lengths[i % 10]; ++j) {
      const double a = j % 71 == 0 ? 0.0 : (j % 2 ? -0.5 : 2.0);
      indices.push_back(j); values.push_back(a);
      if (a == 0.0) continue;
      const double lo = a >= 0 ? a * lower[j] : a * upper[j];
      const double hi = a >= 0 ? a * upper[j] : a * lower[j];
      if (std::isfinite(lo)) expected_lo[i] += lo; else ++expected_ni[i];
      if (std::isfinite(hi)) expected_hi[i] += hi; else ++expected_pi[i];
    }
    ptr.push_back(indices.size());
  }
  DeviceCsrMatrix A{rows, cols, static_cast<int>(values.size()), copy_to_device(std::vector<NnzOffset>(ptr.begin(),ptr.end())),
                    copy_to_device(indices), copy_to_device(values)};
  auto* l = copy_to_device(lower); auto* u = copy_to_device(upper);
  auto* lo = copy_to_device(std::vector<double>(rows));
  auto* hi = copy_to_device(std::vector<double>(rows));
  auto* ni = copy_to_device(std::vector<int>(rows));
  auto* pi = copy_to_device(std::vector<int>(rows));
  const auto saved = runtime::options().adaptive_activity;
  for (bool setting : {false, true, true}) {
    runtime::options().adaptive_activity = setting;
    compute_row_activity_summary(lo, hi, ni, pi, A, l, u, 1e-12);
    check(cudaDeviceSynchronize(), "mixed length activity");
    // Dyadic data allow exact comparisons against an independently computed CPU oracle.
    assert(copy_to_host(lo, rows) == expected_lo);
    assert(copy_to_host(hi, rows) == expected_hi);
    assert(copy_to_host(ni, rows) == expected_ni);
    assert(copy_to_host(pi, rows) == expected_pi);
  }
  std::vector<double> expected_max(rows);
  for (int i=0; i<rows; ++i)
    for (int j=ptr[i]; j<ptr[i+1]; ++j) expected_max[i]=std::fmax(expected_max[i],std::fabs(values[j]));
  const auto saved_extended = runtime::options().extended_row_kernels;
  for (bool setting : {false, true}) {
    runtime::options().extended_row_kernels = setting;
    compute_col_max_abs(lo,A);
    assert(copy_to_host(lo,rows)==expected_max);
  }
  runtime::options().extended_row_kernels = saved_extended;
  const auto saved_reuse = runtime::options().reuse_row_groups;
  runtime::options().reuse_row_groups = true;
  const auto saved_sort = runtime::options().reuse_sortedness;
  runtime::options().reuse_sortedness = true;
  {
    MatrixAnalysisScope scope;
    assert(csr_rows_strictly_sorted(A));
    assert(csr_rows_strictly_sorted(A));
    // Detect disorder in a long row, despite reusing the same pointers.
    const NnzOffset offset=ptr[8]+500;
    const int duplicate=indices[offset-1];
    check(cudaMemcpy(A.colVal+offset,&duplicate,sizeof(int),cudaMemcpyHostToDevice),"duplicate index");
    invalidate_matrix_analysis();
    assert(!csr_rows_strictly_sorted(A));
    check(cudaMemcpy(A.colVal+offset,&indices[offset],sizeof(int),cudaMemcpyHostToDevice),"restore index");
    invalidate_matrix_analysis();
    assert(csr_rows_strictly_sorted(A));
    const auto& first = prepare_adaptive_row_groups(A);
    const auto before = copy_to_host(first.counts,2);
    assert(before[0]==1050 && before[1]==420);
    const auto& again = prepare_adaptive_row_groups(A);
    assert(again.counts==first.counts);
    // Change row lengths in-place without changing address, dimensions or total nnz.
    // A structure-version invalidation must force fresh grouping.
    ptr[9]-=1020;
    check(cudaMemcpy(A.rowPtr,std::vector<NnzOffset>(ptr.begin(),ptr.end()).data(),ptr.size()*sizeof(NnzOffset),cudaMemcpyHostToDevice),"modify lengths");
    invalidate_matrix_analysis();
    const auto after = copy_to_host(prepare_adaptive_row_groups(A).counts,2);
    assert(after[0]==1050 && after[1]==419);
  }
  assert(matrix_analysis_epoch()==0);
  runtime::options().reuse_sortedness = saved_sort;
  runtime::options().reuse_row_groups = saved_reuse;
  runtime::options().adaptive_activity = saved;
  cudaFree(A.rowPtr); cudaFree(A.colVal); cudaFree(A.nzVal);
  cudaFree(l); cudaFree(u); cudaFree(lo); cudaFree(hi); cudaFree(ni); cudaFree(pi);
}

void test_hash_values_match_cpu() {
  using namespace gpu_presolver::presolve;
  const int lengths[]={0,1,8,9,31,32,33,1024,1025,4097};
  std::vector<int> ptr{0},indices;
  std::vector<double> values;
  std::vector<std::uint8_t> keep(2100,1);keep[7]=0;
  for(int row=0;row<2100;++row) {
    const int length=row<10 ? lengths[row] : 1;
    for(int j=0;j<length;++j) { indices.push_back(j);values.push_back(j%2 ? -0.5 : 2.0); }
    ptr.push_back(indices.size());
  }
  DeviceCsrMatrix A{2100,4097,static_cast<int>(values.size()),copy_to_device(std::vector<NnzOffset>(ptr.begin(),ptr.end())),copy_to_device(indices),copy_to_device(values)};
  auto* mask=copy_to_device(keep);
  auto* hashes=copy_to_device(std::vector<unsigned long long>(2100));
  const auto saved = runtime::options().hash_scan;
  auto mix=[](unsigned long long h,unsigned long long x) { return (h^x)*0x100000001b3ULL; };
  for(int rows:{10,2100}) {
    A.rows=rows;A.nnz=ptr[rows];
    const bool use_xor=A.nnz>32LL*rows;
    std::vector<unsigned long long> expected(rows);
    for(int row=0;row<rows;++row) {
      NnzOffset first=ptr[row],last=ptr[row+1];
      if(!keep[row]) {expected[row]=row;continue;}
      if(first==last) {expected[row]=mix(0xcbf29ce484222325ULL,row);continue;}
      auto h=mix(0xcbf29ce484222325ULL,last-first);unsigned long long aggregate=0;
      for(NnzOffset p=first;p<last;++p) {
        double norm=values[p]/values[first];unsigned long long bits;std::memcpy(&bits,&norm,sizeof(bits));
        if(use_xor) aggregate^=mix(mix(0xcbf29ce484222325ULL,indices[p]),bits);
        else h=mix(mix(h,indices[p]),bits);
      }
      expected[row]=use_xor?mix(h,aggregate):h;
    }
    for (bool setting : {false,true}) {
      runtime::options().hash_scan = setting;
      compute_parallel_row_hashes(hashes,A,mask,1e-12);
      assert(copy_to_host(hashes,rows)==expected);
    }
  }
  runtime::options().hash_scan = saved;
  cudaFree(A.rowPtr);cudaFree(A.colVal);cudaFree(A.nzVal);cudaFree(mask);cudaFree(hashes);
}

void test_incremental_live_structure_matches_full_scan() {
  using namespace gpu_presolver::presolve;
  auto require = [](bool condition, const char* what) {
    if (!condition) throw std::runtime_error(what);
  };
  LPInfoGpu lp;
  lp.A = DeviceCsrMatrix{5, 5, 8,
      copy_to_device<NnzOffset>({0, 3, 5, 6, 8, 8}),
      copy_to_device<std::int32_t>({0, 1, 3, 1, 2, 2, 3, 4}),
      copy_to_device<double>({2, 0, 4, 5, 6, 7, 8, 9})};
  lp.AT = DeviceCsrMatrix{5, 5, 8,
      copy_to_device<NnzOffset>({0, 1, 3, 5, 7, 8}),
      copy_to_device<std::int32_t>({0, 0, 1, 1, 2, 0, 3, 3}),
      copy_to_device<double>({2, 0, 5, 6, 7, 4, 8, 9})};
  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1, 1, 1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1, 1, 1, 1});
  auto* old_rows = copy_to_device<std::uint8_t>({1, 1, 1, 1, 1});
  auto* old_cols = copy_to_device<std::uint8_t>({1, 1, 1, 1, 1});
  std::int32_t* touched_rows = nullptr;
  std::int32_t* touched_cols = nullptr;
  check(cudaMalloc(&touched_rows, 5 * sizeof(int)), "touched rows");
  check(cudaMalloc(&touched_cols, 5 * sizeof(int)), "touched cols");

  auto allocate = [](PresolveStatsGpu& s) {
    check(cudaMalloc(&s.row_nnz, 5 * sizeof(int)), "row degree");
    check(cudaMalloc(&s.col_nnz, 5 * sizeof(int)), "col degree");
    check(cudaMalloc(&s.empty_row_mask, 5), "empty row");
    check(cudaMalloc(&s.empty_col_mask, 5), "empty col");
    check(cudaMalloc(&s.singleton_row_mask, 5), "singleton row");
    check(cudaMalloc(&s.singleton_column_mask, 5), "singleton col");
    check(cudaMalloc(&s.singleton_row_col, 5 * sizeof(int)), "row support");
    check(cudaMalloc(&s.singleton_column_row, 5 * sizeof(int)), "col support");
    check(cudaMalloc(&s.singleton_row_val, 5 * sizeof(double)), "row value");
    check(cudaMalloc(&s.singleton_column_val, 5 * sizeof(double)), "col value");
  };
  auto release = [](PresolveStatsGpu& s) {
    cudaFree(s.row_nnz); cudaFree(s.col_nnz);
    cudaFree(s.empty_row_mask); cudaFree(s.empty_col_mask);
    cudaFree(s.singleton_row_mask); cudaFree(s.singleton_column_mask);
    cudaFree(s.singleton_row_col); cudaFree(s.singleton_column_row);
    cudaFree(s.singleton_row_val); cudaFree(s.singleton_column_val);
  };
  PresolveStatsGpu full, delta;
  allocate(full);
  allocate(delta);
  refresh_live_stats(full, lp, plan);
  refresh_live_stats(delta, lp, plan);
  auto compare = [&]() {
    require(full.live_nnz == delta.live_nnz, "live nnz mismatch");
    require(copy_to_host(full.row_nnz, 5) == copy_to_host(delta.row_nnz, 5), "row degree mismatch");
    require(copy_to_host(full.col_nnz, 5) == copy_to_host(delta.col_nnz, 5), "col degree mismatch");
    require(copy_to_host(full.empty_row_mask, 5) == copy_to_host(delta.empty_row_mask, 5), "empty row mismatch");
    require(copy_to_host(full.empty_col_mask, 5) == copy_to_host(delta.empty_col_mask, 5), "empty col mismatch");
    require(copy_to_host(full.singleton_row_mask, 5) == copy_to_host(delta.singleton_row_mask, 5), "singleton row mismatch");
    require(copy_to_host(full.singleton_column_mask, 5) == copy_to_host(delta.singleton_column_mask, 5), "singleton col mismatch");
    require(copy_to_host(full.singleton_row_col, 5) == copy_to_host(delta.singleton_row_col, 5), "row support mismatch");
    require(copy_to_host(full.singleton_column_row, 5) == copy_to_host(delta.singleton_column_row, 5), "col support mismatch");
    require(copy_to_host(full.singleton_row_val, 5) == copy_to_host(delta.singleton_row_val, 5), "row value mismatch");
    require(copy_to_host(full.singleton_column_val, 5) == copy_to_host(delta.singleton_column_val, 5), "col value mismatch");
  };
  compare();
  const std::vector<std::vector<std::uint8_t>> row_states{
      {1, 0, 1, 1, 1}, {1, 0, 1, 1, 1}, {1, 0, 0, 0, 1}, {1, 0, 0, 0, 0}};
  const std::vector<std::vector<std::uint8_t>> col_states{
      {1, 1, 1, 1, 1}, {1, 1, 1, 0, 1}, {1, 1, 0, 0, 1}, {1, 0, 0, 0, 1}};
  for (std::size_t k = 0; k < row_states.size(); ++k) {
    check(cudaMemcpy(plan.keep_row_mask, row_states[k].data(), 5,
                     cudaMemcpyHostToDevice), "update row mask");
    check(cudaMemcpy(plan.keep_col_mask, col_states[k].data(), 5,
                     cudaMemcpyHostToDevice), "update col mask");
    require(update_live_stats_from_deletions(delta, lp, plan, old_rows, old_cols,
                                             touched_rows, touched_cols), "incremental update rejected deletion");
    refresh_live_stats(full, lp, plan);
    compare();
    check(cudaMemcpy(old_rows, plan.keep_row_mask, 5, cudaMemcpyDeviceToDevice),
          "snapshot rows");
    check(cudaMemcpy(old_cols, plan.keep_col_mask, 5, cudaMemcpyDeviceToDevice),
          "snapshot cols");
  }
  const std::uint8_t one = 1;
  check(cudaMemcpy(plan.keep_row_mask + 1, &one, 1, cudaMemcpyHostToDevice),
        "reactivate row");
  require(!update_live_stats_from_deletions(delta, lp, plan, old_rows, old_cols,
                                            touched_rows, touched_cols), "reactivation was accepted");

  // The merged scheduler keeps the authoritative nnz on the device. A stale
  // host cache must neither corrupt repeated updates nor be read back early.
  auto* device_live_nnz = copy_to_device<unsigned long long>({0});
  const std::vector<std::uint8_t> all_live(5, 1);
  check(cudaMemcpy(plan.keep_row_mask, all_live.data(), 5, cudaMemcpyHostToDevice), "reset rows");
  check(cudaMemcpy(plan.keep_col_mask, all_live.data(), 5, cudaMemcpyHostToDevice), "reset cols");
  check(cudaMemcpy(old_rows, plan.keep_row_mask, 5, cudaMemcpyDeviceToDevice), "reset old rows");
  check(cudaMemcpy(old_cols, plan.keep_col_mask, 5, cudaMemcpyDeviceToDevice), "reset old cols");
  delta.live_nnz = -123;
  refresh_live_stats(delta, lp, plan, false, device_live_nnz);
  require(delta.live_nnz == -123, "deferred scan read back nnz");
  require(copy_to_host(device_live_nnz, 1)[0] == 8, "device scan nnz mismatch");
  for (std::size_t k = 0; k < row_states.size(); ++k) {
    check(cudaMemcpy(plan.keep_row_mask, row_states[k].data(), 5, cudaMemcpyHostToDevice), "device update rows");
    check(cudaMemcpy(plan.keep_col_mask, col_states[k].data(), 5, cudaMemcpyHostToDevice), "device update cols");
    require(update_live_stats_from_deletions(delta, lp, plan, old_rows, old_cols,
        touched_rows, touched_cols, false, device_live_nnz), "device update rejected deletion");
    require(delta.live_nnz == -123, "incremental update read back nnz");
    refresh_live_stats(full, lp, plan);
    const auto device_count = copy_to_host(device_live_nnz, 1)[0];
    require(device_count == static_cast<unsigned long long>(full.live_nnz), "device incremental nnz mismatch");
    delta.live_nnz = static_cast<NnzOffset>(device_count);
    compare();
    delta.live_nnz = -123;
    check(cudaMemcpy(old_rows, plan.keep_row_mask, 5, cudaMemcpyDeviceToDevice), "device snapshot rows");
    check(cudaMemcpy(old_cols, plan.keep_col_mask, 5, cudaMemcpyDeviceToDevice), "device snapshot cols");
  }
  require(update_live_stats_from_deletions(delta, lp, plan, old_rows, old_cols,
      touched_rows, touched_cols, true, device_live_nnz), "device readback rejected unchanged masks");
  compare();
  const auto before_reactivation = copy_to_host(device_live_nnz, 1);
  check(cudaMemcpy(plan.keep_row_mask + 1, &one, 1, cudaMemcpyHostToDevice), "device reactivate row");
  require(!update_live_stats_from_deletions(delta, lp, plan, old_rows, old_cols,
      touched_rows, touched_cols, false, device_live_nnz), "device reactivation was accepted");
  require(copy_to_host(device_live_nnz, 1) == before_reactivation, "failed update changed nnz");
  cudaFree(device_live_nnz);

  release(full); release(delta);
  cudaFree(lp.A.rowPtr); cudaFree(lp.A.colVal); cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr); cudaFree(lp.AT.colVal); cudaFree(lp.AT.nzVal);
  cudaFree(plan.keep_row_mask); cudaFree(plan.keep_col_mask);
  cudaFree(old_rows); cudaFree(old_cols);
  cudaFree(touched_rows); cudaFree(touched_cols);
}

}  // namespace

int main() {
  using gpu_presolver::presolve::DeviceCsrMatrix;
  const double inf = std::numeric_limits<double>::infinity();

  NnzOffset* row_ptr = copy_to_device<NnzOffset>({0, 0, 1, 3});
  std::int32_t* col_val = copy_to_device<std::int32_t>({2, 0, 1});
  double* nz_val = copy_to_device<double>({4.0, 5.0, 6.0});

  DeviceCsrMatrix matrix{3, 3, 3, row_ptr, col_val, nz_val};

  std::int32_t* row_nnz = nullptr;
  std::int32_t* singleton_column = nullptr;
  double* singleton_val = nullptr;
  check(cudaMalloc(&row_nnz, sizeof(std::int32_t) * 3), "cudaMalloc row_nnz");
  check(cudaMalloc(&singleton_column, sizeof(std::int32_t) * 3), "cudaMalloc singleton_column");
  check(cudaMalloc(&singleton_val, sizeof(double) * 3), "cudaMalloc singleton_val");

  gpu_presolver::presolve::compute_row_nnz(row_nnz, matrix);
  gpu_presolver::presolve::compute_singleton_row_support(singleton_column, singleton_val, row_nnz, matrix);
  check(cudaDeviceSynchronize(), "sync");

  const std::vector<std::int32_t> nnz = copy_to_host(row_nnz, 3);
  const std::vector<std::int32_t> support_col = copy_to_host(singleton_column, 3);
  const std::vector<double> support_val = copy_to_host(singleton_val, 3);

  assert((nnz == std::vector<std::int32_t>{0, 1, 2}));
  assert((support_col == std::vector<std::int32_t>{-1, 2, -1}));
  assert(support_val[0] == 0.0);
  assert(support_val[1] == 4.0);
  assert(support_val[2] == 0.0);

  double* l = copy_to_device<double>({-1.0, -inf, 2.0});
  double* u = copy_to_device<double>({3.0, 5.0, inf});
  double* row_min = nullptr;
  double* row_max = nullptr;
  double* row_min_fin = nullptr;
  double* row_max_fin = nullptr;
  std::int32_t* row_min_neg_inf_count = nullptr;
  std::int32_t* row_max_pos_inf_count = nullptr;
  check(cudaMalloc(&row_min, sizeof(double) * 3), "cudaMalloc row_min");
  check(cudaMalloc(&row_max, sizeof(double) * 3), "cudaMalloc row_max");
  check(cudaMalloc(&row_min_fin, sizeof(double) * 3), "cudaMalloc row_min_fin");
  check(cudaMalloc(&row_max_fin, sizeof(double) * 3), "cudaMalloc row_max_fin");
  check(cudaMalloc(&row_min_neg_inf_count, sizeof(std::int32_t) * 3), "cudaMalloc row_min_neg_inf_count");
  check(cudaMalloc(&row_max_pos_inf_count, sizeof(std::int32_t) * 3), "cudaMalloc row_max_pos_inf_count");

  gpu_presolver::presolve::compute_row_activity_bounds(row_min, row_max, matrix, l, u);
  gpu_presolver::presolve::compute_row_activity_summary(
      row_min_fin,
      row_max_fin,
      row_min_neg_inf_count,
      row_max_pos_inf_count,
      matrix,
      l,
      u,
      1.0e-12);
  check(cudaDeviceSynchronize(), "activity sync");

  const std::vector<double> h_row_min = copy_to_host(row_min, 3);
  const std::vector<double> h_row_max = copy_to_host(row_max, 3);
  const std::vector<double> h_row_min_fin = copy_to_host(row_min_fin, 3);
  const std::vector<double> h_row_max_fin = copy_to_host(row_max_fin, 3);
  const std::vector<std::int32_t> h_neg = copy_to_host(row_min_neg_inf_count, 3);
  const std::vector<std::int32_t> h_pos = copy_to_host(row_max_pos_inf_count, 3);
  assert(h_row_min[0] == 0.0);
  assert(h_row_max[0] == 0.0);
  assert(h_row_min[1] == 8.0);
  assert(std::isinf(h_row_max[1]) && h_row_max[1] > 0.0);
  assert(std::isinf(h_row_min[2]) && h_row_min[2] < 0.0);
  assert(h_row_max[2] == 45.0);
  assert(h_row_min_fin[2] == -5.0);
  assert(h_row_max_fin[2] == 45.0);
  assert(h_neg[2] == 1);
  assert(h_pos[2] == 0);

  NnzOffset* at_row_ptr = copy_to_device<NnzOffset>({0, 1, 2, 3});
  std::int32_t* at_col_val = copy_to_device<std::int32_t>({2, 2, 1});
  double* at_nz_val = copy_to_device<double>({5.0, -6.0, 4.0});
  DeviceCsrMatrix transpose{3, 3, 3, at_row_ptr, at_col_val, at_nz_val};
  double* col_max_abs = nullptr;
  check(cudaMalloc(&col_max_abs, sizeof(double) * 3), "cudaMalloc col_max_abs");
  gpu_presolver::presolve::compute_col_max_abs(col_max_abs, transpose);
  check(cudaDeviceSynchronize(), "col max sync");
  const std::vector<double> h_col_max_abs = copy_to_host(col_max_abs, 3);
  assert((h_col_max_abs == std::vector<double>{5.0, 6.0, 4.0}));

  cudaFree(row_ptr);
  cudaFree(col_val);
  cudaFree(nz_val);
  cudaFree(row_nnz);
  cudaFree(singleton_column);
  cudaFree(singleton_val);
  cudaFree(l);
  cudaFree(u);
  cudaFree(row_min);
  cudaFree(row_max);
  cudaFree(row_min_fin);
  cudaFree(row_max_fin);
  cudaFree(row_min_neg_inf_count);
  cudaFree(row_max_pos_inf_count);
  cudaFree(at_row_ptr);
  cudaFree(at_col_val);
  cudaFree(at_nz_val);
  cudaFree(col_max_abs);

  test_adaptive_activity_mixed_lengths();
  test_hash_values_match_cpu();
  test_incremental_live_structure_matches_full_scan();

  std::cout << "test_gpu_presolve_kernels passed\n";
  return 0;
}
