#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

using namespace gpu_presolver::presolve;
namespace {
bool measure = false;
std::size_t downloaded = 0, uploaded = 0;
void count_transfer(std::size_t bytes, cudaMemcpyKind kind) {
  if (!measure) return;
  if (kind == cudaMemcpyDeviceToHost) downloaded += bytes;
  if (kind == cudaMemcpyHostToDevice) uploaded += bytes;
}
void check(cudaError_t e) { assert(e == cudaSuccess); }
template<class T> T* device(const std::vector<T>& v) {
  T* p = nullptr;
  check(cudaMalloc(&p, sizeof(T)*v.size()));
  check(cudaMemcpy(p, v.data(), sizeof(T)*v.size(), cudaMemcpyHostToDevice));
  return p;
}
void begin_measure() { downloaded = uploaded = 0; measure = true; }
void end_measure(std::size_t expected_download) {
  measure = false;
  assert(uploaded == 0 && downloaded == expected_download);
}
int mapped(int i) { return (i+3)%12; }
PresolveRecordGpu record(bool resident) {
  PresolveRecordGpu r;
  r.n0 = r.n1 = 12; r.m0 = r.m1 = 1;
  r.col_red2org.resize(12);
  for (int i = 0; i < 12; ++i) r.col_red2org[i] = mapped(i);
  r.row_red2org = {0};
  if (resident) {
    initialize_device_recovery(r);
    check(cudaMemcpy(r.device->cols.data, r.col_red2org.data(), 12*sizeof(int), cudaMemcpyHostToDevice));
    r.col_red2org.clear(); r.row_red2org.clear();
  }
  return r;
}
void commit(PresolveRecordGpu& r, PresolvePlanGpu& p) {
  LPInfoGpu lp; lp.A.rows = 1; lp.A.cols = 12;
  update_device_recovery(r, p, lp, 1, 12, 0, nullptr, nullptr);
}
void compare_postsolve(const PresolveRecordGpu& resident, const PresolveRecordGpu& host) {
  std::vector<double> input(12);
  std::iota(input.begin(), input.end(), 1.0);
  auto x = device(input), y = device(std::vector<double>{0}), z = device(std::vector<double>(12,0));
  auto a = postsolve_gpu(x,y,z,resident), b = postsolve_gpu(x,y,z,host);
  for (auto pair : {std::pair{a.x_org,b.x_org}, std::pair{a.z_org,b.z_org}}) {
    double va[12],vb[12];
    check(cudaMemcpy(va,pair.first,sizeof(va),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(vb,pair.second,sizeof(vb),cudaMemcpyDeviceToHost));
    assert(std::memcmp(va,vb,sizeof(va)) == 0);
  }
  double ya,yb;
  check(cudaMemcpy(&ya,a.y_org,sizeof(ya),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(&yb,b.y_org,sizeof(yb),cudaMemcpyDeviceToHost));
  assert(ya == yb);
  for (double* p : {a.x_org,a.y_org,a.z_org,b.x_org,b.y_org,b.z_org,x,y,z}) cudaFree(p);
}
void test_pairs(bool shifted) {
  auto parent = device(std::vector<int>{0,0,2});
  auto plus = device(std::vector<double>{2,4,0}), minus = device(std::vector<double>{1,3,0});
  PresolvePlanGpu p; p.tape.resident = true;
  begin_measure();
  capture_variable_pair_recovery_device(p,parent,3,shifted?plus:nullptr,shifted?minus:nullptr);
  end_measure(0);
  assert(p.variable_pair_aggregation_recovery.plus_cols.empty());
  cudaFree(parent); cudaFree(plus); cudaFree(minus);
  auto r = record(true), oracle = record(false);
  begin_measure(); commit(r,p); end_measure(0);
  assert(!p.variable_pair_aggregation_device);
  assert(r.variable_pair_aggregation_recoveries[0].plus_cols.empty());
  VariablePairAggregationRecoveryStep expected;
  expected.plus_cols = {mapped(0),mapped(1),mapped(2)};
  expected.minus_cols = {mapped(3),mapped(4),mapped(5)};
  expected.root_plus_cols = {mapped(0),mapped(0),mapped(2)};
  expected.root_minus_cols = {mapped(3),mapped(3),mapped(5)};
  expected.plus_shift = shifted ? std::vector<double>{2,4,0} : std::vector<double>(3,0);
  expected.minus_shift = shifted ? std::vector<double>{1,3,0} : std::vector<double>(3,0);
  oracle.variable_pair_aggregation_recoveries.push_back(expected);
  oracle.primal_recovery_timeline.push_back({PrimalRecoveryKind::VariablePairAggregation,0,0});
  compare_postsolve(r,oracle);
  export_recovery_record(r);
  const auto& q = r.variable_pair_aggregation_recoveries[0];
  assert(q.plus_cols == expected.plus_cols && q.minus_cols == expected.minus_cols);
  assert(q.root_plus_cols == expected.root_plus_cols && q.root_minus_cols == expected.root_minus_cols);
  assert(q.plus_shift == expected.plus_shift && q.minus_shift == expected.minus_shift);
}
void test_absolute_value() {
  auto t=device(std::vector<int>{0,2}), e=device(std::vector<int>{1,3});
  auto rho=device(std::vector<double>{2,4});
  auto bound=device(std::vector<int>{4}), free=device(std::vector<int>{5});
  auto slack=device(std::vector<int>{6}), linked_t=device(std::vector<int>{0});
  auto factor=device(std::vector<double>{-0.5});
  auto row_block=device(std::vector<int>{1,0,0,0,1,1,1,0});
  auto row_slack=device(std::vector<int>{9,8,9,9,8,-1,10,8});
  auto row_factor=device(std::vector<double>{0.5,-2,1.5,99,3,99,99,4});
  auto removable=device(std::vector<std::uint8_t>{1,1,1,0,1,1,0,1});
  PresolvePlanGpu p; p.tape.resident=true;
  begin_measure();
  capture_absolute_value_splits_device(p,t,e,rho,2,"device-test");
  capture_absolute_value_outer_device(p,bound,free,1,slack,linked_t,factor,1);
  end_measure(0);
  begin_measure();
  capture_absolute_value_max_slacks_device(p,t,row_block,row_slack,row_factor,removable,8,12);
  end_measure(sizeof(int)*4); // One group count and three offsets; no payload.
  assert(p.absolute_value_epigraph_recovery.splits.empty());
  assert(p.absolute_value_epigraph_recovery.outer_pairs.empty());
  assert(p.absolute_value_epigraph_recovery.linked_slacks.empty());
  assert(p.absolute_value_epigraph_recovery.max_slacks.empty());
  for (int* ptr : {t,e,bound,free,slack,linked_t,row_block,row_slack}) cudaFree(ptr);
  for (double* ptr : {rho,factor,row_factor}) cudaFree(ptr);
  cudaFree(removable);
  auto r=record(true), oracle=record(false);
  begin_measure(); commit(r,p); end_measure(0);
  assert(!p.absolute_value_epigraph_device);
  AbsoluteValueEpigraphPrimalRecoveryStep expected;
  expected.pattern="device-test";
  expected.splits={{mapped(0),mapped(1),2},{mapped(2),mapped(3),4}};
  expected.outer_pairs={{mapped(4),mapped(5)}};
  expected.linked_slacks={{mapped(6),mapped(0),-0.5}};
  // First incident row orders groups 9,8; supports retain input row order.
  expected.max_slacks={{mapped(9),{mapped(2),mapped(0)},{0.5,1.5}},
                       {mapped(8),{mapped(0),mapped(2),mapped(0)},{-2,3,4}}};
  oracle.absolute_value_epigraph_recoveries.push_back(expected);
  oracle.primal_recovery_timeline.push_back({PrimalRecoveryKind::AbsoluteValueEpigraph,0,0});
  compare_postsolve(r,oracle);
  export_recovery_record(r);
  const auto& q=r.absolute_value_epigraph_recoveries[0];
  assert(q.pattern == expected.pattern && q.max_slacks.size()==2);
  for (int i=0;i<2;++i) {
    assert(q.splits[i].t_col==expected.splits[i].t_col && q.splits[i].e_col==expected.splits[i].e_col);
    assert(q.splits[i].rho==expected.splits[i].rho);
    assert(q.max_slacks[i].slack_col==expected.max_slacks[i].slack_col);
    assert(q.max_slacks[i].t_cols==expected.max_slacks[i].t_cols);
    assert(q.max_slacks[i].factors==expected.max_slacks[i].factors);
  }
}
} // namespace
extern "C" cudaError_t __real_cudaMemcpy(void*,const void*,std::size_t,cudaMemcpyKind);
extern "C" cudaError_t __wrap_cudaMemcpy(void* d,const void* s,std::size_t n,cudaMemcpyKind k) {
  count_transfer(n,k); return __real_cudaMemcpy(d,s,n,k);
}
extern "C" cudaError_t __real_cudaMemcpyAsync(void*,const void*,std::size_t,cudaMemcpyKind,cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* d,const void* s,std::size_t n,cudaMemcpyKind k,cudaStream_t stream) {
  count_transfer(n,k); return __real_cudaMemcpyAsync(d,s,n,k,stream);
}
int main() {
  test_pairs(false); test_pairs(true); test_absolute_value();
  std::cout << "PASS structural device payloads, stable grouping, mapping, postsolve and transfer counts\n";
}
