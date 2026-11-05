// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/compact_bound_record.hpp"
#include "gpu_preprocessor/lp/presolve/packed_bound_record.hpp"
#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_postsolve.hpp"
#include "../tools/postsolve_replay/snapshot.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace gpu_preprocessor::lp;
using namespace gpu_preprocessor::lp::presolve;
static void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
struct Memory {
  std::vector<void*> pointers;
  ~Memory() {for(void* p:pointers) cudaFree(p);}
  template<class T> T* put(std::vector<T> v) {
    T* p=nullptr;
    if(!v.empty()) {check(cudaMalloc(&p,v.size()*sizeof(T)));pointers.push_back(p);
      check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    return p;
  }
};
static std::vector<double> get(const double* p,int n) {
  std::vector<double> v(n);if(n) check(cudaMemcpy(v.data(),p,n*sizeof(double),cudaMemcpyDeviceToHost));return v;
}
static void release(GpuPostsolveResult& r) {cudaFree(r.x_org);cudaFree(r.y_org);cudaFree(r.z_org);r={};}
static void near(double a,long double b) {assert(std::abs(static_cast<long double>(a)-b)<=2e-12L*(1+std::abs(b)));}

// The unshared format exceeds int32 here, although A has only 70,000 entries.
// An exact primal/dual point also checks all-column joint lifting and signs.
static void capacity_and_snapshot(bool resident) {
  constexpr int n=70000;
  static_assert(std::int64_t(n)*(n+5)>std::numeric_limits<std::int32_t>::max());
  Memory mem;
  std::vector<int> ids(n);for(int j=0;j<n;++j) ids[j]=j;
  LPInfoGpu lp;
  lp.A={1,n,n,mem.put<NnzOffset>({0,n}),mem.put(ids),mem.put(std::vector<double>(n,1))};
  auto lower=mem.put(std::vector<double>(n,0)),upper=mem.put(std::vector<double>(n,1));
  auto owner=mem.put(std::vector<int>(n,0)),none=mem.put(std::vector<int>(n,INT_MAX));
  PresolvePlanGpu plan;plan.tape.resident=resident;
  append_bound_batch_recovery(plan.tape,lp.A,n,nullptr,lower,upper,upper,upper,upper,upper,owner,none);
  PresolveRecordGpu r;r.m0=r.m1=1;r.n0=r.n1=n;initialize_device_recovery(r);
  update_device_recovery(r,plan,lp,1,n,0,nullptr,nullptr);
  validate_device_recovery(r);
  assert(r.device->segments.size()==1);
  const auto& s=r.device->segments[0];
  const bool packed=std::getenv("GPUPREPROCESSOR_PACKED_BOUND_RECORDS")[0]=='1';
  if(packed)assert(s.record_count==1 && s.index_count==packed_bound_header+1+2*packed_bound_words+n+packed_bound_words*n && s.value_count==n);
  else assert(s.record_count==n+2 && s.index_count==(compact_bound_indices+1)*n+3 && s.value_count==n);
  auto x=upper,y=mem.put<double>({0}),z=upper;
  auto restored=postsolve_gpu(x,y,z,r);
  near(get(restored.y_org,1)[0],n);
  for(double v:get(restored.z_org,n)) near(v,1-n);
  assert(get(restored.x_org,n)==std::vector<double>(n,1));
  // c-A^T y-z=0, x=u, y>=0 and z<=0 give zero stationarity,
  // complementarity and primal-dual gap for sum(x)>=n, 0<=x<=1, c=1.
  near(n+get(restored.z_org,n)[0],1);
  export_recovery_record(r);
  auto retained=r.device;r.device.reset();
  auto host=postsolve_gpu(x,y,z,r);
  assert(get(host.y_org,1)==get(restored.y_org,1));
  assert(get(host.z_org,n)==get(restored.z_org,n));
  release(host);r.device=retained;
  diagnostics::PostsolveSnapshot snapshot;
  auto& p=snapshot.pipeline;p.original_rows=p.folded_rows=1;p.original_cols=p.folded_cols=n;
  p.presolve.original_rows=p.presolve.reduced_rows=1;
  p.presolve.original_cols=p.presolve.reduced_cols=n;p.presolve.record=std::move(r);
  p.presolve.reduced_lp.A.rows=1;p.presolve.reduced_lp.A.cols=n;
  snapshot.original.A.rows=1;snapshot.original.A.cols=n;
  snapshot.x=mem.put(std::vector<double>(n,1));snapshot.y=mem.put<double>({0});snapshot.z=mem.put(std::vector<double>(n,1));
  // Snapshot owns its reduced vectors.
  for(auto ptr:{snapshot.x,snapshot.y,snapshot.z})
    mem.pointers.erase(std::find(mem.pointers.begin(),mem.pointers.end(),ptr));
  char directory[]="shared-bound-snapshot-XXXXXX";assert(mkdtemp(directory));
  const std::string path=std::string(directory)+"/state.bin";
  diagnostics::save_snapshot(path,snapshot.x,snapshot.y,snapshot.z,p,snapshot.original,0);
  auto loaded=diagnostics::load_snapshot(path);assert(loaded->format_version==7);
  const auto roundtrip=folding::postsolve_and_unfold_to_host(loaded->x,loaded->y,loaded->z,loaded->pipeline,&loaded->original,0);
  assert(roundtrip.x==get(restored.x_org,n));assert(roundtrip.y==get(restored.y_org,1));assert(roundtrip.z==get(restored.z_org,n));
  release(restored);
}

// Two shared source rows, both signs, missing owners, zero/dead coefficients,
// numerical endpoints, adjacent batches and nonidentity old maps.
static void mixed_batches() {
  constexpr int n=6,m=3;Memory mem;
  std::vector<int> columns{0,1,2,3,4,5,0,1,2,3,4,5};
  std::vector<double> values{2,-3,1,4,0,7,-1,2,5,-2,3,0};
  LPInfoGpu lp;lp.A={m,n,12,mem.put<NnzOffset>({0,6,6,12}),mem.put(columns),mem.put(values)};
  std::vector<int> cm{5,4,3,2,1,0},rm{2,0,1};
  PresolveRecordGpu r;r.m0=r.m1=m;r.n0=r.n1=n;initialize_device_recovery(r);
  check(cudaMemcpy(r.device->cols.data,cm.data(),n*sizeof(int),cudaMemcpyHostToDevice));
  check(cudaMemcpy(r.device->rows.data,rm.data(),m*sizeof(int),cudaMemcpyHostToDevice));
  std::vector<double> incoming{4,-6,10,8,-9,14};
  std::vector<long double> expected_z(incoming.begin(),incoming.end()),expected_y(m,0);
  for(int batch=0;batch<2;++batch) {
    std::vector<double> old(n,0),upper(n,10),next(n,1),exact=next;
    exact[2]=1.00001;
    std::vector<int> lower{0,0,2,2,INT_MAX,0},up{2,0,2,0,2,INT_MAX};
    std::vector<unsigned char> live{1,1,1,1,1,0};
    PresolvePlanGpu plan;plan.tape.resident=true;
    append_bound_batch_recovery(plan.tape,lp.A,n,mem.put(live),mem.put(old),mem.put(upper),
        mem.put(next),mem.put(std::vector<double>(n,9)),mem.put(exact),mem.put(std::vector<double>(n,9)),mem.put(lower),mem.put(up));
    update_device_recovery(r,plan,lp,m,n,0,nullptr,nullptr);
    // Apply the mathematical affine lift to a frozen incoming triple on CPU.
    std::vector<long double> delta(m,0);
    for(int j=0;j<n;++j) if(live[j]) {
      int row=expected_z[j]>0?lower[j]:up[j];if(row==INT_MAX) continue;
      const int base=row==0?0:6;const double pivot=values[base+j];
      if(pivot!=0) delta[row]+=expected_z[j]/pivot;
    }
    for(int row:{0,2}) {
      expected_y[row]+=delta[row];const int base=row==0?0:6;
      for(int j=0;j<n;++j) if(live[j]) expected_z[j]-=values[base+j]*delta[row];
    }
  }
  validate_device_recovery(r);
  auto x=mem.put(std::vector<double>(n,1)),y=mem.put(std::vector<double>(m,0)),z=mem.put(incoming);
  auto out=postsolve_gpu(x,y,z,r);auto oz=get(out.z_org,n),oy=get(out.y_org,m);
  // Reverse chronological replay; the two batches are identical affine maps.
  for(int j=0;j<n;++j) near(oz[cm[j]],expected_z[j]);
  for(int i=0;i<m;++i) near(oy[rm[i]],expected_y[i]);
  export_recovery_record(r);r.device.reset();auto host=postsolve_gpu(x,y,z,r);
  assert(get(host.z_org,n)==oz && get(host.y_org,m)==oy);
  release(host);release(out);
}
int main() {
  // Wide coefficient references retain both words, including the low sign bit.
  int ids[compact_bound_indices]{};
  for(int side=0;side<2;++side) {
    for(NnzOffset ref:{NnzOffset(-1),NnzOffset(0),NnzOffset(INT_MAX)}) {
      set_compact_bound_reference(ids,side,ref);
      assert(compact_bound_reference(ids,side)==ref);
    }
    if constexpr(sizeof(NnzOffset)==8) {
      const auto ref=static_cast<NnzOffset>((std::uint64_t(1)<<40)+0x80000001ULL);
      set_compact_bound_reference(ids,side,ref);
      assert(compact_bound_reference(ids,side)==ref);
    }
  }
  for(const char* packed:{"0","1"}) {
    setenv("GPUPREPROCESSOR_PACKED_BOUND_RECORDS",packed,1);
    capacity_and_snapshot(true);capacity_and_snapshot(false);mixed_batches();
  }
  unsetenv("GPUPREPROCESSOR_PACKED_BOUND_RECORDS");
  check(cudaDeviceSynchronize());
  std::puts("Shared bound capacity, joint affine lift, host export, maps and snapshot roundtrip passed");
}
