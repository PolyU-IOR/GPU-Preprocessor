// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "../tools/postsolve_replay/kkt.hpp"
#include "../tools/postsolve_replay/snapshot.hpp"
#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_postsolve.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_presolve.hpp"
#include "gpu_preprocessor/lp/presolve/structural_dual.hpp"
#include <unistd.h>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
using namespace gpu_preprocessor::lp;
using namespace gpu_preprocessor::lp::presolve;
const double inf=std::numeric_limits<double>::infinity();
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Memory {
  std::vector<void*> pointers;
  ~Memory() { for(void* p:pointers) cudaFree(p); }
  template<class T> T* put(const std::vector<T>& v) {
    if(v.empty()) return nullptr;
    T* p=nullptr; check(cudaMalloc(&p,sizeof(T)*v.size())); pointers.push_back(p);
    check(cudaMemcpy(p,v.data(),sizeof(T)*v.size(),cudaMemcpyHostToDevice)); return p;
  }
  template<class T> std::vector<T> get(const T* p,int n) {
    std::vector<T> v(n); if(n) check(cudaMemcpy(v.data(),p,sizeof(T)*n,cudaMemcpyDeviceToHost)); return v;
  }
  LPInfoGpu model(const std::vector<std::vector<double>>& a,const std::vector<double>& c,
                  const std::vector<double>& l,const std::vector<double>& u,
                  const std::vector<double>& lo,const std::vector<double>& hi) {
    const int m=a.size(),n=c.size(); LPInfoGpu lp;
    auto matrix=[&](bool trans) {
      std::vector<NnzOffset> ptr{0}; std::vector<int> ids; std::vector<double> vals;
      for(int i=0;i<(trans?n:m);++i) {
        for(int j=0;j<(trans?m:n);++j) {
          const double v=trans?a[j][i]:a[i][j]; if(v!=0) { ids.push_back(j); vals.push_back(v); }
        }
        ptr.push_back(ids.size());
      }
      return DeviceCsrMatrix{trans?n:m,trans?m:n,static_cast<NnzOffset>(ids.size()),put(ptr),put(ids),put(vals)};
    };
    lp.A=matrix(false); lp.AT=matrix(true); lp.c=put(c); lp.l=put(l); lp.u=put(u); lp.AL=put(lo); lp.AU=put(hi); return lp;
  }

};
void valid(const LPInfoGpu& lp,const double* x,const double* y,const double* z) {
  auto k=diagnostics::check_kkt(&lp,x,y,z);
  if(!k.passed(1e-12)) diagnostics::print_kkt("test_failure",k,1e-12);
  assert(k.passed(1e-12)); assert(k.primal_inf<1e-12 && k.stationarity_inf<1e-12 && k.complementarity_inf<1e-12);
}

PresolveParams isolated(bool resident) {
  PresolveParams p;
  p.max_iters=1; p.record_postsolve_tape_cpu=!resident;
  p.enable_row_sparsification=false;
  p.enable_column_domination=false;
  p.enable_piecewise_linear_reformulation_variable_pair=false;
  p.enable_piecewise_linear_reformulation_absolute_value=false;
  p.enable_two_row_variable_elimination=false;
  p.enable_zero_cost_product_variables=false;
  p.enable_infeasible_and_fixed_variables=false;
  p.enable_empty_rows=false; p.enable_empty_columns=false;
  p.enable_singleton_rows=false; p.enable_singleton_columns=false;
  p.enable_infeasible_and_redundant_rows=false;
  p.enable_bound_strengthening=false; p.enable_parallel_rows=false;
  p.enable_doubleton_equations=false; p.enable_dual_fix=false;
  p.enable_parallel_columns=false;
  p.piecewise_linear_reformulation_variable_pair_epigraph_min_pairs=1;
  p.piecewise_linear_reformulation_variable_pair_epigraph_probe_pairs=3;
  p.piecewise_linear_reformulation_variable_pair_shifted_antipodal_min_pairs=1;
  p.piecewise_linear_reformulation_variable_pair_shifted_antipodal_probe_pairs=3;
  p.column_domination_min_fixed_cols=1;
  p.column_domination_probe_rows=2;
  p.column_domination_probe_cols=3;
  p.column_domination_probe_nnz=4;
  return p;
}
int cases=0;
PresolveParams projection_params(bool resident) {
  auto p=isolated(resident);
  p.enable_two_row_variable_elimination=true;
  p.enable_zero_cost_product_variables=true;
  p.two_row_variable_elimination_min_candidates=1;
  p.two_row_variable_elimination_min_candidate_ratio=0;
  p.zero_cost_product_variables_min_candidates=1;
  p.zero_cost_product_variables_min_candidate_ratio=0;
  return p;
}
void verify(Memory& mem, const LPInfoGpu& original, const PresolveParams& params,
            const std::vector<double>& expected_x, const std::vector<double>& expected_y,
            int removed_cols, bool roundtrip=false) {
  auto summary=run_gpu_presolve_with_reduced_lp(original,params);
  assert(!summary.has_infeasible && !summary.has_unbounded);
  assert(bool(summary.record.device)==device_recovery_enabled());
  assert(summary.reduced_cols==original.A.cols-removed_cols);
  assert(!summary.record.has_primal_only_projected_auxiliary_reduction);
  if(summary.record.device) assert(summary.record.device->host_staged_bytes==0);
  export_recovery_record(summary.record);
  const auto& rows=summary.record.row_red2org;
  const auto& cols=summary.record.col_red2org;
  std::vector<double> x,y;
  for(int j:cols)x.push_back(expected_x[j]);
  for(int i:rows)y.push_back(expected_y[i]);
  const auto& lp=summary.reduced_lp;
  auto z=mem.get(lp.c,lp.A.cols);
  const auto rp=mem.get(lp.A.rowPtr,lp.A.rows+1);
  const auto ci=mem.get(lp.A.colVal,lp.A.nnz);
  const auto av=mem.get(lp.A.nzVal,lp.A.nnz);
  for(int i=0;i<lp.A.rows;++i)for(auto q=rp[i];q<rp[i+1];++q)z[ci[q]]-=av[q]*y[i];
  auto dx=mem.put(x),dy=mem.put(y),dz=mem.put(z);
  valid(lp,dx,dy,dz);
  auto post=postsolve_gpu(dx,dy,dz,summary.record,static_cast<const LPInfoGpu*>(nullptr));
  mem.pointers.insert(mem.pointers.end(),{post.x_org,post.y_org,post.z_org});
  valid(original,post.x_org,post.y_org,post.z_org);
  const auto actual_x=mem.get(post.x_org,original.A.cols);
  const auto actual_y=mem.get(post.y_org,original.A.rows);
  const auto actual_z=mem.get(post.z_org,original.A.cols);
  for(int j=0;j<original.A.cols;++j)assert(std::abs(actual_x[j]-expected_x[j])<1e-12);
  assert(actual_y==expected_y);
  for(int j=0;j<int(cols.size());++j)assert(actual_z[cols[j]]==z[j]);
  for(int j=0;j<original.A.cols;++j)
    if(std::find(cols.begin(),cols.end(),j)==cols.end())assert(actual_z[j]==0);
  auto projected=postsolve_gpu(dx,dy,dz,summary.record,&original);
  mem.pointers.insert(mem.pointers.end(),{projected.x_org,projected.y_org,projected.z_org});
  valid(original,projected.x_org,projected.y_org,projected.z_org);
  assert(mem.get(projected.x_org,original.A.cols)==actual_x);
  assert(mem.get(projected.y_org,original.A.rows)==actual_y);
  assert(mem.get(projected.z_org,original.A.cols)==actual_z);
  // Snapshot serialization currently supports resident records only.
  if(roundtrip && summary.record.device) {
    char path[]="/tmp/gpup-projection-XXXXXX";
    assert(mkdtemp(path));const std::string file=std::string(path)+"/case.snapshot";
    folding::FoldingPipelineSummary pipe;
    pipe.presolve_enabled=true;pipe.original_rows=original.A.rows;pipe.original_cols=original.A.cols;
    pipe.presolve=std::move(summary);
    diagnostics::save_snapshot(file,dx,dy,dz,pipe,original,0);
    summary=std::move(pipe.presolve);
    auto loaded=diagnostics::load_snapshot(file);
    assert(!loaded->pipeline.presolve.record.has_primal_only_projected_auxiliary_reduction);
    auto replay=postsolve_gpu(loaded->x,loaded->y,loaded->z,loaded->pipeline.presolve.record,
                             static_cast<const LPInfoGpu*>(nullptr));
    mem.pointers.insert(mem.pointers.end(),{replay.x_org,replay.y_org,replay.z_org});
    valid(original,replay.x_org,replay.y_org,replay.z_org);
    assert(mem.get(replay.x_org,original.A.cols)==actual_x);
    assert(mem.get(replay.y_org,original.A.rows)==actual_y);
    assert(mem.get(replay.z_org,original.A.cols)==actual_z);
    unlink(file.c_str());rmdir(path);
  }
  ++cases;
  free_gpu_presolve_reduced_lp(summary);
}
// Preserve nonzero retained multipliers, including either active box endpoint.
double bound_dual(double value) { return value==0?2:(value==1?-2:0); }
void envelope(bool resident,double r,double s,double scale) {
  Memory mem;
  auto lp=mem.model({{-scale,-scale,scale},{-scale,0,scale},{0,-scale,scale},{1,-2,0}},
      {3+bound_dual(r),-6+bound_dual(s),0},{0,0,0},{1,1,1},
      {scale>0?-scale:-inf,scale>0?-inf:0,scale>0?-inf:0,r-2*s},
      {scale>0?inf:-scale,scale>0?0:inf,scale>0?0:inf,r-2*s});
  lp.obj_constant=7;
  auto p=projection_params(resident);p.enable_two_row_variable_elimination=false;
  verify(mem,lp,p,{r,s,std::max(0.0,r+s-1)},{0,0,0,3},1,r==0.5&&s==0.5);
}
void two_row(bool resident,double r,double s,double scale,double lower,bool shared) {
  Memory mem;
  const double f=.25+.5*r-.25*s;
  std::vector<std::vector<double>> a{{-.5*scale,.25*scale,scale},{-.5*scale,.25*scale,scale},{1,-2,0}};
  std::vector<double> lo{scale>0?.25*scale:-inf,scale>0?-inf:2.25*scale,r-2*s};
  std::vector<double> hi{scale>0?inf:.25*scale,scale>0?2.25*scale:inf,r-2*s};
  std::vector<double> c{3+bound_dual(r),-6+bound_dual(s),0},l{0,0,lower},u{1,1,4};
  std::vector<double> x{r,s,std::max(lower,f)},y{0,0,3};
  if(shared) {
    for(auto& row:a)row.push_back(0);
    a.push_back({.5,-.25,0,1});a.push_back({.5,-.25,0,1});
    lo.insert(lo.end(),{-.25,-inf});hi.insert(hi.end(),{inf,1.75});
    c.push_back(0);l.push_back(-1);u.push_back(4);
    x.push_back(-f);y.insert(y.end(),{0,0});
  }
  auto lp=mem.model(a,c,l,u,lo,hi);
  auto p=projection_params(resident);p.enable_zero_cost_product_variables=false;
  verify(mem,lp,p,x,y,shared?2:1,r==0.5&&s==0.5);
}
void chain(bool resident) {
  Memory mem;
  auto lp=mem.model({{-1,-1,0,0,1,0},{-1,0,0,0,1,0},{0,-1,0,0,1,0},
                    {0,0,1,-1,2,-2},{0,0,-1,1,2,2}},
      {-1,-2,0,0,0,0},{0,0,0,0,0,0},{1,1,1,1,1,1},
      {-1,-inf,-inf,-inf,-inf},{inf,0,0,2,4});
  verify(mem,lp,projection_params(resident),{1,1,.5,0,1,.25},{0,0,0,0,0},2,true);
  // Disabling all recording must keep the legacy incomplete-recovery warning.
  auto p=projection_params(resident);p.record_postsolve_tape=false;
  auto summary=run_gpu_presolve_with_reduced_lp(lp,p);
  assert(summary.reduced_cols==4 && summary.record.has_primal_only_projected_auxiliary_reduction);
  free_gpu_presolve_reduced_lp(summary);
}
int main() {
  setvbuf(stdout,nullptr,_IONBF,0);
  for(bool resident:{false,true}) {
    assert(setenv("GPUPREPROCESSOR_DEVICE_RECOVERY",resident?"1":"0",1)==0);
    for(double scale:{1.0,-2.0,0.125})for(double r:{0.0,0.25,0.5,1.0})for(double s:{0.0,0.25,0.5,1.0}) {
      envelope(resident,r,s,scale);
      for(double lower:{-1.0,0.5})two_row(resident,r,s,scale,lower,false);
    }
    two_row(resident,.5,.5,-2,-1,true);
    chain(resident);
  }
  std::printf("Projected auxiliary KKT lifts passed: %d cases\n",cases);
}
