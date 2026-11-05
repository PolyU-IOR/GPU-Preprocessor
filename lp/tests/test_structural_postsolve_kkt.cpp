// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "../tools/postsolve_replay/kkt.hpp"
#include "../tools/postsolve_replay/snapshot.hpp"
#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_postsolve.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_presolve.hpp"
#include "gpu_preprocessor/lp/presolve/structural_dual.hpp"
#include <unistd.h>
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
  PresolvePlanGpu plan(const LPInfoGpu& lp,bool resident) {
    PresolvePlanGpu p; p.tape.resident=resident;
    p.keep_row_mask=put(std::vector<std::uint8_t>(lp.A.rows,1));
    p.keep_col_mask=put(std::vector<std::uint8_t>(lp.A.cols,1));
    p.new_c=put(get(lp.c,lp.A.cols)); p.new_l=put(get(lp.l,lp.A.cols)); p.new_u=put(get(lp.u,lp.A.cols));
    p.new_AL=put(get(lp.AL,lp.A.rows)); p.new_AU=put(get(lp.AU,lp.A.rows)); return p;
  }
  PresolveRecordGpu record(PresolvePlanGpu& p,const LPInfoGpu& lp) {
    PresolveRecordGpu r; r.m0=lp.A.rows; r.n0=lp.A.cols; r.m1=r.m0; r.n1=r.n0;
    const auto rows=get(p.keep_row_mask,r.m0),cols=get(p.keep_col_mask,r.n0);
    std::vector<int> ri(r.m0,-1),ci(r.n0,-1);
    for(int i=0;i<r.m0;++i) if(rows[i]) { ri[i]=r.row_red2org.size(); r.row_red2org.push_back(i); }
    for(int j=0;j<r.n0;++j) if(cols[j]) { ci[j]=r.col_red2org.size(); r.col_red2org.push_back(j); }
    if(p.tape.resident) {
      initialize_device_recovery(r);
      update_device_recovery(r,p,lp,r.row_red2org.size(),r.col_red2org.size(),0,put(ri),put(ci));
    } else r.tape=std::move(p.tape);
    r.m1=r.row_red2org.size(); r.n1=r.col_red2org.size(); return r;
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
struct Case {
  Memory mem;
  LPInfoGpu original;
  GpuPresolveSummary summary;
  std::vector<int> rows,cols;
  std::vector<double> x,y;
  ~Case(){ free_gpu_presolve_reduced_lp(summary); }
  void reduce(const PresolveParams& p) {
    summary=run_gpu_presolve_with_reduced_lp(original,p);
    assert(!summary.has_infeasible && !summary.has_unbounded);
    assert(bool(summary.record.device)==device_recovery_enabled());
    if(summary.record.structural_dual_recoveries.size()!=1) {
      std::fprintf(stderr,"Unexpected checkpoints: %zu; reduced %d x %d\n",
        summary.record.structural_dual_recoveries.size(),summary.reduced_rows,summary.reduced_cols);
    }
    assert(summary.record.structural_dual_recoveries.size()==1);
    if(summary.record.device) assert(summary.record.device->host_staged_bytes==0);
    export_recovery_record(summary.record);
    rows=summary.record.row_red2org;cols=summary.record.col_red2org;
    x.resize(cols.size()); y.resize(rows.size());
  }
  void set_x(int old,double value) {
    for(int j=0;j<int(cols.size());++j) if(cols[j]==old) x[j]=value;
  }
  void set_y(int old,double value) {
    for(int i=0;i<int(rows.size());++i) if(rows[i]==old) y[i]=value;
  }
  double coefficient(int oldrow,int oldcol) {
    auto& a=summary.reduced_lp.A;
    auto rp=mem.get(a.rowPtr,a.rows+1);auto ci=mem.get(a.colVal,a.nnz);
    auto av=mem.get(a.nzVal,a.nnz);
    for(int i=0;i<a.rows;++i)if(rows[i]==oldrow)
      for(auto q=rp[i];q<rp[i+1];++q)if(cols[ci[q]]==oldcol)return av[q];
    return 0;
  }
  GpuPostsolveResult verify(bool snapshot=false) {
    auto& lp=summary.reduced_lp;
    auto z=mem.get(lp.c,lp.A.cols);
    auto rp=mem.get(lp.A.rowPtr,lp.A.rows+1);auto ci=mem.get(lp.A.colVal,lp.A.nnz);
    auto av=mem.get(lp.A.nzVal,lp.A.nnz);
    for(int i=0;i<lp.A.rows;++i)for(auto q=rp[i];q<rp[i+1];++q)z[ci[q]]-=av[q]*y[i];
    auto dx=mem.put(x),dy=mem.put(y),dz=mem.put(z);
    valid(lp,dx,dy,dz);
    auto post=postsolve_gpu(dx,dy,dz,summary.record,static_cast<const LPInfoGpu*>(nullptr));
    mem.pointers.insert(mem.pointers.end(),{post.x_org,post.y_org,post.z_org});
    valid(original,post.x_org,post.y_org,post.z_org);
    // Passing the original model must preserve the exact local lift.
    auto projected=postsolve_gpu(dx,dy,dz,summary.record,&original);
    mem.pointers.insert(mem.pointers.end(),{projected.x_org,projected.y_org,projected.z_org});
    assert(mem.get(projected.z_org,original.A.cols)==mem.get(post.z_org,original.A.cols));
    valid(original,projected.x_org,projected.y_org,projected.z_org);
    if(snapshot) {
      char path[]="/tmp/gpup-structural-XXXXXX";
      assert(mkdtemp(path));std::string file=std::string(path)+"/case.snapshot";
      folding::FoldingPipelineSummary pipe;
      pipe.presolve_enabled=true;pipe.original_rows=original.A.rows;pipe.original_cols=original.A.cols;
      pipe.presolve=std::move(summary);
      diagnostics::save_snapshot(file,dx,dy,dz,pipe,original,0);
      summary=std::move(pipe.presolve);
      auto loaded=diagnostics::load_snapshot(file);
      assert(loaded->format_version==7 && loaded->pipeline.presolve.record.structural_dual_recoveries.size()==1);
      auto replay=postsolve_gpu(loaded->x,loaded->y,loaded->z,loaded->pipeline.presolve.record,
                              static_cast<const LPInfoGpu*>(nullptr));
      mem.pointers.insert(mem.pointers.end(),{replay.x_org,replay.y_org,replay.z_org});
      valid(original,replay.x_org,replay.y_org,replay.z_org);
      assert(mem.get(replay.x_org,original.A.cols)==mem.get(post.x_org,original.A.cols));
      assert(mem.get(replay.y_org,original.A.rows)==mem.get(post.y_org,original.A.rows));
      assert(mem.get(replay.z_org,original.A.cols)==mem.get(post.z_org,original.A.cols));
      unlink(file.c_str());rmdir(path);
    }
    return post;
  }
};
void covering(bool resident,bool chain) {
  std::fprintf(stderr,"covering resident=%d\n",resident);
  Case c;
  c.original=c.mem.model({{1,0,1},{0,1,1}},{1,1,3},{0,0,0},{1,1,1},{1,1},{inf,inf});
  auto p=isolated(resident);p.enable_column_domination=true;p.enable_singleton_rows=chain;c.reduce(p);
  assert(!c.summary.record.has_primal_only_column_domination_reduction);
  c.set_x(0,1);c.set_x(1,1);c.set_y(0,2);c.set_y(1,4);
  auto post=c.verify(resident);
  assert((c.mem.get(post.y_org,2)==std::vector<double>{1,1}));
  assert((c.mem.get(post.z_org,3)==std::vector<double>{0,0,1}));
}
void covering_approximate(bool resident) {
  Case c;
  c.original=c.mem.model({{1,0,1},{0,1,1}},{1,1,3},{0,0,0},{1,1,1},{1,1},{inf,inf});
  auto p=isolated(resident);p.enable_column_domination=true;c.reduce(p);
  auto post=postsolve_gpu(c.mem.put<double>({1,1}),c.mem.put<double>({2,4}),
      c.mem.put<double>({-1+1e-8,-3-1e-8}),c.summary.record,&c.original);
  c.mem.pointers.insert(c.mem.pointers.end(),{post.x_org,post.y_org,post.z_org});
  const auto z=c.mem.get(post.z_org,3);
  assert(std::abs(z[0]-1e-8)<1e-15 && std::abs(z[1]+1e-8)<1e-15 && z[2]==1);
}
void pairs(bool resident,bool epigraph,int active,double edge_scale,bool cycle) {
  std::fprintf(stderr,"pairs resident=%d epi=%d active=%d scale=%g cycle=%d\n",resident,epigraph,active,edge_scale,cycle);
  Case c;const int k=3,n=epigraph?7:6;
  std::vector<std::vector<double>> a;
  std::vector<double> lo,hi,cost(n,0),lower(n,0),upper(n,10);
  if(epigraph){
    cost[6]=1;upper[6]=inf;
    for(int i=0;i<k;++i){std::vector<double> r(n);r[i]=r[i+k]=-1;r[6]=1;a.push_back(r);lo.push_back(0);hi.push_back(inf);}
  }else for(int i=0;i<k;++i){cost[i]=cost[i+k]=i+1;lower[i]=2+2*i;lower[i+k]=1+2*i;upper[i]=lower[i]+10;upper[i+k]=lower[i+k]+10;}
  auto edge=[&](int i,int j,double scale){std::vector<double> r(n);r[i]=r[j+k]=scale;r[j]=r[i+k]=-scale;a.push_back(r);lo.push_back(0);hi.push_back(0);};
  edge(0,1,edge_scale);edge(1,2,-2*edge_scale);if(cycle)edge(2,0,3*edge_scale);
  const double xp=active<0?0:(active==2?0:1),xn=active<0?1:0;
  if(active>0)upper[1]=lower[1]+xp;
  if(active<0)upper[4]=lower[4]+xn;
  std::vector<double> force(n);for(int i=0;i<k;++i){force[i]=1;force[i+k]=-1;}
  int row=a.size();a.push_back(force);lo.push_back(k*(xp-xn+(epigraph?0:1)));hi.push_back(lo.back());
  c.original=c.mem.model(a,cost,lower,upper,lo,hi);
  auto p=isolated(resident);p.enable_piecewise_linear_reformulation_variable_pair=true;c.reduce(p);
  assert(!c.summary.record.has_primal_only_piecewise_linear_reformulation_variable_pair_reduction);
  c.set_x(0,xp+(epigraph?lower[0]:0));c.set_x(3,xn+(epigraph?lower[3]:0));if(epigraph)c.set_x(6,xp+xn);
  if(epigraph)c.set_y(0,1/c.coefficient(0,6));
  double coeff=c.coefficient(row,0);if(coeff==0)coeff=-c.coefficient(row,3);
  double base=epigraph?1:6;
  c.set_y(row,(active<0?-base-1:(active>0?base+1:base))/coeff);
  auto post=c.verify(resident && edge_scale==1 && cycle);
  auto x=c.mem.get(post.x_org,n);
  for(int i=0;i<k;++i){assert(x[i]==xp+lower[i]);assert(x[i+k]==xn+lower[i+k]);}
}
void split(bool resident,double e_lower,double e_upper,double eq_cost) {
  std::fprintf(stderr,"split resident=%d cost=%g\n",resident,eq_cost);
  Case c;
  c.original=c.mem.model({{0,1,2},{1,-1,0},{1,1,0}}, {2,0.5,2*eq_cost},
      {0,e_lower,-inf},{inf,e_upper,inf},{7,0,0},{7,inf,inf});
  auto p=isolated(resident);p.enable_piecewise_linear_reformulation_absolute_value=true;c.reduce(p);
  double e=eq_cost>2.5?e_upper:(eq_cost<-1.5?e_lower:0);
  c.set_x(0,std::max(e,0.0));c.set_x(1,std::max(-e,0.0));c.set_x(2,(7-e)/2);
  c.set_y(0,eq_cost);
  if(eq_cost>2.5)c.set_y(1,2.5-eq_cost);
  if(eq_cost<-1.5)c.set_y(1,-1.5-eq_cost);
  c.verify(resident);
}
void graph(bool resident,double rho) {
  std::fprintf(stderr,"graph resident=%d rho=%g\n",resident,rho);
  Case c;
  c.original=c.mem.model({{0,1,0,0,0},{rho,-rho,0,0,0},{rho,rho,0,0,0},{-2,0,0,0,1},
                         {0,0,0,1,0},{0,0,1,-1,0},{0,0,1,1,0},{0,0,-3,0,1}},
      {1,0,2,0,0},{0,-inf,0,-inf,5},{inf,inf,inf,inf,inf},
      {1,0,0,0,-2,0,0,0},{1,inf,inf,inf,-2,inf,inf,inf});
  auto p=isolated(resident);p.enable_piecewise_linear_reformulation_absolute_value=true;c.reduce(p);
  c.set_x(0,1);c.set_x(1,0);c.set_x(2,0);c.set_x(3,2);
  c.set_y(0,1);c.set_y(4,-2);
  auto post=c.verify(resident);assert(c.mem.get(post.x_org,5)[4]==std::max(5.0,std::max(2.0,6.0)));
}
void outer(bool resident,double source_cost) {
  std::fprintf(stderr,"outer resident=%d cost=%g\n",resident,source_cost);
  Case c;
  // Two free sources share a bound; a retained equality couples one source
  // to the residual. Four active local rows give the link a nonzero multiplier.
  std::vector<std::vector<double>> a(14,std::vector<double>(10));
  for(int i=0;i<2;++i){int f=1+i,r=3*i;a[r][0]=1;a[r][f]=-1;a[r+1][0]=a[r+1][f]=1;a[r+2][0]=-1;a[r+2][f]=1;}
  a[6][1]=1;a[6][3]=1;a[7][3]=-1;a[7][4]=1;a[8][3]=a[8][4]=1;
  a[9][4]=2;a[9][5]=-1;for(int i=0;i<4;++i){a[10+i][5]=-1;a[10+i][6+i]=1;}
  c.original=c.mem.model(a, {20,source_cost,-3,0.5,1,0.75,1,1,1,1},
      {0,-inf,-inf,-inf,0,0,0,0,0,0},{10,inf,inf,inf,inf,inf,inf,inf,inf,inf},
      {0,0,0,0,0,0,1,0,0,0,0,0,0,0},{inf,inf,inf,inf,inf,inf,1,inf,inf,0,inf,inf,inf,inf});
  auto p=isolated(resident);p.enable_piecewise_linear_reformulation_absolute_value=true;c.reduce(p);
  c.set_x(0,0);c.set_x(4,1);c.set_x(3,0);for(int i=0;i<4;++i)c.set_x(6+i,2);
  c.set_y(6,11);for(int i=0;i<4;++i)c.set_y(10+i,1);
  c.verify(resident);
}
int main(){
  setvbuf(stdout,nullptr,_IONBF,0);
  for(bool resident:{false,true}){
    assert(setenv("GPUPREPROCESSOR_DEVICE_RECOVERY",resident?"1":"0",1)==0);
    covering(resident,false);covering(resident,true);covering_approximate(resident);
    for(bool epi:{false,true})for(int active:{-1,0,1,2})for(double scale:{1.0,-2.0})for(bool cycle:{false,true})pairs(resident,epi,active,scale,cycle);
    split(resident,-2,3,0);split(resident,-2,3,4);split(resident,-2,3,-3);split(resident,-inf,inf,0);
    graph(resident,1);graph(resident,2);
    outer(resident,2);outer(resident,12);
  }
  std::puts("Joint structural KKT lifts and snapshot roundtrips passed");
}
