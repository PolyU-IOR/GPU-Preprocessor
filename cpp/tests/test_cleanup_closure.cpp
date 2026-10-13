#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

namespace runtime = gpu_presolver::runtime;
using namespace gpu_presolver::presolve;
namespace {
void check(cudaError_t e) { assert(e == cudaSuccess); }
template<class T> T* upload(const std::vector<T>& v) {
  T* p=nullptr;
  if (!v.empty()) {check(cudaMalloc(&p,v.size()*sizeof(T)));check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
  return p;
}
template<class T> void scalar(std::vector<unsigned char>& out,const T& v) {
  const auto* p=reinterpret_cast<const unsigned char*>(&v);out.insert(out.end(),p,p+sizeof(T));
}
template<class T> void array(std::vector<unsigned char>& out,const std::vector<T>& v) {
  scalar(out,v.size());if(!v.empty()){const auto* p=reinterpret_cast<const unsigned char*>(v.data());out.insert(out.end(),p,p+v.size()*sizeof(T));}
}
template<class T> void device(std::vector<unsigned char>& out,const T* p,std::size_t n) {
  std::vector<T> v(n);if(n)check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));array(out,v);
}
std::vector<unsigned char> snapshot(GpuPresolveSummary& s) {
  std::vector<unsigned char> out;
  scalar(out,s.has_infeasible);scalar(out,s.has_unbounded);scalar(out,s.iterations);
  scalar(out,s.reduced_rows);scalar(out,s.reduced_cols);scalar(out,s.reduced_nnz);
  scalar(out,s.obj_constant_delta);
  const auto& lp=s.reduced_lp;
  for(const auto* a:{&lp.A,&lp.AT}) {
    scalar(out,a->rows);scalar(out,a->cols);scalar(out,a->nnz);
    device(out,a->rowPtr,a->rowPtr?a->rows+1:0);device(out,a->colVal,a->nnz);device(out,a->nzVal,a->nnz);
  }
  device(out,lp.c,lp.A.cols);device(out,lp.l,lp.A.cols);device(out,lp.u,lp.A.cols);
  device(out,lp.AL,lp.A.rows);device(out,lp.AU,lp.A.rows);scalar(out,lp.obj_constant);
  export_recovery_record(s.record);const auto& r=s.record;
  array(out,r.row_org2red);array(out,r.row_red2org);array(out,r.col_org2red);array(out,r.col_red2org);
  array(out,r.fixed_idx);array(out,r.fixed_val);array(out,r.removed_row_idx);array(out,r.removed_col_idx);
  scalar(out,r.obj_constant_old);scalar(out,r.obj_constant_new);
  const auto& t=r.tape;array(out,t.types);array(out,t.dual_modes);array(out,t.indices);array(out,t.vals);
  array(out,t.index_starts);array(out,t.value_starts);
  return out;
}

void stress(int groups, int seed, int scenario, bool tape) {
  const int n=4*groups+2;
  std::vector<int> perm(n);std::iota(perm.begin(),perm.end(),0);
  std::mt19937 rng(seed);std::shuffle(perm.begin(),perm.end(),rng);
  std::vector<NnzOffset> rp{0};std::vector<int> ci;
  std::vector<double> av,al,au,lo(n,0),up(n,5),c(n);
  const double costs[]={1e16,1,-1e16,-1,2,-2,0,3};
  for(int j=0;j<n;++j)c[j]=costs[rng()%8];
  auto row=[&](std::vector<int> cols,double l,double u){
    std::sort(cols.begin(),cols.end());
    for(int j:cols){ci.push_back(j);av.push_back(1);}
    rp.push_back(ci.size());al.push_back(l);au.push_back(u);
  };
  for(int g=0;g<groups;++g){
    const int a=perm[4*g],b=perm[4*g+1],d=perm[4*g+2],e=perm[4*g+3];
    row({a},1,1);row({a,b},2,scenario==4 ? 2+1e-10 : 2);row({b,d},2,2);
    double l=1,u=3;
    if(g==groups-1 && scenario==1)l=u=20; // late singleton infeasibility
    if(g==groups-1 && scenario==2){u=up[e]=std::numeric_limits<double>::infinity();c[e]=-1;}
    row({d,e},l,u);
  }
  if(scenario==3 && groups>1) row({perm[1],perm[5]},-100,100); // overlapping nonzero shifts: retain boundary
  row({perm[n-2],perm[n-1]},0,8); // retain a coupled residual model
  const int m=al.size();
  std::vector<NnzOffset> tp(n+1);for(int j:ci)++tp[j+1];for(int j=0;j<n;++j)tp[j+1]+=tp[j];
  auto cursor=tp;std::vector<int> ti(ci.size());std::vector<double> tv(av.size());
  for(int i=0;i<m;++i)for(auto k=rp[i];k<rp[i+1];++k){auto q=cursor[ci[k]]++;ti[q]=i;tv[q]=av[k];}
  LPInfoGpu lp;lp.A={m,n,NnzOffset(av.size()),upload(rp),upload(ci),upload(av)};
  lp.AT={n,m,NnzOffset(av.size()),upload(tp),upload(ti),upload(tv)};
  lp.c=upload(c);lp.l=upload(lo);lp.u=upload(up);lp.AL=upload(al);lp.AU=upload(au);
  lp.obj_constant=1e16;
  PresolveParams p;p.enable_startup=true;p.max_iters=4;
  p.record_postsolve_tape_cpu=tape;p.enable_structure_specific_rules=false;
  p.enable_infeasible_and_redundant_rows=false;p.enable_bound_strengthening=false;
  p.enable_parallel_rows=false;p.enable_parallel_columns=false;
  p.enable_singleton_columns=false;
  p.enable_doubleton_equations=false;p.enable_dual_fix=false;p.enable_absolute_value_epigraph_reduction=false;
  runtime::options().device_recovery = false;auto a=run_gpu_presolve_with_reduced_lp(lp,p);
  auto expected=snapshot(a);
  runtime::options().device_recovery = true;auto b=run_gpu_presolve_with_reduced_lp(lp,p);
  auto actual=snapshot(b);
  if(expected!=actual){std::cerr<<"stress mismatch groups="<<groups<<" seed="<<seed<<" scenario="<<scenario<<" tape="<<tape<<'\n';std::abort();}
  free_gpu_presolve_reduced_lp(a);free_gpu_presolve_reduced_lp(b);
  for(auto* matrix:{&lp.A,&lp.AT}){cudaFree(matrix->rowPtr);cudaFree(matrix->colVal);cudaFree(matrix->nzVal);}
  cudaFree(lp.c);cudaFree(lp.l);cudaFree(lp.u);cudaFree(lp.AL);cudaFree(lp.AU);
}
}
int main(){
  for(int groups:{7,1024})for(int seed:{7,42,91})for(int scenario=0;scenario<5;++scenario)
    for(bool tape:{false,true})stress(groups,seed,scenario,tape);
  std::cout<<"PASS 60 cleanup-closure host/device differential stress cases: permuted chains, cancellation, late terminal states\n";
}
