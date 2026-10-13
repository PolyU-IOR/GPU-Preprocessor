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

void ordinary_case(int groups, int seed, int scenario, bool tape) {
  const double inf=std::numeric_limits<double>::infinity();
  const int n=5*groups+2;
  std::vector<int> perm(n);std::iota(perm.begin(),perm.end(),0);
  std::mt19937 rng(seed);std::shuffle(perm.begin(),perm.end(),rng);
  std::vector<NnzOffset> rp{0};std::vector<int> ci;std::vector<double> av,al,au;
  std::vector<double> lo(n,0),up(n,5),c(n,0);
  auto row=[&](std::vector<std::pair<int,double>> entries,double l,double u){
    std::sort(entries.begin(),entries.end());
    for(auto e:entries){ci.push_back(e.first);av.push_back(e.second);}
    rp.push_back(ci.size());al.push_back(l);au.push_back(u);
  };
  for(int g=0;g<groups;++g){
    const int a=perm[5*g],b=perm[5*g+1],d=perm[5*g+2],e=perm[5*g+3],f=perm[5*g+4];
    lo[a]=up[a]=1;c[a]=(g%2?1e16:-1e16);
    lo[b]=1;up[b]=5;c[b]=(g%2?1:1e16);
    c[e]=-1;c[f]=0;lo[f]=up[f]=0;
    // d normally belongs only to the empty-column prefix. Scenario 2 adds
    // a second finite dual contributor to force the safe prefix boundary.
    lo[d]=1;c[d]=1;
    std::vector<std::pair<int,double>> entries{{a,1},{b,1},{e,1}};
    double lhs=-inf,rhs=4;
    if(scenario==1 || scenario==4){lo[b]=-inf;c[b]=0;}
    if(scenario==2){entries.push_back({d,0.5});rhs=4.5;}
    if(scenario==8){entries.push_back({d,1});rhs=5;} // certified exact integer overlap
    if(scenario==3)rhs=-10;
    if(scenario==4)up[e]=inf; // late empty-column unboundedness after infinite dual fix
    if(scenario==5)lo[b]=-inf; // dual unboundedness after successful prefix
    if(scenario==7)entries.back().second=0; // stored zero remains structural
    if(scenario==9){entries.pop_back();rhs=0;} // late empty-row infeasibility
    if(scenario==10){lo[b]=-inf;c[b]=0;entries.push_back({d,1});rhs=5;} // finite+infinite fixes in one row
    if(scenario==11){lo[b]=0;up[b]=inf;c[b]=0;c[e]=1;lhs=1;rhs=inf;} // positive infinite fix
    if(scenario==12)entries.back().second=1e-12;
    row(entries,lhs,rhs);
  }
  if(scenario!=6)row({{perm[n-2],1},{perm[n-1],1}},1,1);
  else {lo[perm[n-2]]=up[perm[n-2]]=0;lo[perm[n-1]]=up[perm[n-1]]=0;}
  const int m=al.size();
  std::vector<NnzOffset> tp(n+1);for(int j:ci)++tp[j+1];for(int j=0;j<n;++j)tp[j+1]+=tp[j];
  auto cursor=tp;std::vector<int> ti(ci.size());std::vector<double> tv(av.size());
  for(int i=0;i<m;++i)for(auto k=rp[i];k<rp[i+1];++k){auto q=cursor[ci[k]]++;ti[q]=i;tv[q]=av[k];}
  LPInfoGpu lp;lp.A={m,n,NnzOffset(av.size()),upload(rp),upload(ci),upload(av)};
  lp.AT={n,m,NnzOffset(av.size()),upload(tp),upload(ti),upload(tv)};
  lp.c=upload(c);lp.l=upload(lo);lp.u=upload(up);lp.AL=upload(al);lp.AU=upload(au);lp.obj_constant=1e16;
  PresolveParams p;p.enable_startup=false;p.max_iters=1;
  p.record_postsolve_tape_cpu=tape;p.enable_structure_specific_rules=false;
  p.enable_infeasible_and_redundant_rows=false;p.enable_bound_strengthening=false;
  p.enable_parallel_rows=false;p.enable_parallel_columns=false;
  p.enable_singleton_columns=false;
  p.enable_doubleton_equations=false;p.enable_dual_fix=true;p.enable_absolute_value_epigraph_reduction=false;
  runtime::options().reuse_cleanup_workspace = false;
  auto a=run_gpu_presolve_with_reduced_lp(lp,p);auto expected=snapshot(a);
  for (bool mode : {true}) {
    runtime::options().reuse_cleanup_workspace = mode;
    auto b=run_gpu_presolve_with_reduced_lp(lp,p);auto actual=snapshot(b);
    if(expected!=actual){
      std::cerr<<"ordinary mismatch groups="<<groups<<" seed="<<seed<<" scenario="<<scenario<<" tape="<<tape<<" mode="<<mode<<" bytes="<<expected.size()<<":"<<actual.size()<<'\n';
      for(size_t k=0;k<std::min(expected.size(),actual.size());++k)if(expected[k]!=actual[k]){std::cerr<<"first byte="<<k<<'\n';break;}
      std::abort();
    }
    free_gpu_presolve_reduced_lp(b);
  }
  free_gpu_presolve_reduced_lp(a);
  for(auto* matrix:{&lp.A,&lp.AT}){cudaFree(matrix->rowPtr);cudaFree(matrix->colVal);cudaFree(matrix->nzVal);}
  cudaFree(lp.c);cudaFree(lp.l);cudaFree(lp.u);cudaFree(lp.AL);cudaFree(lp.AU);
}
}
int main(){
  runtime::options().device_recovery = true;
  int cases=0;
  for(int groups:{1,7,128,1024})for(int seed:{7,42})for(int scenario=0;scenario<13;++scenario)
    for(bool tape:{false,true}){ordinary_case(groups,seed,scenario,tape);++cases;}
  std::cout<<"PASS "<<cases<<" cleanup workspace on/off comparisons\n";
}
