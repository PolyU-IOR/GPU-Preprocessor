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

void startup_case(int groups,int seed,int scenario,bool tape) {
 const double inf=std::numeric_limits<double>::infinity();
 const int width=scenario==8?1056:64, stride=width+8;
 const int n=stride*groups;
 std::vector<int> perm(n);std::iota(perm.begin(),perm.end(),0);
 std::mt19937 rng(seed);std::shuffle(perm.begin(),perm.end(),rng);
 std::vector<NnzOffset> rp{0};std::vector<int> ci;std::vector<double> av,al,au;
 std::vector<double> lo(n,0),up(n,10),c(n,1);
 auto row=[&](std::vector<std::pair<int,double>> entries,double l,double u){
  std::sort(entries.begin(),entries.end());for(auto e:entries){ci.push_back(e.first);av.push_back(e.second);}
  rp.push_back(ci.size());al.push_back(l);au.push_back(u);
 };
 for(int g=0;g<groups;++g){
  row({},scenario==1 && g==0?1:0,2);
  // Many singleton rows move the logical average row length across 32.
  for(int k=0;k<100;++k)row({{perm[stride*g+width+k%8],1}},2,scenario==2 && g==0?1:8);
  row({{perm[stride*g+width],1},{perm[stride*g+width+1],1}},-inf,scenario==3 && g==0?1:100);
  std::vector<std::pair<int,double>> entries;
  int len=scenario==6?33:scenario==9?8:width;
  for(int k=0;k<len;++k)entries.push_back({perm[stride*g+k],k%3==0?2.:1.});
  row(entries,40,60);row(entries,41,59);
  // A successful parallel group and a failing group can execute together.
  if(scenario==4 && g==groups-1)row(entries,70,80);
  if(scenario==5)row(entries,50,50);
  if(scenario==7){entries.back().second=0;row(entries,40,60);row(entries,42,58);}
 }
 const int m=al.size();
 std::vector<NnzOffset> tp(n+1);for(int j:ci)++tp[j+1];for(int j=0;j<n;++j)tp[j+1]+=tp[j];
 auto cursor=tp;std::vector<int> ti(ci.size());std::vector<double> tv(av.size());
 for(int i=0;i<m;++i)for(auto k=rp[i];k<rp[i+1];++k){auto q=cursor[ci[k]]++;ti[q]=i;tv[q]=av[k];}
 LPInfoGpu lp;lp.A={m,n,NnzOffset(av.size()),upload(rp),upload(ci),upload(av)};
 lp.AT={n,m,NnzOffset(av.size()),upload(tp),upload(ti),upload(tv)};
 lp.c=upload(c);lp.l=upload(lo);lp.u=upload(up);lp.AL=upload(al);lp.AU=upload(au);
 PresolveParams p;p.enable_startup=true;p.max_iters=2;p.record_postsolve_tape_cpu=tape;
 p.enable_structure_specific_rules=false;p.enable_bound_strengthening=false;
 p.enable_infeasible_and_fixed_variables=false;p.enable_empty_columns=false;
 p.enable_parallel_columns=false;p.enable_singleton_columns=false;
 p.enable_doubleton_equations=false;p.enable_dual_fix=false;
 if (groups == 1 && seed == 7 && scenario == 0 && tape) {
   auto startup_only = p; startup_only.max_iters = 1;
   // Isolate structural Startup rules by explicitly disabling Cleanup.
   startup_only.enable_empty_rows = startup_only.enable_singleton_rows = false;
   auto untouched = run_gpu_presolve_with_reduced_lp(lp, startup_only);
   assert(untouched.reduced_rows == m && untouched.reduced_cols == n);
   assert(untouched.reduced_nnz == NnzOffset(av.size()));
   export_recovery_record(untouched.record);
   assert(untouched.record.tape.types.empty());
   free_gpu_presolve_reduced_lp(untouched);
 }
 runtime::options().device_recovery = false;
 auto a=run_gpu_presolve_with_reduced_lp(lp,p);auto expected=snapshot(a);
 runtime::options().device_recovery = true;
 auto b=run_gpu_presolve_with_reduced_lp(lp,p);auto actual=snapshot(b);
 if(expected!=actual){std::cerr<<"startup mismatch groups="<<groups<<" seed="<<seed<<" scenario="<<scenario<<" tape="<<tape<<'\n';std::abort();}
 free_gpu_presolve_reduced_lp(a);free_gpu_presolve_reduced_lp(b);
 for(auto* a:{&lp.A,&lp.AT}){cudaFree(a->rowPtr);cudaFree(a->colVal);cudaFree(a->nzVal);}
 cudaFree(lp.c);cudaFree(lp.l);cudaFree(lp.u);cudaFree(lp.AL);cudaFree(lp.AU);
}
}
int main(){int cases=0;for(int groups:{1,7,33})for(int seed:{7,42})for(int scenario=0;scenario<10;++scenario)for(bool tape:{false,true}){startup_case(groups,seed,scenario,tape);++cases;}std::cout<<"PASS "<<cases<<" startup differential cases\n";}
