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
void run_case(bool startup,int scenario,bool tape,int disabled) {
  // Fixing x0/x5 exposes both a singleton row and an empty row. A separate
  // singleton exposes an empty column; x2/x3 keep a nonempty residual model.
  const int m=5,n=6;
  std::vector<NnzOffset> rp{0,2,4,4,5,7};
  std::vector<int> ci{0,1,2,3,4,0,5};
  std::vector<double> av{1,1,1,-1,1,1,1};
  std::vector<double> lo(n,0),up(n,10),c{0,2,0,0,1,0};
  lo[0]=up[0]=1;lo[5]=up[5]=1;
  std::vector<double> al{3,-2,0,1,2},au{3,4,0,2,2};
  if(scenario==1)al[2]=au[2]=1; // empty-row infeasibility after successful singleton stage
  if(scenario==2){lo[4]=-std::numeric_limits<double>::infinity();al[3]=lo[4];} // late unbounded empty column
  if(scenario==3)up[1]=1; // combined trivial row stage must discard its empty-row deletion
  if(scenario==4)av[4]=0.0; // a stored zero is still one structural entry
  if(scenario==6){lo[2]=up[2]=0;lo[3]=up[3]=0;} // tail can eliminate the entire LP
  if(scenario==5)av[4]=1e-12; // small coefficients must not become structural zeros
  std::vector<NnzOffset> tp(n+1);for(int j:ci)++tp[j+1];for(int j=0;j<n;++j)tp[j+1]+=tp[j];
  auto cursor=tp;std::vector<int> ti(ci.size());std::vector<double> tv(av.size());
  for(int i=0;i<m;++i)for(auto p=rp[i];p<rp[i+1];++p){auto q=cursor[ci[p]]++;ti[q]=i;tv[q]=av[p];}
  LPInfoGpu lp;lp.A={m,n,NnzOffset(av.size()),upload(rp),upload(ci),upload(av)};
  lp.AT={n,m,NnzOffset(av.size()),upload(tp),upload(ti),upload(tv)};
  lp.c=upload(c);lp.l=upload(lo);lp.u=upload(up);lp.AL=upload(al);lp.AU=upload(au);
  PresolveParams p;p.enable_startup=startup;p.max_iters=4;
  p.record_postsolve_tape_cpu=tape;p.enable_structure_specific_rules=false;
  p.enable_infeasible_and_redundant_rows=false;p.enable_bound_strengthening=false;
  p.enable_parallel_rows=false;p.enable_parallel_columns=false;
  p.enable_singleton_columns=false;
  p.enable_doubleton_equations=false;p.enable_dual_fix=false;p.enable_absolute_value_epigraph_reduction=false;
  p.enable_empty_rows=disabled!=1;p.enable_singleton_rows=disabled!=2;p.enable_empty_columns=disabled!=3;
  runtime::options().device_recovery = false;auto a=run_gpu_presolve_with_reduced_lp(lp,p);
  auto expected=snapshot(a);
  runtime::options().device_recovery = true;auto b=run_gpu_presolve_with_reduced_lp(lp,p);
  auto actual=snapshot(b);
  if(expected!=actual){std::cerr<<"mismatch startup="<<startup<<" scenario="<<scenario<<" tape="<<tape<<" disabled="<<disabled<<'\n';std::abort();}
  free_gpu_presolve_reduced_lp(a);free_gpu_presolve_reduced_lp(b);
  for(auto* matrix:{&lp.A,&lp.AT}){cudaFree(matrix->rowPtr);cudaFree(matrix->colVal);cudaFree(matrix->nzVal);}
  cudaFree(lp.c);cudaFree(lp.l);cudaFree(lp.u);cudaFree(lp.AL);cudaFree(lp.AU);
}
}
int main(){
  const auto saved = runtime::options().device_recovery;
  for(bool startup:{false,true})for(int scenario=0;scenario<7;++scenario)
    for(bool tape:{false,true})for(int disabled=0;disabled<4;++disabled)run_case(startup,scenario,tape,disabled);
  runtime::options().device_recovery = saved;
  std::cout<<"PASS 112 cleanup host/device differential cases (LP, status, bounds, objective, recovery ordering)\n";
}
