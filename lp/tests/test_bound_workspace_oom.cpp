// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/rules/rule_bound_strengthening.hpp"
#include "gpu_preprocessor/lp/presolve/cleanup_workspace.hpp"
#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_postsolve.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>
#include <stdexcept>
namespace gp=gpu_preprocessor::lp::presolve;
namespace {bool inject=false;int matches=0,failures=0;constexpr int n=2048;}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
extern "C" cudaError_t __real_cudaMallocAsync(void**,std::size_t,cudaStream_t);
extern "C" cudaError_t __wrap_cudaMallocAsync(void** p,std::size_t bytes,cudaStream_t stream){
 if(inject && gp::bound_strengthening_workspace_in_use() && bytes==n*sizeof(double) && ++matches==2){
  inject=false;++failures;std::size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
  auto e=__real_cudaMallocAsync(p,total+1024*1024,stream);assert(e==cudaErrorMemoryAllocation);return e;
 }
 return __real_cudaMallocAsync(p,bytes,stream);
}
struct Memory {
 std::vector<void*> pointers;
 ~Memory(){for(auto p:pointers)cudaFree(p);}
 template<class T>T* put(std::vector<T> h){T* p=nullptr;check(cudaMalloc(&p,h.size()*sizeof(T)));pointers.push_back(p);check(cudaMemcpy(p,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
};
std::vector<double> read(const double* p,int count){std::vector<double> h(count);check(cudaMemcpy(h.data(),p,count*sizeof(double),cudaMemcpyDeviceToHost));return h;}
std::vector<double> run(bool fail){
 Memory mem;std::vector<int> ids(n),zero(n,0);std::vector<NnzOffset> ptr(n+1);
 for(int j=0;j<n;++j){ids[j]=j;ptr[j]=j;}ptr[n]=n;
 gp::LPInfoGpu lp;lp.A={1,n,n,mem.put<NnzOffset>({0,n}),mem.put(ids),mem.put(std::vector<double>(n,1))};
 lp.AT={n,1,n,mem.put(ptr),mem.put(zero),mem.put(std::vector<double>(n,1))};
 // Nonzero costs keep all columns eligible for bound strengthening.
 std::vector<double> cost(n,1);cost[0]=-1;
 lp.c=mem.put(cost);lp.l=mem.put(std::vector<double>(n,0));lp.u=mem.put(std::vector<double>(n,INFINITY));
 lp.AL=mem.put<double>({-INFINITY});lp.AU=mem.put<double>({10});
 gp::PresolvePlanGpu plan;plan.tape.resident=true;
 plan.keep_row_mask=mem.put<std::uint8_t>({1});plan.keep_col_mask=mem.put(std::vector<std::uint8_t>(n,1));
 plan.new_c=mem.put(cost);plan.new_l=mem.put(std::vector<double>(n,0));plan.new_u=mem.put(std::vector<double>(n,INFINITY));
 plan.new_AL=mem.put<double>({-INFINITY});plan.new_AU=mem.put<double>({10});
 gp::PresolveStatsGpu stats;stats.row_nnz=mem.put<int>({n});gp::PresolveParams params;
 gp::CleanupWorkspaceScope cache;
 gp::cleanup_scratch::active_pool->reserve(0,4096);
 inject=fail;matches=failures=0;
 gp::apply_rule_bound_strengthening(plan,lp,stats,params);
 assert(!gp::bound_strengthening_workspace_in_use() && !gp::bound_strengthening_workspace_prepared());
 assert(failures==int(fail));check(cudaPeekAtLastError());
 auto upper=read(plan.new_u,n);for(auto x:upper)assert(x>=10 && x<10+1e-5);
 gp::PresolveRecordGpu record;record.m0=record.m1=1;record.n0=record.n1=n;
 gp::initialize_device_recovery(record);gp::update_device_recovery(record,plan,lp,1,n,0,nullptr,nullptr);
 std::vector<double> x(n,0),z(n,0);x[0]=10;z[0]=-1;
 auto point=gp::postsolve_gpu(mem.put(x),mem.put<double>({0}),mem.put(z),record,&lp);
 auto out=read(point.x_org,n),y=read(point.y_org,1),zz=read(point.z_org,n);
 assert(y[0]==-1 && zz[0]==0);for(int j=1;j<n;++j)assert(zz[j]==1);
 out.insert(out.end(),y.begin(),y.end());out.insert(out.end(),zz.begin(),zz.end());
 cudaFree(point.x_org);cudaFree(point.y_org);cudaFree(point.z_org);
 gp::release_bound_strengthening_workspace();return out;
}
int main(){auto before=run(false),after=run(true);assert(before.size()==after.size());assert(!std::memcmp(before.data(),after.data(),before.size()*sizeof(double)));std::puts("Bound active-workspace OOM retry and exact recovery PASS");}
