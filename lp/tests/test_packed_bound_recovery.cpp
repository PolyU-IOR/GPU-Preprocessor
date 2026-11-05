// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_postsolve.hpp"
#include "gpu_preprocessor/lp/presolve/packed_bound_record.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace gpu_preprocessor::lp::presolve;
namespace {
constexpr int n=4097,m=17;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Memory {
 std::vector<void*> pointers;
 ~Memory(){for(auto p:pointers)cudaFree(p);}
 template<class T>T* put(const std::vector<T>& v){T* p=nullptr;if(!v.empty()){check(cudaMalloc(&p,v.size()*sizeof(T)));pointers.push_back(p);check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}return p;}
};
std::vector<double> read(const double* p,int count){std::vector<double> v(count);if(count)check(cudaMemcpy(v.data(),p,count*sizeof(double),cudaMemcpyDeviceToHost));return v;}
void same(const std::vector<double>& a,const std::vector<double>& b){assert(a.size()==b.size());assert(!std::memcmp(a.data(),b.data(),a.size()*sizeof(double)));}
void run(int sides,bool reverse,bool nonfinite){
 Memory mem;std::vector<NnzOffset> ptr{0};std::vector<int> col;std::vector<double> val;
 for(int r=0;r<m;++r){for(int j=0;j<n;++j){const double a=j%19?((r+j)%2?-1.0:1.0)*(1+(j%3)):0;
  col.push_back(j);val.push_back(a);if(j%257==0){col.push_back(j);val.push_back(a);}}
  ptr.push_back(col.size());}
 LPInfoGpu lp;lp.A={m,n,NnzOffset(col.size()),mem.put(ptr),mem.put(col),mem.put(val)};
 std::vector<double> lower(n,0),upper(n,10),next_l(n,1),next_u(n,9),exact(n,1);
 std::vector<int> lr(n),ur(n);std::vector<unsigned char> live(n);
 for(int j=0;j<n;++j){lr[j]=(sides&1)&&j%13?j%m:INT_MAX;ur[j]=(sides&2)&&j%23?(j*7)%m:INT_MAX;live[j]=j%11!=0;}
 auto old_l=mem.put(lower),old_u=mem.put(upper),new_l=mem.put(next_l),new_u=mem.put(next_u),exact_l=mem.put(exact);
 auto dlr=mem.put(lr),dur=mem.put(ur);auto dlive=mem.put(live);
 std::vector<double> xv(n),yv(m),zv(n);
 for(int j=0;j<n;++j){xv[j]=j%7?j/13.0:-0.0;zv[j]=j%4==0?1e16:j%4==1?1:j%4==2?-1e16:-0.0;}
 if(nonfinite)zv[1]=std::numeric_limits<double>::infinity();
 for(int i=0;i<m;++i)yv[i]=i/11.0;
 auto x=mem.put(xv),y=mem.put(yv),z=mem.put(zv);std::vector<double> reference;
 for(const char* packed:{"0","1"}){
  setenv("GPUPREPROCESSOR_PACKED_BOUND_RECORDS",packed,1);
  PresolveRecordGpu record;record.m0=record.m1=m;record.n0=record.n1=n;initialize_device_recovery(record);
  if(reverse){std::vector<int> cm(n),rm(m);for(int j=0;j<n;++j)cm[j]=n-1-j;for(int r=0;r<m;++r)rm[r]=m-1-r;
    check(cudaMemcpy(record.device->cols.data,cm.data(),n*sizeof(int),cudaMemcpyHostToDevice));check(cudaMemcpy(record.device->rows.data,rm.data(),m*sizeof(int),cudaMemcpyHostToDevice));}
  PresolvePlanGpu plan;plan.tape.resident=true;
  append_bound_batch_recovery(plan.tape,lp.A,n,dlive,old_l,old_u,new_l,new_u,exact_l,new_u,dlr,dur);
  update_device_recovery(record,plan,lp,m,n,0,nullptr,nullptr);validate_device_recovery(record);
  if(packed[0]=='1')assert(record.device->records==1);
  for(bool host:{false,true}){
   auto retained=record.device;
   if(host){export_recovery_record(record);record.device.reset();}
   for(const char* parallel:{"0","1"}){
    setenv("GPUPREPROCESSOR_PARALLEL_BOUND",parallel,1);auto point=postsolve_gpu(x,y,z,record);
    auto out=read(point.x_org,n),yy=read(point.y_org,m),zz=read(point.z_org,n);out.insert(out.end(),yy.begin(),yy.end());out.insert(out.end(),zz.begin(),zz.end());
    if(reference.empty())reference=out;else same(reference,out);
    cudaFree(point.x_org);cudaFree(point.y_org);cudaFree(point.z_org);
   }
   record.device=retained;
  }
 }
}
}
int main(){
 for(int side=0;side<2;++side){int words[packed_bound_words]{};for(NnzOffset ref:{NnzOffset(-1),NnzOffset(0),NnzOffset(INT_MAX)}){set_packed_bound_offset(words,ref);assert(packed_bound_offset(words)==ref);}
  if constexpr(sizeof(NnzOffset)==8){const auto wide=static_cast<NnzOffset>((std::uint64_t(1)<<40)+0x80000001ULL);set_packed_bound_offset(words,wide);assert(packed_bound_offset(words)==wide);}}
 for(int sides=0;sides<4;++sides)for(bool reverse:{false,true})run(sides,reverse,false);
 run(3,false,true);
 unsetenv("GPUPREPROCESSOR_PACKED_BOUND_RECORDS");unsetenv("GPUPREPROCESSOR_PARALLEL_BOUND");
 std::puts("Packed/compact bound records, one-sided references, missing pivots, maps, host/device and parallel/serial bytes PASS");
}
