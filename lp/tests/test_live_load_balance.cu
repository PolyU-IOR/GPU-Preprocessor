// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/live_matrix.cuh"
#include "gpu_preprocessor/lp/presolve/gpu_presolve_kernels.hpp"
#include <thread>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace gpu_preprocessor::lp::presolve;
void check(cudaError_t e) { if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::abort();} }
template<int Width> __global__ void visit_rows(int n,const int* indices,
    const std::uint8_t* keep,int* positions,double* sums,bool masked) {
  const int lane=Width==32?threadIdx.x&31:threadIdx.x;
  const int row=Width==32?blockIdx.x*8+threadIdx.x/32:blockIdx.x;
  __shared__ LiveRowStorage<Width> storage;
  int count=0; double sum=0;
  for_each_live_entry<Width>(NnzOffset(row)*n,NnzOffset(row+1)*n,indices,
      masked?keep:nullptr,storage,[&](NnzOffset p) {
    positions[NnzOffset(row)*n+count*Width+lane]=p-row*n;
    // A changed lane assignment/order changes cancellation in these sums.
    sum+=(p%3==0?1e16:p%3==1?-1e16:1.0); ++count;
  });
  sums[row*Width+lane]=sum;
}
template<int Width> void run(int n,int pattern,bool masked) {
  constexpr int rows=8;
  std::vector<int> ids(rows*n),out(rows*n,-1);
  std::vector<std::uint8_t> mask(rows*n);
  for(int row=0;row<rows;++row)for(int j=0;j<n;++j){
    const int p=row*n+j;ids[p]=p;
    switch((pattern+row)%6) {
      case 0:mask[p]=1;break;
      case 1:mask[p]=0;break;
      case 2:mask[p]=j%3==0;break;
      case 3:mask[p]=(unsigned(j)*2654435761u%17)<7;break;
      case 4:mask[p]=j==n-1;break;
      case 5:mask[p]=j>n/3 && j<n-7 && j%5;break;
    }
  }
  int *di,*dp;std::uint8_t* dm;double* ds;
  check(cudaMalloc(&di,ids.size()*sizeof(int)));check(cudaMalloc(&dp,out.size()*sizeof(int)));
  check(cudaMalloc(&dm,mask.size()));check(cudaMalloc(&ds,rows*Width*sizeof(double)));
  check(cudaMemcpy(di,ids.data(),ids.size()*sizeof(int),cudaMemcpyHostToDevice));
  check(cudaMemcpy(dm,mask.data(),mask.size(),cudaMemcpyHostToDevice));
  check(cudaMemset(dp,0xff,out.size()*sizeof(int)));
  visit_rows<Width><<<Width==32?1:rows,256>>>(n,di,dm,dp,ds,masked);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  std::vector<double> sums(rows*Width);
  check(cudaMemcpy(out.data(),dp,out.size()*sizeof(int),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(sums.data(),ds,sums.size()*sizeof(double),cudaMemcpyDeviceToHost));
  for(int row=0;row<rows;++row){
    std::vector<double> expected(Width,0);int count=0;
    for(int j=0;j<n;++j)if(!masked || mask[row*n+j]){
      if(out[row*n+count]!=j)std::abort();
      const int p=row*n+j;
      expected[count%Width]+=(p%3==0?1e16:p%3==1?-1e16:1.0);++count;
    }
    for(int j=count;j<n;++j)if(out[row*n+j]!=-1)std::abort();
    if(std::memcmp(expected.data(),sums.data()+row*Width,Width*sizeof(double)))std::abort();
  }
  check(cudaFree(di));check(cudaFree(dp));check(cudaFree(dm));check(cudaFree(ds));
}
// Keep stream-ordered work outstanding when the worker's TLS cache is
// destroyed. No worker-side synchronization may hide an unordered free.
__global__ void hold_stream() {
  const auto start = clock64();
  while (clock64() - start < 5000000ULL) {}
}
__global__ void check_groups(const int* counts, int rows, int* error) {
  if (counts[0] != rows/2 || counts[1] != rows/2) atomicExch(error, 1);
}
void adaptive_workspace_lifetime() {
  int devices = 0; check(cudaGetDeviceCount(&devices));
  std::vector<int*> errors(devices);
  for (int device=0; device<devices; ++device) {
    check(cudaSetDevice(device));
    check(cudaMalloc(&errors[device],sizeof(int)));
    check(cudaMemset(errors[device],0,sizeof(int)));
  }
  std::thread worker([&] {
    MatrixAnalysisScope scope;
    // Two cache slots: populate both, grow both, then switch devices when
    // available and return to the first. Each release follows queued users.
    for (int visit=0; visit<(devices>1 ? 3 : 1); ++visit) {
      const int device = visit==1 ? 1 : 0;
      check(cudaSetDevice(device));
      for (int phase=0; phase<4; ++phase) {
        const int rows = phase<2 ? 2 : 64;
        std::vector<NnzOffset> ptr(1,0);
        for (int r=0;r<rows;++r) ptr.push_back(ptr.back()+(r%2 ? 1025 : 9));
        NnzOffset* dptr = nullptr;
        check(cudaMallocAsync(&dptr,ptr.size()*sizeof(NnzOffset),nullptr));
        check(cudaMemcpy(dptr,ptr.data(),ptr.size()*sizeof(NnzOffset),cudaMemcpyHostToDevice));
        hold_stream<<<1,1>>>();check(cudaGetLastError());
        invalidate_matrix_analysis();
        DeviceCsrMatrix matrix{rows,2048,ptr.back(),dptr,nullptr,nullptr};
        const auto& groups = prepare_adaptive_row_groups(matrix);
        check_groups<<<1,1>>>(groups.counts,rows,errors[device]);check(cudaGetLastError());
        check(cudaFreeAsync(dptr,nullptr));
      }
    }
  });
  worker.join();
  for (int device=0;device<devices;++device) {
    check(cudaSetDevice(device));check(cudaDeviceSynchronize());
    int error=0;check(cudaMemcpy(&error,errors[device],sizeof(int),cudaMemcpyDeviceToHost));
    if(error)std::abort();check(cudaFree(errors[device]));
  }
  check(cudaSetDevice(0));
  std::puts("PASS adaptive cache growth, device ownership and pending-work thread exit");
}
int main(){
  adaptive_workspace_lifetime();
  for(int n:{1,7,8,9,31,32,33,255,256,257,1023,1024,1025,4099,65539})
    for(bool masked:{false,true}){run<32>(n,0,masked);run<256>(n,0,masked);}
  std::puts("PASS stable masked lane assignment and exact accumulation (warp/block)");
}
