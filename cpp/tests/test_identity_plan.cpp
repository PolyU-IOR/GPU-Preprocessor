#include "gpu_presolver/presolve/identity_plan.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace gpu_presolver::presolve;
static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<class T> T* device(const std::vector<T>& v) {
  T* p = nullptr;
  if (!v.empty()) {
    check(cudaMalloc(&p, sizeof(T)*v.size()));
    check(cudaMemcpy(p, v.data(), sizeof(T)*v.size(), cudaMemcpyHostToDevice));
  }
  return p;
}
template<class T> void same_device(const T* p, const std::vector<T>& expected) {
  if (expected.empty()) return;
  std::vector<T> actual(expected.size());
  check(cudaMemcpy(actual.data(), p, sizeof(T)*actual.size(), cudaMemcpyDeviceToHost));
  assert(std::memcmp(actual.data(),expected.data(),sizeof(T)*actual.size())==0);
}

int main() {
  for (int m : {0, 1, 255, 256, 257, 4097}) for (int n : {0, 1, 257}) {
    PresolvePlanGpu p;
    std::vector<std::uint8_t> rows(m, 2), cols(n, 255);
    p.keep_row_mask=device(rows);p.keep_col_mask=device(cols);
    assert(detail::plan_keeps_all_indices(p,m,n));
    for (auto mask : {p.keep_row_mask,p.keep_col_mask}) {
      int size=mask==p.keep_row_mask ? m : n;
      if (!size) continue;
      for (int pos : {0,size-1}) {
        std::uint8_t value=0;
        check(cudaMemcpy(mask+pos,&value,1,cudaMemcpyHostToDevice));
        assert(!detail::plan_keeps_all_indices(p,m,n));
        value=2;check(cudaMemcpy(mask+pos,&value,1,cudaMemcpyHostToDevice));
      }
    }
    check(cudaFree(p.keep_row_mask));check(cudaFree(p.keep_col_mask));
    p.keep_row_mask=nullptr;p.keep_col_mask=nullptr;
    // Every combination of optional plan vectors, with bit-sensitive values.
    for (int present=0;present<32;++present) {
      LPInfoGpu lp;lp.A.rows=m;lp.A.cols=n;
      double** dst[]={&lp.c,&lp.l,&lp.u,&lp.AL,&lp.AU};
      double** src[]={&p.new_c,&p.new_l,&p.new_u,&p.new_AL,&p.new_AU};
      std::vector<double> old[5],next[5];
      const std::uint64_t patterns[]={0,0x8000000000000000ULL,
        0x7ff8000000000001ULL,0x7ff8000000000002ULL,
        0x7ff0000000000000ULL,0xfff0000000000000ULL,0x3ff0000000000000ULL};
      for (int k=0;k<5;++k) {
        int size=k<3?n:m;old[k].resize(size);next[k].resize(size);
        for(int i=0;i<size;++i) {
          auto a=patterns[(i+k)%7],b=patterns[(i+k+(i%3!=0))%7];
          std::memcpy(&old[k][i],&a,8);std::memcpy(&next[k][i],&b,8);
        }
        *dst[k]=device(old[k]);*src[k]=(present&(1<<k))?device(next[k]):nullptr;
      }
      detail::commit_identity_plan_vectors(p,lp);
      for (int k=0;k<5;++k) {
        same_device(*dst[k],(present&(1<<k))?next[k]:old[k]);
        if(*src[k])same_device(*src[k],next[k]);
        check(cudaFree(*dst[k]));check(cudaFree(*src[k]));*src[k]=nullptr;
      }
    }
  }
  check(cudaDeviceSynchronize());
  std::cout<<"identity masks and all optional-vector combinations preserve exact bits\n";
}
