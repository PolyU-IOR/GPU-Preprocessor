#include "gpu_presolver/presolve/fixed_col_tape.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
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

static void same_tape(const PostsolveTape& a,const PostsolveTape& b) {
  assert(a.types==b.types && a.dual_modes==b.dual_modes &&
    a.indices==b.indices && a.index_starts==b.index_starts && a.value_starts==b.value_starts);
  assert(a.vals.size()==b.vals.size());
  if(!a.vals.empty())assert(std::memcmp(a.vals.data(),b.vals.data(),a.vals.size()*8)==0);
}
int main() {
  DeviceCsrMatrix at;
  at.rows=5;at.cols=3;at.nnz=8;
  at.rowPtr=device(std::vector<NnzOffset>{0,2,3,5,7,8});
  at.colVal=device(std::vector<int>{0,1,2,0,2,1,2,0});
  at.nzVal=device(std::vector<double>{-0.0,8,2,3,4,5,6,7});
  auto fixed=device(std::vector<std::uint8_t>{2,1,1,1,0});
  auto excluded=device(std::vector<std::uint8_t>{0,0,0,1,0});
  auto rows=device(std::vector<std::uint8_t>{2,0,1});
  auto values=device(std::vector<double>{1,1,1,999,-0.0});
  auto costs=device(std::vector<double>{1e16,1,-1e16,99,7});
  PostsolveTape oracle;
  auto compute=[&](PostsolveTape* t,FixedColObjective mode) {
    return append_compact_fixed_col_tape_from_device(t,fixed,excluded,values,rows,costs,at,5,
                                                    "fixed tape test",mode);
  };
  assert(compute(&oracle,FixedColObjective::Compute)==1.0);
  assert((oracle.indices==std::vector<int>{0,0,1,2,2,0,2}));
  assert((oracle.index_starts==std::vector<NnzOffset>{0,2,4,7}));
  for (bool resident : {false,true}) for(auto mode : {FixedColObjective::Compute,FixedColObjective::Skip}) {
    PostsolveTape tape;tape.resident=resident;
    assert(compute(&tape,mode)==(mode==FixedColObjective::Compute?1.0:0.0));
    if(resident) {
      assert(tape.types.empty() && tape.vals.empty() && tape.segments.size()==1);
      export_local_recovery_tape(tape);
    }
    same_tape(tape,oracle);
  }
  assert(compute(nullptr,FixedColObjective::Compute)==1.0);
  assert(compute(nullptr,FixedColObjective::Skip)==0.0);
  check(cudaMemset(fixed,0,5));
  for(bool resident : {false,true}) {
    PostsolveTape tape;tape.resident=resident;
    assert(compute(&tape,FixedColObjective::Skip)==0.0);
    assert(tape.types.empty() && tape.segments.empty());
  }
  assert(append_compact_fixed_col_tape_from_device(nullptr,nullptr,nullptr,nullptr,nullptr,
                                                  nullptr,at,0,"empty")==0.0);
  for(auto p:{static_cast<void*>(at.rowPtr),static_cast<void*>(at.colVal),static_cast<void*>(at.nzVal),
              static_cast<void*>(fixed),static_cast<void*>(excluded),static_cast<void*>(rows),
              static_cast<void*>(values),static_cast<void*>(costs)})check(cudaFree(p));
  check(cudaDeviceSynchronize());
  std::cout<<"host/device record-only tapes match; stable objective cancellation preserved\n";
}
