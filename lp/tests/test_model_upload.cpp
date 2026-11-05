// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/model_upload.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace gp = gpu_preprocessor::lp::presolve;
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
bool fail_scratch = false;
extern "C" cudaError_t __real_cudaMalloc(void**, std::size_t);
extern "C" cudaError_t __wrap_cudaMalloc(void** out, std::size_t bytes) {
  if (fail_scratch) {
    std::size_t free = 0, total = 0; check(cudaMemGetInfo(&free,&total));
    auto status = __real_cudaMalloc(out,total+1024*1024);
    assert(status == cudaErrorMemoryAllocation); return status;
  }
  return __real_cudaMalloc(out,bytes);
}
struct Array {
  double* p = nullptr;
  explicit Array(std::size_t n) { check(cudaMalloc(&p,n*sizeof(double))); }
  ~Array() { cudaFree(p); }
};
std::vector<double> pattern(std::size_t n, std::uint64_t bits) {
  std::vector<double> h(n);
  for (auto& x : h) std::memcpy(&x,&bits,sizeof(bits));
  return h;
}
void compare(const double* p, const std::vector<double>& expected) {
  std::vector<double> result(expected.size());
  check(cudaMemcpy(result.data(),p,result.size()*sizeof(double),cudaMemcpyDeviceToHost));
  assert(std::memcmp(result.data(),expected.data(),result.size()*sizeof(double)) == 0);
}
int main() {
  const std::size_t n = 131091;
  Array input(n), output(n);
  gp::LPInfoGpu model;
  model.A.rows = model.A.cols = n; model.A.nnz = n;
  model.A.nzVal = model.c = model.AL = model.AU = model.l = model.u = input.p;
  for (std::uint64_t bits : {0ULL,0x8000000000000000ULL,0x3ff4000000000000ULL,
       0x7ff0000000000000ULL,0xfff0000000000000ULL,1ULL,
       0x7ff8000000001234ULL,0x7ff0000000000001ULL,0x7fefffffffffffffULL}) {
    auto h = pattern(n,bits);
    check(cudaMemcpy(input.p,h.data(),n*sizeof(double),cudaMemcpyHostToDevice));
    const auto cache = gp::capture_model_upload_hints(model);
    for (const auto* field : {&cache.values,&cache.c,&cache.AL,&cache.AU,&cache.l,&cache.u})
      assert(field->count == n && field->uniform && field->bits == bits);
    compare(input.p,h);
    assert(gp::upload_with_hint(output.p,h.data(),n,cache.c)); compare(output.p,h);
    assert(gp::try_upload_uniform_double(output.p,h.data(),n)); compare(output.p,h);

    // A mismatch away from the sampled positions must still reject the fill.
    std::uint64_t other = bits ^ 1; std::memcpy(&h[17],&other,sizeof(other));
    check(cudaMemcpy(input.p,h.data(),n*sizeof(double),cudaMemcpyHostToDevice));
    const auto mixed = gp::capture_model_upload_hints(model);
    assert(mixed.c.count == n && !mixed.c.uniform && !mixed.values.uniform);
    const auto marker = pattern(n,0xdeadbeefcafebabeULL);
    check(cudaMemcpy(output.p,marker.data(),n*sizeof(double),cudaMemcpyHostToDevice));
    assert(!gp::try_upload_uniform_double(output.p,h.data(),n)); compare(output.p,marker);
    compare(input.p,h);
    assert(gp::upload_with_hint(output.p,h.data(),n,mixed.c)); compare(output.p,h);
    auto invalid = mixed.c; invalid.chunks.pop_back();
    check(cudaMemcpy(output.p,marker.data(),n*sizeof(double),cudaMemcpyHostToDevice));
    assert(!gp::upload_with_hint(output.p,h.data(),n,invalid)); compare(output.p,marker);
  }
  // Mixed small dictionaries use exact packed indices, including tail bytes.
  for (int colors : {2,4,5}) {
    const std::uint64_t values[] = {0,0x8000000000000000ULL,0x7ff0000000000000ULL,
                                  0x7ff8000000001234ULL,0x7ff0000000000001ULL};
    std::vector<double> h(n);
    for (std::size_t i=0;i<n;++i) std::memcpy(&h[i],&values[i%colors],sizeof(double));
    check(cudaMemcpy(input.p,h.data(),n*sizeof(double),cudaMemcpyHostToDevice));
    const auto cache = gp::capture_model_upload_hints(model);
    assert(!cache.c.uniform && (cache.c.packed.empty() == (colors>4)));
    assert(gp::upload_with_hint(output.p,h.data(),n,cache.c)); compare(output.p,h); compare(input.p,h);
    if (colors==4) {
      const auto marker=pattern(n,0xdeadbeefcafebabeULL);
      check(cudaMemcpy(output.p,marker.data(),n*sizeof(double),cudaMemcpyHostToDevice));
      fail_scratch=true;
      assert(!gp::upload_with_hint(output.p,h.data(),n,cache.c));
      fail_scratch=false; check(cudaPeekAtLastError()); compare(output.p,marker);
      // A fifth value invalidates its chunk; other chunks remain packed.
      std::memcpy(&h[17],&values[4],sizeof(double));
      check(cudaMemcpy(input.p,h.data(),n*sizeof(double),cudaMemcpyHostToDevice));
      const auto partial=gp::capture_model_upload_hints(model);
      assert(!partial.c.packed.empty() && partial.c.chunks[0].count==0);
      assert(gp::upload_with_hint(output.p,h.data(),n,partial.c)); compare(output.p,h);
    }
  }
  // +0 and -0 must never be merged by a floating-point equality test.
  auto signed_zero = pattern(n,0);
  std::uint64_t minus_zero = 0x8000000000000000ULL;
  std::memcpy(&signed_zero[17],&minus_zero,sizeof(minus_zero));
  assert(!gp::try_upload_uniform_double(output.p,signed_zero.data(),n));
  auto h = pattern(n,0x7ff0000000000000ULL);
  check(cudaMemcpy(input.p,h.data(),n*sizeof(double),cudaMemcpyHostToDevice));
  fail_scratch = true;
  const auto omitted = gp::capture_model_upload_hints(model);
  fail_scratch = false;
  assert(!omitted.c.count && !omitted.c.uniform);
  check(cudaPeekAtLastError()); compare(input.p,h);
  model.A.rows = model.A.cols = 17; model.A.nnz = 17;
  const auto tiny = gp::capture_model_upload_hints(model);
  assert(!tiny.c.count && !tiny.values.count);
  assert(!gp::try_upload_uniform_double(output.p,h.data(),17));
  check(cudaDeviceSynchronize());
  std::cout << "Exact uniform upload, nonuniform fallback, signed zero/NaN payloads and real scratch OOM PASS\n";
}
