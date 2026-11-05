// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "../src/presolve/rules/parallel_signature_common.cuh"

#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sig = gpu_preprocessor::lp::presolve::parallel_signature;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void test_modes_and_scratch() {
  for (const char* option : {"GPUPREPROCESSOR_PARALLEL_ROWS_SIGNATURE",
                              "GPUPREPROCESSOR_PARALLEL_COLUMNS_SIGNATURE"}) {
    const char* old = std::getenv(option);
    const bool had_value = old != nullptr;
    const std::string saved = old ? old : "";
    gpu_preprocessor::lp::presolve::DeviceCsrMatrix a;
    a.rows = 17;
    a.nnz = 32;
    unsetenv(option);
    assert(sig::enabled(a, option));
    const auto required = sig::scratch_bytes(a.rows, option);
    std::vector<unsigned long long> storage((required + 7) / 8);
    sig::Workspace w(storage.data(), a.rows);
    const auto end = reinterpret_cast<unsigned char*>(w.counts + 2);
    assert(std::size_t(end - reinterpret_cast<unsigned char*>(storage.data())) == required);
    assert(reinterpret_cast<std::uintptr_t>(w.counts) % alignof(unsigned long long) == 0);
    setenv(option, "", 1);
    assert(sig::enabled(a, option) && sig::scratch_bytes(a.rows, option) == required);
    setenv(option, "0", 1);
    assert(!sig::enabled(a, option) && sig::scratch_bytes(a.rows, option) == 0);
    setenv(option, "auto", 1);
    assert(!sig::enabled(a, option));
    a.rows = 16384;
    a.nnz = 1000000;
    assert(sig::enabled(a, option));
    a.nnz = 999999;
    assert(!sig::enabled(a, option));
    a.rows = 125000;
    a.nnz = 1000000;
    assert(!sig::enabled(a, option));
    a.nnz++;
    assert(sig::enabled(a, option));
    for (const char* mode : {"", "1", "auto"}) {
      setenv(option, mode, 1);
      a.rows = sig::max_dimensions + 1;
      assert(!sig::enabled(a, option) && sig::scratch_bytes(a.rows, option) == 0);
    }
    if (had_value) setenv(option, saved.c_str(), 1);
    else unsetenv(option);
  }
}

void test_bucket_fallbacks() {
  constexpr int n = 8;
  void* memory = nullptr;
  check(cudaMalloc(&memory, sig::scratch_bytes(n, "GPUPREPROCESSOR_TEST_SIGNATURE")));
  sig::Workspace w(memory, n);
  std::uint8_t* keep = nullptr;
  check(cudaMalloc(&keep, n));
  const std::vector<std::uint8_t> masks{1, 1, 1, 1, 1, 1, 0, 1};
  check(cudaMemcpy(keep, masks.data(), n, cudaMemcpyHostToDevice));
  // A neighboring bucket, an uncertain peer, a singleton, a deleted record,
  // and an unprojectable record exercise distinct candidate paths.
  const std::vector<sig::Projection> projections{
      {1, 0, 0, 2}, {1, 1, -1, 2}, {3, LLONG_MIN, LLONG_MIN, 4},
      {3, 20, 20, 4}, {5, 30, 30, 3}, {7, 40, 40, 3},
      {0, LLONG_MIN, LLONG_MIN, 0}, {0, LLONG_MIN, LLONG_MIN, 0}};
  check(cudaMemcpy(w.projections, projections.data(), sizeof(sig::Projection) * n,
                   cudaMemcpyHostToDevice));
  check(cudaMemset(w.keys, 0, w.size * sizeof(unsigned long long)));
  check(cudaMemset(w.owners, 0xff, w.size * sizeof(int)));
  check(cudaMemset(w.uncertain_support, 0, w.size * sizeof(int)));
  check(cudaMemset(w.overflow, 0, sizeof(int)));
  sig::build_buckets<<<1, 256>>>(w, keep, n);
  for (bool retain_deleted : {false, true}) {
    if (retain_deleted) sig::mark_candidates<true><<<1, 256>>>(w, keep, n);
    else sig::mark_candidates<false><<<1, 256>>>(w, keep, n);
    check(cudaGetLastError());
    std::vector<std::uint8_t> selected(n);
    check(cudaMemcpy(selected.data(), w.candidates, n, cudaMemcpyDeviceToHost));
    assert((selected == std::vector<std::uint8_t>{1, 1, 1, 1, 0, 0,
                                               std::uint8_t(retain_deleted), 1}));
  }
  const int overflow = 1;
  check(cudaMemcpy(w.overflow, &overflow, sizeof(int), cudaMemcpyHostToDevice));
  sig::mark_candidates<true><<<1, 256>>>(w, keep, n);
  std::vector<std::uint8_t> selected(n);
  check(cudaMemcpy(selected.data(), w.candidates, n, cudaMemcpyDeviceToHost));
  assert((selected == std::vector<std::uint8_t>(n, 1)));
  cudaFree(keep);
  cudaFree(memory);
}

int main() {
  test_modes_and_scratch();
  test_bucket_fallbacks();
  std::cout << "test_parallel_signature passed\n";
}
