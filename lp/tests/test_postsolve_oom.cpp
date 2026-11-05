// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/gpu_postsolve.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {
bool tracking = false;
int allocations = 0, copies = 0, fail_allocation = 0, fail_copy = 0;
std::set<void*> live;
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}

extern "C" cudaError_t __real_cudaMalloc(void**, std::size_t);
extern "C" cudaError_t __real_cudaFree(void*);
extern "C" cudaError_t __real_cudaFreeAsync(void*, cudaStream_t);
extern "C" cudaError_t __real_cudaMemcpy(void*, const void*, std::size_t, cudaMemcpyKind);
extern "C" cudaError_t __wrap_cudaMalloc(void** out, std::size_t size) {
  if (tracking && ++allocations == fail_allocation) {
    *out = nullptr;
    return cudaErrorMemoryAllocation;
  }
  const auto status = __real_cudaMalloc(out, size);
  if (tracking && status == cudaSuccess) live.insert(*out);
  return status;
}
extern "C" cudaError_t __wrap_cudaFree(void* p) {
  const auto status = __real_cudaFree(p);
  if (status == cudaSuccess) live.erase(p);
  return status;
}
extern "C" cudaError_t __wrap_cudaFreeAsync(void* p, cudaStream_t stream) {
  const auto status = __real_cudaFreeAsync(p, stream);
  if (status == cudaSuccess) live.erase(p);
  return status;
}
extern "C" cudaError_t __wrap_cudaMemcpy(void* out, const void* in, std::size_t size,
                                         cudaMemcpyKind kind) {
  if (tracking && ++copies == fail_copy) return cudaErrorInvalidValue;
  return __real_cudaMemcpy(out, in, size, kind);
}

int main() {
  using namespace gpu_preprocessor::lp::presolve;
  PresolveRecordGpu record;
  record.m0 = record.m1 = 1;
  record.n0 = 2; record.n1 = 1;
  record.row_red2org = record.row_org2red = {0};
  record.col_red2org = {0}; record.col_org2red = {0, -1};
  record.fixed_idx = {1}; record.fixed_val = {2.0};
  record.tape.types = {int(PostsolveReductionType::FixedVariable)};
  record.tape.index_starts = {0, 1}; record.tape.value_starts = {0, 2};
  record.tape.dual_modes = {std::uint8_t(PostsolveDualMode::Exact)};
  record.tape.indices = {1}; record.tape.vals = {2.0, 0.0};
  double* input = nullptr;
  check(cudaMalloc(&input, sizeof(double)));
  const double value = 3.0;
  check(cudaMemcpy(input, &value, sizeof(value), cudaMemcpyHostToDevice));
  auto attempt = [&](int allocation, int copy) {
    allocations = copies = 0;
    fail_allocation = allocation; fail_copy = copy;
    tracking = true;
    bool threw = false;
    try {
      const auto result = postsolve_gpu(input, input, input, record);
      check(cudaFree(result.x_org)); check(cudaFree(result.y_org)); check(cudaFree(result.z_org));
    } catch (const std::runtime_error&) { threw = true; }
    tracking = false;
    check(cudaDeviceSynchronize());
    if (!live.empty()) {
      std::cerr << "postsolve leaked " << live.size() << " allocations after allocation="
                << allocation << " copy=" << copy << '\n';
      throw std::runtime_error("postsolve exception cleanup failed");
    }
    if (threw != (allocation != 0 || copy != 0))
      throw std::runtime_error("injected failure was not propagated");
    double unchanged = 0;
    check(cudaMemcpy(&unchanged, input, sizeof(value), cudaMemcpyDeviceToHost));
    if (unchanged != value) throw std::runtime_error("borrowed input changed");
  };
  auto sweep = [&](const char* name) {
    attempt(0, 0);
    const int allocation_count = allocations, copy_count = copies;
    for (int i = 1; i <= allocation_count; ++i) attempt(i, 0);
    for (int i = 1; i <= copy_count; ++i) attempt(0, i);
    attempt(0, 0);
    std::cout << name << ": " << allocation_count << " allocation failures and " << copy_count
              << " transfer failures leave no owned buffers or modified inputs\n";
  };
  sweep("ordinary");
  PiecewiseLinearReformulationVariablePairRecoveryStep pair;
  pair.plus_cols = pair.root_plus_cols = {0};
  pair.minus_cols = pair.root_minus_cols = {1};
  pair.plus_shift = pair.minus_shift = {0.0};
  record.piecewise_linear_reformulation_variable_pair_recoveries.push_back(pair);
  record.primal_recovery_timeline.push_back({PrimalRecoveryKind::PiecewiseLinearReformulationVariablePair, 0, 0});
  sweep("variable pair");
  record.piecewise_linear_reformulation_variable_pair_recoveries.clear();
  record.primal_recovery_timeline.clear();
  PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep absolute;
  absolute.splits.push_back({0, 1, 1.0});
  absolute.outer_pairs.push_back({0, 1});
  absolute.linked_slacks.push_back({1, 0, 1.0});
  AbsoluteValueMaxSlackRecovery maximum;
  maximum.slack_col = 1; maximum.t_cols = {0}; maximum.factors = {1.0};
  absolute.max_slacks.push_back(maximum);
  record.piecewise_linear_reformulation_absolute_value_recoveries.push_back(absolute);
  record.primal_recovery_timeline.push_back({PrimalRecoveryKind::PiecewiseLinearReformulationAbsoluteValue, 0, 0});
  sweep("absolute value");
  check(cudaFree(input));
}
