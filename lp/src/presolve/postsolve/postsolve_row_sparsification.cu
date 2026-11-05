// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/rules/rule_row_sparsification.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <new>
#include <stdexcept>

namespace gpu_preprocessor::lp::presolve {
namespace {

void checked(cudaError_t error) {
  if (error == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

__global__ void restore_row_duals(int rows, const int* parents, const int* alpha,
                                  const double* input, double* output) {
  for (auto row = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       row < rows; row += std::int64_t(blockDim.x) * gridDim.x) {
    const int parent = parents[row];
    // Sorted adjacency gives each parent at most one child.
    if (parent >= 0) output[parent] = input[parent] - alpha[row] * input[row];
  }
}

}  // namespace

void restore_startup_row_duals(const StartupRowSparsification& state, double* y) {
  const int rows = state.transformed.A.rows;
  if (rows == 0) return;
  struct Snapshot {
    double* data = nullptr;
    ~Snapshot() { if (data) cudaFreeAsync(data, 0); }
  } snapshot;
  checked(cudaMallocAsync(&snapshot.data, std::size_t(rows) * sizeof(double), 0));
  checked(cudaMemcpyAsync(snapshot.data, y, std::size_t(rows) * sizeof(double),
                          cudaMemcpyDeviceToDevice, 0));
  // A' = S A, so y = S^T y'. Read an immutable y' to apply S^T once;
  // reading partially restored ancestors would overcount a chain.
  constexpr int threads = 256;
  const int blocks = int(std::min<std::int64_t>(8192, (std::int64_t(rows) + threads - 1) / threads));
  restore_row_duals<<<blocks, threads>>>(rows, state.parents, state.alpha, snapshot.data, y);
  checked(cudaGetLastError());
}

}  // namespace gpu_preprocessor::lp::presolve
