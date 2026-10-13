#include "gpu_presolver/presolve/identity_plan.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

namespace gpu_presolver::presolve::detail {
namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

__global__ void detect_removed(const std::uint8_t* rows, const std::uint8_t* cols,
                               int m, int n, int* removed) {
  const auto i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool drop = (i < m && rows[i] == 0) || (i < n && cols[i] == 0);
  // One atomic per affected block, instead of contending once per entry.
  const int block_drop = __syncthreads_or(drop);
  if (threadIdx.x == 0 && block_drop) atomicExch(removed, 1);
}

__device__ void assign_changed(double* dst, const double* src, std::int64_t i) {
  if (src && __double_as_longlong(dst[i]) != __double_as_longlong(src[i])) {
    dst[i] = src[i];
  }
}

__global__ void commit_vectors(int m, int n, double* c, double* l, double* u,
                               double* al, double* au, const double* next_c,
                               const double* next_l, const double* next_u,
                               const double* next_al, const double* next_au) {
  const auto i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    assign_changed(c, next_c, i);
    assign_changed(l, next_l, i);
    assign_changed(u, next_u, i);
  }
  if (i < m) {
    assign_changed(al, next_al, i);
    assign_changed(au, next_au, i);
  }
}
}  // namespace

bool plan_keeps_all_indices(const PresolvePlanGpu& plan, std::int32_t m, std::int32_t n) {
  const auto size = std::max(m, n);
  if (size == 0) return true;
  RecoveryArray<int> flag;
  check(cudaMallocAsync(reinterpret_cast<void**>(&flag.data), sizeof(int), nullptr));
  check(cudaMemsetAsync(flag.data, 0, sizeof(int), nullptr));
  detect_removed<<<(std::int64_t(size) + 255) / 256, 256>>>(
      plan.keep_row_mask, plan.keep_col_mask, m, n, flag.data);
  check(cudaGetLastError());
  int removed = 0;
  check(cudaMemcpy(&removed, flag.data, sizeof(int), cudaMemcpyDeviceToHost));
  return removed == 0;
}

void commit_identity_plan_vectors(const PresolvePlanGpu& plan, LPInfoGpu& lp) {
  const auto size = std::max(lp.A.rows, lp.A.cols);
  if (size == 0) return;
  commit_vectors<<<(std::int64_t(size) + 255) / 256, 256>>>(
      lp.A.rows, lp.A.cols, lp.c, lp.l, lp.u, lp.AL, lp.AU,
      plan.new_c, plan.new_l, plan.new_u, plan.new_AL, plan.new_AU);
  check(cudaGetLastError());
}
}  // namespace gpu_presolver::presolve::detail
