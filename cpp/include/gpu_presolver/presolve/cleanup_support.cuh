#pragma once
#include "gpu_presolver/presolve/cleanup_workspace.hpp"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
namespace gpu_presolver::presolve {
// Compact contributions in the same stable column order/length as a former
// physical Apply. The caller retains its existing reduction implementation.
template<class Sum>
inline double sum_cleanup_domain(const double* values, int n,
                                 const std::uint8_t* keep, Sum sum) {
  if (!keep || n == 0) return sum(values, n);
  auto check = [](cudaError_t e) { if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e)); };
  CleanupScratchFrame selection;
  double* compact = selection.get<double>(8, n);
  int* count = selection.get<int>(9, 1);
  void* scratch = nullptr;
  std::size_t bytes = 0;
  check(cub::DeviceSelect::Flagged(nullptr, bytes, values, keep, compact, count, n));
  scratch = selection.get<unsigned char>(10, bytes);
  check(cub::DeviceSelect::Flagged(scratch, bytes, values, keep, compact, count, n));
  int active_n = 0;
  check(cudaMemcpy(&active_n, count, sizeof(int), cudaMemcpyDeviceToHost));
  const double result = sum(compact, active_n);
  return result;
}
}
