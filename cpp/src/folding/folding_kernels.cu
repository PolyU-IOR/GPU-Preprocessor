#include "folding_internal.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gpu_presolver::folding {
namespace {

constexpr int kFoldingThreads = 256;
constexpr double kTwoPi = 6.283185307179586476925286766559;

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

std::uint64_t mix64_host(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__device__ std::uint64_t mix64_device(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__device__ double unit_from_mix(std::uint64_t x) {
  const std::uint64_t mixed = mix64_device(x);
  const double unit = static_cast<double>(mixed >> 11) * 0x1.0p-53;
  return unit > 0.0 ? unit : 0x1.0p-53;
}

std::uint64_t color_seed(std::int32_t round, std::uint64_t side) {
  return mix64_host(static_cast<std::uint64_t>(round) ^ (side * 0x9e3779b97f4a7c15ULL));
}

struct ColorPairLess {
  __host__ __device__ bool operator()(const ColorPair& lhs, const ColorPair& rhs) const {
    if (lhs.color_id != rhs.color_id) {
      return lhs.color_id < rhs.color_id;
    }
    return lhs.color_sig < rhs.color_sig;
  }
};

__global__ void kernel_pack_color_and_perm(ColorPair* color, std::int32_t* perm,
                                           const std::int32_t* ids,
                                           const double* sig, std::int32_t n) {
  const std::int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    color[i] = {ids[i], static_cast<float>(sig[i])};
    perm[i] = i;
  }
}

__global__ void kernel_mark_color_start(std::int32_t* color_start,
                                        const ColorPair* color,
                                        double tol,
                                        std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    if (i == 0) {
      color_start[i] = 1;
    } else {
      const bool changed_id = color[i].color_id != color[i - 1].color_id;
      const bool changed_sig = static_cast<double>(color[i].color_sig) >
                               static_cast<double>(color[i - 1].color_sig) + tol;
      color_start[i] = (changed_id || changed_sig) ? 1 : 0;
    }
  }
}

__global__ void kernel_write_color_id(std::int32_t* color_id,
                                      const std::int32_t* perm,
                                      const std::int32_t* prefix,
                                      std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    color_id[perm[i]] = prefix[i];
  }
}

__global__ void kernel_write_color_sig(double* color_sig,
                                       const std::int32_t* color_id,
                                       std::uint64_t seed,
                                       std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    const std::uint64_t color = static_cast<std::uint64_t>(color_id[i]);
    const double u1 = unit_from_mix(seed ^ (color * 0xbf58476d1ce4e5b9ULL));
    const double u2 = unit_from_mix((seed + 0x9e3779b97f4a7c15ULL) ^
                                    (color * 0x94d049bb133111ebULL));
    color_sig[i] = sqrt(-2.0 * log(u1)) * cos(kTwoPi * u2);
  }
}

__global__ void kernel_write_color_id_and_sig(std::int32_t* ids, double* sig,
                                             const std::int32_t* perm,
                                             const std::int32_t* prefix,
                                             std::uint64_t seed, std::int32_t n) {
  const std::int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const auto member = perm[i];
    ids[member] = prefix[i];
    const std::uint64_t color = static_cast<std::uint64_t>(prefix[i]);
    const double u1 = unit_from_mix(seed ^ (color * 0xbf58476d1ce4e5b9ULL));
    const double u2 = unit_from_mix((seed + 0x9e3779b97f4a7c15ULL) ^
                                    (color * 0x94d049bb133111ebULL));
    sig[member] = sqrt(-2.0 * log(u1)) * cos(kTwoPi * u2);
  }
}

__global__ void kernel_collect_color_counts(std::int32_t* counts,
                                            const std::int32_t* row_prefix,
                                            const std::int32_t* col_prefix,
                                            std::int32_t rows, std::int32_t cols) {
  counts[0] = rows ? row_prefix[rows - 1] : 0;
  counts[1] = cols ? col_prefix[cols - 1] : 0;
}

// Blocks live until the invocation ends. Reuse in the asynchronous path is
// safe because every consumer and subsequent reuse runs in the default stream.
struct ColorScratchAllocator {
  using value_type = char;
  FoldingWorkspace& workspace;
  char* allocate(std::ptrdiff_t n) {
    FoldingScratchBlock* best = nullptr;
    for (auto& block : workspace.color_scratch)
      if (!block.in_use && block.bytes >= static_cast<std::size_t>(n) &&
          (!best || block.bytes < best->bytes)) best = &block;
    if (best) { best->in_use = true; return best->pointer; }
    char* pointer = nullptr;
    throw_if_cuda_error(cudaMalloc(&pointer, static_cast<std::size_t>(n)), "cudaMalloc color scratch");
    try { workspace.color_scratch.push_back({pointer,static_cast<std::size_t>(n),true}); }
    catch (...) { cudaFree(pointer); throw; }
    return pointer;
  }
  void deallocate(char* pointer, std::size_t) {
    for (auto& block : workspace.color_scratch)
      if (block.pointer == pointer) { block.in_use = false; return; }
    throw std::runtime_error("Unknown color scratch allocation");
  }
};

std::int32_t update_color_id(FoldingWorkspace& workspace, std::int32_t* color_id,
                             std::int32_t* color_start,
                             std::int32_t* perm,
                             ColorPair* color,
                             const double* color_sig,
                             double tol,
                             std::int32_t n, bool defer_count = false,
                             double* output_sig = nullptr, std::uint64_t seed = 0) {
  if (n <= 0) {
    return 0;
  }
  const int blocks = (n + kFoldingThreads - 1) / kFoldingThreads;
  ColorScratchAllocator allocator{workspace};
  thrust::device_ptr<ColorPair> color_ptr(color);
  thrust::device_ptr<std::int32_t> perm_ptr(perm);
  kernel_pack_color_and_perm<<<blocks, kFoldingThreads>>>(color, perm, color_id, color_sig, n);
  throw_if_cuda_error(cudaGetLastError(), "kernel_pack_color_and_perm");
  if (defer_count)
    thrust::sort_by_key(thrust::cuda::par_nosync(allocator), color_ptr, color_ptr + n, perm_ptr, ColorPairLess{});
  else
    thrust::sort_by_key(thrust::cuda::par(allocator), color_ptr, color_ptr + n, perm_ptr, ColorPairLess{});
  kernel_mark_color_start<<<blocks, kFoldingThreads>>>(color_start, color, tol, n);
  throw_if_cuda_error(cudaGetLastError(), "kernel_mark_color_start");
  if (defer_count)
    thrust::inclusive_scan(thrust::cuda::par_nosync(allocator), thrust::device_pointer_cast(color_start),
                           thrust::device_pointer_cast(color_start + n), thrust::device_pointer_cast(color_start));
  else
    thrust::inclusive_scan(thrust::cuda::par(allocator), thrust::device_pointer_cast(color_start),
                           thrust::device_pointer_cast(color_start + n), thrust::device_pointer_cast(color_start));

  std::int32_t new_color_num = 0;
  if (!defer_count) throw_if_cuda_error(
      cudaMemcpy(&new_color_num, color_start + (n - 1), sizeof(std::int32_t), cudaMemcpyDeviceToHost),
      "cudaMemcpy update_color_id");
  if (output_sig)
    kernel_write_color_id_and_sig<<<blocks, kFoldingThreads>>>(color_id, output_sig, perm, color_start, seed, n);
  else
    kernel_write_color_id<<<blocks, kFoldingThreads>>>(color_id, perm, color_start, n);
  throw_if_cuda_error(cudaGetLastError(), "kernel_write_color_id");
  return new_color_num;
}

void write_color_sig(double* color_sig,
                     const std::int32_t* color_id,
                     std::int32_t n,
                     std::uint64_t seed) {
  if (n <= 0) return;
  const int blocks = (n + kFoldingThreads - 1) / kFoldingThreads;
  kernel_write_color_sig<<<blocks, kFoldingThreads>>>(color_sig, color_id, seed, n);
  throw_if_cuda_error(cudaGetLastError(), "kernel_write_color_sig");
}

}  // namespace

void init_workspace(FoldingWorkspace& workspace, std::int32_t num_row, std::int32_t num_col) {
  workspace.num_row = num_row;
  workspace.num_col = num_col;
  workspace.num_row_color = 1;
  workspace.num_col_color = 1;
  throw_if_cuda_error(cudaMalloc(&workspace.row_color, sizeof(ColorPair) * static_cast<std::size_t>(num_row)),
                      "cudaMalloc row_color");
  throw_if_cuda_error(cudaMalloc(&workspace.col_color, sizeof(ColorPair) * static_cast<std::size_t>(num_col)),
                      "cudaMalloc col_color");
  throw_if_cuda_error(cudaMalloc(&workspace.row_color_id, sizeof(std::int32_t) * static_cast<std::size_t>(num_row)),
                      "cudaMalloc row_color_id");
  throw_if_cuda_error(cudaMalloc(&workspace.col_color_id, sizeof(std::int32_t) * static_cast<std::size_t>(num_col)),
                      "cudaMalloc col_color_id");
  throw_if_cuda_error(cudaMalloc(&workspace.row_color_sig, sizeof(double) * static_cast<std::size_t>(num_row)),
                      "cudaMalloc row_color_sig");
  throw_if_cuda_error(cudaMalloc(&workspace.col_color_sig, sizeof(double) * static_cast<std::size_t>(num_col)),
                      "cudaMalloc col_color_sig");
  throw_if_cuda_error(cudaMalloc(&workspace.row_color_start, sizeof(std::int32_t) * static_cast<std::size_t>(num_row)),
                      "cudaMalloc row_color_start");
  throw_if_cuda_error(cudaMalloc(&workspace.col_color_start, sizeof(std::int32_t) * static_cast<std::size_t>(num_col)),
                      "cudaMalloc col_color_start");
  throw_if_cuda_error(cudaMalloc(&workspace.row_perm, sizeof(std::int32_t) * static_cast<std::size_t>(num_row)),
                      "cudaMalloc row_perm");
  throw_if_cuda_error(cudaMalloc(&workspace.col_perm, sizeof(std::int32_t) * static_cast<std::size_t>(num_col)),
                      "cudaMalloc col_perm");
  if (num_row > 0) {
    throw_if_cuda_error(cudaMemset(workspace.row_color_id, 0, sizeof(std::int32_t) * static_cast<std::size_t>(num_row)),
                        "cudaMemset row_color_id");
    throw_if_cuda_error(cudaMemset(workspace.row_color_sig, 0, sizeof(double) * static_cast<std::size_t>(num_row)),
                        "cudaMemset row_color_sig");
  }
  if (num_col > 0) {
    throw_if_cuda_error(cudaMemset(workspace.col_color_id, 0, sizeof(std::int32_t) * static_cast<std::size_t>(num_col)),
                        "cudaMemset col_color_id");
    throw_if_cuda_error(cudaMemset(workspace.col_color_sig, 0, sizeof(double) * static_cast<std::size_t>(num_col)),
                        "cudaMemset col_color_sig");
  }
}

void free_workspace(FoldingWorkspace& workspace) {
  cudaFree(workspace.device_color_counts);
  cudaFree(workspace.row_color_count);
  cudaFree(workspace.col_color_count);
  for (auto& block : workspace.color_scratch) cudaFree(block.pointer);
  cudaFree(workspace.row_color);
  cudaFree(workspace.col_color);
  cudaFree(workspace.row_color_id);
  cudaFree(workspace.col_color_id);
  cudaFree(workspace.row_color_sig);
  cudaFree(workspace.col_color_sig);
  cudaFree(workspace.row_color_start);
  cudaFree(workspace.col_color_start);
  cudaFree(workspace.row_perm);
  cudaFree(workspace.col_perm);
  workspace = FoldingWorkspace{};
}

void folding_read_color_counts(FoldingWorkspace& w) {
  if (!w.device_color_counts)
    throw_if_cuda_error(cudaMalloc(&w.device_color_counts, 2 * sizeof(std::int32_t)), "cudaMalloc batched counts");
  // The blocking D2H copy is the round's host decision barrier. Pageable stack
  // storage suffices for these two integers and avoids pin/unpin overhead.
  std::int32_t counts[2];
  kernel_collect_color_counts<<<1,1>>>(w.device_color_counts, w.row_color_start,
                                     w.col_color_start, w.num_row, w.num_col);
  throw_if_cuda_error(cudaGetLastError(), "kernel_collect_color_counts");
  throw_if_cuda_error(cudaMemcpy(counts, w.device_color_counts,
                                 2 * sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy batched color counts");
  w.num_row_color = counts[0];
  w.num_col_color = counts[1];
}

std::int32_t folding_update_color_and_sig(FoldingWorkspace& w, bool row,
                                         const double* sig, double tol,
                                         std::int32_t round, std::uint64_t side,
                                         bool defer_count, bool write_sig) {
  return update_color_id(w, row ? w.row_color_id : w.col_color_id,
      row ? w.row_color_start : w.col_color_start, row ? w.row_perm : w.col_perm,
      row ? w.row_color : w.col_color, sig, tol, row ? w.num_row : w.num_col,
      defer_count, write_sig ? (row ? w.row_color_sig : w.col_color_sig) : nullptr,
      color_seed(round, side));
}

std::int32_t folding_update_row_color_id(FoldingWorkspace& workspace,
                                         const double* color_sig,
                                         double tol) {
  return update_color_id(workspace, workspace.row_color_id,
                         workspace.row_color_start,
                         workspace.row_perm,
                         workspace.row_color,
                         color_sig,
                         tol,
                         workspace.num_row);
}

std::int32_t folding_update_col_color_id(FoldingWorkspace& workspace,
                                         const double* color_sig,
                                         double tol) {
  return update_color_id(workspace, workspace.col_color_id,
                         workspace.col_color_start,
                         workspace.col_perm,
                         workspace.col_color,
                         color_sig,
                         tol,
                         workspace.num_col);
}

void folding_write_row_color_sig(FoldingWorkspace& workspace, std::int32_t round, std::uint64_t side) {
  write_color_sig(workspace.row_color_sig,
                  workspace.row_color_id,
                  workspace.num_row,
                  color_seed(round, side));
}

void folding_write_col_color_sig(FoldingWorkspace& workspace, std::int32_t round, std::uint64_t side) {
  write_color_sig(workspace.col_color_sig,
                  workspace.col_color_id,
                  workspace.num_col,
                  color_seed(round, side));
}

}  // namespace gpu_presolver::folding
