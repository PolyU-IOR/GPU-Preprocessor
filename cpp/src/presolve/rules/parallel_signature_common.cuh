#pragma once

#include "gpu_presolver/detail/runtime_options.hpp"

#include "gpu_presolver/presolve/presolve_structs.hpp"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cfloat>
#include <climits>
#include <cstddef>
#include <cstdint>

// Internal infrastructure shared by the row and column projection kernels.
// Support traversal remains separate: rows retain stored tiny coefficients,
// whereas columns follow the legacy hash's zero-tolerance filtering.
namespace gpu_presolver::presolve::parallel_signature {

constexpr double bin_scale = 1024.0;
constexpr int max_probe = 64;
constexpr std::size_t max_dimensions = 2 * 1024 * 1024;

inline bool enabled(const DeviceCsrMatrix& a, runtime::SignatureMode mode) {
  if (mode == runtime::SignatureMode::Disabled) return false;
  if (std::size_t(a.rows) > max_dimensions) return false;
  if (mode == runtime::SignatureMode::Automatic)
    return a.rows >= 16384 && a.nnz >= 1000000 && a.nnz > NnzOffset(a.rows) * 8;
  return true;
}

inline std::size_t capacity(std::size_t dimensions) {
  std::size_t result = 1;
  while (result < dimensions * 2) result <<= 1;
  return result;
}

template<class T>
inline std::size_t reserve(std::size_t& bytes, std::size_t count) {
  bytes = (bytes + alignof(T) - 1) & ~(alignof(T) - 1);
  const auto offset = bytes;
  bytes += sizeof(T) * count;
  return offset;
}

template<class T>
inline T* at(void* memory, std::size_t offset) {
  return reinterpret_cast<T*>(static_cast<unsigned char*>(memory) + offset);
}

struct Projection {
  unsigned long long support;
  long long x, y;
  NnzOffset length;
};

struct Workspace {
  Projection* projections;
  std::uint8_t* candidates;
  unsigned long long* keys;
  int* owners;
  int* uncertain_support;
  int* overflow;
  unsigned long long* counts;
  std::size_t size;

  Workspace(void* memory, std::size_t count) : size(capacity(count)) {
    std::size_t offset = 0;
    projections = at<Projection>(memory, reserve<Projection>(offset, count));
    candidates = at<std::uint8_t>(memory, reserve<std::uint8_t>(offset, count));
    keys = at<unsigned long long>(memory, reserve<unsigned long long>(offset, size));
    owners = at<int>(memory, reserve<int>(offset, size));
    uncertain_support = at<int>(memory, reserve<int>(offset, size));
    overflow = at<int>(memory, reserve<int>(offset, 1));
    counts = at<unsigned long long>(memory, reserve<unsigned long long>(offset, 2));
  }
};

inline std::size_t scratch_bytes(std::size_t dimensions, runtime::SignatureMode mode) {
  if ((mode == runtime::SignatureMode::Disabled) || dimensions > max_dimensions) return 0;
  // Mirror Workspace's aligned layout, including both profiling counters.
  std::size_t bytes = 0;
  reserve<Projection>(bytes, dimensions);
  reserve<std::uint8_t>(bytes, dimensions);
  reserve<unsigned long long>(bytes, capacity(dimensions));
  reserve<int>(bytes, capacity(dimensions));
  reserve<int>(bytes, capacity(dimensions));
  reserve<int>(bytes, 1);
  reserve<unsigned long long>(bytes, 2);
  return bytes;
}

__device__ inline unsigned long long mix(unsigned long long x) {
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27; x *= 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__device__ inline unsigned long long support_key(NnzOffset length, int first_col) {
  return mix(static_cast<unsigned long long>(length) ^
             (static_cast<unsigned long long>(first_col) << 32)) | 1ULL;
}

__device__ inline unsigned long long bucket_key(unsigned long long support, long long x, long long y) {
  return mix(support ^ mix(static_cast<unsigned long long>(x)) ^
             mix(static_cast<unsigned long long>(y) + 0x9e3779b97f4a7c15ULL)) | 1ULL;
}

__device__ inline double weight(int col, unsigned long long seed) {
  // Exact conversion and power-of-two scaling, with |r_j| <= 1. The vector is
  // implicit: no O(n) random-vector allocation or regeneration per rule call.
  return static_cast<double>(static_cast<int>(mix(static_cast<unsigned>(col) ^ seed) >> 32)) * 0x1p-31;
}

__device__ inline void finish(Projection* output, int row, NnzOffset length, int first_col,
                       double x, double y, double abs_sum, NnzOffset terms, int width) {
  Projection result{support_key(length, first_col), LLONG_MIN, LLONG_MIN, length};
  // Includes each lane's sequential sum, product/FMA rounding, the reduction
  // tree and rounding in abs_sum. Deliberately loose, with a wide safety margin.
  const double error = 64.0 * DBL_EPSILON * (static_cast<double>(terms) + width + 4) * abs_sum;
  if (isfinite(x) && isfinite(y) && isfinite(abs_sum) &&
      error * bin_scale <= 0.125 &&
      fabs(x * bin_scale) < 0x1p50 && fabs(y * bin_scale) < 0x1p50) {
    result.x = __double2ll_rd(x * bin_scale);
    result.y = __double2ll_rd(y * bin_scale);
  }
  output[row] = result;
}

static __global__ void build_buckets(Workspace w, const std::uint8_t* keep_row, int m) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || !keep_row[row]) return;
  const Projection value = w.projections[row];
  if (!value.support) return;
  if (value.x == LLONG_MIN) {
    // A hashed flag can only keep extra candidates. Mark the whole support
    // class so a reliable row cannot discard an uncertain parallel partner.
    atomicExch(w.uncertain_support + (value.support & (w.size - 1)), 1);
    return;
  }
  const auto key = bucket_key(value.support, value.x, value.y);
  for (int probe = 0; probe < max_probe; ++probe) {
    const auto slot = (key + probe) & (w.size - 1);
    const auto previous = atomicCAS(w.keys + slot, 0ULL, key);
    if (previous != 0 && previous != key) continue;
    const int owner = atomicCAS(w.owners + slot, -1, row);
    if (owner != -1 && owner != row) atomicExch(w.owners + slot, -2);
    return;
  }
  // Global fallback: even rows whose own insertion succeeded must be kept.
  atomicExch(w.overflow, 1);
}

template<bool KeepDeleted>
static __global__ void mark_candidates(Workspace w, const std::uint8_t* keep_row, int m) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m) return;
  w.candidates[row] = 0;
  if (!keep_row[row]) { w.candidates[row] = KeepDeleted; return; }
  const Projection value = w.projections[row];
  if (*w.overflow || !value.support ||
      w.uncertain_support[value.support & (w.size-1)]) {
    w.candidates[row] = 1; return;
  }
  // Reliable projections of identical canonical vectors differ by less than
  // 1/4 bin, so all possible peers lie in these nine buckets.
  for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) {
    const auto key = bucket_key(value.support, value.x + dx, value.y + dy);
    for (int probe = 0; probe < max_probe; ++probe) {
      const auto slot = (key + probe) & (w.size - 1);
      const auto previous = w.keys[slot];
      if (!previous) break;
      if (previous == key) {
        const int owner = w.owners[slot];
        if (owner != row && owner != -1) { w.candidates[row] = 1; return; }
        break;
      }
    }
  }
}

struct IsCandidate {
  __device__ bool operator()(std::uint8_t value) const { return value != 0; }
};

static __global__ void count_nnz(Workspace w, int m) {
  using Reduce = cub::BlockReduce<unsigned long long, 256>;
  __shared__ typename Reduce::TempStorage storage;
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const auto nnz = row < m ? static_cast<unsigned long long>(w.projections[row].length) : 0ULL;
  const auto candidate_nnz = row < m && w.candidates[row] ? nnz : 0ULL;
  const auto total = Reduce(storage).Sum(nnz); __syncthreads();
  const auto selected = Reduce(storage).Sum(candidate_nnz);
  if (!threadIdx.x) { atomicAdd(w.counts, total); atomicAdd(w.counts+1, selected); }
}

}  // namespace gpu_presolver::presolve::parallel_signature
