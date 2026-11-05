// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gpu_preprocessor::lp::folding::detail {

inline void check_chunk_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<class T> struct FoldingBuffer {
  T* pointer = nullptr;
  FoldingBuffer() = default;
  explicit FoldingBuffer(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::length_error("folding buffer size overflow");
    if (count) check_chunk_cuda(cudaMalloc(&pointer, count * sizeof(T)));
  }
  ~FoldingBuffer() { cudaFree(pointer); }
  FoldingBuffer(const FoldingBuffer&) = delete;
  FoldingBuffer& operator=(const FoldingBuffer&) = delete;
  FoldingBuffer(FoldingBuffer&& other) noexcept : pointer(std::exchange(other.pointer, nullptr)) {}
  FoldingBuffer& operator=(FoldingBuffer&& other) noexcept {
    std::swap(pointer, other.pointer);
    return *this;
  }
};

struct FoldingEntries {
  FoldingBuffer<std::int64_t> keys;
  FoldingBuffer<double> values;
  std::size_t count = 0;
  FoldingEntries() = default;
  explicit FoldingEntries(std::size_t capacity) : keys(capacity), values(capacity) {}
};

// Emit a contiguous original-nnz interval into scratch, then combine equal
// color-pair keys. max_unique bounds the key domain, not the original nnz.
// Retain even zero partial sums: cancellation and tolerance are resolved only
// after every chunk has contributed. Allocation failures release all scratch.
template<class Emit>
FoldingEntries chunked_fold_entries(std::int64_t nnz, std::size_t chunk_size,
                                    std::size_t max_unique, Emit emit) {
  if (nnz < 0 || chunk_size == 0 || (nnz && max_unique == 0))
    throw std::invalid_argument("invalid chunked folding dimensions");
  FoldingEntries result;
  if (nnz == 0) return result;
  chunk_size = std::min<std::uint64_t>(chunk_size, nnz);
  FoldingEntries input(chunk_size), unique(std::min(chunk_size, max_unique));
  for (std::int64_t begin = 0; begin < nnz;) {
    const std::size_t count = std::min<std::uint64_t>(chunk_size, nnz - begin);
    emit(begin, count, input.keys.pointer, input.values.pointer);
    check_chunk_cuda(cudaGetLastError());
    auto keys = thrust::device_pointer_cast(input.keys.pointer);
    auto values = thrust::device_pointer_cast(input.values.pointer);
    thrust::sort_by_key(keys, keys + count, values);
    auto unique_keys = thrust::device_pointer_cast(unique.keys.pointer);
    auto unique_values = thrust::device_pointer_cast(unique.values.pointer);
    const auto end = thrust::reduce_by_key(keys, keys + count, values, unique_keys, unique_values);
    unique.count = static_cast<std::size_t>(end.first - unique_keys);
    if (result.count == 0) {
      result = FoldingEntries(unique.count);
      check_chunk_cuda(cudaMemcpy(result.keys.pointer, unique.keys.pointer,
                                  unique.count * sizeof(std::int64_t), cudaMemcpyDeviceToDevice));
      check_chunk_cuda(cudaMemcpy(result.values.pointer, unique.values.pointer,
                                  unique.count * sizeof(double), cudaMemcpyDeviceToDevice));
      result.count = unique.count;
    } else {
      if (result.count > std::numeric_limits<std::size_t>::max() - unique.count)
        throw std::length_error("folding merge size overflow");
      const auto merged_count = result.count + unique.count;
      FoldingEntries merged(merged_count), next(std::min(merged_count, max_unique));
      auto old_keys = thrust::device_pointer_cast(result.keys.pointer);
      auto old_values = thrust::device_pointer_cast(result.values.pointer);
      auto merged_keys = thrust::device_pointer_cast(merged.keys.pointer);
      auto merged_values = thrust::device_pointer_cast(merged.values.pointer);
      thrust::merge_by_key(old_keys, old_keys + result.count,
                           unique_keys, unique_keys + unique.count,
                           old_values, unique_values, merged_keys, merged_values);
      const auto next_end = thrust::reduce_by_key(merged_keys, merged_keys + merged_count,
          merged_values, thrust::device_pointer_cast(next.keys.pointer),
          thrust::device_pointer_cast(next.values.pointer));
      next.count = static_cast<std::size_t>(next_end.first - thrust::device_pointer_cast(next.keys.pointer));
      result = std::move(next);
    }
    begin += count;
  }
  return result;
}

}  // namespace gpu_preprocessor::lp::folding::detail
