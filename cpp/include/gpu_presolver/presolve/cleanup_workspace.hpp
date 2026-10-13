#pragma once
#include "gpu_presolver/detail/runtime_options.hpp"
#include <cuda_runtime.h>
#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>

namespace gpu_presolver::presolve {
// Implemented in a CUDA translation unit; CUB's query cannot be compiled by
// the host-only integration file.
std::size_t cleanup_reduction_temp_bytes(const double* input,
                                         double* output, int count);
std::size_t cleanup_selection_temp_bytes(const double* input,
                                         const std::uint8_t* flags,
                                         double* output,
                                         int* selected_count,
                                         int count);
namespace cleanup_scratch {
inline constexpr std::size_t deferred_live_nnz_slot = 15;
inline constexpr std::size_t deferred_entry_cols_slot = 16;
inline constexpr std::size_t deferred_reference_bounds_slot = 17;
inline void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
// Shared slots: two column double arrays, a mask, two row double arrays,
// flags, reduction scratch/result, selection output/count/scratch, and
// auxiliary masks/counts for dual fixing and redundant-row cleanup. Slots
// 15..17 hold scheduler state across consecutive deferred apply boundaries.
// No output/tape owns these bytes.
struct Pool {
  std::array<void*, 18> ptr{};
  std::array<std::size_t, 18> capacity{};
  int device = -1;
  Pool() { check(cudaGetDevice(&device)); }
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;
  void* reserve(std::size_t slot, std::size_t bytes) {
    if (slot >= ptr.size()) throw std::out_of_range("cleanup scratch slot");
    if (!bytes) bytes=1;
    if (bytes > capacity[slot]) {
      check(cudaFree(ptr[slot])); ptr[slot]=nullptr; capacity[slot]=0;
      check(cudaMalloc(&ptr[slot],bytes)); capacity[slot]=bytes;
    }
    return ptr[slot];
  }
  ~Pool() {
    int previous=-1;
    if (cudaGetDevice(&previous)!=cudaSuccess) return;
    if (previous!=device && cudaSetDevice(device)!=cudaSuccess) return;
    for (void* p:ptr) if(p) cudaFree(p);
    if(previous!=device) cudaSetDevice(previous);
  }
};
inline thread_local Pool* active_pool=nullptr;
// A prepared pool is claimed by the next presolve scope on this thread only.
inline thread_local Pool* prepared_pool=nullptr;
} // namespace cleanup_scratch

// Allocates the cleanup scratch before the caller starts its presolve timer.
// The inner presolve scope borrows this pool once; nested presolves still own
// independent pools. Keep this object alive through the presolve call.
class PreparedCleanupWorkspace {
  cleanup_scratch::Pool* previous_=cleanup_scratch::prepared_pool;
  std::unique_ptr<cleanup_scratch::Pool> owned_;
 public:
  PreparedCleanupWorkspace(std::size_t rows, std::size_t cols) {
    if (!runtime::options().reuse_cleanup_workspace) return;
    if (rows>std::numeric_limits<std::size_t>::max()/sizeof(double) ||
        cols>std::numeric_limits<std::size_t>::max()/sizeof(double))
      throw std::length_error("cleanup scratch dimensions");
    owned_=std::make_unique<cleanup_scratch::Pool>();
    owned_->reserve(0,cols*sizeof(double));
    owned_->reserve(1,cols*sizeof(double));
    owned_->reserve(2,std::max(rows,cols));
    owned_->reserve(3,rows*sizeof(double));
    owned_->reserve(4,rows*sizeof(double));
    owned_->reserve(5,4*sizeof(int));
    owned_->reserve(7,sizeof(double));
    owned_->reserve(11,std::max(rows,cols));
    owned_->reserve(12,std::max(rows,sizeof(int)));
    owned_->reserve(13,rows*sizeof(unsigned int));
    owned_->reserve(14,rows);
    owned_->reserve(cleanup_scratch::deferred_live_nnz_slot,
                    sizeof(unsigned long long));
    owned_->reserve(cleanup_scratch::deferred_entry_cols_slot, cols);
    owned_->reserve(cleanup_scratch::deferred_reference_bounds_slot,
                    2 * sizeof(double) * cols);
    cleanup_scratch::prepared_pool=owned_.get();
  }
  bool enabled() const { return owned_!=nullptr; }
  double* reduction_result() const {
    return owned_ ? static_cast<double*>(owned_->ptr[7]) : nullptr;
  }
  void reserve_reduction_scratch(std::size_t bytes) {
    if (owned_) owned_->reserve(6,bytes);
  }
  void prepare_selection(const double* input, int count) {
    if (!owned_ || count <= 0) return;
    auto* output = static_cast<double*>(owned_->reserve(8, sizeof(double) * std::size_t(count)));
    auto* selected_count = static_cast<int*>(owned_->reserve(9, sizeof(int)));
    const auto bytes = cleanup_selection_temp_bytes(
        input, static_cast<const std::uint8_t*>(owned_->ptr[2]),
        output, selected_count, count);
    owned_->reserve(10, bytes);
  }
  ~PreparedCleanupWorkspace() { cleanup_scratch::prepared_pool=previous_; }
  PreparedCleanupWorkspace(const PreparedCleanupWorkspace&)=delete;
  PreparedCleanupWorkspace& operator=(const PreparedCleanupWorkspace&)=delete;
};

// Owns a cache for exactly one presolve, including exceptional/terminal exits.
// Nested presolves get independent storage; the outer cache is restored.
class CleanupWorkspaceScope {
  cleanup_scratch::Pool* previous_=cleanup_scratch::active_pool;
  cleanup_scratch::Pool* owned_=nullptr;
 public:
  CleanupWorkspaceScope() {
    if (runtime::options().reuse_cleanup_workspace) {
      if (cleanup_scratch::prepared_pool) {
        cleanup_scratch::active_pool=cleanup_scratch::prepared_pool;
        cleanup_scratch::prepared_pool=nullptr;
      } else {
        owned_=new cleanup_scratch::Pool;
        cleanup_scratch::active_pool=owned_;
      }
    } else {
      cleanup_scratch::active_pool=nullptr;
    }
  }
  ~CleanupWorkspaceScope() { cleanup_scratch::active_pool=previous_; delete owned_; }
  CleanupWorkspaceScope(const CleanupWorkspaceScope&)=delete;
  CleanupWorkspaceScope& operator=(const CleanupWorkspaceScope&)=delete;
};

// Without a presolve scope (e.g. direct rule unit tests), retain local lifetime.
// Rules run sequentially on the default stream and retain their existing
// synchronization. Reduction helpers borrow slots 6/7; selection borrows 8..10.
class CleanupScratchFrame {
  cleanup_scratch::Pool* pool_=cleanup_scratch::active_pool;
  std::array<void*,15> local_{};
 public:
  template<class T> T* get(std::size_t slot,std::size_t count) {
    if(slot>=local_.size()) throw std::out_of_range("cleanup scratch slot");
    if(count>std::numeric_limits<std::size_t>::max()/sizeof(T))
      throw std::length_error("cleanup scratch byte count");
    const auto bytes=count*sizeof(T);
    if(pool_) return static_cast<T*>(pool_->reserve(slot,bytes));
    if(local_[slot]) throw std::logic_error("cleanup scratch slot already borrowed");
    cleanup_scratch::check(cudaMalloc(&local_[slot],bytes?bytes:1));
    return static_cast<T*>(local_[slot]);
  }
  ~CleanupScratchFrame() { for(void* p:local_) if(p) cudaFree(p); }
  CleanupScratchFrame()=default;
  CleanupScratchFrame(const CleanupScratchFrame&)=delete;
  CleanupScratchFrame& operator=(const CleanupScratchFrame&)=delete;
};
} // namespace gpu_presolver::presolve
