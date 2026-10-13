#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace gpu_presolver::presolve {
namespace duplicate_rule_scratch {

struct Storage {
  void* ptr = nullptr;
  std::size_t capacity = 0;
};

inline thread_local Storage* prepared = nullptr;
inline thread_local Storage* active = nullptr;

inline void check(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

}  // namespace duplicate_rule_scratch

// Both rules execute sequentially, so their temporary arrays can occupy the
// same allocation. The preparation and release surround the presolve timer.
class PreparedDuplicateRuleWorkspace {
  duplicate_rule_scratch::Storage storage_;
  duplicate_rule_scratch::Storage* previous_ = duplicate_rule_scratch::prepared;
  int device_ = -1;

 public:
  explicit PreparedDuplicateRuleWorkspace(std::size_t bytes) {
    if (bytes == 0) return;
    duplicate_rule_scratch::check(cudaGetDevice(&device_), "duplicate workspace device");
    duplicate_rule_scratch::check(cudaMalloc(&storage_.ptr, bytes),
                                  "prepare duplicate rule workspace");
    storage_.capacity = bytes;
    duplicate_rule_scratch::prepared = &storage_;
  }
  ~PreparedDuplicateRuleWorkspace() {
    duplicate_rule_scratch::prepared = previous_;
    if (storage_.ptr) {
      int current = -1;
      if (cudaGetDevice(&current) == cudaSuccess) {
        if (current == device_ || cudaSetDevice(device_) == cudaSuccess) {
          cudaFree(storage_.ptr);
          if (current != device_) cudaSetDevice(current);
        }
      }
    }
  }
  PreparedDuplicateRuleWorkspace(const PreparedDuplicateRuleWorkspace&) = delete;
  PreparedDuplicateRuleWorkspace& operator=(const PreparedDuplicateRuleWorkspace&) = delete;
};

// Claim the prepared buffer only for this presolve invocation. A nested
// presolve without its own preparation falls back to local rule allocations.
class DuplicateRuleWorkspaceScope {
  duplicate_rule_scratch::Storage* previous_active_ = duplicate_rule_scratch::active;
  duplicate_rule_scratch::Storage* previous_prepared_ = duplicate_rule_scratch::prepared;

 public:
  DuplicateRuleWorkspaceScope() {
    duplicate_rule_scratch::active = duplicate_rule_scratch::prepared;
    duplicate_rule_scratch::prepared = nullptr;
  }
  ~DuplicateRuleWorkspaceScope() {
    duplicate_rule_scratch::active = previous_active_;
    duplicate_rule_scratch::prepared = previous_prepared_;
  }
  DuplicateRuleWorkspaceScope(const DuplicateRuleWorkspaceScope&) = delete;
  DuplicateRuleWorkspaceScope& operator=(const DuplicateRuleWorkspaceScope&) = delete;
};

class DuplicateRuleScratchFrame {
  void* local_ = nullptr;

 public:
  void* get(std::size_t bytes) {
    auto* pool = duplicate_rule_scratch::active;
    if (pool && bytes <= pool->capacity) return pool->ptr;
    duplicate_rule_scratch::check(cudaMalloc(&local_, bytes),
                                  "allocate duplicate rule scratch");
    return local_;
  }
  ~DuplicateRuleScratchFrame() { if (local_) cudaFree(local_); }
  DuplicateRuleScratchFrame(const DuplicateRuleScratchFrame&) = delete;
  DuplicateRuleScratchFrame& operator=(const DuplicateRuleScratchFrame&) = delete;
  DuplicateRuleScratchFrame() = default;
};

}  // namespace gpu_presolver::presolve
