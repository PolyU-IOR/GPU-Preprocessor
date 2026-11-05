// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/offset.hpp"
#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_bound_strengthening.hpp"
#include "gpu_preprocessor/lp/presolve/cleanup_workspace.hpp"
#include "gpu_preprocessor/lp/detail/runtime_options.hpp"

#include "gpu_preprocessor/lp/presolve/fixed_variable_tape.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_presolve_kernels.hpp"

#include <cub/cub.cuh>
#include <thrust/functional.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_preprocessor::lp::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

template <class T>
cudaError_t bound_workspace_malloc(T** pointer, std::size_t bytes) {
  auto status = cudaMalloc(reinterpret_cast<void**>(pointer), bytes);
  if (status != cudaErrorMemoryAllocation) return status;
  if (cudaPeekAtLastError() == cudaErrorMemoryAllocation) cudaGetLastError();
  // This rule may already own live candidate/support arrays: only evict the
  // other rule's inactive scratch, never the bound workspace itself.
  cleanup_scratch::reclaim_unused();
  status = cudaStreamSynchronize(nullptr);
  if (status != cudaSuccess) return status;
  // Async apply/recovery frees can leave unused pages reserved in the pool.
  // Return those pages before retrying an ordinary cudaMalloc allocation.
  int device = -1;
  cudaMemPool_t pool = nullptr;
  status = cudaGetDevice(&device);
  if (status == cudaSuccess) status = cudaDeviceGetMemPool(&pool, device);
  if (status == cudaSuccess) status = cudaMemPoolTrimTo(pool, 0);
  if (status != cudaSuccess) return status;
  return cudaMalloc(reinterpret_cast<void**>(pointer), bytes);
}

struct BoundStrengtheningWorkspace {
  double* candidate_l = nullptr;
  double* candidate_u = nullptr;
  double* finalized_l = nullptr;
  double* finalized_u = nullptr;
  double* row_min_fin = nullptr;
  double* row_max_fin = nullptr;
  std::int32_t* row_min_neg_inf_count = nullptr;
  std::int32_t* row_max_pos_inf_count = nullptr;
  double* col_max_abs = nullptr;
  std::int32_t* record_prefix = nullptr;
  std::uint8_t* lower_changed = nullptr;
  std::uint8_t* upper_changed = nullptr;
  std::uint8_t* fixed_mask = nullptr;
  double* fixed_val = nullptr;
  std::int32_t* flags_device = nullptr;
  double* row_shift = nullptr;
  std::int32_t* row_nnz_after = nullptr;
  std::int32_t* support_l_row = nullptr;
  std::int32_t* support_u_row = nullptr;
  std::size_t row_capacity = 0;
  std::size_t col_capacity = 0;
  int owner_device = -1;
  std::uint64_t col_max_epoch = 0;
  const NnzOffset* col_max_ptr = nullptr;
  const double* col_max_values = nullptr;
  int col_max_rows = -1;

  BoundStrengtheningWorkspace() = default;
  BoundStrengtheningWorkspace(const BoundStrengtheningWorkspace&) = delete;
  BoundStrengtheningWorkspace& operator=(const BoundStrengtheningWorkspace&) = delete;

  void release_rows() noexcept {
    cudaFree(row_min_fin);
    cudaFree(row_max_fin);
    cudaFree(row_min_neg_inf_count);
    cudaFree(row_max_pos_inf_count);
    cudaFree(row_shift);
    cudaFree(row_nnz_after);
    row_min_fin = nullptr;
    row_max_fin = nullptr;
    row_min_neg_inf_count = nullptr;
    row_max_pos_inf_count = nullptr;
    row_shift = nullptr;
    row_nnz_after = nullptr;
    row_capacity = 0;
  }

  void release_cols() noexcept {
    col_max_epoch = 0;
    cudaFree(candidate_l);
    cudaFree(candidate_u);
    cudaFree(finalized_l);
    cudaFree(finalized_u);
    cudaFree(col_max_abs);
    cudaFree(record_prefix);
    cudaFree(lower_changed);
    cudaFree(upper_changed);
    cudaFree(fixed_mask);
    cudaFree(fixed_val);
    cudaFree(support_l_row);
    cudaFree(support_u_row);
    candidate_l = nullptr;
    candidate_u = nullptr;
    finalized_l = nullptr;
    finalized_u = nullptr;
    col_max_abs = nullptr;
    record_prefix = nullptr;
    lower_changed = nullptr;
    upper_changed = nullptr;
    fixed_mask = nullptr;
    fixed_val = nullptr;
    support_l_row = nullptr;
    support_u_row = nullptr;
    col_capacity = 0;
  }

  void release() noexcept {
    int restore_device = -1;
    const bool restore = owner_device >= 0 &&
                         cudaGetDevice(&restore_device) == cudaSuccess &&
                         restore_device != owner_device &&
                         cudaSetDevice(owner_device) == cudaSuccess;
    release_rows();
    release_cols();
    cudaFree(flags_device);
    flags_device = nullptr;
    owner_device = -1;
    if (restore) {
      cudaSetDevice(restore_device);
    }
  }

  void prepare_device() {
    int current_device = -1;
    throw_if_cuda_error(cudaGetDevice(&current_device),
                        "cudaGetDevice primal workspace");
    if (owner_device >= 0 && owner_device != current_device) {
      release();
    }
    owner_device = current_device;
  }

  void ensure(std::int32_t m, std::int32_t n) {
    prepare_device();
    try {
      if (flags_device == nullptr) {
        throw_if_cuda_error(bound_workspace_malloc(&flags_device, sizeof(std::int32_t) * 5),
                            "cudaMalloc primal workspace flags");
      }
      const std::size_t required_rows = static_cast<std::size_t>(m);
      if (required_rows > row_capacity) {
        release_rows();
        try {
          throw_if_cuda_error(bound_workspace_malloc(&row_min_fin, sizeof(double) * required_rows),
                              "cudaMalloc primal workspace row_min_fin");
          throw_if_cuda_error(bound_workspace_malloc(&row_max_fin, sizeof(double) * required_rows),
                              "cudaMalloc primal workspace row_max_fin");
          throw_if_cuda_error(
              bound_workspace_malloc(&row_min_neg_inf_count, sizeof(std::int32_t) * required_rows),
              "cudaMalloc primal workspace row_min_neg_inf_count");
          throw_if_cuda_error(
              bound_workspace_malloc(&row_max_pos_inf_count, sizeof(std::int32_t) * required_rows),
              "cudaMalloc primal workspace row_max_pos_inf_count");
          throw_if_cuda_error(bound_workspace_malloc(&row_shift, sizeof(double) * required_rows),
                              "cudaMalloc primal workspace row_shift");
          throw_if_cuda_error(bound_workspace_malloc(&row_nnz_after,
                                         sizeof(std::int32_t) * required_rows),
                              "cudaMalloc primal workspace row_nnz_after");
          row_capacity = required_rows;
        } catch (...) {
          release_rows();
          throw;
        }
      }
      const std::size_t required_cols = static_cast<std::size_t>(n);
      if (required_cols > col_capacity) {
        release_cols();
        try {
          throw_if_cuda_error(bound_workspace_malloc(&candidate_l, sizeof(double) * required_cols),
                              "cudaMalloc primal workspace candidate_l");
          throw_if_cuda_error(bound_workspace_malloc(&candidate_u, sizeof(double) * required_cols),
                              "cudaMalloc primal workspace candidate_u");
          throw_if_cuda_error(bound_workspace_malloc(&finalized_l, sizeof(double) * required_cols),
                              "cudaMalloc primal workspace finalized_l");
          throw_if_cuda_error(bound_workspace_malloc(&finalized_u, sizeof(double) * required_cols),
                              "cudaMalloc primal workspace finalized_u");
          throw_if_cuda_error(bound_workspace_malloc(&col_max_abs, sizeof(double) * required_cols),
                              "cudaMalloc primal workspace col_max_abs");
          throw_if_cuda_error(bound_workspace_malloc(&lower_changed, required_cols),
                              "cudaMalloc primal workspace lower_changed");
          throw_if_cuda_error(bound_workspace_malloc(&upper_changed, required_cols),
                              "cudaMalloc primal workspace upper_changed");
          throw_if_cuda_error(bound_workspace_malloc(&fixed_mask, required_cols),
                              "cudaMalloc primal workspace fixed_mask");
          throw_if_cuda_error(bound_workspace_malloc(&fixed_val, sizeof(double) * required_cols),
                              "cudaMalloc primal workspace fixed_val");
          throw_if_cuda_error(bound_workspace_malloc(&support_l_row,
                                         sizeof(std::int32_t) * required_cols),
                              "cudaMalloc primal workspace support_l_row");
          throw_if_cuda_error(bound_workspace_malloc(&support_u_row,
                                         sizeof(std::int32_t) * required_cols),
                              "cudaMalloc primal workspace support_u_row");
          col_capacity = required_cols;
        } catch (...) {
          release_cols();
          throw;
        }
      }
    } catch (...) {
      if (flags_device == nullptr) {
        owner_device = -1;
      }
      throw;
    }
  }

  ~BoundStrengtheningWorkspace() { release(); }
};

struct DeviceScratchBuffer {
  void* data = nullptr;
  std::size_t capacity = 0;
  int owner_device = -1;

  void release() noexcept {
    int restore_device = -1;
    const bool restore = owner_device >= 0 &&
                         cudaGetDevice(&restore_device) == cudaSuccess &&
                         restore_device != owner_device &&
                         cudaSetDevice(owner_device) == cudaSuccess;
    cudaFree(data);
    data = nullptr;
    capacity = 0;
    owner_device = -1;
    if (restore) {
      cudaSetDevice(restore_device);
    }
  }

  void ensure(std::size_t bytes, const char* context) {
    int current_device = -1;
    throw_if_cuda_error(cudaGetDevice(&current_device), context);
    if (owner_device >= 0 && owner_device != current_device) {
      release();
    }
    owner_device = current_device;
    if (bytes <= capacity) {
      return;
    }
    release();
    owner_device = current_device;
    try {
      throw_if_cuda_error(bound_workspace_malloc(&data, bytes), context);
    } catch (...) {
      release();
      throw;
    }
    capacity = bytes;
  }

  ~DeviceScratchBuffer() { release(); }
};

BoundStrengtheningWorkspace& bound_strengthening_workspace() {
  static thread_local BoundStrengtheningWorkspace workspace;
  return workspace;
}

thread_local bool externally_prepared_workspace = false;
thread_local unsigned active_workspace_users = 0;
struct BoundWorkspaceUse {
  BoundWorkspaceUse() { ++active_workspace_users; }
  ~BoundWorkspaceUse() { --active_workspace_users; }
  BoundWorkspaceUse(const BoundWorkspaceUse&) = delete;
  BoundWorkspaceUse& operator=(const BoundWorkspaceUse&) = delete;
};

DeviceScratchBuffer& bound_strengthening_scan_scratch() {
  static thread_local DeviceScratchBuffer scratch;
  return scratch;
}

template <typename T>
void inclusive_scan_i32(T* values, std::int32_t n, const char* context) {
  if (n <= 0) {
    return;
  }
  DeviceScratchBuffer& scratch = bound_strengthening_scan_scratch();
  void* temp_storage = nullptr;
  std::size_t temp_bytes = 0;
  throw_if_cuda_error(
      cub::DeviceScan::InclusiveSum(temp_storage, temp_bytes, values, values, n), context);
  scratch.ensure(temp_bytes, context);
  throw_if_cuda_error(
      cub::DeviceScan::InclusiveSum(scratch.data, temp_bytes, values, values, n), context);
}

void append_postsolve_record(PostsolveTape& tape,
                             PostsolveReductionType type,
                             const std::vector<std::int32_t>& indices,
                             const std::vector<double>& vals,
                             PostsolveDualMode dual_mode) {
  tape.types.push_back(static_cast<std::int32_t>(type));
  tape.indices.insert(tape.indices.end(), indices.begin(), indices.end());
  tape.vals.insert(tape.vals.end(), vals.begin(), vals.end());
  tape.index_starts.push_back(checked_nnz_offset(tape.indices.size()));
  tape.value_starts.push_back(checked_nnz_offset(tape.vals.size()));
  tape.dual_modes.push_back(static_cast<std::uint8_t>(dual_mode));
}

__device__ double atomic_max_double(double* address, double value) {
  auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed = 0;
  do {
    assumed = old;
    const double current = __longlong_as_double(static_cast<long long>(assumed));
    if (current >= value) {
      break;
    }
    old = atomicCAS(
        address_as_ull,
        assumed,
        static_cast<unsigned long long int>(__double_as_longlong(value)));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
}

__device__ double atomic_min_double(double* address, double value) {
  auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed = 0;
  do {
    assumed = old;
    const double current = __longlong_as_double(static_cast<long long>(assumed));
    if (current <= value) {
      break;
    }
    old = atomicCAS(
        address_as_ull,
        assumed,
        static_cast<unsigned long long int>(__double_as_longlong(value)));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
}

__device__ void _term_interval_bound_strengthening(double aij,
                                                  double lj,
                                                  double uj,
                                                  double* term_min,
                                                  double* term_max) {
  if (aij >= 0.0) {
    *term_min = aij * lj;
    *term_max = aij * uj;
  } else {
    *term_min = aij * uj;
    *term_max = aij * lj;
  }
}

__device__ double _row_residual_min_from_summary(double row_min_fin,
                                                 std::int32_t row_min_neg_inf_count,
                                                 double term_min) {
  if (isfinite(term_min)) {
    return row_min_neg_inf_count == 0 ? (row_min_fin - term_min) : -INFINITY;
  }
  if (term_min < 0.0) {
    return row_min_neg_inf_count > 1 ? -INFINITY : row_min_fin;
  }
  return row_min_fin;
}

__device__ double _row_residual_max_from_summary(double row_max_fin,
                                                 std::int32_t row_max_pos_inf_count,
                                                 double term_max) {
  if (isfinite(term_max)) {
    return row_max_pos_inf_count == 0 ? (row_max_fin - term_max) : INFINITY;
  }
  if (term_max > 0.0) {
    return row_max_pos_inf_count > 1 ? INFINITY : row_max_fin;
  }
  return row_max_fin;
}

__global__ void _kernel_bound_strengthening_candidates(std::int32_t* infeasible_flag,
                                                      double* candidate_l,
                                                      double* candidate_u,
                                                      const double* row_min_fin,
                                                      const double* row_max_fin,
                                                      const std::int32_t* row_min_neg_inf_count,
                                                      const std::int32_t* row_max_pos_inf_count,
                                                      const std::uint8_t* keep_row,
    const std::uint8_t* keep_col,
                                                      const std::int32_t* row_nnz,
                                                      const double* AL,
                                                      const double* AU,
                                                      const double* l_cur,
                                                      const double* u_cur,
                                                      const NnzOffset* row_ptr,
                                                      const std::int32_t* col_val,
                                                      const double* nz_val,
                                                      double zero_tol,
                                                      std::int32_t m, int max_length) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < m && (max_length == 0 || (keep_col ? row_nnz[i] : row_ptr[i + 1] - row_ptr[i]) <= max_length) && keep_row[i] != std::uint8_t{0} && row_nnz[i] > 1) {
    const NnzOffset row_start = row_ptr[i];
    const NnzOffset row_stop = row_ptr[i + 1];
    const double lower_i = AL[i];
    const double upper_i = AU[i];
    const double row_min_fin_i = row_min_fin[i];
    const double row_max_fin_i = row_max_fin[i];
    const std::int32_t row_min_neg_inf_count_i = row_min_neg_inf_count[i];
    const std::int32_t row_max_pos_inf_count_i = row_max_pos_inf_count[i];
    for (NnzOffset p = row_start; p < row_stop; ++p) {
      const std::int32_t j = col_val[p];
      if (keep_col && !keep_col[j]) continue;
      const double a = nz_val[p];
      if (fabs(a) <= zero_tol) {
        continue;
      }

      const double old_l = l_cur[j];
      const double old_u = u_cur[j];
      double term_min = 0.0;
      double term_max = 0.0;
      _term_interval_bound_strengthening(a, old_l, old_u, &term_min, &term_max);

      const double rest_min = _row_residual_min_from_summary(
          row_min_fin_i,
          row_min_neg_inf_count_i,
          term_min);
      const double rest_max = _row_residual_max_from_summary(
          row_max_fin_i,
          row_max_pos_inf_count_i,
          term_max);

      double implied_l = -INFINITY;
      double implied_u = INFINITY;
      if (a > 0.0) {
        if (isfinite(lower_i) && isfinite(rest_max)) {
          implied_l = (lower_i - rest_max) / a;
        }
        if (isfinite(upper_i) && isfinite(rest_min)) {
          implied_u = (upper_i - rest_min) / a;
        }
      } else {
        if (isfinite(upper_i) && isfinite(rest_min)) {
          implied_l = (upper_i - rest_min) / a;
        }
        if (isfinite(lower_i) && isfinite(rest_max)) {
          implied_u = (lower_i - rest_max) / a;
        }
      }

      double new_l = old_l;
      double new_u = old_u;
      if (isfinite(implied_l)) {
        new_l = fmax(new_l, implied_l);
      }
      if (isfinite(implied_u)) {
        new_u = fmin(new_u, implied_u);
      }
      if (new_l > old_l) {
        atomic_max_double(&candidate_l[j], new_l);
      }
      if (new_u < old_u) {
        atomic_min_double(&candidate_u[j], new_u);
      }
    }
  }
  (void)infeasible_flag;
}

__global__ void _kernel_bound_strengthening_candidates_block(
    std::int32_t* infeasible_flag,
    double* candidate_l,
    double* candidate_u,
    const double* row_min_fin,
    const double* row_max_fin,
    const std::int32_t* row_min_neg_inf_count,
    const std::int32_t* row_max_pos_inf_count,
    const std::uint8_t* keep_row,
    const std::uint8_t* keep_col,
    const std::int32_t* row_nnz,
    const double* AL,
    const double* AU,
    const double* l_cur,
    const double* u_cur,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    double zero_tol,
    std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x);
  if (i >= m || keep_row[i] == std::uint8_t{0} || row_nnz[i] <= 1) {
    return;
  }

  const NnzOffset row_start = row_ptr[i];
  const NnzOffset row_stop = row_ptr[i + 1];
  const double lower_i = AL[i];
  const double upper_i = AU[i];
  const double row_min_fin_i = row_min_fin[i];
  const double row_max_fin_i = row_max_fin[i];
  const std::int32_t row_min_neg_inf_count_i = row_min_neg_inf_count[i];
  const std::int32_t row_max_pos_inf_count_i = row_max_pos_inf_count[i];
  for (NnzOffset p = row_start + static_cast<NnzOffset>(threadIdx.x);
       p < row_stop;
       p += static_cast<std::int32_t>(blockDim.x)) {
    const std::int32_t j = col_val[p];
      if (keep_col && !keep_col[j]) continue;
    const double a = nz_val[p];
    if (fabs(a) <= zero_tol) {
      continue;
    }

    const double old_l = l_cur[j];
    const double old_u = u_cur[j];
    double term_min = 0.0;
    double term_max = 0.0;
    _term_interval_bound_strengthening(a, old_l, old_u, &term_min, &term_max);
    const double rest_min = _row_residual_min_from_summary(
        row_min_fin_i, row_min_neg_inf_count_i, term_min);
    const double rest_max = _row_residual_max_from_summary(
        row_max_fin_i, row_max_pos_inf_count_i, term_max);

    double implied_l = -INFINITY;
    double implied_u = INFINITY;
    if (a > 0.0) {
      if (isfinite(lower_i) && isfinite(rest_max)) {
        implied_l = (lower_i - rest_max) / a;
      }
      if (isfinite(upper_i) && isfinite(rest_min)) {
        implied_u = (upper_i - rest_min) / a;
      }
    } else {
      if (isfinite(upper_i) && isfinite(rest_min)) {
        implied_l = (upper_i - rest_min) / a;
      }
      if (isfinite(lower_i) && isfinite(rest_max)) {
        implied_u = (lower_i - rest_max) / a;
      }
    }

    double new_l = old_l;
    double new_u = old_u;
    if (isfinite(implied_l)) {
      new_l = fmax(new_l, implied_l);
    }
    if (isfinite(implied_u)) {
      new_u = fmin(new_u, implied_u);
    }
    if (new_l > old_l) {
      atomic_max_double(&candidate_l[j], new_l);
    }
    if (new_u < old_u) {
      atomic_min_double(&candidate_u[j], new_u);
    }
  }
  (void)infeasible_flag;
}

template<int Width>
__global__ void _kernel_bound_strengthening_candidates_grouped(
    std::int32_t* infeasible_flag,
    double* candidate_l,
    double* candidate_u,
    const double* row_min_fin,
    const double* row_max_fin,
    const std::int32_t* row_min_neg_inf_count,
    const std::int32_t* row_max_pos_inf_count,
    const std::uint8_t* keep_row,
    const std::uint8_t* keep_col,
    const std::int32_t* row_nnz,
    const double* AL,
    const double* AU,
    const double* l_cur,
    const double* u_cur,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    double zero_tol,
    std::int32_t m, const int* rows, const int* count) {
  const int lane = Width == 32 ? (threadIdx.x & 31) : threadIdx.x;
  const int group = Width == 32 ? blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32 : blockIdx.x;
  const NnzOffset stride = Width == 32 ? gridDim.x * (blockDim.x / 32) : gridDim.x;
  for (NnzOffset q = group; q < *count; q += stride) {
  const std::int32_t i = rows[q];
  if (i >= m || keep_row[i] == std::uint8_t{0} || row_nnz[i] <= 1) {
    continue;
  }

  const NnzOffset row_start = row_ptr[i];
  const NnzOffset row_stop = row_ptr[i + 1];
  const double lower_i = AL[i];
  const double upper_i = AU[i];
  const double row_min_fin_i = row_min_fin[i];
  const double row_max_fin_i = row_max_fin[i];
  const std::int32_t row_min_neg_inf_count_i = row_min_neg_inf_count[i];
  const std::int32_t row_max_pos_inf_count_i = row_max_pos_inf_count[i];
  for (NnzOffset p = row_start + lane;
       p < row_stop;
       p += Width) {
    const std::int32_t j = col_val[p];
      if (keep_col && !keep_col[j]) continue;
    const double a = nz_val[p];
    if (fabs(a) <= zero_tol) {
      continue;
    }

    const double old_l = l_cur[j];
    const double old_u = u_cur[j];
    double term_min = 0.0;
    double term_max = 0.0;
    _term_interval_bound_strengthening(a, old_l, old_u, &term_min, &term_max);
    const double rest_min = _row_residual_min_from_summary(
        row_min_fin_i, row_min_neg_inf_count_i, term_min);
    const double rest_max = _row_residual_max_from_summary(
        row_max_fin_i, row_max_pos_inf_count_i, term_max);

    double implied_l = -INFINITY;
    double implied_u = INFINITY;
    if (a > 0.0) {
      if (isfinite(lower_i) && isfinite(rest_max)) {
        implied_l = (lower_i - rest_max) / a;
      }
      if (isfinite(upper_i) && isfinite(rest_min)) {
        implied_u = (upper_i - rest_min) / a;
      }
    } else {
      if (isfinite(upper_i) && isfinite(rest_min)) {
        implied_l = (upper_i - rest_min) / a;
      }
      if (isfinite(lower_i) && isfinite(rest_max)) {
        implied_u = (lower_i - rest_max) / a;
      }
    }

    double new_l = old_l;
    double new_u = old_u;
    if (isfinite(implied_l)) {
      new_l = fmax(new_l, implied_l);
    }
    if (isfinite(implied_u)) {
      new_u = fmin(new_u, implied_u);
    }
    if (new_l > old_l) {
      atomic_max_double(&candidate_l[j], new_l);
    }
    if (new_u < old_u) {
      atomic_min_double(&candidate_u[j], new_u);
    }
  }
  }
  (void)infeasible_flag;
}

__global__ void _kernel_finalize_bound_strengthening(std::int32_t* flags,
                                                    double* finalized_l,
                                                    double* finalized_u,
                                                    std::uint8_t* lower_changed,
                                                    std::uint8_t* upper_changed,
                                                    std::uint8_t* fixed_mask,
                                                    double* fixed_val,
                                                    const double* candidate_l,
                                                    const double* candidate_u,
                                                    const double* old_l,
                                                    const double* old_u,
                                                    const std::uint8_t* keep_col,
                                                    const double* reference_l,
                                                    const double* reference_u,
                                                    const double* c,
                                                    const double* col_max_abs,
                                                    std::int32_t preserve_free_zero_cost,
                                                    double zero_tol,
                                                    double feas_tol,
                                                    double upper_relaxation_factor,
                                                    std::int32_t fixing_only,
                                                    std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j < n) {
    if (!keep_col[j]) {
      finalized_l[j] = old_l[j];
      finalized_u[j] = old_u[j];
      lower_changed[j] = upper_changed[j] = fixed_mask[j] = 0;
      fixed_val[j] = 0.0;
      return;
    }
    const double lj = old_l[j];
    const double uj = old_u[j];
    double cand_l = candidate_l[j];
    double cand_u = candidate_u[j];
    const double vmax = col_max_abs[j];

    double new_l = lj;
    double new_u = uj;
    std::uint8_t lower_changed_j = std::uint8_t{0};
    std::uint8_t upper_changed_j = std::uint8_t{0};
    std::uint8_t fixed_j = std::uint8_t{0};
    double fixed_at = 0.0;

    double exact_l = lj;
    double exact_u = uj;
    if (isfinite(cand_l)) {
      exact_l = fmax(exact_l, cand_l);
    }
    if (isfinite(cand_u)) {
      exact_u = fmin(exact_u, cand_u);
    }
    if (exact_l > exact_u + feas_tol) {
      atomicMax(&flags[0], 1);
      return;
    }

    // Preserve both bounds of every exactly zero-cost column.
    // Keep the infeasibility check above, but do not tighten or near-fix here.
    // Other presolve rules remain unchanged.
    if (c != nullptr && c[j] == 0.0) {
      finalized_l[j] = lj;
      finalized_u[j] = uj;
      lower_changed[j] = upper_changed[j] = fixed_mask[j] = 0;
      fixed_val[j] = 0.0;
      return;
    }

    if (fixing_only != 0) {
      // Only a finite zero-width intersection supplies a new fixing.
      // Rejecting other candidates leaves the pre-batch box and its rows intact.
      const bool fixed = isfinite(exact_l) && exact_l == exact_u &&
                         (exact_l > lj || exact_u < uj);
      finalized_l[j] = fixed ? exact_l : lj;
      finalized_u[j] = fixed ? exact_u : uj;
      lower_changed[j] = upper_changed[j] = 0;
      fixed_mask[j] = fixed ? std::uint8_t{1} : std::uint8_t{0};
      fixed_val[j] = fixed ? exact_l : 0.0;
      if (fixed) atomicMax(&flags[2], 1);
      return;
    }

    // Retain useful propagation (including near-fixing and all one-sided
    // tightenings), but avoid materializing a wide artificial box around a
    // free zero-cost auxiliary.  The rows still enforce that interval, so
    // avoiding a doubly finite explicit box preserves the LP while avoiding
    // an expensive projection geometry for first-order solvers.
    // old_l/old_u are updated in-place between fixed-point rounds.  Use the
    // bounds at plan entry to remember that a column started free; otherwise
    // a lower bound found in one round followed by an upper bound in the next
    // would bypass the redundant-box guard.
    const bool was_free = !isfinite(reference_l[j]) && !isfinite(reference_u[j]);
    const bool candidate_is_box = isfinite(cand_l) && isfinite(cand_u);
    const bool zero_cost = c != nullptr && fabs(c[j]) <= zero_tol;
    const bool nearly_fixed = candidate_is_box &&
                              (cand_u <= cand_l ||
                               (cand_u - cand_l) * vmax <= feas_tol);
    if (preserve_free_zero_cost != 0 && was_free && candidate_is_box &&
        zero_cost && !nearly_fixed) {
      cand_l = lj;
      cand_u = uj;
    }

    if (isfinite(cand_l) && cand_l > lj) {
      if (isfinite(uj)) {
        if (cand_l >= uj + feas_tol) {
          atomicMax(&flags[0], 1);
          return;
        }
        if (cand_l >= uj || (uj - cand_l) * vmax <= feas_tol) {
          fixed_j = std::uint8_t{1};
          fixed_at = uj;
          new_l = uj;
          new_u = uj;
        }
      }
    }

    if (fixed_j == std::uint8_t{0} && isfinite(cand_l) && cand_l > lj) {
      const bool finite_lb_tightening =
          !isfinite(lj) ||
          ((cand_l - lj > feas_tol * 1.0e4) &&
           (cand_l - lj > 1.0e-2 * fabs(lj)));
      if (finite_lb_tightening) {
        if (cand_l != nearbyint(cand_l)) {
          cand_l -= 0.5 * feas_tol * fabs(cand_l);
        }
        new_l = cand_l;
        lower_changed_j = std::uint8_t{1};
      }
    }

    if (fixed_j == std::uint8_t{0} && isfinite(cand_u) && cand_u < uj) {
      if (isfinite(new_l)) {
        if (cand_u <= new_l - feas_tol) {
          atomicMax(&flags[0], 1);
          return;
        }
        if (cand_u <= new_l || (cand_u - new_l) * vmax <= feas_tol) {
          fixed_j = std::uint8_t{1};
          fixed_at = new_l;
          new_u = new_l;
        }
      }
    }

    if (fixed_j == std::uint8_t{0} && isfinite(cand_u) && cand_u < uj) {
      const bool finite_ub_tightening =
          !isfinite(uj) ||
          ((uj - cand_u > feas_tol * 1.0e4) &&
           (uj - cand_u > 1.0e-2 * fabs(uj)));
      if (finite_ub_tightening) {
        if (cand_u != nearbyint(cand_u)) {
          cand_u += upper_relaxation_factor * feas_tol * fabs(cand_u);
        }
        new_u = cand_u;
        upper_changed_j = std::uint8_t{1};
      }
    }

    if (fixed_j != std::uint8_t{0}) {
      lower_changed_j = std::uint8_t{0};
      upper_changed_j = std::uint8_t{0};
      atomicMax(&flags[2], 1);
    }
    if (lower_changed_j != std::uint8_t{0} || upper_changed_j != std::uint8_t{0}) {
      atomicMax(&flags[1], 1);
    }

    finalized_l[j] = new_l;
    finalized_u[j] = new_u;
    lower_changed[j] = lower_changed_j;
    upper_changed[j] = upper_changed_j;
    fixed_mask[j] = fixed_j;
    fixed_val[j] = fixed_at;
  }
}

__global__ void _kernel_capture_bound_changes(std::uint8_t* lower, std::uint8_t* upper,
    const double* old_l, const double* old_u, const double* new_l, const double* new_u, int n) {
  const auto j=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(j<std::uint64_t(n)) {lower[j]=new_l[j]>old_l[j];upper[j]=new_u[j]<old_u[j];}
}

__global__ void _kernel_apply_primal_bounds_and_fixed(std::uint8_t* keep_col,
                                                      double* new_l,
                                                      double* new_u,
                                                      const double* finalized_l,
                                                      const double* finalized_u,
                                                      const std::uint8_t* fixed_mask,
                                                      std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j < n) {
    new_l[j] = finalized_l[j];
    new_u[j] = finalized_u[j];
    if (fixed_mask[j] != std::uint8_t{0}) {
      keep_col[j] = std::uint8_t{0};
    }
  }
}

template<int Width>
__global__ void _kernel_capture_bound_strengthening_support_rows_by_col(
    std::int32_t* support_l_row,
    std::int32_t* support_u_row,
    const std::uint8_t* lower_changed,
    const std::uint8_t* upper_changed,
    const double* candidate_l,
    const double* candidate_u,
    const double* row_min_fin,
    const double* row_max_fin,
    const std::int32_t* row_min_neg_inf_count,
    const std::int32_t* row_max_pos_inf_count,
    const std::uint8_t* keep_row,
    const std::int32_t* row_nnz,
    const double* AL,
    const double* AU,
    const double* l_cur,
    const double* u_cur,
    const NnzOffset* at_row_ptr,
    const std::int32_t* at_col_val,
    const double* at_nz_val,
    double zero_tol,
    double tol,
    double support_tol,
    std::int32_t n, int max_length, const int* rows, const int* count) {
  const int lane=Width==1 ? 0 : (Width==32 ? threadIdx.x&31 : threadIdx.x);
  const int group=Width==1 ? blockIdx.x*blockDim.x+threadIdx.x : (Width==32 ? blockIdx.x*8+threadIdx.x/32 : blockIdx.x);
  const NnzOffset stride=Width==1 ? gridDim.x*blockDim.x : (Width==32 ? gridDim.x*8 : gridDim.x);
  using Reduce=cub::BlockReduce<int,256>;
  __shared__ typename Reduce::TempStorage storage;
  for(NnzOffset q=group;q<(Width==1 ? n : *count);q+=stride) {
  const int j=Width==1 ? q : rows[q];
  if constexpr(Width==1) {
    if(max_length && at_row_ptr[j+1]-at_row_ptr[j]>max_length) continue;
  }
  std::int32_t support_l = INT_MAX;
  std::int32_t support_u = INT_MAX;
  const bool need_l = lower_changed[j] != std::uint8_t{0};
  const bool need_u = upper_changed[j] != std::uint8_t{0};
  if (need_l || need_u) {
    const double old_l = l_cur[j];
    const double old_u = u_cur[j];
    const double cand_l = candidate_l[j];
    const double cand_u = candidate_u[j];
    for (NnzOffset p = at_row_ptr[j]+lane; p < at_row_ptr[j + 1]; p+=Width) {
      const std::int32_t i = at_col_val[p];
      if (keep_row[i] == std::uint8_t{0} || row_nnz[i] <= 1) {
        continue;
      }
      const double a = at_nz_val[p];
      if (fabs(a) <= zero_tol) {
        continue;
      }
      double term_min = 0.0;
      double term_max = 0.0;
      _term_interval_bound_strengthening(a, old_l, old_u, &term_min, &term_max);
      const double rest_min = _row_residual_min_from_summary(
          row_min_fin[i], row_min_neg_inf_count[i], term_min);
      const double rest_max = _row_residual_max_from_summary(
          row_max_fin[i], row_max_pos_inf_count[i], term_max);
      double implied_l = -INFINITY;
      double implied_u = INFINITY;
      if (a > 0.0) {
        if (isfinite(AL[i]) && isfinite(rest_max)) {
          implied_l = (AL[i] - rest_max) / a;
        }
        if (isfinite(AU[i]) && isfinite(rest_min)) {
          implied_u = (AU[i] - rest_min) / a;
        }
      } else {
        if (isfinite(AU[i]) && isfinite(rest_min)) {
          implied_l = (AU[i] - rest_min) / a;
        }
        if (isfinite(AL[i]) && isfinite(rest_max)) {
          implied_u = (AL[i] - rest_max) / a;
        }
      }
      if (need_l && cand_l > old_l + tol && isfinite(implied_l) &&
          fabs(implied_l - cand_l) <= support_tol) {
        support_l = min(support_l, i);
      }
      if (need_u && cand_u < old_u - tol && isfinite(implied_u) &&
          fabs(implied_u - cand_u) <= support_tol) {
        support_u = min(support_u, i);
      }
    }
  }
  if constexpr(Width==32) {
    for(int off=16;off;off/=2) {
      support_l=min(support_l,__shfl_down_sync(0xffffffffu,support_l,off));
      support_u=min(support_u,__shfl_down_sync(0xffffffffu,support_u,off));
    }
  } else if constexpr(Width==256) {
    support_l=Reduce(storage).Reduce(support_l,thrust::minimum<int>());__syncthreads();
    support_u=Reduce(storage).Reduce(support_u,thrust::minimum<int>());__syncthreads();
  }
  if(lane==0) { support_l_row[j]=support_l;support_u_row[j]=support_u; }
  if constexpr(Width==256) __syncthreads();
  }
}

__global__ void _kernel_primal_fixed_row_shift(double* row_shift,
                                               const std::uint8_t* fixed_mask,
                                               const double* fixed_val,
                                               const std::uint8_t* keep_row,
                                               const NnzOffset* at_row_ptr,
                                               const std::int32_t* at_col_val,
                                               const double* at_nz_val,
                                               std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j < n && fixed_mask[j] != std::uint8_t{0}) {
    const double vj = fixed_val[j];
    const NnzOffset p_start = at_row_ptr[j];
    const NnzOffset p_stop = at_row_ptr[j + 1];
    for (NnzOffset p = p_start; p < p_stop; ++p) {
      const std::int32_t row = at_col_val[p];
      if (keep_row[row] != std::uint8_t{0}) {
        atomicAdd(&row_shift[row], at_nz_val[p] * vj);
      }
    }
  }
}

__global__ void _kernel_apply_row_shift(double* AL,
                                        double* AU,
                                        const double* row_shift,
                                        std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < m) {
    AL[i] -= row_shift[i];
    AU[i] -= row_shift[i];
  }
}

__global__ void _kernel_row_nnz_after_fixed(std::int32_t* row_nnz_after,
                                            const std::uint8_t* keep_row,
                                            const std::uint8_t* keep_col_after,
                                            const NnzOffset* row_ptr,
                                            const std::int32_t* col_val,
                                            std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < m) {
    std::int32_t count = 0;
    if (keep_row[i] != std::uint8_t{0}) {
      const NnzOffset p_start = row_ptr[i];
      const NnzOffset p_stop = row_ptr[i + 1];
      for (NnzOffset p = p_start; p < p_stop; ++p) {
        if (keep_col_after[col_val[p]] != std::uint8_t{0}) {
          ++count;
        }
      }
    }
    row_nnz_after[i] = count;
  }
}

__global__ void _kernel_remove_feasible_empty_rows_after_fixed(std::int32_t* flags,
                                                               std::uint8_t* keep_row,
                                                               const double* AL,
                                                               const double* AU,
                                                               const std::int32_t* row_nnz_after,
                                                               double feas_tol,
                                                               std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < m && keep_row[i] != std::uint8_t{0} && row_nnz_after[i] == 0) {
    if (AL[i] <= feas_tol && AU[i] >= -feas_tol) {
      keep_row[i] = std::uint8_t{0};
      atomicMax(&flags[4], 1);
    } else {
      atomicMax(&flags[3], 1);
    }
  }
}

}  // namespace

void prepare_bound_strengthening_workspace(std::int32_t rows, std::int32_t cols) {
  if (externally_prepared_workspace) {
    throw std::logic_error("implied variable bounds workspace already prepared");
  }
  bound_strengthening_workspace().ensure(rows, cols);
  externally_prepared_workspace = true;
}

bool bound_strengthening_workspace_prepared() noexcept {
  return externally_prepared_workspace;
}

bool bound_strengthening_workspace_in_use() noexcept {
  return active_workspace_users != 0;
}

void release_bound_strengthening_workspace() noexcept {
  externally_prepared_workspace = false;
  bound_strengthening_workspace().release();
  bound_strengthening_scan_scratch().release();
}

void apply_rule_bound_strengthening(PresolvePlanGpu& plan,
                                   const LPInfoGpu& lp,
                                   const PresolveStatsGpu& stats,
                                   const PresolveParams& pparams) {
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }

  const std::int32_t m = lp.A.rows;
  const std::int32_t n = lp.A.cols;
  if (m == 0 || n == 0) {
    return;
  }

  BoundWorkspaceUse workspace_use;
  BoundStrengtheningWorkspace& workspace = bound_strengthening_workspace();
  workspace.ensure(m, n);
  double* candidate_l = workspace.candidate_l;
  double* candidate_u = workspace.candidate_u;
  double* finalized_l = workspace.finalized_l;
  double* finalized_u = workspace.finalized_u;
  double* row_min_fin = workspace.row_min_fin;
  double* row_max_fin = workspace.row_max_fin;
  std::int32_t* row_min_neg_inf_count = workspace.row_min_neg_inf_count;
  std::int32_t* row_max_pos_inf_count = workspace.row_max_pos_inf_count;
  double* col_max_abs = workspace.col_max_abs;
  std::uint8_t* lower_changed = workspace.lower_changed;
  std::uint8_t* upper_changed = workspace.upper_changed;
  std::uint8_t* fixed_mask = workspace.fixed_mask;
  double* fixed_val = workspace.fixed_val;
  std::int32_t* flags_device = workspace.flags_device;
  double* row_shift = workspace.row_shift;
  std::int32_t* row_nnz_after = workspace.row_nnz_after;

  throw_if_cuda_error(cudaMemcpy(candidate_l, plan.new_l, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy primal candidate_l");
  throw_if_cuda_error(cudaMemcpy(candidate_u, plan.new_u, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy primal candidate_u");
  if (!adaptive_reuse_enabled()) {
    throw_if_cuda_error(cudaMemcpy(finalized_l, plan.new_l, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice), "cudaMemcpy primal finalized_l");
    throw_if_cuda_error(cudaMemcpy(finalized_u, plan.new_u, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice), "cudaMemcpy primal finalized_u");
  }
  throw_if_cuda_error(cudaMemset(flags_device, 0, sizeof(std::int32_t) * 5), "cudaMemset primal flags");
  throw_if_cuda_error(cudaMemset(row_shift, 0, sizeof(double) * static_cast<std::size_t>(m)), "cudaMemset primal row_shift");

  compute_row_activity_summary(
      row_min_fin,
      row_max_fin,
      row_min_neg_inf_count,
      row_max_pos_inf_count,
      lp.A,
      plan.new_l,
      plan.new_u,
      pparams.zero_tol, plan.keep_row_mask,
      stats.live_masks ? plan.keep_col_mask : nullptr,
      stats.live_masks ? stats.row_nnz : nullptr);

  const int blocks_m = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const bool use_block_per_row =
      static_cast<std::int64_t>(lp.A.nnz) >
      32LL * static_cast<std::int64_t>(m);
  if (adaptive_row_dispatch_enabled()) {
    const auto& groups = prepare_adaptive_row_groups(lp.A, stats.live_masks ? stats.row_nnz : nullptr);
    _kernel_bound_strengthening_candidates<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
        flags_device, candidate_l, candidate_u, row_min_fin, row_max_fin,
        row_min_neg_inf_count, row_max_pos_inf_count, plan.keep_row_mask,
        stats.live_masks ? plan.keep_col_mask : nullptr, stats.row_nnz, plan.new_AL, plan.new_AU, plan.new_l, plan.new_u,
        lp.A.rowPtr, lp.A.colVal, lp.A.nzVal, pparams.zero_tol, m, ADAPTIVE_SHORT_ROW);
    _kernel_bound_strengthening_candidates_grouped<32>
        <<<std::min(1024, (m + 7) / 8), GPU_PRESOLVE_THREADS>>>(
        flags_device, candidate_l, candidate_u, row_min_fin, row_max_fin,
        row_min_neg_inf_count, row_max_pos_inf_count, plan.keep_row_mask,
        stats.live_masks ? plan.keep_col_mask : nullptr, stats.row_nnz, plan.new_AL, plan.new_AU, plan.new_l, plan.new_u,
        lp.A.rowPtr, lp.A.colVal, lp.A.nzVal, pparams.zero_tol, m,
        groups.medium, groups.counts);
    _kernel_bound_strengthening_candidates_grouped<256>
        <<<std::min(256, m), GPU_PRESOLVE_THREADS>>>(
        flags_device, candidate_l, candidate_u, row_min_fin, row_max_fin,
        row_min_neg_inf_count, row_max_pos_inf_count, plan.keep_row_mask,
        stats.live_masks ? plan.keep_col_mask : nullptr, stats.row_nnz, plan.new_AL, plan.new_AU, plan.new_l, plan.new_u,
        lp.A.rowPtr, lp.A.colVal, lp.A.nzVal, pparams.zero_tol, m,
        groups.long_rows, groups.counts + 1);
    throw_if_cuda_error(cudaGetLastError(), "adaptive bound strengthening candidates");
  } else if (use_block_per_row) {
    _kernel_bound_strengthening_candidates_block<<<m, GPU_PRESOLVE_THREADS>>>(
        flags_device, candidate_l, candidate_u, row_min_fin, row_max_fin,
        row_min_neg_inf_count, row_max_pos_inf_count, plan.keep_row_mask,
        stats.live_masks ? plan.keep_col_mask : nullptr, stats.row_nnz, plan.new_AL, plan.new_AU, plan.new_l, plan.new_u,
        lp.A.rowPtr, lp.A.colVal, lp.A.nzVal, pparams.zero_tol, m);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_bound_strengthening_candidates_block");
  } else {
    _kernel_bound_strengthening_candidates<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
        flags_device, candidate_l, candidate_u, row_min_fin, row_max_fin,
        row_min_neg_inf_count, row_max_pos_inf_count, plan.keep_row_mask,
        stats.live_masks ? plan.keep_col_mask : nullptr, stats.row_nnz, plan.new_AL, plan.new_AU, plan.new_l, plan.new_u,
        lp.A.rowPtr, lp.A.colVal, lp.A.nzVal, pparams.zero_tol, m, 0);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_bound_strengthening_candidates");
  }

  const auto epoch = matrix_analysis_epoch();
  if (stats.live_masks || !epoch || workspace.col_max_epoch != epoch || workspace.col_max_ptr != lp.AT.rowPtr ||
      workspace.col_max_values != lp.AT.nzVal || workspace.col_max_rows != n) {
    compute_col_max_abs(col_max_abs, lp.AT, stats.live_masks ? plan.keep_row_mask : nullptr);
    workspace.col_max_epoch = stats.live_masks ? 0 : epoch;
    workspace.col_max_ptr = lp.AT.rowPtr;
    workspace.col_max_values = lp.AT.nzVal;
    workspace.col_max_rows = n;
  }

  const int blocks_n = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_finalize_bound_strengthening<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      flags_device,
      finalized_l,
      finalized_u,
      lower_changed,
      upper_changed,
      fixed_mask,
      fixed_val,
      candidate_l,
      candidate_u,
      plan.new_l,
      plan.new_u,
      plan.keep_col_mask,
      plan.propagation_reference_l ? plan.propagation_reference_l : (lp.l ? lp.l : plan.new_l),
      plan.propagation_reference_u ? plan.propagation_reference_u : (lp.u ? lp.u : plan.new_u),
      plan.new_c,
      col_max_abs,
      pparams.bound_strengthening_preserve_free_zero_cost ? 1 : 0,
      pparams.zero_tol,
      pparams.feasibility_tol,
      pparams.bound_strengthening_upper_relaxation_factor,
      runtime::env_enabled("GPUPREPROCESSOR_BOUND_STRENGTHENING_FIX_ONLY") ? 1 : 0,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_finalize_bound_strengthening");

  std::int32_t flags[5] = {0, 0, 0, 0, 0};
  throw_if_cuda_error(cudaMemcpy(flags, flags_device, sizeof(flags), cudaMemcpyDeviceToHost),
                      "cudaMemcpy primal flags after finalize");
  if (flags[0] != 0) {
    plan.has_infeasible = true;
    return;
  }

  if (flags[1] != 0 || flags[2] != 0) {
    if (pparams.record_postsolve_tape) {
      _kernel_capture_bound_changes<<<blocks_n,GPU_PRESOLVE_THREADS>>>(lower_changed,upper_changed,
          plan.new_l,plan.new_u,finalized_l,finalized_u,n);
      throw_if_cuda_error(cudaGetLastError(), "capture bound changes including fixes");
      std::int32_t* support_l_row = workspace.support_l_row;
      std::int32_t* support_u_row = workspace.support_u_row;
      const double support_tol = 0.0;
      _kernel_capture_bound_strengthening_support_rows_by_col<1>
          <<<blocks_n, GPU_PRESOLVE_THREADS>>>(
          support_l_row,
          support_u_row,
          lower_changed,
          upper_changed,
          candidate_l,
          candidate_u,
          row_min_fin,
          row_max_fin,
          row_min_neg_inf_count,
          row_max_pos_inf_count,
          plan.keep_row_mask,
          stats.row_nnz,
          plan.new_AL,
          plan.new_AU,
          plan.new_l,
          plan.new_u,
          lp.AT.rowPtr,
          lp.AT.colVal,
          lp.AT.nzVal,
          pparams.zero_tol,
          0.0,
          support_tol,
          n, adaptive_gather_enabled() ? ADAPTIVE_SHORT_ROW : 0, nullptr, nullptr);
      if(adaptive_gather_enabled()) {
        const auto& groups=prepare_adaptive_row_groups(lp.AT);
        _kernel_capture_bound_strengthening_support_rows_by_col<32>
            <<<std::min(1024,(n+7)/8),256>>>(
          support_l_row,
          support_u_row,
          lower_changed,
          upper_changed,
          candidate_l,
          candidate_u,
          row_min_fin,
          row_max_fin,
          row_min_neg_inf_count,
          row_max_pos_inf_count,
          plan.keep_row_mask,
          stats.row_nnz,
          plan.new_AL,
          plan.new_AU,
          plan.new_l,
          plan.new_u,
          lp.AT.rowPtr,
          lp.AT.colVal,
          lp.AT.nzVal,
          pparams.zero_tol,
          0.0,
          support_tol,
          n, 0, groups.medium, groups.counts);
        _kernel_capture_bound_strengthening_support_rows_by_col<256>
            <<<std::min(256,n),256>>>(
          support_l_row,
          support_u_row,
          lower_changed,
          upper_changed,
          candidate_l,
          candidate_u,
          row_min_fin,
          row_max_fin,
          row_min_neg_inf_count,
          row_max_pos_inf_count,
          plan.keep_row_mask,
          stats.row_nnz,
          plan.new_AL,
          plan.new_AU,
          plan.new_l,
          plan.new_u,
          lp.AT.rowPtr,
          lp.AT.colVal,
          lp.AT.nzVal,
          pparams.zero_tol,
          0.0,
          support_tol,
          n, 0, groups.long_rows, groups.counts+1);
      }
      throw_if_cuda_error(
          cudaGetLastError(),
          "_kernel_capture_bound_strengthening_support_rows_by_col");
      throw_if_cuda_error(cudaDeviceSynchronize(), "capture primal support rows synchronize");

      append_bound_batch_recovery(plan.tape,lp.A,n,plan.keep_col_mask,
          plan.new_l,plan.new_u,finalized_l,finalized_u,candidate_l,candidate_u,
          support_l_row,support_u_row);
    }
    _kernel_apply_primal_bounds_and_fixed<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
        plan.keep_col_mask,
        plan.new_l,
        plan.new_u,
        finalized_l,
        finalized_u,
        fixed_mask,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_apply_primal_bounds_and_fixed");

    if (flags[2] != 0) {
      _kernel_primal_fixed_row_shift<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
          row_shift,
          fixed_mask,
          fixed_val,
          plan.keep_row_mask,
          lp.AT.rowPtr,
          lp.AT.colVal,
          lp.AT.nzVal,
          n);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_primal_fixed_row_shift");
      _kernel_apply_row_shift<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
          plan.new_AL,
          plan.new_AU,
          row_shift,
          m);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_apply_row_shift");
      _kernel_row_nnz_after_fixed<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
          row_nnz_after,
          plan.keep_row_mask,
          plan.keep_col_mask,
          lp.A.rowPtr,
          lp.A.colVal,
          m);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_row_nnz_after_fixed");
      _kernel_remove_feasible_empty_rows_after_fixed<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
          flags_device,
          plan.keep_row_mask,
          plan.new_AL,
          plan.new_AU,
          row_nnz_after,
          pparams.feasibility_tol,
          m);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_remove_feasible_empty_rows_after_fixed");
    }

    throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_bound_strengthening synchronize");
    throw_if_cuda_error(cudaMemcpy(flags, flags_device, sizeof(flags), cudaMemcpyDeviceToHost),
                        "cudaMemcpy primal flags after apply");
    if (flags[3] != 0) {
      plan.has_infeasible = true;
    } else {
      const double obj_delta = flags[2] != 0
          ? append_compact_fixed_variable_tape_from_device(
                pparams.record_postsolve_tape ? &plan.tape : nullptr,
                fixed_mask,
                nullptr,
                fixed_val,
                plan.keep_row_mask,
                plan.new_c,
                lp.AT,
                n,
                "primal packed fixed-col tape/objective")
          : 0.0;
      plan.obj_constant_delta += obj_delta;
      plan.has_row_action = plan.has_row_action || flags[2] != 0;
      plan.has_col_action = plan.has_col_action || flags[2] != 0;
      plan.has_change = true;
    }
  }

}

}  // namespace gpu_preprocessor::lp::presolve
