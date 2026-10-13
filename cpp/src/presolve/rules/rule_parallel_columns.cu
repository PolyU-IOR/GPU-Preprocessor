#include "parallel_signature_common.cuh"
#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/duplicate_rule_workspace.hpp"
#include "gpu_presolver/presolve/rules/rule_parallel_columns.hpp"

#include "gpu_presolver/presolve/fixed_col_tape.hpp"

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include "gpu_presolver/presolve/gpu_presolve_kernels.hpp"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;
constexpr unsigned long long FNV_OFFSET = 0xcbf29ce484222325ULL;
constexpr unsigned long long FNV_PRIME = 0x100000001b3ULL;

template <typename T>
std::size_t reserve_scratch(std::size_t& bytes, std::size_t count) {
  const std::size_t alignment = alignof(T);
  bytes = (bytes + alignment - 1) & ~(alignment - 1);
  const std::size_t offset = bytes;
  bytes += sizeof(T) * count;
  return offset;
}

template <typename T>
T* scratch_at(void* storage, std::size_t offset) {
  return reinterpret_cast<T*>(static_cast<unsigned char*>(storage) + offset);
}

struct HashColLess {
  __host__ __device__ bool operator()(const thrust::tuple<unsigned long long, std::int32_t>& lhs,
                                      const thrust::tuple<unsigned long long, std::int32_t>& rhs) const {
    const unsigned long long lhs_hash = thrust::get<0>(lhs);
    const unsigned long long rhs_hash = thrust::get<0>(rhs);
    if (lhs_hash != rhs_hash) {
      return lhs_hash < rhs_hash;
    }
    return thrust::get<1>(lhs) < thrust::get<1>(rhs);
  }
};

struct ParallelColumnTapeRecord {
  std::int32_t deleted_col;
  std::int32_t kept_col;
  double ratio;
  double deleted_l;
  double deleted_u;
  double kept_l;
  double kept_u;
};

struct RecoveryOrder {
  __host__ __device__ bool operator()(const ParallelColumnTapeRecord &a,
                                      const ParallelColumnTapeRecord &b) const {
    return a.deleted_col < b.deleted_col;
  }
};
__global__ void pack_recovery_segment(int *types, unsigned char *modes, NnzOffset *is,
                                      NnzOffset *vs, int *indices, double *vals,
                                      const ParallelColumnTapeRecord *records, int count) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k > count)
    return;
  is[k] = NnzOffset(2) * k;
  vs[k] = NnzOffset(5) * k;
  if (k == count)
    return;
  const auto r = records[k];
  types[k] = int(PostsolveReductionType::ParallelColumn);
  modes[k] = unsigned(PostsolveDualMode::Minimal);
  indices[NnzOffset(2) * k + 0] = r.deleted_col;
  indices[NnzOffset(2) * k + 1] = r.kept_col;
  vals[NnzOffset(5) * k + 0] = r.ratio;
  vals[NnzOffset(5) * k + 1] = r.deleted_l;
  vals[NnzOffset(5) * k + 2] = r.deleted_u;
  vals[NnzOffset(5) * k + 3] = r.kept_l;
  vals[NnzOffset(5) * k + 4] = r.kept_u;
}

__global__ void _kernel_pack_parallel_column_tape(ParallelColumnTapeRecord* records,
                                               std::int32_t* count,
                                               const std::uint8_t* col_delete,
                                               const std::int32_t* merge_to,
                                               const double* merge_ratio,
                                               const double* merge_from_l,
                                               const double* merge_from_u,
                                               const double* merge_to_l,
                                               const double* merge_to_u,
                                               std::int32_t n) {
  const std::int32_t col =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (col < n && col_delete[col] != std::uint8_t{0}) {
    const std::int32_t slot = atomicAdd(count, 1);
    records[slot] = ParallelColumnTapeRecord{col,
                                          merge_to[col],
                                          merge_ratio[col],
                                          merge_from_l[col],
                                          merge_from_u[col],
                                          merge_to_l[col],
                                          merge_to_u[col]};
  }
}

void throw_if_cuda_error(cudaError_t status, const char* context);

void append_postsolve_record(PostsolveTape& tape,
                             PostsolveReductionType type,
                             const std::int32_t* indices,
                             std::size_t index_count,
                             const double* vals,
                             std::size_t value_count,
                             PostsolveDualMode dual_mode) {
  tape.types.push_back(static_cast<std::int32_t>(type));
  tape.indices.insert(tape.indices.end(), indices, indices + index_count);
  tape.vals.insert(tape.vals.end(), vals, vals + value_count);
  tape.index_starts.push_back(static_cast<NnzOffset>(tape.indices.size()));
  tape.value_starts.push_back(static_cast<NnzOffset>(tape.vals.size()));
  tape.dual_modes.push_back(static_cast<std::uint8_t>(dual_mode));
}

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

__global__ void _kernel_parallel_column_sort_keys(unsigned long long* keys,
                                               const NnzOffset* row_ptr,
                                               const std::int32_t* col_val,
                                               std::int32_t rows) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row < rows) {
    const unsigned long long row_key = static_cast<unsigned long long>(static_cast<std::uint32_t>(row)) << 32;
    for (NnzOffset p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
      keys[p] = row_key | static_cast<unsigned long long>(static_cast<std::uint32_t>(col_val[p]));
    }
  }
}

__global__ void _kernel_parallel_column_unpack_keys(std::int32_t* col_val,
                                                 const unsigned long long* keys,
                                                 NnzOffset nnz) {
  const NnzOffset p = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (p < nnz) {
    col_val[p] = static_cast<std::int32_t>(keys[p] & 0xffffffffULL);
  }
}

__device__ unsigned long long _parallel_column_hash_mix(unsigned long long h,
                                                     unsigned long long x) {
  return (h ^ x) * FNV_PRIME;
}

__device__ unsigned long long _double_bits(double value) {
  return static_cast<unsigned long long>(__double_as_longlong(value));
}

__global__ void _kernel_parallel_column_hashes(unsigned long long* col_hash,
                                            const std::uint8_t* keep_col,
                                            const std::uint8_t* keep_row,
                                            const NnzOffset* row_ptr,
                                            const std::int32_t* row_idx,
                                            const double* row_val,
                                            double coeff_tol,
                                            std::int32_t n,
                                            const std::uint8_t* candidates = nullptr) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j < n) {
    if (candidates && !candidates[j]) return;
    if (keep_col[j] == std::uint8_t{0}) {
      col_hash[j] = static_cast<unsigned long long>(j);
      return;
    }
    const NnzOffset start_j = row_ptr[j];
    const NnzOffset stop_j = row_ptr[j + 1];
    std::int32_t live_len = 0;
    double pivot = 0.0;
    bool pivot_set = false;
    for (NnzOffset p = start_j; p < stop_j; ++p) {
      const std::int32_t row = row_idx[p];
      if (keep_row[row] != std::uint8_t{0}) {
        const double a = row_val[p];
        if (fabs(a) > coeff_tol) {
          ++live_len;
          if (!pivot_set) {
            pivot = a;
            pivot_set = true;
          }
        }
      }
    }
    if (!pivot_set || live_len == 0) {
      col_hash[j] = _parallel_column_hash_mix(FNV_OFFSET, static_cast<unsigned long long>(j));
      return;
    }
    unsigned long long h = _parallel_column_hash_mix(FNV_OFFSET, static_cast<unsigned long long>(live_len));
    for (NnzOffset p = start_j; p < stop_j; ++p) {
      const std::int32_t row = row_idx[p];
      if (keep_row[row] != std::uint8_t{0}) {
        const double a = row_val[p];
        if (fabs(a) > coeff_tol) {
          h = _parallel_column_hash_mix(h, static_cast<unsigned long long>(row));
          h = _parallel_column_hash_mix(h, _double_bits(a / pivot));
        }
      }
    }
    col_hash[j] = h;
  }
}

__device__ NnzOffset _next_live_parallel_column_entry(NnzOffset ptr,
                                                      NnzOffset stop,
                                                      const std::int32_t* row_idx,
                                                      const double* row_val,
                                                      const std::uint8_t* keep_row,
                                                      double zero_tol) {
  NnzOffset cur = ptr;
  while (cur < stop) {
    const std::int32_t row = row_idx[cur];
    if (keep_row[row] != std::uint8_t{0}) {
      const double a = row_val[cur];
      if (fabs(a) > zero_tol) {
        return cur;
      }
    }
    ++cur;
  }
  return stop;
}

__device__ bool _parallel_column_ratio(std::int32_t j,
                                    std::int32_t k,
                                    const NnzOffset* row_ptr,
                                    const std::int32_t* row_idx,
                                    const double* row_val,
                                    const std::uint8_t* keep_row,
                                    double zero_tol,
                                    double coeff_tol,
                                    double* ratio_out) {
  NnzOffset ptr_j = row_ptr[j];
  const NnzOffset stop_j = row_ptr[j + 1];
  NnzOffset ptr_k = row_ptr[k];
  const NnzOffset stop_k = row_ptr[k + 1];
  double ratio = 0.0;
  bool ratio_set = false;

  for (;;) {
    ptr_j = _next_live_parallel_column_entry(ptr_j, stop_j, row_idx, row_val, keep_row, zero_tol);
    ptr_k = _next_live_parallel_column_entry(ptr_k, stop_k, row_idx, row_val, keep_row, zero_tol);
    if (ptr_j >= stop_j || ptr_k >= stop_k) {
      break;
    }
    const std::int32_t row_j = row_idx[ptr_j];
    const std::int32_t row_k = row_idx[ptr_k];
    if (row_j != row_k) {
      return false;
    }
    const double a_j = row_val[ptr_j];
    const double a_k = row_val[ptr_k];
    if (!ratio_set) {
      if (fabs(a_j) <= zero_tol) {
        return false;
      }
      ratio = a_k / a_j;
      if (fabs(ratio) <= zero_tol || !isfinite(ratio)) {
        return false;
      }
      ratio_set = true;
    }
    // A fixed absolute tolerance can erase meaningful small coefficients.
    // In particular, (-1,-eps,...) and (1,-eps,...) are not antiparallel,
    // even when 2*eps < bound_tol. Check each entry at its own scale so a
    // hash collision never turns those columns into a valid merge.
    const double expected = ratio * a_j;
    const double scale = fmax(fabs(a_k), fabs(expected));
    if (!isfinite(expected) || fabs(a_k - expected) > coeff_tol * scale) {
      return false;
    }
    ++ptr_j;
    ++ptr_k;
  }

  ptr_j = _next_live_parallel_column_entry(ptr_j, stop_j, row_idx, row_val, keep_row, zero_tol);
  ptr_k = _next_live_parallel_column_entry(ptr_k, stop_k, row_idx, row_val, keep_row, zero_tol);
  if (ptr_j < stop_j || ptr_k < stop_k || !ratio_set) {
    return false;
  }
  *ratio_out = ratio;
  return true;
}

// Check long candidate columns concurrently. The group kernel still owns all
// bound, deletion, and recovery decisions in its original column order.
// State 0 means not checked, 1 means proportional, and 2 means not proportional.
__global__ void _kernel_preverify_parallel_columns(
    std::uint8_t* state, std::int32_t* verified_partner, double* verified_ratio,
    const unsigned long long* sorted_hash, const std::int32_t* sorted_cols,
    const std::uint8_t* keep_col, const std::uint8_t* keep_row,
    const double* c,
    const NnzOffset* row_ptr, const std::int32_t* row_idx, const double* row_val,
    double zero_tol, double coeff_tol, double obj_tol, std::int32_t n) {
  const std::int32_t b =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (b >= n) return;
  const unsigned long long hash = sorted_hash[b];
  std::int32_t lo = 0;
  std::int32_t hi = b;
  while (lo < hi) {
    const std::int32_t mid = lo + (hi - lo) / 2;
    if (sorted_hash[mid] < hash) lo = mid + 1;
    else hi = mid;
  }
  const std::int32_t s = lo;
  if (b == s) return;
  const std::int32_t k = sorted_cols[b];
  if (!keep_col[k] || row_ptr[k + 1] - row_ptr[k] < 1024) return;
  std::int32_t partner = -1;
  for (std::int32_t a = s; a < b; ++a) {
    const std::int32_t j = sorted_cols[a];
    if (!keep_col[j] || row_ptr[j + 1] - row_ptr[j] < 1024) continue;
    const NnzOffset first_j = _next_live_parallel_column_entry(
        row_ptr[j], row_ptr[j + 1], row_idx, row_val, keep_row, zero_tol);
    const NnzOffset first_k = _next_live_parallel_column_entry(
        row_ptr[k], row_ptr[k + 1], row_idx, row_val, keep_row, zero_tol);
    if (first_j >= row_ptr[j + 1] || first_k >= row_ptr[k + 1]) continue;
    const double first_ratio = row_val[first_k] / row_val[first_j];
    if (isfinite(first_ratio) &&
        fabs(c[k] - first_ratio * c[j]) <= obj_tol) {
      partner = a;
      break;
    }
  }
  if (partner < 0) return;
  const std::int32_t j = sorted_cols[partner];
  double ratio = 0.0;
  const bool proportional = _parallel_column_ratio(
      j, k, row_ptr, row_idx, row_val, keep_row,
      zero_tol, coeff_tol, &ratio);
  if (proportional) verified_ratio[b] = ratio;
  verified_partner[b] = partner;
  state[b] = proportional ? std::uint8_t{1} : std::uint8_t{2};
}

__device__ double _merged_lower_bound_parallel_columns(double target_l,
                                                    double target_u,
                                                    double source_l,
                                                    double source_u,
                                                    double ratio) {
  (void)target_u;
  if (ratio > 0.0) {
    if (isfinite(target_l) && isfinite(source_l)) {
      return target_l + ratio * source_l;
    }
  } else if (isfinite(target_l) && isfinite(source_u)) {
    return target_l + ratio * source_u;
  }
  return -INFINITY;
}

__device__ double _merged_upper_bound_parallel_columns(double target_l,
                                                    double target_u,
                                                    double source_l,
                                                    double source_u,
                                                    double ratio) {
  (void)target_l;
  if (ratio > 0.0) {
    if (isfinite(target_u) && isfinite(source_u)) {
      return target_u + ratio * source_u;
    }
  } else if (isfinite(target_u) && isfinite(source_l)) {
    return target_u + ratio * source_l;
  }
  return INFINITY;
}

__global__ void _kernel_count_small_high_ratio_parallel_groups(
    std::int32_t* count,
    const unsigned long long* sorted_hash,
    const std::int32_t* sorted_cols,
    const std::uint8_t* keep_col,
    const std::uint8_t* keep_row,
    const double* c,
    const NnzOffset* row_ptr,
    const std::int32_t* row_idx,
    const double* row_val,
    double zero_tol,
    double coeff_tol,
    double obj_tol,
    double max_abs_ratio,
    std::int32_t max_group_size,
    std::int32_t stop_after,
    std::int32_t n) {
  const std::int32_t s =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (s >= n || atomicAdd(count, 0) >= stop_after) {
    return;
  }
  const unsigned long long hs = sorted_hash[s];
  if (s > 0 && sorted_hash[s - 1] == hs) {
    return;
  }
  std::int32_t e = s;
  while (e + 1 < n && sorted_hash[e + 1] == hs) {
    ++e;
  }
  const std::int32_t group_size = e - s + 1;
  if (group_size < 2 || group_size > max_group_size) {
    return;
  }
  for (std::int32_t a = s; a < e; ++a) {
    const std::int32_t j = sorted_cols[a];
    if (keep_col[j] == std::uint8_t{0}) {
      continue;
    }
    for (std::int32_t b = a + 1; b <= e; ++b) {
      const std::int32_t k = sorted_cols[b];
      if (keep_col[k] == std::uint8_t{0}) {
        continue;
      }
      double ratio = 0.0;
      if (!_parallel_column_ratio(j, k, row_ptr, row_idx, row_val, keep_row,
                               zero_tol, coeff_tol, &ratio)) {
        continue;
      }
      if (fabs(c[k] - ratio * c[j]) > obj_tol) {
        continue;
      }
      const double abs_ratio = fabs(ratio);
      if (max_abs_ratio < 1.0 || abs_ratio > max_abs_ratio ||
          abs_ratio * max_abs_ratio < 1.0) {
        atomicAdd(count, 1);
        return;
      }
    }
  }
}

__global__ void _kernel_parallel_column_groups(std::int32_t* status_flag,
                                            std::uint8_t* col_delete,
                                            std::uint8_t* fixed_mask,
                                            double* fixed_val,
                                            std::int32_t* merge_to,
                                            double* merge_ratio,
                                            double* merge_from_l,
                                            double* merge_from_u,
                                            double* merge_to_l,
                                            double* merge_to_u,
                                            const std::uint8_t* preverified_state,
                                            const std::int32_t* preverified_partner,
                                            const double* preverified_ratio,
                                            const unsigned long long* sorted_hash,
                                            const std::int32_t* sorted_cols,
                                            const std::uint8_t* keep_col,
                                            const std::uint8_t* keep_row,
                                            const double* c,
                                            double* l,
                                            double* u,
                                            const NnzOffset* row_ptr,
                                            const std::int32_t* row_idx,
                                            const double* row_val,
                                            double zero_tol,
                                            double coeff_tol,
                                            double obj_tol,
                                            double max_abs_ratio,
                                            const std::int32_t* scale_guard_group_count,
                                            std::int32_t scale_guard_min_groups,
                                            std::int32_t n, bool bounded_objective_prefilter) {
  const std::int32_t s = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (s < n) {
    const unsigned long long hs = sorted_hash[s];
    if (s > 0 && sorted_hash[s - 1] == hs) {
      return;
    }
    std::int32_t e = s;
    while (e + 1 < n && sorted_hash[e + 1] == hs) {
      ++e;
    }
    if (e <= s) {
      return;
    }

    for (std::int32_t a = s; a < e; ++a) {
      const std::int32_t j = sorted_cols[a];
      if (keep_col[j] == std::uint8_t{0} || col_delete[j] != std::uint8_t{0} || fixed_mask[j] != std::uint8_t{0}) {
        continue;
      }
      for (std::int32_t b = a + 1; b <= e; ++b) {
        const std::int32_t k = sorted_cols[b];
        if (keep_col[k] == std::uint8_t{0} || col_delete[k] != std::uint8_t{0} || fixed_mask[k] != std::uint8_t{0}) {
          continue;
        }
        // If both variables have finite bounds and their objective ratio
        // differs, none of this rule's merge/fix branches can act. Reject that
        // pair before scanning both whole columns. Use the exact same first
        // live entries, division and objective test as the full verifier.
        if (bounded_objective_prefilter && isfinite(l[j]) && isfinite(u[j]) &&
            isfinite(l[k]) && isfinite(u[k])) {
          const auto first_j = _next_live_parallel_column_entry(row_ptr[j], row_ptr[j+1], row_idx, row_val, keep_row, zero_tol);
          const auto first_k = _next_live_parallel_column_entry(row_ptr[k], row_ptr[k+1], row_idx, row_val, keep_row, zero_tol);
          if (first_j < row_ptr[j+1] && first_k < row_ptr[k+1]) {
            const double first_ratio = row_val[first_k] / row_val[first_j];
            if (fabs(c[k] - first_ratio * c[j]) > obj_tol) continue;
          }
        }
        double ratio = 0.0;
        if (preverified_state != nullptr && preverified_state[b] != std::uint8_t{0} &&
            preverified_partner[b] == a) {
          if (preverified_state[b] == std::uint8_t{2}) continue;
          ratio = preverified_ratio[b];
        } else if (!_parallel_column_ratio(
                       j, k, row_ptr, row_idx, row_val, keep_row,
                       zero_tol, coeff_tol, &ratio)) {
          continue;
        }
        double active_max_abs_ratio = max_abs_ratio;
        if (scale_guard_group_count != nullptr &&
            *scale_guard_group_count < scale_guard_min_groups) {
          active_max_abs_ratio = INFINITY;
        }
        const double abs_ratio = fabs(ratio);
        if (isfinite(active_max_abs_ratio) &&
            (active_max_abs_ratio < 1.0 || abs_ratio > active_max_abs_ratio ||
             abs_ratio * active_max_abs_ratio < 1.0)) {
          continue;
        }

        const double obj_gap = c[k] - ratio * c[j];
        const double target_l = l[j];
        const double target_u = u[j];
        const double source_l = l[k];
        const double source_u = u[k];

        if (fabs(obj_gap) <= obj_tol) {
          merge_to[k] = j;
          merge_ratio[k] = ratio;
          merge_from_l[k] = source_l;
          merge_from_u[k] = source_u;
          merge_to_l[k] = target_l;
          merge_to_u[k] = target_u;
          l[j] = _merged_lower_bound_parallel_columns(target_l, target_u, source_l, source_u, ratio);
          u[j] = _merged_upper_bound_parallel_columns(target_l, target_u, source_l, source_u, ratio);
          col_delete[k] = std::uint8_t{1};
          atomicMax(&status_flag[1], 1);
          continue;
        }

        bool fix_xk_to_lower = false;
        bool fix_xk_to_upper = false;
        bool fix_xj_to_lower = false;
        bool fix_xj_to_upper = false;
        if (obj_gap > obj_tol) {
          if (ratio > 0.0) {
            fix_xk_to_lower = !isfinite(target_u);
            fix_xj_to_upper = !isfinite(source_l);
          } else {
            fix_xk_to_lower = !isfinite(target_l);
            fix_xj_to_lower = !isfinite(source_l);
          }
        } else {
          if (ratio > 0.0) {
            fix_xk_to_upper = !isfinite(target_l);
            fix_xj_to_lower = !isfinite(source_u);
          } else {
            fix_xk_to_upper = !isfinite(target_u);
            fix_xj_to_upper = !isfinite(source_u);
          }
        }

        if (fix_xk_to_lower) {
          if (!isfinite(source_l)) {
            atomicMax(&status_flag[0], 1);
            return;
          }
          fixed_mask[k] = std::uint8_t{1};
          fixed_val[k] = source_l;
          atomicMax(&status_flag[2], 1);
          continue;
        } else if (fix_xk_to_upper) {
          if (!isfinite(source_u)) {
            atomicMax(&status_flag[0], 1);
            return;
          }
          fixed_mask[k] = std::uint8_t{1};
          fixed_val[k] = source_u;
          atomicMax(&status_flag[2], 1);
          continue;
        }

        if (fix_xj_to_lower) {
          if (!isfinite(target_l)) {
            atomicMax(&status_flag[0], 1);
            return;
          }
          fixed_mask[j] = std::uint8_t{1};
          fixed_val[j] = target_l;
          atomicMax(&status_flag[2], 1);
          break;
        } else if (fix_xj_to_upper) {
          if (!isfinite(target_u)) {
            atomicMax(&status_flag[0], 1);
            return;
          }
          fixed_mask[j] = std::uint8_t{1};
          fixed_val[j] = target_u;
          atomicMax(&status_flag[2], 1);
          break;
        }
      }
    }
  }
}

__global__ void _kernel_parallel_column_fixed_row_shift(double* row_shift,
                                                     const std::uint8_t* fixed_mask,
                                                     const double* fixed_val,
                                                     const std::uint8_t* keep_row,
                                                     const NnzOffset* row_ptr,
                                                     const std::int32_t* row_idx,
                                                     const double* row_val,
                                                     std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j < n && fixed_mask[j] != std::uint8_t{0}) {
    const double vj = fixed_val[j];
    const NnzOffset start_j = row_ptr[j];
    const NnzOffset stop_j = row_ptr[j + 1];
    for (NnzOffset p = start_j; p < stop_j; ++p) {
      const std::int32_t row = row_idx[p];
      if (keep_row[row] != std::uint8_t{0}) {
        atomicAdd(&row_shift[row], row_val[p] * vj);
      }
    }
  }
}

__global__ void _kernel_apply_parallel_columns(std::uint8_t* keep_col,
                                            double* AL,
                                            double* AU,
                                            double* new_l,
                                            double* new_u,
                                            double* obj_delta,
                                            const double* c,
                                            const std::uint8_t* col_delete,
                                            const std::uint8_t* fixed_mask,
                                            const double* fixed_val,
                                            const double* row_shift,
                                            std::int32_t m,
                                            std::int32_t n) {
  const NnzOffset q = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (q < n && (col_delete[q] != std::uint8_t{0} || fixed_mask[q] != std::uint8_t{0})) {
    keep_col[q] = std::uint8_t{0};
    if (fixed_mask[q] != std::uint8_t{0}) {
      new_l[q] = fixed_val[q];
      new_u[q] = fixed_val[q];
      atomicAdd(obj_delta, c[q] * fixed_val[q]);
    }
  }
  if (q < m) {
    AL[q] -= row_shift[q];
    AU[q] -= row_shift[q];
  }
}

#include "parallel_column_signature.cuh"

}  // namespace

std::size_t parallel_columns_scratch_bytes(std::size_t rows, std::size_t cols,
                                            bool parallel_verify) {
  std::size_t bytes = 0;
  reserve_scratch<unsigned long long>(bytes, cols);
  reserve_scratch<std::int32_t>(bytes, cols);
  if (parallel_verify) {
    reserve_scratch<std::uint8_t>(bytes, cols);
    reserve_scratch<std::int32_t>(bytes, cols);
    reserve_scratch<double>(bytes, cols);
  }
  reserve_scratch<std::int32_t>(bytes, 3);
  reserve_scratch<std::int32_t>(bytes, 1);
  reserve_scratch<std::uint8_t>(bytes, cols);
  reserve_scratch<std::uint8_t>(bytes, cols);
  reserve_scratch<double>(bytes, cols);
  reserve_scratch<std::int32_t>(bytes, cols);
  for (int i = 0; i < 5; ++i) reserve_scratch<double>(bytes, cols);
  reserve_scratch<double>(bytes, rows);
  reserve_scratch<double>(bytes, 1);
  reserve_scratch<unsigned long long>(bytes,
      (column_signature::scratch_bytes(cols) + 7) / 8);
  return bytes;
}

void apply_rule_parallel_columns(PresolvePlanGpu& plan,
                              const LPInfoGpu& lp,
                              const PresolveStatsGpu& stats,
                              const PresolveParams& pparams) {
  (void)stats;
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }
  const std::int32_t m = lp.A.rows;
  const std::int32_t n = lp.A.cols;
  if (n <= 1) {
    return;
  }
  const auto rule_start = std::chrono::steady_clock::now();
  const bool profile = runtime::options().profile_parallel_columns;
  auto profile_stage = [&](const char* stage, const std::chrono::steady_clock::time_point& stage_start) {
    if (!profile) {
      return;
    }
    throw_if_cuda_error(cudaDeviceSynchronize(), "parallel_columns profile synchronize");
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - stage_start;
    std::cerr << ">>> [parallel_columns C++] " << stage << " = " << elapsed.count() << "s\n";
  };

  if (lp.AT.nnz > 0 && !csr_rows_strictly_sorted(lp.AT)) {
    const auto sort_at_start = std::chrono::steady_clock::now();
    unsigned long long* sort_keys = nullptr;
    throw_if_cuda_error(cudaMalloc(&sort_keys, sizeof(unsigned long long) * static_cast<std::size_t>(lp.AT.nnz)),
                        "cudaMalloc parallel_columns sort keys");
    const int blocks_at_rows = (lp.AT.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    _kernel_parallel_column_sort_keys<<<blocks_at_rows, GPU_PRESOLVE_THREADS>>>(
        sort_keys, lp.AT.rowPtr, lp.AT.colVal, lp.AT.rows);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_parallel_column_sort_keys");
    thrust::sort_by_key(thrust::device_pointer_cast(sort_keys),
                        thrust::device_pointer_cast(sort_keys + lp.AT.nnz),
                        thrust::device_pointer_cast(lp.AT.nzVal));
    throw_if_cuda_error(cudaGetLastError(), "thrust parallel_columns sort AT rows");
    const int blocks_at_nnz = (lp.AT.nnz + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    _kernel_parallel_column_unpack_keys<<<blocks_at_nnz, GPU_PRESOLVE_THREADS>>>(
        lp.AT.colVal, sort_keys, lp.AT.nnz);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_parallel_column_unpack_keys");
    throw_if_cuda_error(cudaDeviceSynchronize(), "parallel_columns sort AT synchronize");
    cudaFree(sort_keys);
    profile_stage("sort_AT_rows", sort_at_start);
  }

  const std::size_t n_size = static_cast<std::size_t>(n);
  const std::size_t m_size = static_cast<std::size_t>(m);
  const bool parallel_verify = runtime::options().parallel_column_preverify;
  std::size_t scratch_bytes = 0;
  const auto col_hash_offset = reserve_scratch<unsigned long long>(scratch_bytes, n_size);
  const auto sorted_cols_offset = reserve_scratch<std::int32_t>(scratch_bytes, n_size);
  const auto preverified_state_offset = parallel_verify
      ? reserve_scratch<std::uint8_t>(scratch_bytes, n_size) : 0;
  const auto preverified_partner_offset = parallel_verify
      ? reserve_scratch<std::int32_t>(scratch_bytes, n_size) : 0;
  const auto preverified_ratio_offset = parallel_verify
      ? reserve_scratch<double>(scratch_bytes, n_size) : 0;
  const auto status_flag_offset = reserve_scratch<std::int32_t>(scratch_bytes, 3);
  const auto scale_guard_group_count_offset =
      reserve_scratch<std::int32_t>(scratch_bytes, 1);
  const auto col_delete_offset = reserve_scratch<std::uint8_t>(scratch_bytes, n_size);
  const auto fixed_mask_offset = reserve_scratch<std::uint8_t>(scratch_bytes, n_size);
  const auto fixed_val_offset = reserve_scratch<double>(scratch_bytes, n_size);
  const auto merge_to_offset = reserve_scratch<std::int32_t>(scratch_bytes, n_size);
  const auto merge_ratio_offset = reserve_scratch<double>(scratch_bytes, n_size);
  const auto merge_from_l_offset = reserve_scratch<double>(scratch_bytes, n_size);
  const auto merge_from_u_offset = reserve_scratch<double>(scratch_bytes, n_size);
  const auto merge_to_l_offset = reserve_scratch<double>(scratch_bytes, n_size);
  const auto merge_to_u_offset = reserve_scratch<double>(scratch_bytes, n_size);
  const auto row_shift_offset = reserve_scratch<double>(scratch_bytes, m_size);
  const auto obj_delta_offset = reserve_scratch<double>(scratch_bytes, 1);
  const bool signature_enabled = column_signature::enabled(lp.AT);
  const auto signature_offset = reserve_scratch<unsigned long long>(scratch_bytes,
      (column_signature::scratch_bytes(n_size) + 7) / 8);
  DuplicateRuleScratchFrame scratch_frame;
  void* scratch_storage = scratch_frame.get(scratch_bytes);
  auto* col_hash = scratch_at<unsigned long long>(scratch_storage, col_hash_offset);
  auto* sorted_cols = scratch_at<std::int32_t>(scratch_storage, sorted_cols_offset);
  auto* preverified_state = parallel_verify
      ? scratch_at<std::uint8_t>(scratch_storage, preverified_state_offset) : nullptr;
  auto* preverified_partner = parallel_verify
      ? scratch_at<std::int32_t>(scratch_storage, preverified_partner_offset) : nullptr;
  auto* preverified_ratio = parallel_verify
      ? scratch_at<double>(scratch_storage, preverified_ratio_offset) : nullptr;
  auto* status_flag = scratch_at<std::int32_t>(scratch_storage, status_flag_offset);
  auto* scale_guard_group_count =
      scratch_at<std::int32_t>(scratch_storage, scale_guard_group_count_offset);
  auto* col_delete = scratch_at<std::uint8_t>(scratch_storage, col_delete_offset);
  auto* fixed_mask = scratch_at<std::uint8_t>(scratch_storage, fixed_mask_offset);
  auto* fixed_val = scratch_at<double>(scratch_storage, fixed_val_offset);
  auto* merge_to = scratch_at<std::int32_t>(scratch_storage, merge_to_offset);
  auto* merge_ratio = scratch_at<double>(scratch_storage, merge_ratio_offset);
  auto* merge_from_l = scratch_at<double>(scratch_storage, merge_from_l_offset);
  auto* merge_from_u = scratch_at<double>(scratch_storage, merge_from_u_offset);
  auto* merge_to_l = scratch_at<double>(scratch_storage, merge_to_l_offset);
  auto* merge_to_u = scratch_at<double>(scratch_storage, merge_to_u_offset);
  auto* row_shift = scratch_at<double>(scratch_storage, row_shift_offset);
  auto* obj_delta_device = scratch_at<double>(scratch_storage, obj_delta_offset);
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t) * 3), "cudaMemset parallel_columns status_flag");
  throw_if_cuda_error(
      cudaMemset(scale_guard_group_count, 0, sizeof(std::int32_t)),
      "cudaMemset parallel_columns scale_guard_group_count");
  throw_if_cuda_error(cudaMemset(col_delete, 0, static_cast<std::size_t>(n)), "cudaMemset parallel_columns col_delete");
  throw_if_cuda_error(cudaMemset(fixed_mask, 0, static_cast<std::size_t>(n)), "cudaMemset parallel_columns fixed_mask");
  throw_if_cuda_error(cudaMemset(merge_to, 0, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      "cudaMemset parallel_columns merge_to");
  throw_if_cuda_error(cudaMemset(row_shift, 0, sizeof(double) * static_cast<std::size_t>(m)), "cudaMemset parallel_columns row_shift");
  throw_if_cuda_error(cudaMemset(obj_delta_device, 0, sizeof(double)), "cudaMemset parallel_columns obj_delta");
  if (parallel_verify) {
    throw_if_cuda_error(cudaMemset(preverified_state, 0, n_size),
                        "cudaMemset parallel_columns preverified_state");
  }

  const int blocks_n = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const std::uint8_t* candidates = nullptr;
  const bool signature_audit = signature_enabled &&
      runtime::options().audit_column_signatures;
  auto hash_start = rule_start;
  if (signature_enabled) {
    column_signature::Workspace signature(
        scratch_at<void>(scratch_storage, signature_offset), n_size);
    column_signature::filter(signature, lp.AT, plan.keep_col_mask,
                             plan.keep_row_mask, pparams.zero_tol);
    candidates = signature.candidates;
    profile_stage("signature", rule_start);
    if (signature_audit) {
      const auto audit_start = std::chrono::steady_clock::now();
      _kernel_parallel_column_hashes<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
          col_hash, plan.keep_col_mask, plan.keep_row_mask,
          lp.AT.rowPtr, lp.AT.colVal, lp.AT.nzVal, pparams.zero_tol, n);
      throw_if_cuda_error(cudaGetLastError(), "parallel columns audit hashes");
      auto hashes = thrust::device_pointer_cast(col_hash);
      auto cols = thrust::device_pointer_cast(sorted_cols);
      thrust::sequence(cols, cols + n, std::int32_t{0});
      auto zipped = thrust::make_zip_iterator(thrust::make_tuple(hashes, cols));
      thrust::sort(zipped, zipped + n, HashColLess{});
      throw_if_cuda_error(cudaMemset(signature.overflow, 0, sizeof(int)),
                          "column signature audit reset");
      column_signature::audit_groups<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
          col_hash, sorted_cols, plan.keep_col_mask, candidates, n, signature.overflow);
      int missing = 0;
      throw_if_cuda_error(cudaMemcpy(&missing, signature.overflow, sizeof(int),
                                    cudaMemcpyDeviceToHost), "column signature audit result");
      std::cerr << ">>> [parallel_columns signature] audit_missing = " << missing << "\n";
      if (missing)
        throw std::runtime_error("parallel column signature omitted original hash-group members");
      profile_stage("signature_audit", audit_start);
    }
    hash_start = std::chrono::steady_clock::now();
  }
  // The audit sorts full hashes; always recompute in original column-id order.
  _kernel_parallel_column_hashes<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      col_hash,
      plan.keep_col_mask,
      plan.keep_row_mask,
      lp.AT.rowPtr,
      lp.AT.colVal,
      lp.AT.nzVal,
      pparams.zero_tol,
      n, candidates);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_parallel_column_hashes");
  throw_if_cuda_error(cudaDeviceSynchronize(), "parallel_columns hash synchronize");
  profile_stage("hash", hash_start);

  const auto sort_start = std::chrono::steady_clock::now();
  auto col_hash_begin = thrust::device_pointer_cast(col_hash);
  auto sorted_cols_begin = thrust::device_pointer_cast(sorted_cols);
  auto* sort_hash = col_hash;
  int sort_count = n;
  if (signature_enabled) {
    const auto first = thrust::make_counting_iterator(std::int32_t{0});
    const auto end = thrust::copy_if(first, first + n,
        thrust::device_pointer_cast(candidates), sorted_cols_begin,
        column_signature::IsCandidate{});
    sort_count = static_cast<int>(end - sorted_cols_begin);
    if (profile) {
      column_signature::Workspace signature(
          scratch_at<void>(scratch_storage, signature_offset), n_size);
      throw_if_cuda_error(cudaMemset(signature.counts, 0, 2*sizeof(unsigned long long)),
                          "column signature nnz reset");
      column_signature::count_nnz<<<blocks_n, GPU_PRESOLVE_THREADS>>>(signature, n);
      unsigned long long counts[2];
      throw_if_cuda_error(cudaMemcpy(counts, signature.counts, sizeof(counts),
                                    cudaMemcpyDeviceToHost), "column signature nnz counts");
      std::cerr << ">>> [parallel_columns signature] candidate_live_nnz = "
                << counts[1] << "/" << counts[0] << "\n";
    }
    // Projection data is no longer needed. Reuse its storage for compact hashes.
    sort_hash = scratch_at<unsigned long long>(scratch_storage, signature_offset);
    auto output = thrust::device_pointer_cast(sort_hash);
    thrust::gather(sorted_cols_begin, end, col_hash_begin, output);
    col_hash_begin = output;
    if (profile || signature_audit)
      std::cerr << ">>> [parallel_columns signature] candidates = "
                << sort_count << "/" << n << "\n";
  } else {
    thrust::sequence(sorted_cols_begin, sorted_cols_begin + n, std::int32_t{0});
  }
  auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(col_hash_begin, sorted_cols_begin));
  thrust::sort(zipped_begin, zipped_begin + sort_count, HashColLess{});
  throw_if_cuda_error(cudaGetLastError(), "thrust parallel_columns sort hashes");
  profile_stage("device_sort", sort_start);

  const double coeff_tol = fmax(pparams.zero_tol, pparams.bound_tol);
  const double obj_tol = fmax(pparams.zero_tol, pparams.bound_tol);
  const bool use_scale_guard_gate =
      isfinite(pparams.parallel_columns_max_abs_ratio) &&
      pparams.parallel_columns_scale_guard_min_groups > 1;
  const auto scale_guard_start = std::chrono::steady_clock::now();
  if (use_scale_guard_gate) {
    _kernel_count_small_high_ratio_parallel_groups<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
        scale_guard_group_count,
        sort_hash,
        sorted_cols,
        plan.keep_col_mask,
        plan.keep_row_mask,
        plan.new_c,
        lp.AT.rowPtr,
        lp.AT.colVal,
        lp.AT.nzVal,
        pparams.zero_tol,
        coeff_tol,
        obj_tol,
        pparams.parallel_columns_max_abs_ratio,
        pparams.parallel_columns_scale_guard_max_group_size,
        pparams.parallel_columns_scale_guard_min_groups,
        sort_count);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_count_small_high_ratio_parallel_groups");
  }
  profile_stage("scale_guard", scale_guard_start);
  const auto parallel_verify_start = std::chrono::steady_clock::now();
  if (parallel_verify) {
    _kernel_preverify_parallel_columns<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
        preverified_state, preverified_partner, preverified_ratio,
        sort_hash, sorted_cols, plan.keep_col_mask, plan.keep_row_mask,
        plan.new_c,
        lp.AT.rowPtr, lp.AT.colVal, lp.AT.nzVal,
        pparams.zero_tol, coeff_tol, obj_tol, sort_count);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_preverify_parallel_columns");
  }
  profile_stage("parallel_verify", parallel_verify_start);
  const auto groups_start = std::chrono::steady_clock::now();
  _kernel_parallel_column_groups<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      status_flag,
      col_delete,
      fixed_mask,
      fixed_val,
      merge_to,
      merge_ratio,
      merge_from_l,
      merge_from_u,
      merge_to_l,
      merge_to_u,
      preverified_state,
      preverified_partner,
      preverified_ratio,
      sort_hash,
      sorted_cols,
      plan.keep_col_mask,
      plan.keep_row_mask,
      plan.new_c,
      plan.new_l,
      plan.new_u,
      lp.AT.rowPtr,
      lp.AT.colVal,
      lp.AT.nzVal,
      pparams.zero_tol,
      coeff_tol,
      obj_tol,
      pparams.parallel_columns_max_abs_ratio,
      use_scale_guard_gate ? scale_guard_group_count : nullptr,
      pparams.parallel_columns_scale_guard_min_groups,
      sort_count, runtime::options().parallel_column_prefilter);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_parallel_column_groups");
  profile_stage("groups", groups_start);

  const auto status_start = std::chrono::steady_clock::now();
  std::int32_t status[3] = {0, 0, 0};
  throw_if_cuda_error(cudaMemcpy(status, status_flag, sizeof(status), cudaMemcpyDeviceToHost),
                      "cudaMemcpy parallel_columns status");
  profile_stage("status", status_start);
  if (status[0] != 0) {
    plan.has_unbounded = true;
  } else if (status[1] != 0 || status[2] != 0) {
    const auto tape_start = std::chrono::steady_clock::now();
    if (status[1] != 0 && pparams.record_postsolve_tape) {
      const auto col_delete_begin = thrust::device_pointer_cast(col_delete);
      const std::int32_t merge_count = static_cast<std::int32_t>(
          thrust::count(col_delete_begin, col_delete_begin + n, std::uint8_t{1}));
      std::vector<ParallelColumnTapeRecord> records(
          static_cast<std::size_t>(plan.tape.resident ? 0 : merge_count));
      ParallelColumnTapeRecord* packed_records = nullptr;
      if (merge_count > 0) {
        throw_if_cuda_error(
            cudaMalloc(&packed_records,
                       sizeof(ParallelColumnTapeRecord) * static_cast<std::size_t>(merge_count)),
            "cudaMalloc packed parallel-column tape");
        throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)),
                            "cudaMemset packed parallel-column count");
        _kernel_pack_parallel_column_tape<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
            packed_records, status_flag, col_delete, merge_to, merge_ratio,
            merge_from_l, merge_from_u, merge_to_l, merge_to_u, n);
        throw_if_cuda_error(cudaGetLastError(), "_kernel_pack_parallel_column_tape");
        if (plan.tape.resident) {
          auto ptr = thrust::device_pointer_cast(packed_records);
          thrust::sort(ptr, ptr + merge_count, RecoveryOrder{});
          auto segment = allocate_recovery_segment(
              merge_count, checked_nnz_offset(2 * std::int64_t(merge_count)),
              checked_nnz_offset(5 * std::int64_t(merge_count)));
          pack_recovery_segment<<<(merge_count + 256) / 256, 256>>>(
              segment.types, segment.dual_modes, segment.index_starts, segment.value_starts,
              segment.indices, segment.vals, packed_records, merge_count);
          throw_if_cuda_error(cudaGetLastError(), "pack duplicate recovery segment");
          append_device_recovery_segment(plan.tape, std::move(segment));
        } else {
          throw_if_cuda_error(cudaMemcpy(records.data(), packed_records,
                                         sizeof(ParallelColumnTapeRecord) * records.size(),
                                         cudaMemcpyDeviceToHost),
                              "cudaMemcpy packed parallel-column tape");
        }
        cudaFree(packed_records);
      }
      std::sort(records.begin(), records.end(),
                [](const ParallelColumnTapeRecord& lhs, const ParallelColumnTapeRecord& rhs) {
                  return lhs.deleted_col < rhs.deleted_col;
                });
      plan.tape.types.reserve(plan.tape.types.size() + records.size());
      plan.tape.indices.reserve(plan.tape.indices.size() + 2 * records.size());
      plan.tape.vals.reserve(plan.tape.vals.size() + 5 * records.size());
      plan.tape.index_starts.reserve(plan.tape.index_starts.size() + records.size());
      plan.tape.value_starts.reserve(plan.tape.value_starts.size() + records.size());
      plan.tape.dual_modes.reserve(plan.tape.dual_modes.size() + records.size());
      for (const ParallelColumnTapeRecord& record : records) {
        const std::int32_t indices[] = {record.deleted_col, record.kept_col};
        const double vals[] = {record.ratio,
                               record.deleted_l,
                               record.deleted_u,
                               record.kept_l,
                               record.kept_u};
        append_postsolve_record(
            plan.tape,
            PostsolveReductionType::ParallelColumn,
            indices,
            2,
            vals,
            5,
            PostsolveDualMode::Minimal);
      }
    }
    if (status[2] != 0) {
      if (pparams.record_postsolve_tape) {
        append_compact_fixed_col_tape_from_device(&plan.tape,
                                                  fixed_mask,
                                                  nullptr,
                                                  fixed_val,
                                                  plan.keep_row_mask,
                                                  plan.new_c,
                                                  lp.AT,
                                                  n,
                                                  "parallel_columns fixed-col tape", FixedColObjective::Skip);
      }
      _kernel_parallel_column_fixed_row_shift<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
          row_shift,
          fixed_mask,
          fixed_val,
          plan.keep_row_mask,
          lp.AT.rowPtr,
          lp.AT.colVal,
          lp.AT.nzVal,
          n);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_parallel_column_fixed_row_shift");
      plan.has_row_action = true;
    }
    profile_stage("tape", tape_start);
    const auto apply_start = std::chrono::steady_clock::now();
    const std::int32_t max_mn = m > n ? m : n;
    const int blocks_mn = (max_mn + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    _kernel_apply_parallel_columns<<<blocks_mn, GPU_PRESOLVE_THREADS>>>(
        plan.keep_col_mask,
        plan.new_AL,
        plan.new_AU,
        plan.new_l,
        plan.new_u,
        obj_delta_device,
        plan.new_c,
        col_delete,
        fixed_mask,
        fixed_val,
        row_shift,
        m,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_apply_parallel_columns");
    throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_parallel_columns synchronize");
    double obj_delta = 0.0;
    throw_if_cuda_error(cudaMemcpy(&obj_delta, obj_delta_device, sizeof(double), cudaMemcpyDeviceToHost),
                        "cudaMemcpy parallel_columns obj_delta");
    profile_stage("apply_delete", apply_start);
    plan.obj_constant_delta += obj_delta;
    plan.has_col_action = true;
    plan.has_change = true;
  }

  profile_stage("total", rule_start);
}

}  // namespace gpu_presolver::presolve
