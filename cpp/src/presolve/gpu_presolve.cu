#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/presolve/cleanup_workspace.hpp"
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/duplicate_rule_workspace.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver/presolve/identity_plan.hpp"
#include "gpu_presolver/presolve/gpu_presolve_kernels.hpp"
#include "gpu_presolver/presolve/stable_mapping.hpp"
#include "gpu_presolver/presolve/transpose64.hpp"

#include "gpu_presolver/presolve/rules/rule_variable_pair_aggregation.hpp"
#include "plan_storage.hpp"
#include "gpu_presolver/presolve/rules/rule_two_row_variable_elimination.hpp"
#include "gpu_presolver/presolve/rules/rule_column_domination.hpp"
#include "gpu_presolver/presolve/rules/rule_infeasible_and_redundant_rows.hpp"
#include "gpu_presolver/presolve/rules/rule_infeasible_and_fixed_variables.hpp"
#include "gpu_presolver/presolve/rules/rule_doubleton_equations.hpp"
#include "gpu_presolver/presolve/rules/rule_dual_fix.hpp"
#include "gpu_presolver/presolve/rules/rule_empty_rows_and_columns.hpp"
#include "gpu_presolver/presolve/rules/rule_parallel_columns.hpp"
#include "gpu_presolver/presolve/rules/rule_parallel_rows.hpp"
#include "gpu_presolver/presolve/rules/rule_mccormick_variable_elimination.hpp"
#include "gpu_presolver/presolve/rules/rule_bound_strengthening.hpp"
#include "gpu_presolver/presolve/rules/rule_singleton_columns.hpp"
#include "gpu_presolver/presolve/rules/rule_singleton_rows.hpp"
#include "gpu_presolver/presolve/rules/rule_absolute_value_epigraph_reduction.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

constexpr std::int32_t SPARSE_FIXED_BOUND_RECORD_THRESHOLD = 65536;
constexpr std::int32_t COMPACT_LONG_ROW_THRESHOLD = 1024;
constexpr std::int32_t PARALLEL_IDENTITY_FILL_THRESHOLD = 1 << 20;
constexpr unsigned PARALLEL_IDENTITY_FILL_MAX_THREADS = 8;

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(
      std::string(context) + ": " + cudaGetErrorString(status));
}

#include "diagnostic_snapshot.hpp"

template <class T>
cudaError_t stream_ordered_malloc(T** pointer, std::size_t bytes) {
  return cudaMallocAsync(reinterpret_cast<void**>(pointer), bytes, nullptr);
}

cudaError_t stream_ordered_free(void* pointer) {
  return pointer == nullptr ? cudaSuccess : cudaFreeAsync(pointer, nullptr);
}

template <typename T>
void inclusive_scan_i32(T* values, std::int32_t n, const char* context) {
  if (n <= 0) {
    return;
  }
  void* temp_storage = nullptr;
  std::size_t temp_bytes = 0;
  throw_if_cuda_error(
      cub::DeviceScan::InclusiveSum(temp_storage, temp_bytes, values, values, n),
      context);
  CleanupScratchFrame scratch;
  temp_storage = scratch.get<unsigned char>(6, temp_bytes);
  throw_if_cuda_error(
      cub::DeviceScan::InclusiveSum(temp_storage, temp_bytes, values, values, n),
      context);
}

__global__ void _kernel_fill_u8(std::uint8_t* data, std::uint8_t value, std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j < n) {
    data[j] = value;
  }
}

__global__ void _kernel_mask_to_prefix_i32(std::int32_t* out,
                                           const std::uint8_t* mask,
                                           std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j < n) {
    out[j] = mask[j] != std::uint8_t{0} ? 1 : 0;
  }
}

__global__ void _kernel_old_to_new_from_prefix(std::int32_t* old_to_new,
                                               const std::uint8_t* mask,
                                               const std::int32_t* prefix,
                                               std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j < n) {
    old_to_new[j] = mask[j] != std::uint8_t{0} ? prefix[j] - 1 : -1;
  }
}

__global__ void _kernel_count_compacted_rows(NnzOffset* row_counts_new,
                                             const std::int32_t* row_old_to_new,
                                             const std::int32_t* col_old_to_new,
                                             const NnzOffset* row_ptr,
                                             const std::int32_t* col_val,
                                             std::int32_t* long_rows,
                                             std::int32_t* long_row_count,
                                             std::int32_t long_row_threshold,
                                             std::int32_t m_old,
                                             std::int32_t* medium_rows,
                                             std::int32_t* medium_row_count,
                                             std::int32_t medium_row_threshold) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row < m_old) {
    const std::int32_t row_new = row_old_to_new[row];
    if (row_new < 0) {
      return;
    }
    if (row_ptr[row + 1] - row_ptr[row] > long_row_threshold) {
      const std::int32_t slot = atomicAdd(long_row_count, 1);
      // Medium rows grow from the front, long rows from the back. Since a
      // kept row belongs to at most one queue, their slots cannot overlap.
      long_rows[medium_rows != nullptr ? m_old - 1 - slot : slot] = row;
      return;
    }
    if (row_ptr[row + 1] - row_ptr[row] > medium_row_threshold) {
      const std::int32_t slot = atomicAdd(medium_row_count, 1);
      medium_rows[slot] = row;
      return;
    }
    std::int32_t count = 0;
    for (NnzOffset p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
      if (col_old_to_new[col_val[p]] >= 0) {
        ++count;
      }
    }
    row_counts_new[row_new] = count;
  }
}

// Each warp owns one row. Integer ballots preserve entry order and copy
// coefficient bits unchanged, independently of queue insertion order.
__global__ void _kernel_compact_medium_rows(
    NnzOffset* counts, std::int32_t* cols_out, double* vals_out,
    const NnzOffset* ptr_out, const std::int32_t* row_map,
    const std::int32_t* col_map, const NnzOffset* ptr,
    const std::int32_t* cols, const double* vals,
    const std::int32_t* rows, std::int32_t row_count) {
  const int lane = threadIdx.x & 31;
  const NnzOffset q = (NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  if (q >= row_count) return;
  const int row = rows[q], row_new = row_map[row];
  int count = 0;
  for (std::int64_t chunk = ptr[row]; chunk < ptr[row + 1]; chunk += 32) {
    const std::int64_t p = chunk + lane;
    const int col = p < ptr[row + 1] ? col_map[cols[p]] : -1;
    const unsigned mask = __ballot_sync(0xffffffffu, col >= 0);
    if (cols_out != nullptr && col >= 0) {
      const unsigned preceding = mask & ((1u << lane) - 1u);
      const NnzOffset out = ptr_out[row_new] + count + __popc(preceding);
      cols_out[out] = col;
      vals_out[out] = vals[p];
    }
    count += __popc(mask);
  }
  if (counts != nullptr && lane == 0) counts[row_new] = count;
}

__global__ void _kernel_count_compacted_long_rows(
    NnzOffset* row_counts_new,
    const std::int32_t* row_old_to_new,
    const std::int32_t* col_old_to_new,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const std::int32_t* long_rows,
    std::int32_t long_row_count) {
  const std::int32_t long_index = static_cast<std::int32_t>(blockIdx.x);
  if (long_index >= long_row_count) {
    return;
  }
  const std::int32_t row = long_rows[long_index];
  std::int32_t count = 0;
  for (NnzOffset p = row_ptr[row] + static_cast<NnzOffset>(threadIdx.x);
       p < row_ptr[row + 1]; p += static_cast<std::int32_t>(blockDim.x)) {
    count += col_old_to_new[col_val[p]] >= 0 ? 1 : 0;
  }
  __shared__ std::int32_t shared_counts[256];
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
  shared_counts[tid] = count;
  __syncthreads();
  for (NnzOffset stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_counts[tid] += shared_counts[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    row_counts_new[row_old_to_new[row]] = shared_counts[0];
  }
}

__global__ void _kernel_count_compacted_rows_block(
    NnzOffset* row_counts_new,
    const std::int32_t* row_old_to_new,
    const std::int32_t* col_old_to_new,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    std::int32_t m_old) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
  if (row >= m_old) {
    return;
  }
  const std::int32_t row_new = row_old_to_new[row];
  if (row_new < 0) {
    return;
  }
  std::int32_t local_count = 0;
  for (NnzOffset p = row_ptr[row] + static_cast<NnzOffset>(threadIdx.x);
       p < row_ptr[row + 1]; p += static_cast<std::int32_t>(blockDim.x)) {
    local_count += col_old_to_new[col_val[p]] >= 0 ? 1 : 0;
  }
  using BlockReduce = cub::BlockReduce<std::int32_t, 256>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  const std::int32_t count = BlockReduce(reduce_storage).Sum(local_count);
  if (threadIdx.x == 0) {
    row_counts_new[row_new] = count;
  }
}

__global__ void _kernel_row_ptr_from_prefix(NnzOffset* row_ptr,
                                            const NnzOffset* prefix,
                                            std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i == 0) {
    row_ptr[0] = 0;
  }
  if (i < m) {
    row_ptr[i + 1] = prefix[i];
  }
}

__global__ void _kernel_copy_compacted_rows(std::int32_t* col_val_new,
                                            double* nz_val_new,
                                            const NnzOffset* row_ptr_new,
                                            const std::int32_t* row_old_to_new,
                                            const std::int32_t* col_old_to_new,
                                            const NnzOffset* row_ptr,
                                            const std::int32_t* col_val,
                                            const double* nz_val,
                                            std::int32_t long_row_threshold,
                                            std::int32_t m_old) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row < m_old) {
    const std::int32_t row_new = row_old_to_new[row];
    if (row_new < 0) {
      return;
    }
    if (row_ptr[row + 1] - row_ptr[row] > long_row_threshold) {
      return;
    }
    NnzOffset write = row_ptr_new[row_new];
    for (NnzOffset p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
      const std::int32_t col_new = col_old_to_new[col_val[p]];
      if (col_new >= 0) {
        col_val_new[write] = col_new;
        nz_val_new[write] = nz_val[p];
        ++write;
      }
    }
  }
}

__global__ void _kernel_copy_compacted_long_rows(
    std::int32_t* col_val_new,
    double* nz_val_new,
    const NnzOffset* row_ptr_new,
    const std::int32_t* row_old_to_new,
    const std::int32_t* col_old_to_new,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    const std::int32_t* long_rows,
    std::int32_t long_row_count) {
  const std::int32_t long_index = static_cast<std::int32_t>(blockIdx.x);
  if (long_index >= long_row_count) {
    return;
  }
  const std::int32_t row = long_rows[long_index];
  const std::int32_t row_new = row_old_to_new[row];
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
  __shared__ std::int32_t scan[256];
  __shared__ NnzOffset base;
  if (tid == 0) {
    base = 0;
  }
  __syncthreads();
  for (NnzOffset chunk = row_ptr[row]; chunk < row_ptr[row + 1]; chunk += 256) {
    const NnzOffset p = chunk + tid;
    std::int32_t col_new = -1;
    if (p < row_ptr[row + 1]) {
      col_new = col_old_to_new[col_val[p]];
    }
    scan[tid] = col_new >= 0 ? 1 : 0;
    __syncthreads();
    for (int offset = 1; offset < 256; offset <<= 1) {
      const std::int32_t value =
          scan[tid] + (tid >= offset ? scan[tid - offset] : 0);
      __syncthreads();
      scan[tid] = value;
      __syncthreads();
    }
    if (col_new >= 0) {
      const NnzOffset write = row_ptr_new[row_new] + base + scan[tid] - 1;
      col_val_new[write] = col_new;
      nz_val_new[write] = nz_val[p];
    }
    const std::int32_t chunk_count = scan[255];
    __syncthreads();
    if (tid == 0) {
      base += chunk_count;
    }
    __syncthreads();
  }
}

__global__ void _kernel_copy_compacted_rows_block(
    std::int32_t* col_val_new,
    double* nz_val_new,
    const NnzOffset* row_ptr_new,
    const std::int32_t* row_old_to_new,
    const std::int32_t* col_old_to_new,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    std::int32_t work_count,
    const std::int32_t* row_ids = nullptr) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t row = row_ids == nullptr ? work : row_ids[work];
  const std::int32_t row_new = row_old_to_new[row];
  if (row_new < 0) {
    return;
  }
  using BlockScan = cub::BlockScan<std::int32_t, 256>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ NnzOffset row_write_offset;
  if (threadIdx.x == 0) {
    row_write_offset = 0;
  }
  __syncthreads();
  const NnzOffset row_stop = row_ptr[row + 1];
  for (NnzOffset chunk = row_ptr[row]; chunk < row_stop;
       chunk += 256) {
    const NnzOffset p = chunk + static_cast<NnzOffset>(threadIdx.x);
    const std::int32_t col_new =
        p < row_stop ? col_old_to_new[col_val[p]] : -1;
    const std::int32_t keep = col_new >= 0 ? 1 : 0;
    std::int32_t prefix = 0;
    std::int32_t chunk_count = 0;
    BlockScan(scan_storage).ExclusiveSum(keep, prefix, chunk_count);
    if (keep != 0) {
      const NnzOffset write =
          row_ptr_new[row_new] + row_write_offset + prefix;
      col_val_new[write] = col_new;
      nz_val_new[write] = nz_val[p];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      row_write_offset += chunk_count;
    }
    __syncthreads();
  }
}

__global__ void _kernel_gather_by_old_to_new(double* dst,
                                             const double* src,
                                             const std::int32_t* old_to_new,
                                             std::int32_t n_old) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j < n_old) {
    const std::int32_t mapped = old_to_new[j];
    if (mapped >= 0) {
      dst[mapped] = src[j];
    }
  }
}

__global__ void _kernel_identity_red2org_from_old_to_new(
    std::int32_t* red2org,
    std::int32_t* removed_fixed_count,
    const std::int32_t* old_to_new,
    const double* l,
    const double* u,
    double bound_tol,
    std::int32_t n_old) {
  using BlockReduce = cub::BlockReduce<std::int32_t, 256>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  const NnzOffset old_index =
      static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  std::int32_t removed_fixed = 0;
  if (old_index < n_old) {
    const NnzOffset new_index = old_to_new[old_index];
    if (new_index >= 0) {
      red2org[new_index] = old_index;
    } else if (isfinite(l[old_index]) && isfinite(u[old_index]) &&
               fabs(l[old_index] - u[old_index]) <= bound_tol) {
      removed_fixed = 1;
    }
  }
  const std::int32_t block_count = BlockReduce(reduce_storage).Sum(removed_fixed);
  if (threadIdx.x == 0 && block_count > 0) {
    atomicAdd(removed_fixed_count, block_count);
  }
}

struct StatsRequirements {
  bool row_nnz = false;
  bool empty_rows = false;
  bool singleton_rows = false;
  bool col_nnz = false;
  bool empty_columns = false;
  bool singleton_columns = false;
};

void allocate_stats(PresolveStatsGpu& stats,
                    std::int32_t m,
                    std::int32_t n,
                    const StatsRequirements& needs) {
  auto align_up = [](std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
  };
  struct SideOffsets {
    std::size_t nnz = 0;
    std::size_t empty_mask = 0;
    std::size_t singleton_mask = 0;
    std::size_t singleton_index = 0;
    std::size_t singleton_value = 0;
  };
  auto layout_side = [&](std::size_t count,
                         bool need_nnz,
                         bool need_empty,
                         bool need_singleton,
                         SideOffsets& side,
                         std::size_t& offset) {
    if (need_nnz) {
      offset = align_up(offset, alignof(std::int32_t));
      side.nnz = offset;
      offset += sizeof(std::int32_t) * count;
    }
    if (need_empty) {
      side.empty_mask = offset;
      offset += count;
    }
    if (need_singleton) {
      side.singleton_mask = offset;
      offset += count;
      offset = align_up(offset, alignof(std::int32_t));
      side.singleton_index = offset;
      offset += sizeof(std::int32_t) * count;
      offset = align_up(offset, alignof(double));
      side.singleton_value = offset;
      offset += sizeof(double) * count;
    }
  };

  const std::size_t row_count = static_cast<std::size_t>(m);
  const std::size_t col_count = static_cast<std::size_t>(n);
  SideOffsets row_offsets;
  SideOffsets col_offsets;
  std::size_t total_bytes = 0;
  const bool need_rows = needs.row_nnz || needs.empty_rows || needs.singleton_rows;
  const bool need_cols = needs.col_nnz || needs.empty_columns || needs.singleton_columns;
  if (need_rows && row_count > 0) {
    layout_side(row_count,
                needs.row_nnz,
                needs.empty_rows,
                needs.singleton_rows,
                row_offsets,
                total_bytes);
  }
  if (need_cols && col_count > 0) {
    layout_side(col_count,
                needs.col_nnz,
                needs.empty_columns,
                needs.singleton_columns,
                col_offsets,
                total_bytes);
  }
  if (total_bytes == 0) {
    return;
  }

  throw_if_cuda_error(stream_ordered_malloc(&stats.contiguous_storage, total_bytes),
                      "cudaMalloc stats contiguous storage");
  auto* storage = static_cast<unsigned char*>(stats.contiguous_storage);
  if (need_rows && row_count > 0) {
    if (needs.row_nnz) {
      stats.row_nnz = reinterpret_cast<std::int32_t*>(storage + row_offsets.nnz);
    }
    if (needs.empty_rows) stats.empty_row_mask = storage + row_offsets.empty_mask;
    if (needs.singleton_rows) {
      stats.singleton_row_mask = storage + row_offsets.singleton_mask;
      stats.singleton_row_col =
          reinterpret_cast<std::int32_t*>(storage + row_offsets.singleton_index);
      stats.singleton_row_val =
          reinterpret_cast<double*>(storage + row_offsets.singleton_value);
    }
  }
  if (need_cols && col_count > 0) {
    if (needs.col_nnz) {
      stats.col_nnz = reinterpret_cast<std::int32_t*>(storage + col_offsets.nnz);
    }
    if (needs.empty_columns) stats.empty_col_mask = storage + col_offsets.empty_mask;
    if (needs.singleton_columns) {
      stats.singleton_column_mask = storage + col_offsets.singleton_mask;
      stats.singleton_column_row =
          reinterpret_cast<std::int32_t*>(storage + col_offsets.singleton_index);
      stats.singleton_column_val =
          reinterpret_cast<double*>(storage + col_offsets.singleton_value);
    }
  }
}

void free_stats(PresolveStatsGpu& stats) {
  if (stats.contiguous_storage != nullptr) {
    stream_ordered_free(stats.contiguous_storage);
  } else {
    stream_ordered_free(stats.row_nnz);
    stream_ordered_free(stats.empty_row_mask);
    stream_ordered_free(stats.singleton_row_mask);
    stream_ordered_free(stats.singleton_row_col);
    stream_ordered_free(stats.singleton_row_val);
    stream_ordered_free(stats.col_nnz);
    stream_ordered_free(stats.empty_col_mask);
    stream_ordered_free(stats.singleton_column_mask);
    stream_ordered_free(stats.singleton_column_row);
    stream_ordered_free(stats.singleton_column_val);
  }
}

}  // namespace

namespace detail {

void allocate_plan(PresolvePlanGpu& plan,
                   const LPInfoGpu& lp,
                   const PlanRequirements& needs) {
  plan.tape.resident = device_recovery_enabled();
  const std::int32_t m = lp.A.rows;
  const std::int32_t n = lp.A.cols;
  const bool profile_alloc = runtime::options().profile_allocations;
  const auto alloc_start = std::chrono::steady_clock::now();
  auto after_mask_alloc = alloc_start;
  auto after_vector_alloc = alloc_start;
  auto after_mask_fill = alloc_start;
  auto after_row_copy = alloc_start;
  auto after_col_copy = alloc_start;
  const std::size_t row_mask_bytes = static_cast<std::size_t>(m);
  const std::size_t col_mask_bytes = static_cast<std::size_t>(n);
  const std::size_t row_vector_bytes = sizeof(double) * static_cast<std::size_t>(m);
  const std::size_t col_vector_bytes = sizeof(double) * static_cast<std::size_t>(n);
  auto align_up = [](std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
  };
  std::size_t offset = row_mask_bytes + col_mask_bytes;
  std::size_t c_offset = 0;
  std::size_t l_offset = 0;
  std::size_t u_offset = 0;
  std::size_t AL_offset = 0;
  std::size_t AU_offset = 0;
  auto reserve_vector = [&](bool required, std::size_t bytes, std::size_t& result) {
    if (!required) return;
    offset = align_up(offset, alignof(double));
    result = offset;
    offset += bytes;
  };
  reserve_vector(needs.objective, col_vector_bytes, c_offset);
  reserve_vector(needs.col_bounds, col_vector_bytes, l_offset);
  reserve_vector(needs.col_bounds, col_vector_bytes, u_offset);
  reserve_vector(needs.row_bounds, row_vector_bytes, AL_offset);
  reserve_vector(needs.row_bounds, row_vector_bytes, AU_offset);
  const std::size_t total_bytes = offset;
  if (total_bytes > 0) {
    throw_if_cuda_error(stream_ordered_malloc(&plan.contiguous_storage, total_bytes),
                        "cudaMalloc plan contiguous storage");
    auto* storage = static_cast<unsigned char*>(plan.contiguous_storage);
    plan.keep_row_mask = storage;
    plan.keep_col_mask = storage + row_mask_bytes;
    if (needs.objective) plan.new_c = reinterpret_cast<double*>(storage + c_offset);
    if (needs.col_bounds) {
      plan.new_l = reinterpret_cast<double*>(storage + l_offset);
      plan.new_u = reinterpret_cast<double*>(storage + u_offset);
    }
    if (needs.row_bounds) {
      plan.new_AL = reinterpret_cast<double*>(storage + AL_offset);
      plan.new_AU = reinterpret_cast<double*>(storage + AU_offset);
    }
  }
  if (profile_alloc) {
    after_mask_alloc = std::chrono::steady_clock::now();
  }
  if (profile_alloc) {
    after_vector_alloc = std::chrono::steady_clock::now();
  }
  constexpr int threads = 256;
  if (m > 0) {
    const int blocks_m = (m + threads - 1) / threads;
    _kernel_fill_u8<<<blocks_m, threads>>>(plan.keep_row_mask, std::uint8_t{1}, m);
    throw_if_cuda_error(cudaGetLastError(), "fill keep_row_mask");
  }
  if (n > 0) {
    const int blocks_n = (n + threads - 1) / threads;
    _kernel_fill_u8<<<blocks_n, threads>>>(plan.keep_col_mask, std::uint8_t{1}, n);
    throw_if_cuda_error(cudaGetLastError(), "fill keep_col_mask");
  }
  if (profile_alloc) {
    throw_if_cuda_error(cudaDeviceSynchronize(), "profile plan mask fill synchronize");
    after_mask_fill = std::chrono::steady_clock::now();
  }
  if (m > 0 && needs.row_bounds) {
    throw_if_cuda_error(cudaMemcpy(plan.new_AL, lp.AL, sizeof(double) * static_cast<std::size_t>(m), cudaMemcpyDeviceToDevice),
                        "cudaMemcpy plan AL");
    throw_if_cuda_error(cudaMemcpy(plan.new_AU, lp.AU, sizeof(double) * static_cast<std::size_t>(m), cudaMemcpyDeviceToDevice),
                        "cudaMemcpy plan AU");
  }
  if (profile_alloc) {
    after_row_copy = std::chrono::steady_clock::now();
  }
  if (n > 0 && needs.objective) {
    throw_if_cuda_error(cudaMemcpy(plan.new_c, lp.c, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                        "cudaMemcpy plan c");
  }
  if (n > 0 && needs.col_bounds) {
    throw_if_cuda_error(cudaMemcpy(plan.new_l, lp.l, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                        "cudaMemcpy plan l");
    throw_if_cuda_error(cudaMemcpy(plan.new_u, lp.u, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                        "cudaMemcpy plan u");
  }
  if (profile_alloc) {
    after_col_copy = std::chrono::steady_clock::now();
    const std::chrono::duration<double> mask_alloc_elapsed = after_mask_alloc - alloc_start;
    const std::chrono::duration<double> vector_alloc_elapsed = after_vector_alloc - after_mask_alloc;
    const std::chrono::duration<double> mask_fill_elapsed = after_mask_fill - after_vector_alloc;
    const std::chrono::duration<double> row_copy_elapsed = after_row_copy - after_mask_fill;
    const std::chrono::duration<double> col_copy_elapsed = after_col_copy - after_row_copy;
    const std::chrono::duration<double> total_elapsed = after_col_copy - alloc_start;
    std::cerr << ">>> [GPU Presolve C++ alloc] rows=" << m << " cols=" << n
              << " total=" << total_elapsed.count() << "s"
              << " mask_malloc=" << mask_alloc_elapsed.count() << "s"
              << " vector_malloc=" << vector_alloc_elapsed.count() << "s"
              << " mask_fill=" << mask_fill_elapsed.count() << "s"
              << " row_copy=" << row_copy_elapsed.count() << "s"
              << " col_copy=" << col_copy_elapsed.count() << "s\n";
  }
}

void free_plan(PresolvePlanGpu& plan) {
  plan.absolute_value_epigraph_device.reset();
  plan.variable_pair_aggregation_device.reset();
  const bool profile_alloc = runtime::options().profile_allocations;
  const auto free_start = std::chrono::steady_clock::now();
  if (plan.contiguous_storage != nullptr) {
    stream_ordered_free(plan.contiguous_storage);
  } else {
    stream_ordered_free(plan.keep_row_mask);
    stream_ordered_free(plan.keep_col_mask);
    stream_ordered_free(plan.new_c);
    stream_ordered_free(plan.new_l);
    stream_ordered_free(plan.new_u);
    stream_ordered_free(plan.new_AL);
    stream_ordered_free(plan.new_AU);
  }
  if (plan.has_new_A) {
    stream_ordered_free(plan.new_A.rowPtr);
    stream_ordered_free(plan.new_A.colVal);
    stream_ordered_free(plan.new_A.nzVal);
  }
  if (profile_alloc) {
    throw_if_cuda_error(cudaDeviceSynchronize(), "profile plan free synchronize");
    const std::chrono::duration<double> free_elapsed =
        std::chrono::steady_clock::now() - free_start;
    std::cerr << ">>> [GPU Presolve C++ alloc] free=" << free_elapsed.count() << "s\n";
  }
}

}  // namespace detail

namespace {
using detail::PlanRequirements;
using detail::allocate_plan;
using detail::free_plan;

std::int32_t build_prefix_from_mask(std::int32_t* prefix,
                                    const std::uint8_t* mask,
                                    std::int32_t n) {
  if (n == 0) {
    return 0;
  }
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  _kernel_mask_to_prefix_i32<<<blocks, threads>>>(prefix, mask, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_mask_to_prefix_i32");
  inclusive_scan_i32(prefix, n, "cub inclusive scan mask");
  std::int32_t count = 0;
  throw_if_cuda_error(cudaMemcpy(&count, prefix + n - 1, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy prefix count");
  return count;
}

std::int32_t* build_old_to_new(const std::uint8_t* mask,
                               std::int32_t n,
                               std::int32_t* count_out) {
  if (n == 0) {
    *count_out = 0;
    return nullptr;
  }
  CleanupScratchFrame scratch;
  auto* prefix = scratch.get<std::int32_t>(0, n);
  std::int32_t* old_to_new = nullptr;
  throw_if_cuda_error(stream_ordered_malloc(&old_to_new, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      "cudaMalloc old_to_new");
  const std::int32_t count = build_prefix_from_mask(prefix, mask, n);
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  _kernel_old_to_new_from_prefix<<<blocks, threads>>>(old_to_new, mask, prefix, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_old_to_new_from_prefix");
  *count_out = count;
  return old_to_new;
}

double* compact_vector(const double* src,
                       const std::int32_t* old_to_new,
                       std::int32_t n_old,
                       std::int32_t n_new) {
  double* dst = nullptr;
  if (n_new > 0) {
    throw_if_cuda_error(stream_ordered_malloc(&dst, sizeof(double) * static_cast<std::size_t>(n_new)),
                        "cudaMalloc compact vector");
  }
  if (n_old > 0 && n_new > 0) {
    constexpr int threads = 256;
    const int blocks = (n_old + threads - 1) / threads;
    _kernel_gather_by_old_to_new<<<blocks, threads>>>(dst, src, old_to_new, n_old);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_gather_by_old_to_new");
  }
  return dst;
}

template <class T>
std::vector<T> copy_device_array(const T* device, std::int32_t count, const char* context) {
  std::vector<T> values(static_cast<std::size_t>(count));
  if (count > 0) {
    throw_if_cuda_error(cudaMemcpy(values.data(), device, sizeof(T) * static_cast<std::size_t>(count),
                                   cudaMemcpyDeviceToHost),
                        context);
  }
  return values;
}

std::vector<std::int32_t> compact_identity_red2org_on_gpu(
    const std::int32_t* old_to_new,
    const double* l,
    const double* u,
    std::int32_t n_old,
    std::int32_t n_new,
    std::int32_t* removed_fixed_count,
    const char* context) {
  if (n_new <= 0) {
    *removed_fixed_count = 0;
    return {};
  }
  std::int32_t* red2org_d = nullptr;
  std::int32_t* removed_fixed_count_d = nullptr;
  throw_if_cuda_error(
      stream_ordered_malloc(
          &red2org_d,
          sizeof(std::int32_t) * static_cast<std::size_t>(n_new)),
      context);
  throw_if_cuda_error(stream_ordered_malloc(&removed_fixed_count_d,
                                            sizeof(std::int32_t)),
                      context);
  throw_if_cuda_error(cudaMemset(removed_fixed_count_d, 0,
                                 sizeof(std::int32_t)),
                      context);
  constexpr int threads = 256;
  const int blocks = (n_old + threads - 1) / threads;
  _kernel_identity_red2org_from_old_to_new<<<blocks, threads>>>(
      red2org_d, removed_fixed_count_d, old_to_new, l, u, 1.0e-9, n_old);
  throw_if_cuda_error(cudaGetLastError(),
                      "_kernel_identity_red2org_from_old_to_new");
  std::vector<std::int32_t> red2org(static_cast<std::size_t>(n_new));
  throw_if_cuda_error(
      cudaMemcpy(red2org.data(), red2org_d,
                 sizeof(std::int32_t) * static_cast<std::size_t>(n_new),
                 cudaMemcpyDeviceToHost),
      context);
  throw_if_cuda_error(cudaMemcpy(removed_fixed_count, removed_fixed_count_d,
                                 sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      context);
  stream_ordered_free(red2org_d);
  stream_ordered_free(removed_fixed_count_d);
  return red2org;
}

__global__ void _kernel_removed_fixed_bound_record_counts(std::int32_t* selected_scan,
                                                          const std::uint8_t* keep_col,
                                                          const double* l,
                                                          const double* u,
                                                          double bound_tol,
                                                          std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (col < n) {
    const bool removed = keep_col[col] == std::uint8_t{0};
    const bool fixed = isfinite(l[col]) && isfinite(u[col]) && fabs(l[col] - u[col]) <= bound_tol;
    selected_scan[col] = removed && fixed ? 1 : 0;
  }
}

__global__ void _kernel_pack_removed_fixed_bound_records(std::int32_t* packed_cols,
                                                         double* packed_vals,
                                                         const std::int32_t* selected_scan,
                                                         const std::uint8_t* keep_col,
                                                         const double* l,
                                                         const double* u,
                                                         double bound_tol,
                                                         std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (col >= n) {
    return;
  }
  const bool removed = keep_col[col] == std::uint8_t{0};
  const bool fixed = isfinite(l[col]) && isfinite(u[col]) && fabs(l[col] - u[col]) <= bound_tol;
  if (!removed || !fixed) {
    return;
  }
  const std::int32_t prev_selected = col == 0 ? 0 : selected_scan[col - 1];
  const NnzOffset out = selected_scan[col] - 1;
  if (selected_scan[col] == prev_selected || out < 0) {
    return;
  }
  packed_cols[out] = col;
  packed_vals[out] = 0.5 * (l[col] + u[col]);
}

__global__ void _kernel_gather_removed_bound_values(double* lower_out,
                                                    double* upper_out,
                                                    const std::int32_t* local_cols,
                                                    const double* l,
                                                    const double* u,
                                                    std::int32_t count) {
  const std::int32_t k = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (k < count) {
    const std::int32_t col = local_cols[k];
    lower_out[k] = l[col];
    upper_out[k] = u[col];
  }
}

__global__ void _kernel_removed_col_counts(std::int32_t* selected_scan,
                                           const std::uint8_t* keep_col,
                                           std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (col < n) {
    selected_scan[col] = keep_col[col] == std::uint8_t{0} ? 1 : 0;
  }
}

__global__ void _kernel_pack_removed_cols(std::int32_t* packed_cols,
                                          const std::int32_t* selected_scan,
                                          const std::uint8_t* keep_col,
                                          std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (col >= n || keep_col[col] != std::uint8_t{0}) {
    return;
  }
  const std::int32_t prev_selected = col == 0 ? 0 : selected_scan[col - 1];
  const NnzOffset out = selected_scan[col] - 1;
  if (selected_scan[col] == prev_selected || out < 0) {
    return;
  }
  packed_cols[out] = col;
}

std::vector<std::int32_t> pack_removed_cols_sparse(const std::uint8_t* keep_col,
                                                   std::int32_t n,
                                                   std::int32_t removed_count,
                                                   const char* context) {
  if (n <= 0 || removed_count <= 0) {
    return {};
  }

  std::int32_t* selected_scan = nullptr;
  std::int32_t* packed_cols = nullptr;
  throw_if_cuda_error(cudaMalloc(&selected_scan, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      context);
  throw_if_cuda_error(cudaMalloc(&packed_cols,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(removed_count)),
                      context);

  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  _kernel_removed_col_counts<<<blocks, threads>>>(selected_scan, keep_col, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_removed_col_counts");
  inclusive_scan_i32(selected_scan, n, context);
  _kernel_pack_removed_cols<<<blocks, threads>>>(packed_cols, selected_scan, keep_col, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_pack_removed_cols");

  std::vector<std::int32_t> host_cols(static_cast<std::size_t>(removed_count));
  throw_if_cuda_error(cudaMemcpy(host_cols.data(), packed_cols,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(removed_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  cudaFree(selected_scan);
  cudaFree(packed_cols);
  return host_cols;
}

void append_removed_fixed_bound_records_sparse(
    PresolveRecordGpu& record,
    const std::vector<std::int32_t>& removed_local_cols,
    const double* l,
    const double* u,
    const std::vector<std::int32_t>& old_col_red2org,
    const char* context) {
  const std::int32_t count = static_cast<std::int32_t>(removed_local_cols.size());
  if (count <= 0) {
    return;
  }

  std::int32_t* local_cols_d = nullptr;
  double* lower_d = nullptr;
  double* upper_d = nullptr;
  const std::size_t cols_bytes = sizeof(std::int32_t) * static_cast<std::size_t>(count);
  const std::size_t vals_bytes = sizeof(double) * static_cast<std::size_t>(count);
  throw_if_cuda_error(cudaMalloc(&local_cols_d, cols_bytes), context);
  throw_if_cuda_error(cudaMalloc(&lower_d, vals_bytes), context);
  throw_if_cuda_error(cudaMalloc(&upper_d, vals_bytes), context);
  throw_if_cuda_error(cudaMemcpy(local_cols_d, removed_local_cols.data(), cols_bytes,
                                 cudaMemcpyHostToDevice),
                      context);

  constexpr int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  _kernel_gather_removed_bound_values<<<blocks, threads>>>(
      lower_d, upper_d, local_cols_d, l, u, count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_gather_removed_bound_values");

  std::vector<double> lower(static_cast<std::size_t>(count));
  std::vector<double> upper(static_cast<std::size_t>(count));
  throw_if_cuda_error(cudaMemcpy(lower.data(), lower_d, vals_bytes, cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(upper.data(), upper_d, vals_bytes, cudaMemcpyDeviceToHost),
                      context);

  for (std::int32_t k = 0; k < count; ++k) {
    const double lo = lower[static_cast<std::size_t>(k)];
    const double hi = upper[static_cast<std::size_t>(k)];
    if (!isfinite(lo) || !isfinite(hi) || fabs(lo - hi) > 1.0e-9) {
      continue;
    }
    const std::int32_t local_col = removed_local_cols[static_cast<std::size_t>(k)];
    if (local_col >= 0 && local_col < static_cast<std::int32_t>(old_col_red2org.size())) {
      record.fixed_idx.push_back(old_col_red2org[static_cast<std::size_t>(local_col)]);
      record.fixed_val.push_back(0.5 * (lo + hi));
    }
  }

  cudaFree(local_cols_d);
  cudaFree(lower_d);
  cudaFree(upper_d);
}

void append_removed_fixed_bound_records(PresolveRecordGpu& record,
                                        const std::uint8_t* keep_col,
                                        const double* l,
                                        const double* u,
                                        const std::vector<std::int32_t>& old_col_red2org,
                                        std::int32_t n,
                                        const char* context) {
  if (n <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  std::int32_t* selected_scan = nullptr;
  throw_if_cuda_error(cudaMalloc(&selected_scan, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      context);
  _kernel_removed_fixed_bound_record_counts<<<blocks, threads>>>(
      selected_scan, keep_col, l, u, 1.0e-9, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_removed_fixed_bound_record_counts");
  inclusive_scan_i32(selected_scan, n, context);
  std::int32_t count = 0;
  throw_if_cuda_error(cudaMemcpy(&count, selected_scan + n - 1, sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      context);
  if (count <= 0) {
    cudaFree(selected_scan);
    return;
  }

  std::int32_t* packed_cols = nullptr;
  double* packed_vals = nullptr;
  throw_if_cuda_error(cudaMalloc(&packed_cols, sizeof(std::int32_t) * static_cast<std::size_t>(count)),
                      context);
  throw_if_cuda_error(cudaMalloc(&packed_vals, sizeof(double) * static_cast<std::size_t>(count)),
                      context);
  _kernel_pack_removed_fixed_bound_records<<<blocks, threads>>>(
      packed_cols, packed_vals, selected_scan, keep_col, l, u, 1.0e-9, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_pack_removed_fixed_bound_records");

  if (runtime::options().direct_fixed_values &&
      old_col_red2org.size() == static_cast<std::size_t>(n)) {
    const std::size_t index_start = record.fixed_idx.size();
    const std::size_t value_start = record.fixed_val.size();
    record.fixed_idx.resize(index_start + count);
    record.fixed_val.resize(value_start + count);
    auto* indices = record.fixed_idx.data() + index_start;
    throw_if_cuda_error(cudaMemcpy(indices, packed_cols, sizeof(std::int32_t) * count,
                                   cudaMemcpyDeviceToHost), context);
    throw_if_cuda_error(cudaMemcpy(record.fixed_val.data() + value_start, packed_vals,
                                   sizeof(double) * count, cudaMemcpyDeviceToHost), context);
    const auto remap = [&](unsigned, std::size_t begin, std::size_t end) {
      for (std::size_t k = begin; k < end; ++k) indices[k] = old_col_red2org[indices[k]];
    };
    if (count >= (1 << 20)) detail::mapping_parallel_for(count, remap);
    else remap(0, 0, count);
  } else {
    std::vector<std::int32_t> host_cols(static_cast<std::size_t>(count));
    std::vector<double> host_vals(static_cast<std::size_t>(count));
    throw_if_cuda_error(cudaMemcpy(host_cols.data(), packed_cols,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(count),
                                   cudaMemcpyDeviceToHost),
                        context);
    throw_if_cuda_error(cudaMemcpy(host_vals.data(), packed_vals,
                                   sizeof(double) * static_cast<std::size_t>(count),
                                   cudaMemcpyDeviceToHost),
                        context);
    record.fixed_idx.reserve(record.fixed_idx.size() + static_cast<std::size_t>(count));
    record.fixed_val.reserve(record.fixed_val.size() + static_cast<std::size_t>(count));
    for (std::int32_t k = 0; k < count; ++k) {
      const std::int32_t local_col = host_cols[static_cast<std::size_t>(k)];
      if (local_col >= 0 && local_col < static_cast<std::int32_t>(old_col_red2org.size())) {
        record.fixed_idx.push_back(old_col_red2org[static_cast<std::size_t>(local_col)]);
        record.fixed_val.push_back(host_vals[static_cast<std::size_t>(k)]);
      }
    }
  }
  cudaFree(selected_scan);
  cudaFree(packed_cols);
  cudaFree(packed_vals);
}

std::vector<std::int32_t> make_identity_i32(std::int32_t n) {
  std::vector<std::int32_t> values(static_cast<std::size_t>(n));
  if (values.size() <
      static_cast<std::size_t>(PARALLEL_IDENTITY_FILL_THRESHOLD)) {
    for (std::int32_t i = 0; i < n; ++i) {
      values[static_cast<std::size_t>(i)] = i;
    }
    return values;
  }

  const unsigned hardware_threads = std::thread::hardware_concurrency();
  const unsigned worker_count = std::min<unsigned>(
      PARALLEL_IDENTITY_FILL_MAX_THREADS,
      hardware_threads == 0 ? 1U : hardware_threads);
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  const std::int64_t chunk =
      (static_cast<std::int64_t>(n) + worker_count - 1) / worker_count;
  try {
    for (unsigned worker = 0; worker < worker_count; ++worker) {
      const NnzOffset begin = static_cast<NnzOffset>(
          std::min<std::int64_t>(static_cast<std::int64_t>(n),
                                 static_cast<std::int64_t>(worker) * chunk));
      const NnzOffset end = static_cast<NnzOffset>(
          std::min<std::int64_t>(static_cast<std::int64_t>(n),
                                 static_cast<std::int64_t>(worker + 1) * chunk));
      workers.emplace_back([&values, begin, end]() {
        for (std::int32_t i = begin; i < end; ++i) {
          values[static_cast<std::size_t>(i)] = i;
        }
      });
    }
  } catch (...) {
    for (std::thread& worker : workers) {
      worker.join();
    }
    throw;
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
  return values;
}

PresolveRecordGpu make_initial_record(const LPInfoGpu& lp) {
  PresolveRecordGpu record;
  record.m0 = lp.A.rows;
  record.n0 = lp.A.cols;
  record.m1 = lp.A.rows;
  record.n1 = lp.A.cols;
  if (device_recovery_enabled())
    initialize_device_recovery(record);
  else {
    record.row_red2org = make_identity_i32(lp.A.rows);
    record.col_red2org = make_identity_i32(lp.A.cols);
  }
  record.obj_constant_old = lp.obj_constant;
  record.obj_constant_new = lp.obj_constant;
  return record;
}

void rebuild_org2red(std::vector<std::int32_t>& org2red,
                     const std::vector<std::int32_t>& red2org,
                     std::int32_t original_count) {
  org2red.assign(static_cast<std::size_t>(original_count), -1);
  for (std::int32_t red = 0; red < static_cast<std::int32_t>(red2org.size()); ++red) {
    const std::int32_t org = red2org[static_cast<std::size_t>(red)];
    if (org >= 0 && org < original_count) {
      org2red[static_cast<std::size_t>(org)] = red;
    }
  }
}

AbsoluteValueEpigraphPrimalRecoveryStep globalize_absolute_value_epigraph_recovery(
    const AbsoluteValueEpigraphPrimalRecoveryStep& local,
    const std::vector<std::int32_t>& col_red2org) {
  AbsoluteValueEpigraphPrimalRecoveryStep global = local;
  for (AbsoluteValueEpigraphSplitRecovery& split : global.splits) {
    split.t_col = col_red2org[static_cast<std::size_t>(split.t_col)];
    split.e_col = col_red2org[static_cast<std::size_t>(split.e_col)];
  }
  for (AbsoluteValueOuterPairRecovery& pair : global.outer_pairs) {
    pair.bound_col = col_red2org[static_cast<std::size_t>(pair.bound_col)];
    pair.free_col = col_red2org[static_cast<std::size_t>(pair.free_col)];
  }
  for (AbsoluteValueLinkedSlackRecovery& slack : global.linked_slacks) {
    slack.slack_col = col_red2org[static_cast<std::size_t>(slack.slack_col)];
    slack.t_col = col_red2org[static_cast<std::size_t>(slack.t_col)];
  }
  for (AbsoluteValueMaxSlackRecovery& slack : global.max_slacks) {
    slack.slack_col = col_red2org[static_cast<std::size_t>(slack.slack_col)];
    for (std::int32_t& t_col : slack.t_cols) {
      t_col = col_red2org[static_cast<std::size_t>(t_col)];
    }
  }
  return global;
}

VariablePairAggregationRecoveryStep globalize_variable_pair_aggregation_recovery(
    const VariablePairAggregationRecoveryStep& local,
    const std::vector<std::int32_t>& col_red2org) {
  VariablePairAggregationRecoveryStep global = local;
  const std::size_t count = global.plus_cols.size();
  for (std::size_t i = 0; i < count; ++i) {
    global.plus_cols[i] =
        col_red2org[static_cast<std::size_t>(global.plus_cols[i])];
    global.minus_cols[i] =
        col_red2org[static_cast<std::size_t>(global.minus_cols[i])];
    global.root_plus_cols[i] =
        col_red2org[static_cast<std::size_t>(global.root_plus_cols[i])];
    global.root_minus_cols[i] =
        col_red2org[static_cast<std::size_t>(global.root_minus_cols[i])];
  }
  return global;
}

std::int32_t map_local_col_to_global(std::int32_t col,
                                     const std::vector<std::int32_t>& col_red2org) {
  if (col < 0 || col >= static_cast<std::int32_t>(col_red2org.size())) {
    return col;
  }
  return col_red2org[static_cast<std::size_t>(col)];
}

std::int32_t map_local_row_to_global(std::int32_t row,
                                     const std::vector<std::int32_t>& row_red2org) {
  if (row < 0 || row >= static_cast<std::int32_t>(row_red2org.size())) {
    return row;
  }
  return row_red2org[static_cast<std::size_t>(row)];
}

// Opt-in diagnostic timings. No synchronization or output when disabled.
class DetailStageTimer {
 public:
  explicit DetailStageTimer(const char* section) : section_(section),
      enabled_(runtime::options().profile_details) {
    if (enabled_) last_ = std::chrono::steady_clock::now();
  }
  void mark(const char* stage, bool synchronize = false) {
    if (!enabled_) return;
    if (synchronize) throw_if_cuda_error(cudaDeviceSynchronize(), "detail profile synchronize");
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - last_).count();
    std::cerr << ">>> [GPU Presolve C++ detail-profile] section=" << section_
              << " stage=" << stage << " time=" << seconds << "s\n";
    last_ = std::chrono::steady_clock::now();
  }
 private:
  const char* section_;
  bool enabled_;
  std::chrono::steady_clock::time_point last_;
};

void remap_postsolve_tape_indices(PostsolveTape& dest, const PostsolveTape& src,
                                  std::size_t old_indices,
                                  const std::vector<std::int32_t>& row_red2org,
                                  const std::vector<std::int32_t>& col_red2org) {
  struct IndexSegment {
    std::vector<std::int32_t>& values;
    std::size_t offset;
    std::size_t count;

    bool empty() const { return count == 0; }
    std::size_t size() const { return count; }
    std::int32_t& operator[](std::size_t index) { return values[offset + index]; }
  };

  const auto remap_range = [&](NnzOffset begin, NnzOffset end) {
  for (std::int32_t k = begin; k < end; ++k) {
    const NnzOffset idx0 = src.index_starts[static_cast<std::size_t>(k)];
    const NnzOffset idx1 = src.index_starts[static_cast<std::size_t>(k + 1)];
    const std::size_t index_offset =
        old_indices + static_cast<std::size_t>(idx0);
    IndexSegment indices{dest.indices, index_offset,
                         static_cast<std::size_t>(idx1 - idx0)};

    switch (static_cast<PostsolveReductionType>(src.types[static_cast<std::size_t>(k)])) {
      case PostsolveReductionType::FixedCol:
        if (!indices.empty()) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
          for (std::size_t p = 1; p < indices.size(); ++p) {
            indices[p] = map_local_row_to_global(indices[p], row_red2org);
          }
        }
        break;
      case PostsolveReductionType::FixedColInf:
        if (indices.size() >= 2) {
          indices[1] = map_local_col_to_global(indices[1], col_red2org);
          std::size_t p = 2;
          while (p < indices.size()) {
            const NnzOffset row_len = indices[p++];
            for (std::int32_t t = 0; t < row_len && p < indices.size(); ++t, ++p) {
              indices[p] = map_local_col_to_global(indices[p], col_red2org);
            }
          }
        }
        break;
      case PostsolveReductionType::SubCol:
        if (indices.size() >= 3) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
          indices[1] = map_local_row_to_global(indices[1], row_red2org);
          const std::int32_t support_count = indices[2];
          for (std::int32_t t = 0; t < support_count && 3 + t < static_cast<std::int32_t>(indices.size()); ++t) {
            indices[static_cast<std::size_t>(3 + t)] =
                map_local_col_to_global(indices[static_cast<std::size_t>(3 + t)], col_red2org);
          }
        }
        break;
      case PostsolveReductionType::ParallelColumn:
        if (indices.size() >= 2) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
          indices[1] = map_local_col_to_global(indices[1], col_red2org);
        }
        break;
      case PostsolveReductionType::ParallelRow:
        if (indices.size() >= 2) {
          indices[0] = map_local_row_to_global(indices[0], row_red2org);
          indices[1] = map_local_row_to_global(indices[1], row_red2org);
        }
        break;
      case PostsolveReductionType::DeletedRow:
        if (!indices.empty()) {
          indices[0] = map_local_row_to_global(indices[0], row_red2org);
          if (indices.size() >= 2) {
            indices[1] = map_local_col_to_global(indices[1], col_red2org);
          }
        }
        break;
      case PostsolveReductionType::LhsChange:
      case PostsolveReductionType::RhsChange:
      case PostsolveReductionType::EqToIneq:
        if (!indices.empty()) {
          indices[0] = map_local_row_to_global(indices[0], row_red2org);
        }
        break;
      case PostsolveReductionType::BoundChangeNoRow:
        if (!indices.empty()) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
        }
        break;
      case PostsolveReductionType::BoundChangeTheRow:
        if (indices.size() >= 2) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
          indices[1] = map_local_row_to_global(indices[1], row_red2org);
        }
        break;
      case PostsolveReductionType::DoubletonEquation:
        if (indices.size() >= 4) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
          indices[1] = map_local_col_to_global(indices[1], col_red2org);
          indices[2] = map_local_row_to_global(indices[2], row_red2org);
          const std::int32_t support_count = indices[3];
          for (std::int32_t t = 0; t < support_count && 4 + t < static_cast<std::int32_t>(indices.size()); ++t) {
            indices[static_cast<std::size_t>(4 + t)] =
                map_local_row_to_global(indices[static_cast<std::size_t>(4 + t)], row_red2org);
          }
        }
        break;
      case PostsolveReductionType::FmeCol:
        if (indices.size() >= 3) {
          const std::int32_t involved_count = indices[0];
          indices[1] = map_local_col_to_global(indices[1], col_red2org);
          for (std::int32_t t = 0; t < involved_count && 2 + t < static_cast<std::int32_t>(indices.size()); ++t) {
            indices[static_cast<std::size_t>(2 + t)] =
                map_local_row_to_global(indices[static_cast<std::size_t>(2 + t)], row_red2org);
          }
          const std::int32_t side_count_pos = 2 + involved_count;
          if (side_count_pos < static_cast<std::int32_t>(indices.size())) {
            const std::int32_t side_col_count = indices[static_cast<std::size_t>(side_count_pos)];
            for (std::int32_t t = 0; t < side_col_count &&
                                     side_count_pos + 1 + t < static_cast<std::int32_t>(indices.size()); ++t) {
              indices[static_cast<std::size_t>(side_count_pos + 1 + t)] =
                  map_local_col_to_global(indices[static_cast<std::size_t>(side_count_pos + 1 + t)], col_red2org);
            }
          }
        }
        break;
      case PostsolveReductionType::AddedRow:
      case PostsolveReductionType::AddedRows:
        break;
    }

  }
  };
  // Only disjoint index segments are written. Record order and all floating
  // values remain untouched; maps and source offsets are read-only.
  const auto count = static_cast<NnzOffset>(src.types.size());
  const unsigned workers_count = runtime::options().parallel_tape_build &&
          src.indices.size() >= (1U << 20) && count >= 1024
      ? std::min(8U, std::max(1U, std::thread::hardware_concurrency())) : 1U;
  if (workers_count == 1) {
    remap_range(0, count);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(workers_count - 1);
    const auto boundary = [&](unsigned i) {
      return static_cast<std::int32_t>(static_cast<std::int64_t>(count) * i / workers_count);
    };
    try {
      for (unsigned i = 1; i < workers_count; ++i) {
        workers.emplace_back(remap_range, boundary(i), boundary(i + 1));
      }
      remap_range(0, boundary(1));
    } catch (...) {
      for (auto& worker : workers) worker.join();
      throw;
    }
    for (auto& worker : workers) worker.join();
  }

}

void append_globalized_postsolve_tape(PostsolveTape& dest,
                                      const PostsolveTape& src,
                                      const std::vector<std::int32_t>& row_red2org,
                                      const std::vector<std::int32_t>& col_red2org) {
  if (src.types.empty()) {
    return;
  }
  DetailStageTimer detail("tape");
  const std::size_t old_records = dest.types.size();
  const std::size_t old_indices = dest.indices.size();
  const std::size_t old_values = dest.vals.size();
  const bool tape_headroom = runtime::options().reserve_tape_headroom;
  const auto reserve_for_append = [tape_headroom](auto& values, std::size_t added) {
    const std::size_t required = values.size() + added;
    if (required <= values.capacity()) {
      return;
    }
    const std::size_t doubled = values.capacity() <=
            std::numeric_limits<std::size_t>::max() / 2
        ? values.capacity() * 2
        : std::numeric_limits<std::size_t>::max();
    // A large first append otherwise fills capacity exactly and the following
    // tiny append copies the entire history. Reserve bounded, untouched slack.
    const std::size_t extra = tape_headroom && required >= (1U << 20) / sizeof(values[0])
        ? std::min(required / 4, (64U << 20) / sizeof(values[0])) : 0;
    values.reserve(std::max(required + extra, doubled));
  };
  reserve_for_append(dest.types, src.types.size());
  reserve_for_append(dest.index_starts, src.types.size());
  reserve_for_append(dest.value_starts, src.types.size());
  reserve_for_append(dest.dual_modes, src.dual_modes.size());
  reserve_for_append(dest.indices, src.indices.size());
  reserve_for_append(dest.vals, src.vals.size());

  detail.mark("reserve");

  // Preserve the exact chronological representation, but append each flat
  // array only once.  The previous per-record vector::insert calls repeated
  // millions of tiny range operations for singleton-heavy models.
  dest.types.insert(dest.types.end(), src.types.begin(), src.types.end());
  dest.dual_modes.insert(dest.dual_modes.end(),
                         src.dual_modes.begin(), src.dual_modes.end());
  dest.indices.insert(dest.indices.end(), src.indices.begin(), src.indices.end());
  dest.vals.insert(dest.vals.end(), src.vals.begin(), src.vals.end());

  detail.mark("append_arrays");

  dest.index_starts.resize(old_records + src.types.size() + 1);
  dest.value_starts.resize(old_records + src.types.size() + 1);
  for (std::size_t k = 1; k <= src.types.size(); ++k) {
    dest.index_starts[old_records + k] = static_cast<NnzOffset>(
        old_indices + static_cast<std::size_t>(src.index_starts[k]));
    dest.value_starts[old_records + k] = static_cast<NnzOffset>(
        old_values + static_cast<std::size_t>(src.value_starts[k]));
  }

  detail.mark("offsets");

  remap_postsolve_tape_indices(dest, src, old_indices, row_red2org, col_red2org);
  detail.mark("remap");
}

// Private to a single presolve invocation. Rule order and recovery checkpoints
// use cumulative record counts; public records are flattened before return.
struct HostTapeChunks {
  bool enabled = runtime::options().chunked_tape;
  std::size_t records = 0, indices = 0, values = 0;
  std::vector<PostsolveTape> segments;

  void append(PostsolveTape& source,
              const std::vector<std::int32_t>& rows,
              const std::vector<std::int32_t>& cols) {
    if (source.types.empty()) return;
    if (source.types.size() > INT64_MAX - records ||
        source.indices.size() > INT64_MAX - indices ||
        source.vals.size() > INT64_MAX - values)
      throw std::runtime_error("chunked postsolve tape exceeds 32-bit range");
    // The source starts and types remain read-only while disjoint index
    // segments are globalized in place, before moving ownership from the plan.
    remap_postsolve_tape_indices(source, source, 0, rows, cols);
    records += source.types.size();indices += source.indices.size();values += source.vals.size();
    segments.push_back(std::move(source));
    source = PostsolveTape{};
  }

  void finish(PostsolveTape& dest) {
    if (!enabled || segments.empty()) return;
    DetailStageTimer timer("tape_chunks");
    dest = std::move(segments.front());
    // One allocation per final array; no repeated movement of historical data.
    dest.types.reserve(records);dest.dual_modes.reserve(records);
    dest.index_starts.reserve(records+1);dest.value_starts.reserve(records+1);
    dest.indices.reserve(indices);dest.vals.reserve(values);
    for (std::size_t q=1; q<segments.size(); ++q) {
      auto& src=segments[q];
      const auto old_records=dest.types.size();
      const auto old_indices=dest.indices.size(), old_values=dest.vals.size();
      dest.types.insert(dest.types.end(),src.types.begin(),src.types.end());
      dest.dual_modes.insert(dest.dual_modes.end(),src.dual_modes.begin(),src.dual_modes.end());
      dest.indices.insert(dest.indices.end(),src.indices.begin(),src.indices.end());
      dest.vals.insert(dest.vals.end(),src.vals.begin(),src.vals.end());
      dest.index_starts.resize(old_records+src.types.size()+1);
      dest.value_starts.resize(old_records+src.types.size()+1);
      for(std::size_t k=1;k<=src.types.size();++k) {
        dest.index_starts[old_records+k]=static_cast<NnzOffset>(old_indices+src.index_starts[k]);
        dest.value_starts[old_records+k]=static_cast<NnzOffset>(old_values+src.value_starts[k]);
      }
      src=PostsolveTape{};
    }
    segments.clear();
    timer.mark("flatten");
  }
};

void update_record_from_plan(PresolveRecordGpu &record, PresolvePlanGpu &plan,
                             const LPInfoGpu &old_lp, std::int32_t m_new, std::int32_t n_new,
                             double obj_constant_new, const std::int32_t *row_old_to_new,
                             const std::int32_t *col_old_to_new, HostTapeChunks &tape_chunks) {
  if (record.device) {
    update_device_recovery(record, plan, old_lp, m_new, n_new, obj_constant_new, row_old_to_new,
                           col_old_to_new);
    return;
  }
  const auto source_records=plan.tape.types.size(), source_indices=plan.tape.indices.size(), source_values=plan.tape.vals.size();
  DetailStageTimer detail("mapping");
  const bool profile_record = runtime::options().profile_records;
  const auto record_start = std::chrono::steady_clock::now();
  const bool has_removed_rows = m_new < old_lp.A.rows;
  const bool has_removed_cols = n_new < old_lp.A.cols;
  const std::int32_t removed_row_count = has_removed_rows ? (old_lp.A.rows - m_new) : 0;
  const std::int32_t removed_col_count = has_removed_cols ? (old_lp.A.cols - n_new) : 0;
  const bool sparse_row_update =
      has_removed_rows && removed_row_count <= SPARSE_FIXED_BOUND_RECORD_THRESHOLD;
  const bool sparse_col_update =
      has_removed_cols && removed_col_count <= SPARSE_FIXED_BOUND_RECORD_THRESHOLD;
  const bool record_removed_indices = runtime::options().record_removed_indices;
  const bool identity_gpu_col_update =
      has_removed_cols && n_new > 0 && old_lp.A.cols >= 1000000 &&
      removed_col_count >= old_lp.A.cols / 8 &&
      record.col_red2org.size() == static_cast<std::size_t>(record.n0) &&
      !record_removed_indices;
  const std::vector<std::uint8_t> keep_row = has_removed_rows && !sparse_row_update
      ? copy_device_array(plan.keep_row_mask, old_lp.A.rows, "cudaMemcpy record keep_row")
      : std::vector<std::uint8_t>{};
  const std::vector<std::uint8_t> keep_col =
      has_removed_cols && !sparse_col_update && !identity_gpu_col_update
      ? copy_device_array(plan.keep_col_mask, old_lp.A.cols, "cudaMemcpy record keep_col")
      : std::vector<std::uint8_t>{};
  const auto after_keep_copy = std::chrono::steady_clock::now();

  const std::vector<std::int32_t>& old_row_red2org = record.row_red2org;
  const std::vector<std::int32_t>& old_col_red2org = record.col_red2org;
  const auto after_old_mapping_copy = std::chrono::steady_clock::now();
  std::vector<std::int32_t> next_row_red2org;
  std::vector<std::int32_t> next_col_red2org;
  std::vector<std::int32_t> removed_local_rows;
  std::vector<std::int32_t> removed_local_cols;
  std::int32_t identity_removed_fixed_count = -1;
  if (has_removed_rows) {
    next_row_red2org.reserve(static_cast<std::size_t>(m_new));
    if (record_removed_indices) {
      record.removed_row_idx.reserve(record.removed_row_idx.size() +
                                     static_cast<std::size_t>(old_lp.A.rows - m_new));
    }
    if (sparse_row_update) {
      removed_local_rows.reserve(static_cast<std::size_t>(removed_row_count));
    }
  }
  if (has_removed_cols) {
    next_col_red2org.reserve(static_cast<std::size_t>(n_new));
    if (record_removed_indices) {
      record.removed_col_idx.reserve(record.removed_col_idx.size() +
                                     static_cast<std::size_t>(old_lp.A.cols - n_new));
    }
    if (sparse_col_update) {
      removed_local_cols.reserve(static_cast<std::size_t>(removed_col_count));
    }
  }

  detail.mark("keep_copy_and_reserve");
  if (has_removed_rows) {
    if (sparse_row_update) {
      removed_local_rows = pack_removed_cols_sparse(plan.keep_row_mask,
                                                    old_lp.A.rows,
                                                    removed_row_count,
                                                    "cudaMemcpy sparse removed rows");
      NnzOffset range_begin = 0;
      for (const std::int32_t local_row : removed_local_rows) {
        if (local_row < range_begin ||
            local_row >= static_cast<std::int32_t>(old_row_red2org.size())) {
          continue;
        }
        next_row_red2org.insert(next_row_red2org.end(),
                                old_row_red2org.begin() + range_begin,
                                old_row_red2org.begin() + local_row);
        if (record_removed_indices) {
          record.removed_row_idx.push_back(
              old_row_red2org[static_cast<std::size_t>(local_row)]);
        }
        range_begin = local_row + 1;
      }
      next_row_red2org.insert(next_row_red2org.end(),
                              old_row_red2org.begin() + range_begin,
                              old_row_red2org.end());
    } else if (runtime::options().parallel_mapping && old_row_red2org.size() >= (1U << 20)) {
      detail::compact_mapping_stable(old_row_red2org, keep_row, m_new, next_row_red2org,
          record_removed_indices ? &record.removed_row_idx : nullptr);
    } else {
      next_row_red2org.resize(static_cast<std::size_t>(m_new));
      std::size_t next_row = 0;
      for (std::int32_t row = 0; row < old_lp.A.rows; ++row) {
        const NnzOffset global =
            old_row_red2org[static_cast<std::size_t>(row)];
        if (keep_row[static_cast<std::size_t>(row)] != std::uint8_t{0}) {
          next_row_red2org[next_row++] = global;
        } else {
          if (record_removed_indices) {
            record.removed_row_idx.push_back(global);
          }
        }
      }
      if (next_row != static_cast<std::size_t>(m_new)) {
        throw std::runtime_error("row mapping compaction count mismatch");
      }
    }
  }
  detail.mark("row_mapping");
  if (has_removed_cols) {
    if (identity_gpu_col_update) {
      next_col_red2org = compact_identity_red2org_on_gpu(
          col_old_to_new,
          plan.new_l != nullptr ? plan.new_l : old_lp.l,
          plan.new_u != nullptr ? plan.new_u : old_lp.u,
          old_lp.A.cols, n_new, &identity_removed_fixed_count,
          "cudaMemcpy compact identity column mapping");
    } else if (sparse_col_update) {
      removed_local_cols = pack_removed_cols_sparse(plan.keep_col_mask,
                                                    old_lp.A.cols,
                                                    removed_col_count,
                                                    "cudaMemcpy sparse removed cols");
      NnzOffset range_begin = 0;
      for (const std::int32_t local_col : removed_local_cols) {
        if (local_col < range_begin ||
            local_col >= static_cast<std::int32_t>(old_col_red2org.size())) {
          continue;
        }
        next_col_red2org.insert(next_col_red2org.end(),
                                old_col_red2org.begin() + range_begin,
                                old_col_red2org.begin() + local_col);
        if (record_removed_indices) {
          record.removed_col_idx.push_back(old_col_red2org[static_cast<std::size_t>(local_col)]);
        }
        range_begin = local_col + 1;
      }
      next_col_red2org.insert(next_col_red2org.end(),
                              old_col_red2org.begin() + range_begin,
                              old_col_red2org.end());
    } else if (runtime::options().parallel_mapping && old_col_red2org.size() >= (1U << 20)) {
      detail::compact_mapping_stable(old_col_red2org, keep_col, n_new, next_col_red2org,
          record_removed_indices ? &record.removed_col_idx : nullptr);
    } else {
      next_col_red2org.resize(static_cast<std::size_t>(n_new));
      std::size_t next_col = 0;
      for (std::int32_t col = 0; col < old_lp.A.cols; ++col) {
        const NnzOffset global =
            old_col_red2org[static_cast<std::size_t>(col)];
        if (keep_col[static_cast<std::size_t>(col)] != std::uint8_t{0}) {
          next_col_red2org[next_col++] = global;
        } else {
          if (record_removed_indices) {
            record.removed_col_idx.push_back(global);
          }
        }
      }
      if (next_col != static_cast<std::size_t>(n_new)) {
        throw std::runtime_error("column mapping compaction count mismatch");
      }
    }
    detail.mark("col_mapping");
    if (plan.cleanup_recovery_captured ||
        (identity_gpu_col_update && identity_removed_fixed_count == 0)) {
      // The identity compaction kernel already checked every removed column.
      // A nonzero count falls through to the original stable packing path.
    } else if (sparse_col_update) {
      append_removed_fixed_bound_records_sparse(record,
                                               removed_local_cols,
                                               plan.new_l != nullptr ? plan.new_l : old_lp.l,
                                               plan.new_u != nullptr ? plan.new_u : old_lp.u,
                                               old_col_red2org,
                                               "cudaMemcpy sparse record fixed bounds");
    } else {
      append_removed_fixed_bound_records(record,
                                         plan.keep_col_mask,
                                         plan.new_l != nullptr ? plan.new_l : old_lp.l,
                                         plan.new_u != nullptr ? plan.new_u : old_lp.u,
                                         old_col_red2org,
                                         old_lp.A.cols,
                                         "cudaMemcpy record fixed bounds");
    }
  }
  detail.mark("fixed_records");
  const auto after_removed_mapping = std::chrono::steady_clock::now();

  const bool has_primal_recovery =
      plan.has_absolute_value_epigraph_recovery ||
      plan.has_variable_pair_aggregation_recovery;
  if (plan.has_absolute_value_epigraph_recovery &&
      plan.has_variable_pair_aggregation_recovery) {
    throw std::runtime_error(
        "a presolve plan cannot commit structural and antipodal primal "
        "recoveries at the same chronological position");
  }
  if (has_primal_recovery && !plan.tape.types.empty()) {
    throw std::runtime_error(
        "a primal-recovery plan must not also contain ordinary postsolve "
        "records; split it into chronological plans");
  }
  const auto tape_size=tape_chunks.enabled?tape_chunks.records:record.tape.types.size();
  if (tape_size > static_cast<std::size_t>(INT64_MAX)) {
    throw std::runtime_error("postsolve tape exceeds 32-bit checkpoint range");
  }
  const NnzOffset tape_position =
      static_cast<NnzOffset>(tape_size);
  if (plan.has_absolute_value_epigraph_recovery) {
    if (record.absolute_value_epigraph_recoveries.size() >=
        static_cast<std::size_t>(INT64_MAX)) {
      throw std::runtime_error(
          "structural recovery payload exceeds 32-bit checkpoint range");
    }
    const NnzOffset payload_index = static_cast<NnzOffset>(
        record.absolute_value_epigraph_recoveries.size());
    record.absolute_value_epigraph_recoveries.push_back(
        globalize_absolute_value_epigraph_recovery(plan.absolute_value_epigraph_recovery,
                                      old_col_red2org));
    record.primal_recovery_timeline.push_back(PrimalRecoveryCheckpoint{
        PrimalRecoveryKind::AbsoluteValueEpigraph, payload_index, tape_position});
  }
  if (plan.has_variable_pair_aggregation_recovery) {
    if (record.variable_pair_aggregation_recoveries.size() >=
        static_cast<std::size_t>(INT64_MAX)) {
      throw std::runtime_error(
          "antipodal recovery payload exceeds 32-bit checkpoint range");
    }
    const NnzOffset payload_index = static_cast<NnzOffset>(
        record.variable_pair_aggregation_recoveries.size());
    record.variable_pair_aggregation_recoveries.push_back(
        globalize_variable_pair_aggregation_recovery(plan.variable_pair_aggregation_recovery,
                                     old_col_red2org));
    record.primal_recovery_timeline.push_back(PrimalRecoveryCheckpoint{
        PrimalRecoveryKind::VariablePairAggregation, payload_index,
        tape_position});
    record.has_primal_only_variable_pair_aggregation_reduction = true;
  }
  if (plan.has_column_domination_reduction) {
    record.has_primal_only_column_domination_reduction = true;
  }
  if (plan.has_projected_auxiliary_reduction) {
    record.has_primal_only_projected_auxiliary_reduction = true;
  }
  if (tape_chunks.enabled) tape_chunks.append(plan.tape, old_row_red2org, old_col_red2org);
  else append_globalized_postsolve_tape(record.tape, plan.tape, old_row_red2org, old_col_red2org);
  // Keep the canonical tape append-only on the host during presolve.  Growing
  // one contiguous device mirror after every rule copies the entire tape each
  // time a capacity boundary is crossed; on large models that turns a tiny
  // final append into hundreds of milliseconds of historical D2D copies.
  // The complete device tape is materialized once, in canonical order, after
  // the last presolve rule has committed.
  const auto after_tape = std::chrono::steady_clock::now();

  if (has_removed_rows) {
    record.row_red2org = std::move(next_row_red2org);
  }
  if (has_removed_cols) {
    record.col_red2org = std::move(next_col_red2org);
  }
  record.m1 = m_new;
  record.n1 = n_new;
  record.obj_constant_new = obj_constant_new;
  const auto after_rebuild = std::chrono::steady_clock::now();
  if (profile_record) {
    const std::chrono::duration<double> keep_copy_elapsed = after_keep_copy - record_start;
    const std::chrono::duration<double> old_mapping_copy_elapsed = after_old_mapping_copy - after_keep_copy;
    const std::chrono::duration<double> removed_mapping_elapsed = after_removed_mapping - after_old_mapping_copy;
    const std::chrono::duration<double> tape_elapsed = after_tape - after_removed_mapping;
    const std::chrono::duration<double> rebuild_elapsed = after_rebuild - after_tape;
    const std::chrono::duration<double> total_elapsed = after_rebuild - record_start;
    std::cerr << ">>> [GPU Presolve C++ record-profile]"
              << " dims=(" << old_lp.A.rows << "," << old_lp.A.cols << ")->("
              << m_new << "," << n_new << ")"
              << " removed_rows=" << (old_lp.A.rows - m_new)
              << " removed_cols=" << (old_lp.A.cols - n_new)
              << " tape_records=" << source_records
              << " tape_indices=" << source_indices
              << " tape_vals=" << source_values
              << " keep_copy=" << keep_copy_elapsed.count() << "s"
              << " old_mapping_copy=" << old_mapping_copy_elapsed.count() << "s"
              << " removed_mapping=" << removed_mapping_elapsed.count() << "s"
              << " tape=" << tape_elapsed.count() << "s"
              << " rebuild=" << rebuild_elapsed.count() << "s"
              << " total=" << total_elapsed.count() << "s\n";
  }
}

DeviceCsrMatrix transpose_csr_gpu(const DeviceCsrMatrix& A) {
  return gpu_presolver::presolve::transpose_csr64(A);
}

DeviceCsrMatrix compact_csr_by_masks_gpu(const DeviceCsrMatrix& A,
                                         const std::int32_t* row_old_to_new,
                                         const std::int32_t* col_old_to_new,
                                         std::int32_t m_new,
                                         std::int32_t n_new) {
  DetailStageTimer detail("compact");
  DeviceCsrMatrix out;
  out.rows = m_new;
  out.cols = n_new;
  throw_if_cuda_error(stream_ordered_malloc(&out.rowPtr, sizeof(NnzOffset) * static_cast<std::size_t>(m_new + 1)),
                      "cudaMalloc compact rowPtr");
  if (m_new == 0 || n_new == 0) {
    throw_if_cuda_error(cudaMemset(out.rowPtr, 0, sizeof(NnzOffset) * static_cast<std::size_t>(m_new + 1)),
                        "cudaMemset empty compact rowPtr");
    return out;
  }
  NnzOffset* row_counts = nullptr;
  std::int32_t* long_rows = nullptr;
  std::int32_t* medium_rows = nullptr;
  std::int32_t* long_row_count = nullptr;
  // A full 256-thread block per row only pays off for genuinely dense rows.
  // A low threshold oversubscribes models such as dlr2 (millions of rows with
  // only 11--15 entries each), while s82's roughly 80 entries per row still
  // benefits from the cooperative path.
  const bool use_block_per_row =
      static_cast<std::int64_t>(A.nnz) >
      32LL * static_cast<std::int64_t>(A.rows);
  const bool compact_warp = !use_block_per_row && A.rows >= 4096 &&
      A.nnz >= 65536 && runtime::options().warp_compaction;
  const std::int32_t medium_threshold = compact_warp ? 8 : INT_MAX;
  const bool compact_long = runtime::options().long_row_compaction;
  // A matrix below 10M nnz can still contain extremely long rows. Keep the
  // same per-row threshold, but do not gate long-row work on total nnz.
  const bool use_long_row_kernels = !use_block_per_row &&
      (compact_warp || compact_long || A.nnz >= 10000000);
  const std::int32_t long_row_threshold =
      use_long_row_kernels ? COMPACT_LONG_ROW_THRESHOLD : INT_MAX;
  throw_if_cuda_error(stream_ordered_malloc(&row_counts, sizeof(NnzOffset) * static_cast<std::size_t>(m_new)),
                      "cudaMalloc compact row_counts");
  throw_if_cuda_error(cudaMemset(row_counts, 0, sizeof(NnzOffset) * static_cast<std::size_t>(m_new)),
                      "cudaMemset compact row_counts");
  if (use_long_row_kernels) {
    throw_if_cuda_error(stream_ordered_malloc(&long_rows, sizeof(std::int32_t) * static_cast<std::size_t>(A.rows)),
                        "cudaMalloc compact long_rows");
    throw_if_cuda_error(stream_ordered_malloc(&long_row_count, sizeof(std::int32_t) * (compact_warp ? 2 : 1)),
                        "cudaMalloc compact long_row_count");
    throw_if_cuda_error(cudaMemset(long_row_count, 0, sizeof(std::int32_t) * (compact_warp ? 2 : 1)),
                        "cudaMemset compact long_row_count");
  }
  if (compact_warp) medium_rows = long_rows;
  detail.mark("allocation", true);
  constexpr int threads = 256;
  const int old_row_blocks = (A.rows + threads - 1) / threads;
  if (use_block_per_row) {
    _kernel_count_compacted_rows_block<<<A.rows, threads>>>(
        row_counts, row_old_to_new, col_old_to_new, A.rowPtr, A.colVal, A.rows);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_count_compacted_rows_block");
  } else {
    _kernel_count_compacted_rows<<<old_row_blocks, threads>>>(
        row_counts, row_old_to_new, col_old_to_new, A.rowPtr, A.colVal,
        long_rows, long_row_count, long_row_threshold, A.rows,
        medium_rows, compact_warp ? long_row_count + 1 : nullptr, medium_threshold);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_count_compacted_rows");
  }
  std::int32_t queue_counts[2] = {0, 0};
  if (use_long_row_kernels) {
    throw_if_cuda_error(cudaMemcpy(queue_counts, long_row_count,
                                   sizeof(std::int32_t) * (compact_warp ? 2 : 1), cudaMemcpyDeviceToHost),
                        "cudaMemcpy compact long_row_count");
  }
  const auto long_row_count_host = queue_counts[0];
  const auto medium_count = queue_counts[1];
  if (medium_count > 0) {
    _kernel_compact_medium_rows<<<(medium_count + 7) / 8, threads>>>(
        row_counts, nullptr, nullptr, nullptr, row_old_to_new, col_old_to_new,
        A.rowPtr, A.colVal, A.nzVal, medium_rows, medium_count);
    throw_if_cuda_error(cudaGetLastError(), "count compact medium_rows");
  }
  if (long_row_count_host > 0) {
    _kernel_count_compacted_long_rows<<<long_row_count_host, threads>>>(
        row_counts, row_old_to_new, col_old_to_new, A.rowPtr, A.colVal,
        long_rows + (compact_warp ? A.rows - long_row_count_host : 0), long_row_count_host);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_count_compacted_long_rows");
  }
  detail.mark("count_rows", true);
  inclusive_scan_i32(row_counts, m_new, "cub inclusive scan compact");
  NnzOffset nnz_new = 0;
  throw_if_cuda_error(cudaMemcpy(&nnz_new, row_counts + m_new - 1, sizeof(NnzOffset), cudaMemcpyDeviceToHost),
                      "cudaMemcpy compact nnz");
  out.nnz = nnz_new;
  const int new_row_blocks = (m_new + threads - 1) / threads;
  _kernel_row_ptr_from_prefix<<<new_row_blocks, threads>>>(out.rowPtr, row_counts, m_new);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_row_ptr_from_prefix compact");
  detail.mark("scan_and_nnz", true);
  if (nnz_new > 0) {
    throw_if_cuda_error(stream_ordered_malloc(&out.colVal, sizeof(std::int32_t) * static_cast<std::size_t>(nnz_new)),
                        "cudaMalloc compact colVal");
    throw_if_cuda_error(stream_ordered_malloc(&out.nzVal, sizeof(double) * static_cast<std::size_t>(nnz_new)),
                        "cudaMalloc compact nzVal");
    detail.mark("allocate_output", true);
    if (use_block_per_row) {
      _kernel_copy_compacted_rows_block<<<A.rows, threads>>>(
          out.colVal, out.nzVal, out.rowPtr, row_old_to_new, col_old_to_new,
          A.rowPtr, A.colVal, A.nzVal, A.rows);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_copy_compacted_rows_block");
    } else {
      _kernel_copy_compacted_rows<<<old_row_blocks, threads>>>(
          out.colVal, out.nzVal, out.rowPtr, row_old_to_new, col_old_to_new,
          A.rowPtr, A.colVal, A.nzVal, std::min(long_row_threshold, medium_threshold), A.rows);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_copy_compacted_rows");
      if (medium_count > 0) {
        _kernel_compact_medium_rows<<<(medium_count + 7) / 8, threads>>>(
            nullptr, out.colVal, out.nzVal, out.rowPtr, row_old_to_new, col_old_to_new,
            A.rowPtr, A.colVal, A.nzVal, medium_rows, medium_count);
        throw_if_cuda_error(cudaGetLastError(), "copy compact medium_rows");
      }
      if (long_row_count_host > 0) {
        if (compact_long) {
          // CUB's integer scan preserves stable element positions with fewer
          // barriers than the legacy shared-memory scan for every 256 entries.
          _kernel_copy_compacted_rows_block<<<long_row_count_host, threads>>>(
              out.colVal, out.nzVal, out.rowPtr, row_old_to_new, col_old_to_new,
              A.rowPtr, A.colVal, A.nzVal, long_row_count_host, long_rows + (compact_warp ? A.rows - long_row_count_host : 0));
        } else {
          _kernel_copy_compacted_long_rows<<<long_row_count_host, threads>>>(
              out.colVal, out.nzVal, out.rowPtr, row_old_to_new, col_old_to_new,
              A.rowPtr, A.colVal, A.nzVal, long_rows + (compact_warp ? A.rows - long_row_count_host : 0), long_row_count_host);
        }
        throw_if_cuda_error(cudaGetLastError(), "_kernel_copy_compacted_long_rows");
      }
    }
  }
  detail.mark("copy_rows", true);
  stream_ordered_free(row_counts);
  stream_ordered_free(long_rows);
  stream_ordered_free(long_row_count);
  return out;
}

struct WorkingLp {
  LPInfoGpu lp;
  bool owns = false;

  ~WorkingLp() { free_owned(); }

  void free_owned(bool stream_ordered = false) {
    if (!owns) {
      return;
    }
    const auto release = [stream_ordered](void* ptr) {
      if (stream_ordered) stream_ordered_free(ptr);
      else cudaFree(ptr);
    };
    release(lp.A.rowPtr); release(lp.A.colVal); release(lp.A.nzVal);
    release(lp.AT.rowPtr); release(lp.AT.colVal); release(lp.AT.nzVal);
    release(lp.c); release(lp.AL); release(lp.AU); release(lp.l); release(lp.u);
    owns = false;
  }
};

void apply_plan_to_working_lp(WorkingLp& current,
                              PresolvePlanGpu& plan,
                              PresolveRecordGpu* record,
                              HostTapeChunks& tape_chunks) {
  trace_recovery_plan(plan, current.lp);
  const bool profile_apply = runtime::options().profile_apply;
  const auto apply_start = std::chrono::steady_clock::now();
  auto sync_for_profile = [&](const char* context) {
    if (profile_apply) {
      throw_if_cuda_error(cudaDeviceSynchronize(), context);
    }
  };
  // Borrowed input storage and replacement matrices retain the materialization
  // path. Action flags alone do not imply that an index was removed.
  if (current.owns && !plan.has_new_A &&
      detail::plan_keeps_all_indices(plan, current.lp.A.rows, current.lp.A.cols)) {
    const auto after_masks = std::chrono::steady_clock::now();
    if (record != nullptr) {
      // Capture old bounds/maps before any in-place write. With identity maps,
      // both recovery backends skip map compaction and accept null permutations.
      update_record_from_plan(*record, plan, current.lp, current.lp.A.rows, current.lp.A.cols,
                              current.lp.obj_constant + plan.obj_constant_delta,
                              nullptr, nullptr, tape_chunks);
    }
    sync_for_profile("profile identity apply record synchronize");
    const auto after_record = std::chrono::steady_clock::now();
    detail::commit_identity_plan_vectors(plan, current.lp);
    sync_for_profile("profile identity apply vectors synchronize");
    const auto after_vectors = std::chrono::steady_clock::now();
    current.lp.obj_constant += plan.obj_constant_delta;
    if (profile_apply) {
      const std::chrono::duration<double> masks_elapsed = after_masks - apply_start;
      const std::chrono::duration<double> record_elapsed = after_record - after_masks;
      const std::chrono::duration<double> vectors_elapsed = after_vectors - after_record;
      const std::chrono::duration<double> total_elapsed = after_vectors - apply_start;
      std::cerr << ">>> [GPU Presolve C++ apply-profile] structural_unchanged=true"
                << " dims=(" << current.lp.A.rows << "," << current.lp.A.cols << ")->("
                << current.lp.A.rows << "," << current.lp.A.cols << ")"
                << " mask_check=" << masks_elapsed.count() << "s"
                << " old_to_new=0s"
                << " vectors=" << vectors_elapsed.count() << "s"
                << " record=" << record_elapsed.count() << "s"
                << " free=0s"
                << " total=" << total_elapsed.count() << "s\n";
    }
    return;
  }
  std::int32_t m_new = 0;
  std::int32_t n_new = 0;
  std::int32_t* row_old_to_new =
      build_old_to_new(plan.keep_row_mask, current.lp.A.rows, &m_new);
  std::int32_t* col_old_to_new =
      build_old_to_new(plan.keep_col_mask, current.lp.A.cols, &n_new);
  sync_for_profile("profile apply build_old_to_new synchronize");
  const auto after_old_to_new = std::chrono::steady_clock::now();
  const double* c_source = plan.new_c != nullptr ? plan.new_c : current.lp.c;
  const double* l_source = plan.new_l != nullptr ? plan.new_l : current.lp.l;
  const double* u_source = plan.new_u != nullptr ? plan.new_u : current.lp.u;
  const double* AL_source = plan.new_AL != nullptr ? plan.new_AL : current.lp.AL;
  const double* AU_source = plan.new_AU != nullptr ? plan.new_AU : current.lp.AU;
  // Any materialization can replace/reorder coefficients and recycle addresses.
  invalidate_matrix_analysis();
  const DeviceCsrMatrix& source_A = plan.has_new_A ? plan.new_A : current.lp.A;
  DeviceCsrMatrix A_new = compact_csr_by_masks_gpu(
      source_A, row_old_to_new, col_old_to_new, m_new, n_new);
  sync_for_profile("profile apply compact A synchronize");
  const auto after_compact_a = std::chrono::steady_clock::now();
  DeviceCsrMatrix AT_new = transpose_csr_gpu(A_new);
  sync_for_profile("profile apply transpose synchronize");
  const auto after_transpose = std::chrono::steady_clock::now();
  double* c_new = compact_vector(c_source, col_old_to_new, current.lp.A.cols, n_new);
  double* l_new = compact_vector(l_source, col_old_to_new, current.lp.A.cols, n_new);
  double* u_new = compact_vector(u_source, col_old_to_new, current.lp.A.cols, n_new);
  double* AL_new = compact_vector(AL_source, row_old_to_new, current.lp.A.rows, m_new);
  double* AU_new = compact_vector(AU_source, row_old_to_new, current.lp.A.rows, m_new);
  sync_for_profile("profile apply compact vectors synchronize");
  const auto after_vectors = std::chrono::steady_clock::now();
  if (record != nullptr) {
    update_record_from_plan(*record, plan, current.lp, m_new, n_new,
                            current.lp.obj_constant + plan.obj_constant_delta, row_old_to_new,
                            col_old_to_new, tape_chunks);
  }
  sync_for_profile("profile apply record synchronize");
  const auto after_record = std::chrono::steady_clock::now();
  stream_ordered_free(row_old_to_new);
  stream_ordered_free(col_old_to_new);

  current.free_owned(true);
  sync_for_profile("profile apply free synchronize");
  const auto after_free = std::chrono::steady_clock::now();
  current.lp.A = A_new;
  current.lp.AT = AT_new;
  current.lp.c = c_new;
  current.lp.l = l_new;
  current.lp.u = u_new;
  current.lp.AL = AL_new;
  current.lp.AU = AU_new;
  current.lp.obj_constant += plan.obj_constant_delta;
  current.owns = true;
  if (profile_apply) {
    const std::chrono::duration<double> old_to_new_elapsed = after_old_to_new - apply_start;
    const std::chrono::duration<double> compact_a_elapsed = after_compact_a - after_old_to_new;
    const std::chrono::duration<double> transpose_elapsed = after_transpose - after_compact_a;
    const std::chrono::duration<double> vectors_elapsed = after_vectors - after_transpose;
    const std::chrono::duration<double> record_elapsed = after_record - after_vectors;
    const std::chrono::duration<double> free_elapsed = after_free - after_record;
    const std::chrono::duration<double> total_elapsed = after_free - apply_start;
    std::cerr << ">>> [GPU Presolve C++ apply-profile] structural_unchanged=false"
              << " dims=(" << source_A.rows << "," << source_A.cols << ")->("
              << m_new << "," << n_new << ")"
              << " nnz=" << source_A.nnz << "->" << A_new.nnz
              << " old_to_new=" << old_to_new_elapsed.count() << "s"
              << " compact_A=" << compact_a_elapsed.count() << "s"
              << " transpose=" << transpose_elapsed.count() << "s"
              << " vectors=" << vectors_elapsed.count() << "s"
              << " record=" << record_elapsed.count() << "s"
              << " free=" << free_elapsed.count() << "s"
              << " total=" << total_elapsed.count() << "s\n";
  }
}

bool has_good_nnz_progress(NnzOffset nnz_before,
                           NnzOffset nnz_after,
                           NnzOffset minimum_decrease) {
  return nnz_before >= minimum_decrease &&
         nnz_after <= nnz_before - minimum_decrease;
}

GpuPresolveSummary run_gpu_presolve_impl(const LPInfoGpu& lp,
                                         const PresolveParams& input_params,
                                         bool keep_reduced_lp) {
  MatrixAnalysisScope matrix_analysis_scope;
  CleanupWorkspaceScope cleanup_workspace_scope;
  DuplicateRuleWorkspaceScope duplicate_rule_workspace_scope;
  PresolveParams effective_params = input_params;
  if (!effective_params.enable_structure_specific_rules) {
    effective_params.enable_column_domination = false;
    effective_params.enable_variable_pair_aggregation = false;
    effective_params.enable_two_row_variable_elimination = false;
    effective_params.enable_mccormick_variable_elimination = false;
    effective_params.enable_absolute_value_epigraph_reduction = false;
  }
  const PresolveParams& params = effective_params;

  struct RuleWorkspaceReleaseGuard {
    ~RuleWorkspaceReleaseGuard() {
      if (!bound_strengthening_workspace_prepared()) {
        release_bound_strengthening_workspace();
      }
      release_singleton_columns_workspace();
    }
  } rule_workspace_release_guard;

  const bool profile_record =
      runtime::options().profile_records;
  const auto initial_record_start = std::chrono::steady_clock::now();
  GpuPresolveSummary summary;
  HostTapeChunks tape_chunks;
  summary.original_rows = lp.A.rows;
  summary.original_cols = lp.A.cols;
  summary.record = make_initial_record(lp);
  if (profile_record) {
    const std::chrono::duration<double> initial_record_elapsed =
        std::chrono::steady_clock::now() - initial_record_start;
    std::cerr << ">>> [GPU Presolve C++ record-profile]"
              << " initial_record=" << initial_record_elapsed.count()
              << "s rows=" << lp.A.rows << " cols=" << lp.A.cols << "\n";
  }

  WorkingLp current;
  current.lp = lp;

  bool terminal = false;
  const bool profile_rules = runtime::options().profile_rules;
  const bool incremental_live_stats = runtime::options().incremental_live_stats;
  const bool profile_live_stats = runtime::options().profile_live_stats;
  std::int64_t full_stats_refreshes = 0;
  std::int64_t incremental_stats_updates = 0;
  // Complete initial recovery-map GPU work before timing the presolve core.
  throw_if_cuda_error(cudaDeviceSynchronize(), "presolve initialization complete");
  const auto presolve_start = std::chrono::steady_clock::now();
  const bool has_time_limit = params.max_time > 0.0 && std::isfinite(params.max_time);
  auto time_exceeded = [&]() -> bool {
    if (!has_time_limit) {
      return false;
    }
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - presolve_start;
    return elapsed.count() >= params.max_time;
  };

  // These rules construct a replacement matrix and retain their own APPLY boundary.
  auto run_materialized_rule = [&](bool enabled, const char* name, auto rule) -> bool {
    if (!enabled || terminal || time_exceeded()) return false;
    const auto start = std::chrono::steady_clock::now();
    PresolvePlanGpu plan;
    PresolveStatsGpu stats{};
    allocate_plan(plan, current.lp, PlanRequirements{true, true, true});
    try {
      rule(plan, current.lp, stats, params);
      summary.has_infeasible = plan.has_infeasible;
      summary.has_unbounded = plan.has_unbounded;
      terminal = plan.has_infeasible || plan.has_unbounded;
      const bool changed = !terminal && plan.has_change;
      if (changed) {
        apply_plan_to_working_lp(current, plan, &summary.record, tape_chunks);
        if (std::string(name) == "startup_absolute_value_epigraph")
          write_diagnostic_snapshot(name, current.lp);
        if (profile_rules) std::cerr << ">>> [schedule-apply] boundary=" << name << '\n';
      }
      if (profile_rules) {
        std::cerr << ">>> [GPU Presolve C++ profile] phase=" << name
                  << " changed=" << (changed ? "true" : "false")
                  << " time=" << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start).count() << "s\n";
      }
      free_plan(plan);
      return changed;
    } catch (...) {
      free_plan(plan);
      throw;
    }
  };

  struct DeferredState {
    PresolvePlanGpu plan;
    PresolveStatsGpu stats;
    std::uint8_t* entry_cols = nullptr;
    double* reference_bounds = nullptr;
    std::uint8_t* stats_rows = nullptr;
    std::uint8_t* stats_cols = nullptr;
    std::int32_t* touched_rows = nullptr;
    std::int32_t* touched_cols = nullptr;
    unsigned long long* device_live_nnz = nullptr;
    bool borrowed_buffers = false;
    bool active = false, pending = false, stats_dirty = false, nnz_dirty = false;
    void reset() {
      free_plan(plan); free_stats(stats);
      if (!borrowed_buffers) {
        cudaFree(entry_cols); cudaFree(reference_bounds); cudaFree(device_live_nnz);
      }
      cudaFree(stats_rows); cudaFree(stats_cols);
      cudaFree(touched_rows); cudaFree(touched_cols);
      plan = {}; stats = {}; entry_cols = nullptr; reference_bounds = nullptr;
      stats_rows = stats_cols = nullptr;
      touched_rows = touched_cols = nullptr;
      device_live_nnz = nullptr;
      borrowed_buffers = false;
      active = pending = stats_dirty = nnz_dirty = false;
    }
    ~DeferredState() { reset(); }
  } deferred;
  auto prepare_deferred = [&]() {
    if (deferred.active) return;
    allocate_plan(deferred.plan, current.lp, PlanRequirements{});
    allocate_stats(deferred.stats, current.lp.A.rows, current.lp.A.cols,
                   StatsRequirements{true, true, true, true, true, true});
    const int m = current.lp.A.rows;
    const int n = current.lp.A.cols;
    if (incremental_live_stats) {
      if (m) {
        throw_if_cuda_error(cudaMalloc(&deferred.stats_rows, m), "schedule stats row snapshot");
        throw_if_cuda_error(cudaMalloc(&deferred.touched_rows, sizeof(int) * std::size_t(m)),
                            "schedule touched rows");
      }
      if (n) {
        throw_if_cuda_error(cudaMalloc(&deferred.stats_cols, n), "schedule stats col snapshot");
        throw_if_cuda_error(cudaMalloc(&deferred.touched_cols, sizeof(int) * std::size_t(n)),
                            "schedule touched cols");
      }
    }
    if (auto* pool = cleanup_scratch::active_pool) {
      deferred.borrowed_buffers = true;
      deferred.device_live_nnz = static_cast<unsigned long long*>(
          pool->reserve(cleanup_scratch::deferred_live_nnz_slot,
                        sizeof(*deferred.device_live_nnz)));
      if (n) {
        deferred.entry_cols = static_cast<std::uint8_t*>(
            pool->reserve(cleanup_scratch::deferred_entry_cols_slot, n));
        deferred.reference_bounds = static_cast<double*>(
            pool->reserve(cleanup_scratch::deferred_reference_bounds_slot,
                          2 * sizeof(double) * n));
      }
    } else {
      throw_if_cuda_error(cudaMalloc(&deferred.device_live_nnz, sizeof(*deferred.device_live_nnz)),
                          "schedule live nnz");
      if (n) {
        throw_if_cuda_error(cudaMalloc(&deferred.entry_cols, n), "schedule entry columns");
        throw_if_cuda_error(cudaMalloc(&deferred.reference_bounds, 2 * sizeof(double) * n),
                            "schedule reference bounds");
      }
    }
    deferred.plan.rule_entry_keep_col = deferred.entry_cols;
    deferred.plan.cleanup_entry_keep_col = deferred.entry_cols;
    deferred.active = deferred.stats_dirty = deferred.nnz_dirty = true;
  };
  auto refresh_deferred_stats = [&]() {
    if (deferred.stats_dirty) {
      bool updated = false;
      if (incremental_live_stats && deferred.stats.live_masks) {
        updated = update_live_stats_from_deletions(
            deferred.stats, current.lp, deferred.plan, deferred.stats_rows,
            deferred.stats_cols, deferred.touched_rows, deferred.touched_cols,
            false, deferred.device_live_nnz);
      }
      if (updated) ++incremental_stats_updates;
      else {
        refresh_live_stats(deferred.stats, current.lp, deferred.plan, false,
                           deferred.device_live_nnz);
        ++full_stats_refreshes;
      }
      if (incremental_live_stats) {
        const int m = current.lp.A.rows;
        const int n = current.lp.A.cols;
        if (m) throw_if_cuda_error(cudaMemcpy(deferred.stats_rows, deferred.plan.keep_row_mask,
            m, cudaMemcpyDeviceToDevice), "snapshot live row mask");
        if (n) throw_if_cuda_error(cudaMemcpy(deferred.stats_cols, deferred.plan.keep_col_mask,
            n, cudaMemcpyDeviceToDevice), "snapshot live col mask");
      }
      deferred.stats_dirty = false;
    }
  };
  auto live_nnz = [&]() -> NnzOffset {
    if (!deferred.active) return current.lp.A.nnz;
    refresh_deferred_stats();
    if (deferred.nnz_dirty) {
      unsigned long long count = 0;
      throw_if_cuda_error(cudaMemcpy(&count, deferred.device_live_nnz, sizeof(count),
                                    cudaMemcpyDeviceToHost), "schedule read live nnz");
      deferred.stats.live_nnz = static_cast<NnzOffset>(count);
      deferred.nnz_dirty = false;
    }
    return deferred.stats.live_nnz;
  };
  // Count actual new deletions using flags already copied by each rule.
  // The serial survives per-rule resets and Apply, without extra GPU work.
  std::uint64_t deletion_serial = 0;
  auto deferred_rule = [&](bool enabled, const char* name, auto rule) -> bool {
    if (!enabled || terminal || time_exceeded()) return false;
    prepare_deferred();
    refresh_deferred_stats();
    auto& plan = deferred.plan;
    const int n = current.lp.A.cols;
    if (n) throw_if_cuda_error(cudaMemcpy(deferred.entry_cols, plan.keep_col_mask, n,
                                          cudaMemcpyDeviceToDevice), "schedule entry mask");
    plan.has_change = plan.has_row_action = plan.has_col_action = false;
    plan.has_redundant_row_deletion = plan.has_row_deletion = false;
    plan.obj_constant_delta = 0;
    if (std::string(name) == "parallel_columns")
      write_diagnostic_snapshot("before_parallel_columns", current.lp, &plan);
    rule(plan, deferred.stats);
    if (std::string(name) == "parallel_columns")
      write_diagnostic_snapshot("after_parallel_columns", current.lp, &plan);
    summary.has_infeasible = plan.has_infeasible;
    summary.has_unbounded = plan.has_unbounded;
    terminal = plan.has_infeasible || plan.has_unbounded;
    const bool changed = plan.has_change && !terminal;
    if (changed) {
      if (plan.has_row_deletion || plan.has_col_action) ++deletion_serial;
      current.lp.obj_constant += plan.obj_constant_delta;
      if (summary.record.device) {
        capture_cleanup_recovery(summary.record, plan, current.lp, current.lp.obj_constant);
      } else {
        // The explicit host recovery backend keeps the same logical boundaries.
        update_record_from_plan(summary.record, plan, current.lp, current.lp.A.rows, n,
                                current.lp.obj_constant, nullptr, nullptr, tape_chunks);
        if (plan.has_col_action) {
          const auto before = copy_device_array(deferred.entry_cols, n, "schedule entry mask host");
          const auto after = copy_device_array(plan.keep_col_mask, n, "schedule current mask host");
          std::vector<std::int32_t> removed;
          for (int j = 0; j < n; ++j) if (before[j] && !after[j]) removed.push_back(j);
          append_removed_fixed_bound_records_sparse(summary.record, removed, plan.new_l, plan.new_u,
              summary.record.col_red2org, "schedule fixed records");
        }
        plan.tape = {};
        plan.cleanup_recovery_captured = true;
      }
      deferred.pending = true;
      if (plan.has_row_action || plan.has_col_action) {
        deferred.stats_dirty = true;
        deferred.nnz_dirty = true;
      }
    }
    plan.obj_constant_delta = 0;
    if (profile_rules) std::cerr << ">>> [schedule-rule] rule=" << name << " changed=" << changed << '\n';
    return changed;
  };
  auto cleanup = [&]() -> bool {
    if (profile_rules) std::cerr << ">>> [schedule-phase] phase=cleanup\n";
    bool changed = deferred_rule(params.enable_infeasible_and_fixed_variables, "infeasible_and_fixed_variables",
        [&](auto& p, const auto&) { apply_rule_infeasible_and_fixed_variables(p, current.lp, params); });
    changed = deferred_rule(params.enable_empty_columns, "empty_columns",
        [&](auto& p, const auto& s) { apply_rule_empty_columns(p, current.lp, s, params); }) || changed;
    changed = deferred_rule(params.enable_empty_rows, "empty_rows",
        [&](auto& p, const auto& s) { apply_rule_empty_rows(p, current.lp, s, params); }) || changed;
    changed = deferred_rule(params.enable_singleton_rows, "singleton_rows",
        [&](auto& p, const auto& s) { apply_rule_singleton_rows(p, current.lp, s, params); }) || changed;
    changed = deferred_rule(params.enable_singleton_columns, "singleton_columns",
        [&](auto& p, const auto& s) { apply_rule_singleton_columns(p, current.lp, s, params); }) || changed;
    return changed;
  };
  auto materialize = [&](const char* boundary) {
    if (deferred.pending) {
      deferred.plan.has_change = true;
      apply_plan_to_working_lp(current, deferred.plan, &summary.record, tape_chunks);
      if (profile_rules) std::cerr << ">>> [schedule-apply] boundary=" << boundary << '\n';
    }
    deferred.reset();
  };
  if (params.enable_startup && params.max_iters > 0 && !time_exceeded()) {
    // The first three structural rules only delete
    // rows/columns and share a plan; each detector sees the preceding masks.
    PresolvePlanGpu batch;
    bool batch_allocated = false;
    auto prepare_batch = [&]() {
      if (!batch_allocated) {
        allocate_plan(batch, current.lp, PlanRequirements{false, true, false});
        batch_allocated = true;
      }
    };
    auto batch_rule = [&](bool enabled, const char* name, auto analyze, auto build, auto release) {
      if (!enabled || terminal || time_exceeded()) return;
      const auto start = std::chrono::steady_clock::now();
      auto analysis = analyze();
      try {
        if (analysis.applicable) {
          prepare_batch();
          build(batch, current.lp, analysis, params);
        }
        if (profile_rules) {
          std::cerr << ">>> [GPU Presolve C++ profile] phase=startup_" << name
                    << " candidates=" << analysis.candidate_count
                    << " time=" << std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - start).count() << "s\n";
        }
        release(analysis);
      } catch (...) {
        release(analysis);
        throw;
      }
    };
    try {
      batch_rule(params.enable_column_domination, "column_domination",
          [&]() { return analyze_column_domination(current.lp, params); },
          build_column_domination_plan, free_column_domination_analysis);
      batch_rule(params.enable_two_row_variable_elimination, "two_row_variable_elimination",
          [&]() { return analyze_two_row_variable_elimination(
              current.lp, params, batch_allocated ? &batch : nullptr); },
          build_two_row_variable_elimination_plan, free_two_row_variable_elimination_analysis);
      batch_rule(params.enable_mccormick_variable_elimination, "mccormick_variable_elimination",
          [&]() { return analyze_mccormick_variable_elimination(
              current.lp, params, batch_allocated ? &batch : nullptr); },
          build_mccormick_variable_elimination_plan, free_mccormick_variable_elimination_analysis);
      // Commit accepted reductions even if the time limit expired between rules.
      if (batch.has_change) {
        apply_plan_to_working_lp(current, batch, &summary.record, tape_chunks);
        if (profile_rules) std::cerr << ">>> [schedule-apply] boundary=startup_projection_batch\n";
      }
      free_plan(batch);
    } catch (...) {
      free_plan(batch);
      throw;
    }

    if (!terminal && !time_exceeded() && params.enable_variable_pair_aggregation) {
      const auto phase_start = std::chrono::steady_clock::now();
      const auto m_before = current.lp.A.rows;
      const auto n_before = current.lp.A.cols;
      const auto nnz_before = current.lp.A.nnz;
      PresolvePlanGpu plan;
      const auto selected = build_variable_pair_aggregation_plan(
          plan, current.lp, params, time_exceeded);
      summary.has_infeasible = plan.has_infeasible;
      summary.has_unbounded = plan.has_unbounded;
      terminal = plan.has_infeasible || plan.has_unbounded;
      const bool changed = !terminal && plan.has_change;
      if (changed) {
        apply_plan_to_working_lp(current, plan, &summary.record, tape_chunks);
        if (profile_rules) std::cerr << ">>> [schedule-apply] boundary=startup_variable_pair_aggregation\n";
      }
      free_plan(plan);
      if (profile_rules) {
        throw_if_cuda_error(cudaDeviceSynchronize(), "profile variable-pair aggregation");
        const char* name = selected == VariablePairAggregationCase::Epigraph ? "epigraph" :
            selected == VariablePairAggregationCase::ShiftedAntipodal ? "shifted_antipodal" : "none";
        std::cerr << ">>> [GPU Presolve C++ profile] phase=startup_variable_pair_aggregation"
                  << " case=" << name << " changed=" << (changed ? "true" : "false")
                  << " dims=(" << m_before << "," << n_before << ")->("
                  << current.lp.A.rows << "," << current.lp.A.cols << ") nnz="
                  << nnz_before << "->" << current.lp.A.nnz
                  << " time=" << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - phase_start).count() << "s\n";
      }
    }

    if (params.enable_absolute_value_epigraph_reduction && !terminal && !time_exceeded() &&
        absolute_value_epigraph_prefix_screen_passes(current.lp)) {
      run_materialized_rule(true, "startup_absolute_value_epigraph",
                            apply_rule_absolute_value_epigraph_reduction);
    }

    ++summary.iterations;
  }

  auto propagation = [&]() -> bool {
    bool bounds_changed = false;
    if (profile_rules) std::cerr << ">>> [schedule-phase] phase=propagation\n";
    if (params.enable_bound_strengthening && !terminal && !time_exceeded()) {
      prepare_deferred();
      auto& plan = deferred.plan;
      const int n = current.lp.A.cols;
      if (n) {
        throw_if_cuda_error(cudaMemcpy(deferred.reference_bounds, plan.new_l, sizeof(double)*n,
                                      cudaMemcpyDeviceToDevice), "propagation lower reference");
        throw_if_cuda_error(cudaMemcpy(deferred.reference_bounds+n, plan.new_u, sizeof(double)*n,
                                      cudaMemcpyDeviceToDevice), "propagation upper reference");
        plan.propagation_reference_l = deferred.reference_bounds;
        plan.propagation_reference_u = deferred.reference_bounds+n;
      }
      bounds_changed = deferred_rule(true, "bound_strengthening", [&](auto& p, const auto& s) {
            apply_rule_bound_strengthening(p, current.lp, s, params);
          });
    }
    deferred_rule(params.enable_infeasible_and_redundant_rows, "infeasible_and_redundant_rows",
        [&](auto& p, const auto& s) { apply_rule_infeasible_and_redundant_rows(p, current.lp, s, params); });
    deferred_rule(params.enable_dual_fix, "dual_fix",
        [&](auto& p, const auto& s) { apply_rule_dual_fix(p, current.lp, s, params); });
    return bounds_changed;
  };

  enum class SchedulePhase { Cleanup, Propagation, Aggregation, Parallel };
  auto phase_name = [](SchedulePhase value) {
    switch (value) {
      case SchedulePhase::Cleanup: return "cleanup";
      case SchedulePhase::Propagation: return "propagation";
      case SchedulePhase::Aggregation: return "aggregation";
      case SchedulePhase::Parallel: return "parallel";
    }
    return "unknown";
  };
  SchedulePhase phase = SchedulePhase::Cleanup;
  constexpr NnzOffset minimum_nnz_decrease = 100;
  int bound_only_rounds = 0;
  auto cleanup_phase = [&](NnzOffset stage_nnz_before) {
    const auto deletion_serial_before = deletion_serial;
    const bool changed = cleanup();
    const bool deleted_rows_or_cols = deletion_serial != deletion_serial_before;
    if (deleted_rows_or_cols) bound_only_rounds = 0;
    const auto nnz_after = live_nnz();
    phase = changed ? SchedulePhase::Cleanup : SchedulePhase::Propagation;
    if (profile_rules) std::cerr << ">>> [schedule-progress] phase=cleanup stage_before=" << stage_nnz_before
        << " after=" << nnz_after << " next=" << phase_name(phase)
        << " pending=" << deferred.pending << " deleted=" << deleted_rows_or_cols
        << " changed=" << changed
        << " bound_only_rounds=" << bound_only_rounds << '\n';
  };
  // Fix L before the first Cleanup. Every Cleanup invocation runs through the
  // main loop and repeats on any new update without consuming iteration budget.
  const NnzOffset initial_live_nnz =
      params.max_iters > 0 && !terminal && !time_exceeded() ? live_nnz() : 0;
  const int configured_bound_only_limit =
      std::max(1, params.bound_strengthening_max_bound_only_rounds);
  const int bound_only_limit = [&]() {
    int limit = configured_bound_only_limit;
    if (params.bound_strengthening_bound_only_nnz_round_budget > 0 && initial_live_nnz > 0) {
      limit = static_cast<int>(std::min<std::int64_t>(limit,
          std::max<std::int64_t>(1,
              params.bound_strengthening_bound_only_nnz_round_budget / initial_live_nnz)));
    }
    // Retain the old minimum-two-round exception for this matrix shape.
    const std::int64_t m = current.lp.A.rows, n = current.lp.A.cols;
    if (m >= 512 && m <= 2048 && n >= 128*m && n <= 256*m &&
        initial_live_nnz >= 2048*m && initial_live_nnz <= 8192*m) {
      limit = std::max(limit, std::min(2, configured_bound_only_limit));
    }
    return limit;
  }();
  if (profile_rules) std::cerr << ">>> [schedule-bound-budget] initial_nnz=" << initial_live_nnz
      << " limit=" << bound_only_limit << '\n';
  while (params.max_iters > 0 && !terminal && !time_exceeded()) {
    // Cleanup can finish even after Startup or another phase used the budget.
    // Only Propagation, Aggregation and Parallel consume main-loop iterations.
    if (phase != SchedulePhase::Cleanup) {
      if (summary.iterations >= params.max_iters) {
        summary.stopped_at_iteration_limit = true;
        break;
      }
      ++summary.iterations;
    }
    // Refresh this reference on every execution, including consecutive Cleanup.
    const NnzOffset stage_nnz_before = live_nnz();
    if (phase == SchedulePhase::Cleanup) {
      cleanup_phase(stage_nnz_before);
    } else if (phase == SchedulePhase::Propagation) {
      const auto deletion_serial_before = deletion_serial;
      const bool bounds_changed = propagation();
      const bool deleted_rows_or_cols = deletion_serial != deletion_serial_before;
      bound_only_rounds = deleted_rows_or_cols ? 0 : bound_only_rounds + 1;
      materialize("1");
      const auto nnz_after = live_nnz();
      const bool nnz_progress = has_good_nnz_progress(stage_nnz_before, nnz_after, minimum_nnz_decrease);
      const bool bound_progress = bounds_changed && bound_only_rounds < bound_only_limit;
      phase = nnz_progress || bound_progress
          ? SchedulePhase::Cleanup : SchedulePhase::Aggregation;
      if (profile_rules) std::cerr << ">>> [schedule-progress] phase=propagation stage_before=" << stage_nnz_before
          << " after=" << nnz_after << " next=" << phase_name(phase)
          << " pending=" << deferred.pending
          << " bounds_changed=" << bounds_changed << " deleted=" << deleted_rows_or_cols
          << " bound_only_rounds=" << bound_only_rounds << " limit=" << bound_only_limit
          << " nnz_progress=" << nnz_progress << " bound_progress=" << bound_progress << '\n';
      if (phase == SchedulePhase::Aggregation) bound_only_rounds = 0;
    } else if (phase == SchedulePhase::Aggregation) {
      if (profile_rules) std::cerr << ">>> [schedule-phase] phase=aggregation\n";
      if (profile_rules) std::cerr << ">>> [schedule-phase] phase=doubleton_equations\n";
      // Invoke the rule once; it processes one independent batch.
      run_materialized_rule(params.enable_doubleton_equations, "doubleton_equations",
                            apply_rule_doubleton_equations);
      const auto nnz_after = live_nnz();
      phase = has_good_nnz_progress(stage_nnz_before, nnz_after, minimum_nnz_decrease)
          ? SchedulePhase::Cleanup : SchedulePhase::Parallel;
      if (profile_rules) std::cerr << ">>> [schedule-progress] phase=aggregation stage_before=" << stage_nnz_before
          << " after=" << nnz_after
          << " next=" << phase_name(phase)
          << " pending=" << deferred.pending << '\n';
    } else {
      if (profile_rules) std::cerr << ">>> [schedule-phase] phase=parallel\n";
      bool changed = deferred_rule(params.enable_parallel_rows, "parallel_rows",
          [&](auto& p, const auto& s) { apply_rule_parallel_rows(p, current.lp, s, params); });
      changed = deferred_rule(params.enable_parallel_columns, "parallel_columns",
          [&](auto& p, const auto& s) { apply_rule_parallel_columns(p, current.lp, s, params); }) || changed;
      materialize("2");
      // Count only reductions made by the two Parallel rules.
      const auto nnz_after = live_nnz();
      const bool continue_loop = changed &&
          has_good_nnz_progress(stage_nnz_before, nnz_after, minimum_nnz_decrease);
      if (profile_rules) std::cerr << ">>> [schedule-progress] phase=parallel stage_before=" << stage_nnz_before
          << " after=" << nnz_after << " next=" << (continue_loop ? "cleanup" : "end")
          << " pending=" << deferred.pending << '\n';
      if (!continue_loop) break;
      phase = SchedulePhase::Cleanup;
    }
  }
  // A time/iteration limit can interrupt deferred work before its next Apply.
  // Drain successful pending work so the returned LP and recovery maps agree.
  materialize("finalize");
  write_diagnostic_snapshot("final", current.lp);

  if (profile_live_stats) {
    std::cerr << ">>> [live-structure] full_refreshes=" << full_stats_refreshes
              << " incremental_updates=" << incremental_stats_updates << '\n';
  }

  if (summary.record.device && runtime::options().recovery_stats) {
    const auto &d = *summary.record.device;
    std::cerr << ">>> [device-recovery] segments=" << d.segments.size() << " records=" << d.records
              << " indices=" << d.indices << " values=" << d.values
              << " host_staged_bytes=" << d.host_staged_bytes << "\n";
  }
  // Canonicalization is part of presolve timing, including terminal exits.
  tape_chunks.finish(summary.record.tape);
  if (summary.record.tape_gpu.record_count != 0 ||
      summary.record.tape_gpu.index_count != 0 ||
      summary.record.tape_gpu.value_count != 0) {
    throw std::runtime_error(
        "GPU postsolve tape must remain deferred during presolve");
  }
  // Uploading the canonical tape is a postsolve concern.  postsolve_gpu owns
  // the lazy upload path, so presolve does not wait on a transfer that cannot
  // be consumed until after the reduced solve has finished.
  throw_if_cuda_error(cudaDeviceSynchronize(), "run_gpu_presolve_impl synchronize");
  summary.reduced_rows = current.lp.A.rows;
  summary.reduced_cols = current.lp.A.cols;
  summary.reduced_nnz = current.lp.A.nnz;
  summary.obj_constant_delta = current.lp.obj_constant - lp.obj_constant;
  summary.record.m1 = summary.reduced_rows;
  summary.record.n1 = summary.reduced_cols;
  summary.record.obj_constant_new = current.lp.obj_constant;
  if (!summary.record.device) {
    rebuild_org2red(summary.record.row_org2red, summary.record.row_red2org, summary.record.m0);
    rebuild_org2red(summary.record.col_org2red, summary.record.col_red2org, summary.record.n0);
  }
  const std::chrono::duration<double> presolve_elapsed =
      std::chrono::steady_clock::now() - presolve_start;
  summary.elapsed_seconds = presolve_elapsed.count();
  summary.has_change = current.owns;
  if (keep_reduced_lp) {
    summary.reduced_lp = current.lp;
    summary.owns_reduced_lp = current.owns;
    current.owns = false;
  } else {
    current.free_owned();
  }
  return summary;
}

}  // namespace

GpuPresolveSummary run_gpu_presolve_with_record(const LPInfoGpu& lp,
                                                const PresolveParams& params) {
  return run_gpu_presolve_impl(lp, params, false);
}

GpuPresolveSummary run_gpu_presolve_with_reduced_lp(const LPInfoGpu& lp,
                                                    const PresolveParams& params) {
  return run_gpu_presolve_impl(lp, params, true);
}

void free_gpu_presolve_reduced_lp(GpuPresolveSummary& summary) {
  if (!summary.owns_reduced_lp) {
    summary.reduced_lp = LPInfoGpu{};
    return;
  }
  cudaFree(summary.reduced_lp.A.rowPtr);
  cudaFree(summary.reduced_lp.A.colVal);
  cudaFree(summary.reduced_lp.A.nzVal);
  cudaFree(summary.reduced_lp.AT.rowPtr);
  cudaFree(summary.reduced_lp.AT.colVal);
  cudaFree(summary.reduced_lp.AT.nzVal);
  cudaFree(summary.reduced_lp.c);
  cudaFree(summary.reduced_lp.AL);
  cudaFree(summary.reduced_lp.AU);
  cudaFree(summary.reduced_lp.l);
  cudaFree(summary.reduced_lp.u);
  summary.reduced_lp = LPInfoGpu{};
  summary.owns_reduced_lp = false;
}

}  // namespace gpu_presolver::presolve
