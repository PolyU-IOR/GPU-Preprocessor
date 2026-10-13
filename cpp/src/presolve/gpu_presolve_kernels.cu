#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/gpu_presolve_kernels.hpp"
#include "gpu_presolver/presolve/cleanup_workspace.hpp"

#include "gpu_presolver/presolve/live_matrix.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/functional.h>
#include <algorithm>
#include <array>
#include <cstdlib>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

__global__ void _kernel_compute_row_nnz(std::int32_t* row_nnz,
                                        const NnzOffset* rowPtr,
                                        std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < m) {
    row_nnz[i] = rowPtr[i + 1] - rowPtr[i];
  }
}

__global__ void _kernel_singleton_row_support(std::int32_t* singleton_column,
                                              double* singleton_val,
                                              const std::int32_t* row_nnz,
                                              const NnzOffset* rowPtr,
                                              const std::int32_t* colVal,
                                              const double* nzVal,
                                              std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < m) {
    const std::int32_t nnz_i = row_nnz[i];
    if (nnz_i == 1) {
      const NnzOffset p = rowPtr[i];
      singleton_column[i] = colVal[p];
      singleton_val[i] = nzVal[p];
    } else {
      singleton_column[i] = -1;
      singleton_val[i] = 0.0;
    }
  }
}

__global__ void _kernel_compute_row_activity_bounds(double* row_min,
                                                    double* row_max,
                                                    const NnzOffset* rowPtr,
                                                    const std::int32_t* colVal,
                                                    const double* nzVal,
                                                    const double* l,
                                                    const double* u,
                                                    std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < m) {
    double acc_min = 0.0;
    double acc_max = 0.0;
    const NnzOffset row_start = rowPtr[i];
    const NnzOffset row_stop = rowPtr[i + 1];
    for (NnzOffset p = row_start; p < row_stop; ++p) {
      const std::int32_t col = colVal[p];
      const double a = nzVal[p];
      const double lj = l[col];
      const double uj = u[col];
      if (a >= 0.0) {
        acc_min += a * lj;
        acc_max += a * uj;
      } else {
        acc_min += a * uj;
        acc_max += a * lj;
      }
    }
    row_min[i] = acc_min;
    row_max[i] = acc_max;
  }
}

__global__ void _kernel_compute_row_activity_summary(double* row_min_fin,
                                                     double* row_max_fin,
                                                     std::int32_t* row_min_neg_inf_count,
                                                     std::int32_t* row_max_pos_inf_count,
                                                     const NnzOffset* rowPtr,
                                                     const std::int32_t* colVal,
                                                     const double* nzVal,
                                                     const double* l,
                                                     const double* u,
                                                     double zero_tol,
                                                     std::int32_t m, int max_length,
                                                     const std::uint8_t* keep_row,
                                                     const std::uint8_t* keep_col,
                                                     const std::int32_t* live_degrees) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < m && (max_length == 0 || (live_degrees ? live_degrees[i] : rowPtr[i + 1] - rowPtr[i]) <= max_length)) {
    double acc_min_fin = 0.0;
    double acc_max_fin = 0.0;
    std::int32_t neg_inf_count = 0;
    std::int32_t pos_inf_count = 0;
    const NnzOffset row_start = rowPtr[i];
    const NnzOffset row_stop = rowPtr[i + 1];
    for (NnzOffset p = row_start; p < row_stop; ++p) {
      if ((keep_row && !keep_row[i]) || (keep_col && !keep_col[colVal[p]])) continue;
      const double a = nzVal[p];
      if (fabs(a) <= zero_tol) {
        continue;
      }
      const std::int32_t col = colVal[p];
      const double lj = l[col];
      const double uj = u[col];

      const double term_min = a >= 0.0 ? (a * lj) : (a * uj);
      const double term_max = a >= 0.0 ? (a * uj) : (a * lj);

      if (isfinite(term_min)) {
        acc_min_fin += term_min;
      } else if (term_min < 0.0) {
        ++neg_inf_count;
      }

      if (isfinite(term_max)) {
        acc_max_fin += term_max;
      } else if (term_max > 0.0) {
        ++pos_inf_count;
      }
    }
    row_min_fin[i] = acc_min_fin;
    row_max_fin[i] = acc_max_fin;
    row_min_neg_inf_count[i] = neg_inf_count;
    row_max_pos_inf_count[i] = pos_inf_count;
  }
}

__global__ void _kernel_compute_col_max_abs(double* col_max_abs,
                                            const NnzOffset* rowPtr,
                                            const double* nzVal,
                                            std::int32_t n, int max_length,
                                            const std::int32_t* row_indices,
                                            const std::uint8_t* keep_row) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j < n && (max_length == 0 || rowPtr[j + 1] - rowPtr[j] <= max_length)) {
    double vmax = 0.0;
    const NnzOffset row_start = rowPtr[j];
    const NnzOffset row_stop = rowPtr[j + 1];
    for (NnzOffset p = row_start; p < row_stop; ++p) {
      if (keep_row && !keep_row[row_indices[p]]) continue;
      vmax = fmax(vmax, fabs(nzVal[p]));
    }
    col_max_abs[j] = vmax;
  }
}


__global__ void group_adaptive_rows(const NnzOffset* row_ptr, int m, const int* live_degrees,
                                   std::int32_t* medium, std::int32_t* long_rows,
                                   std::int32_t* counts) {
  const int row = NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & 31;
  const int length = row < m ? (live_degrees ? live_degrees[row] : row_ptr[row + 1] - row_ptr[row]) : 0;
  // One atomic per warp/group, including only rows that need cooperative work.
  for (int group = 0; group < 2; ++group) {
    const bool selected = row < m && (group == 0
        ? length > ADAPTIVE_SHORT_ROW && length <= ADAPTIVE_LONG_ROW
        : length > ADAPTIVE_LONG_ROW);
    const unsigned mask = __ballot_sync(0xffffffffu, selected);
    if (mask) {
      const int leader = __ffs(mask) - 1;
      NnzOffset offset = lane == leader ? atomicAdd(counts + group, __popc(mask)) : 0;
      offset = __shfl_sync(0xffffffffu, offset, leader);
      if (selected) {
        const int rank = __popc(mask & ((1u << lane) - 1u));
        (group == 0 ? medium : long_rows)[offset + rank] = row;
      }
    }
  }
}

template<int Width>
__global__ void cooperative_activity_summary(
    double* min_fin, double* max_fin, int* min_inf, int* max_inf,
    const NnzOffset* row_ptr, const int* col_val, const double* values,
    const double* lower, const double* upper, double zero_tol,
    const int* rows, const int* count,
    const std::uint8_t* keep_row, const std::uint8_t* keep_col, const int* live_degrees) {
  const int lane = Width == 32 ? (threadIdx.x & 31) : threadIdx.x;
  const int group = Width == 32 ? blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32 : blockIdx.x;
  const NnzOffset stride = Width == 32 ? gridDim.x * (blockDim.x / 32) : gridDim.x;
  using DoubleReduce = cub::BlockReduce<double, 256>;
  using IntReduce = cub::BlockReduce<int, 256>;
  __shared__ typename DoubleReduce::TempStorage double_storage;
  __shared__ typename IntReduce::TempStorage int_storage;
  __shared__ LiveRowStorage<Width> live_storage;
  for (NnzOffset q = group; q < *count; q += stride) {
    const int row = rows[q];
    double lo = 0.0, hi = 0.0;
    int ni = 0, pi = 0;
    const auto* mask = live_degrees && live_degrees[row] == row_ptr[row+1]-row_ptr[row]
        ? nullptr : keep_col;
    if (!keep_row || keep_row[row])
      for_each_live_entry<Width>(row_ptr[row], row_ptr[row+1], col_val, mask,
                                live_storage, [&](NnzOffset p) {
      const double a = values[p];
      if (fabs(a) <= zero_tol) return;
      const int col = col_val[p];
      const double tlo = a >= 0.0 ? a * lower[col] : a * upper[col];
      const double thi = a >= 0.0 ? a * upper[col] : a * lower[col];
      if (isfinite(tlo)) lo += tlo; else if (tlo < 0.0) ++ni;
      if (isfinite(thi)) hi += thi; else if (thi > 0.0) ++pi;
    });
    if constexpr (Width == 32) {
      for (int offset = 16; offset; offset /= 2) {
        lo += __shfl_down_sync(0xffffffffu, lo, offset);
        hi += __shfl_down_sync(0xffffffffu, hi, offset);
        ni += __shfl_down_sync(0xffffffffu, ni, offset);
        pi += __shfl_down_sync(0xffffffffu, pi, offset);
      }
    } else {
      lo = DoubleReduce(double_storage).Sum(lo); __syncthreads();
      hi = DoubleReduce(double_storage).Sum(hi); __syncthreads();
      ni = IntReduce(int_storage).Sum(ni); __syncthreads();
      pi = IntReduce(int_storage).Sum(pi); __syncthreads();
    }
    if (lane == 0) {
      min_fin[row] = lo; max_fin[row] = hi;
      min_inf[row] = ni; max_inf[row] = pi;
    }
    if constexpr (Width != 32) __syncthreads();
  }
}

template<int Width>
__global__ void cooperative_col_max(double* output, const NnzOffset* ptr, const double* values,
                                     const int* rows, const int* count,
                                     const int* indices, const std::uint8_t* keep_row) {
  const int lane = Width == 32 ? (threadIdx.x & 31) : threadIdx.x;
  const int group = Width == 32 ? blockIdx.x * 8 + threadIdx.x / 32 : blockIdx.x;
  const NnzOffset stride = Width == 32 ? gridDim.x * 8 : gridDim.x;
  using Reduce = cub::BlockReduce<double, 256>;
  __shared__ typename Reduce::TempStorage storage;
  for (NnzOffset q = group; q < *count; q += stride) {
    const int row = rows[q];
    double result = 0;
    for (NnzOffset p = ptr[row] + lane; p < ptr[row+1]; p += Width)
      if (!keep_row || keep_row[indices[p]]) result = fmax(result, fabs(values[p]));
    if constexpr (Width == 32) {
      for (int offset = 16; offset; offset /= 2)
        result = fmax(result, __shfl_down_sync(0xffffffffu, result, offset));
    } else {
      result = Reduce(storage).Reduce(result, thrust::maximum<double>());
      __syncthreads();
    }
    if (lane == 0) output[row] = result;
    if constexpr (Width != 32) __syncthreads();
  }
}

__global__ void check_sorted_short(const NnzOffset* ptr, const int* indices, int m, int* bad) {
  const int row=blockIdx.x*blockDim.x+threadIdx.x;
  if (row>=m || ptr[row+1]-ptr[row]>ADAPTIVE_SHORT_ROW) return;
  for (NnzOffset p=ptr[row]+1;p<ptr[row+1];++p)
    if(indices[p-1]>=indices[p]) { atomicExch(bad,1);break; }
}
template<int Width>
__global__ void check_sorted_grouped(const NnzOffset* ptr,const int* indices,
                                     const int* rows,const int* count,int* bad) {
  const int lane=Width==32 ? threadIdx.x&31 : threadIdx.x;
  const int group=Width==32 ? blockIdx.x*8+threadIdx.x/32 : blockIdx.x;
  const NnzOffset stride=Width==32 ? gridDim.x*8 : gridDim.x;
  for(NnzOffset q=group;q<*count;q+=stride) {
    const int row=rows[q];
    for(NnzOffset p=ptr[row]+1+lane;p<ptr[row+1];p+=Width)
      if(indices[p-1]>=indices[p]) { atomicExch(bad,1);break; }
  }
}

struct AdaptiveWorkspace {
  AdaptiveRowGroups groups;
  std::size_t medium_capacity = 0, long_capacity = 0;
  int device = -1;
  const NnzOffset* row_ptr = nullptr;
  int rows = -1; NnzOffset nnz = -1;
  std::uint64_t epoch = 0;
  ~AdaptiveWorkspace() {
    int current = -1;
    if (cudaGetDevice(&current) == cudaSuccess && device >= 0) {
      cudaSetDevice(device);
      cudaFree(groups.medium); cudaFree(groups.long_rows); cudaFree(groups.counts);
      cudaSetDevice(current);
    }
  }
};
thread_local bool analysis_scope_active = false;
thread_local std::uint64_t analysis_generation = 1;
}  // namespace

bool adaptive_gather_enabled() {
  return runtime::options().gather_live_rows;
}
bool adaptive_hash_scan_enabled() {
  return runtime::options().hash_scan;
}
bool adaptive_sort_reuse_enabled() {
  return runtime::options().reuse_sortedness;
}
bool adaptive_reuse_enabled() {
  return runtime::options().reuse_row_groups;
}
bool adaptive_extended_enabled() {
  return runtime::options().extended_row_kernels;
}
MatrixAnalysisScope::MatrixAnalysisScope() : previous_(analysis_scope_active) {
  analysis_scope_active = adaptive_reuse_enabled();
  invalidate_matrix_analysis();
}
MatrixAnalysisScope::~MatrixAnalysisScope() {
  invalidate_matrix_analysis();
  analysis_scope_active = previous_;
}
void invalidate_matrix_analysis() { ++analysis_generation; }
std::uint64_t matrix_analysis_epoch() {
  return analysis_scope_active ? analysis_generation : 0;
}

bool adaptive_row_dispatch_enabled() {
  return runtime::options().adaptive_activity;
}

const AdaptiveRowGroups& prepare_adaptive_row_groups(const DeviceCsrMatrix& matrix, const std::int32_t* live_degrees) {
  static thread_local std::array<AdaptiveWorkspace, 2> workspaces;
  static thread_local unsigned next_slot = 0;
  int device = 0;
  throw_if_cuda_error(cudaGetDevice(&device), "adaptive workspace device");
  const auto epoch = matrix_analysis_epoch();
  if (epoch && !live_degrees) {
    for (auto& cached : workspaces)
      if (cached.epoch == epoch && cached.device == device &&
          cached.row_ptr == matrix.rowPtr && cached.rows == matrix.rows && cached.nnz == matrix.nnz)
        return cached.groups;
  }
  auto& workspace = workspaces[next_slot++ % workspaces.size()];
  if (workspace.device != device) {
    if (workspace.device >= 0) {
      throw_if_cuda_error(cudaSetDevice(workspace.device), "adaptive workspace previous device");
      cudaFree(workspace.groups.medium); cudaFree(workspace.groups.long_rows); cudaFree(workspace.groups.counts);
      throw_if_cuda_error(cudaSetDevice(device), "adaptive workspace restore device");
    }
    workspace.groups = {};
    workspace.medium_capacity = workspace.long_capacity = 0;
    workspace.device = device;
  }
  auto ensure = [&](int*& pointer, std::size_t& capacity, std::size_t needed) {
    if (needed <= capacity) return;
    if (pointer) throw_if_cuda_error(cudaFreeAsync(pointer, nullptr), "adaptive workspace free");
    throw_if_cuda_error(cudaMallocAsync(reinterpret_cast<void**>(&pointer), needed * sizeof(int), nullptr), "adaptive workspace allocate");
    capacity = needed;
  };
  // Each queued row consumes at least threshold+1 stored entries.
  ensure(workspace.groups.medium, workspace.medium_capacity,
         std::max<NnzOffset>(1, std::min<NnzOffset>(matrix.rows, matrix.nnz / (ADAPTIVE_SHORT_ROW + 1))));
  ensure(workspace.groups.long_rows, workspace.long_capacity,
         std::max<NnzOffset>(1, std::min<NnzOffset>(matrix.rows, matrix.nnz / (ADAPTIVE_LONG_ROW + 1))));
  if (!workspace.groups.counts)
    throw_if_cuda_error(cudaMallocAsync(reinterpret_cast<void**>(&workspace.groups.counts), 2 * sizeof(int), nullptr), "adaptive counts allocate");
  throw_if_cuda_error(cudaMemsetAsync(workspace.groups.counts, 0, 2 * sizeof(int)), "adaptive counts clear");
  if (matrix.rows)
    group_adaptive_rows<<<(matrix.rows + 255) / 256, 256>>>(matrix.rowPtr, matrix.rows, live_degrees,
        workspace.groups.medium, workspace.groups.long_rows, workspace.groups.counts);
  throw_if_cuda_error(cudaGetLastError(), "group adaptive rows");
  workspace.epoch = live_degrees ? 0 : epoch;
  workspace.row_ptr = matrix.rowPtr;
  workspace.rows = matrix.rows;
  workspace.nnz = matrix.nnz;
  return workspace.groups;
}

bool csr_rows_strictly_sorted(const DeviceCsrMatrix& matrix) {
  if (!adaptive_sort_reuse_enabled()) return false;
  struct Entry { const NnzOffset* ptr=nullptr; const int* indices=nullptr; int rows=-1; NnzOffset nnz=-1; int device=-1; std::uint64_t epoch=0; };
  static thread_local std::array<Entry,4> entries;
  static thread_local unsigned next=0;
  int device=0;
  throw_if_cuda_error(cudaGetDevice(&device),"sorted cache device");
  const auto epoch=matrix_analysis_epoch();
  if(epoch) for(const auto& e:entries)
    if(e.epoch==epoch && e.ptr==matrix.rowPtr && e.indices==matrix.colVal &&
       e.rows==matrix.rows && e.nnz==matrix.nnz && e.device==device) return true;
  if(matrix.rows<=0 || matrix.nnz<=1) return true;
  int* bad=nullptr;
  throw_if_cuda_error(cudaMalloc(&bad,sizeof(int)),"sorted check flag");
  throw_if_cuda_error(cudaMemset(bad,0,sizeof(int)),"sorted check clear");
  const auto& groups=prepare_adaptive_row_groups(matrix);
  check_sorted_short<<<(matrix.rows+255)/256,256>>>(matrix.rowPtr,matrix.colVal,matrix.rows,bad);
  check_sorted_grouped<32><<<std::min(1024,(matrix.rows+7)/8),256>>>(matrix.rowPtr,matrix.colVal,groups.medium,groups.counts,bad);
  check_sorted_grouped<256><<<std::min(256,matrix.rows),256>>>(matrix.rowPtr,matrix.colVal,groups.long_rows,groups.counts+1,bad);
  throw_if_cuda_error(cudaGetLastError(),"sorted check kernels");
  int host_bad=0;
  const auto copy_status=cudaMemcpy(&host_bad,bad,sizeof(int),cudaMemcpyDeviceToHost);
  cudaFree(bad);
  throw_if_cuda_error(copy_status,"sorted check readback");
  if(host_bad) return false;
  entries[next++%entries.size()]={matrix.rowPtr,matrix.colVal,matrix.rows,matrix.nnz,device,epoch};
  return true;
}

namespace {
template<int Width>
__global__ void live_stats_kernel(DeviceCsrMatrix matrix,
    const std::uint8_t* keep_outer, const std::uint8_t* keep_inner,
    int* degree, std::uint8_t* empty, std::uint8_t* singleton,
    int* support, double* coefficient, unsigned long long* total,
    const int* rows, const int* row_count, int max_length) {
  const int lane = Width == 1 ? 0 : Width == 32 ? threadIdx.x & 31 : threadIdx.x;
  const int group = Width == 1 ? blockIdx.x * blockDim.x + threadIdx.x
      : Width == 32 ? blockIdx.x * 8 + threadIdx.x / 32 : blockIdx.x;
  const int stride = Width == 1 ? gridDim.x * blockDim.x
      : Width == 32 ? gridDim.x * 8 : gridDim.x;
  using CountReduce = cub::BlockReduce<int, 256>;
  using PositionReduce = cub::BlockReduce<NnzOffset, 256>;
  __shared__ typename CountReduce::TempStorage count_storage;
  __shared__ typename PositionReduce::TempStorage position_storage;
  const int size = row_count ? *row_count : matrix.rows;
  for (int q = group; q < size; q += stride) {
    const int row = rows ? rows[q] : q;
    const auto first = matrix.rowPtr[row], last = matrix.rowPtr[row+1];
    if (max_length && last-first > max_length) continue;
    const bool active = !keep_outer || keep_outer[row];
    int count = 0;
    NnzOffset position = -1;
    if (active) for (NnzOffset p = first+lane; p < last; p += Width) {
      if (keep_inner && !keep_inner[matrix.colVal[p]]) continue;
      ++count; position = p;
    }
    if constexpr (Width == 32) {
      for (int off = 16; off; off /= 2) {
        count += __shfl_down_sync(0xffffffffu, count, off);
        const auto other = __shfl_down_sync(0xffffffffu, position, off);
        position = position > other ? position : other;
      }
    } else if constexpr (Width == 256) {
      count = CountReduce(count_storage).Sum(count);
      position = PositionReduce(position_storage).Reduce(position, thrust::maximum<NnzOffset>{});
      __syncthreads();
    }
    if (lane == 0) {
      if (degree) degree[row] = count;
      if (empty) empty[row] = active && count == 0;
      if (singleton) singleton[row] = active && count == 1;
      if (support) support[row] = count == 1 ? matrix.colVal[position] : -1;
      if (coefficient) coefficient[row] = count == 1 ? matrix.nzVal[position] : 0;
      if (total) atomicAdd(total, static_cast<unsigned long long>(count));
    }
    if constexpr (Width == 256) __syncthreads();
  }
}

void live_stats(const DeviceCsrMatrix& matrix, const std::uint8_t* keep_outer,
    const std::uint8_t* keep_inner, int* degree, std::uint8_t* empty,
    std::uint8_t* singleton, int* support, double* coefficient, unsigned long long* total) {
  if (!matrix.rows) return;
  const bool adaptive = adaptive_row_dispatch_enabled();
  live_stats_kernel<1><<<(matrix.rows+255)/256,256>>>(matrix, keep_outer, keep_inner,
      degree, empty, singleton, support, coefficient, total, nullptr, nullptr,
      adaptive ? ADAPTIVE_SHORT_ROW : 0);
  if (adaptive) {
    // Traversal cost depends on stored length, including masked-out entries.
    const auto& groups = prepare_adaptive_row_groups(matrix);
    live_stats_kernel<32><<<std::min(1024,(matrix.rows+7)/8),256>>>(matrix,
        keep_outer, keep_inner, degree, empty, singleton, support, coefficient, total,
        groups.medium, groups.counts, 0);
    live_stats_kernel<256><<<std::min(256,matrix.rows),256>>>(matrix,
        keep_outer, keep_inner, degree, empty, singleton, support, coefficient, total,
        groups.long_rows, groups.counts+1, 0);
  }
}
} // namespace

void refresh_live_stats(PresolveStatsGpu& stats, const LPInfoGpu& lp,
                        const PresolvePlanGpu& plan, bool read_live_nnz,
                        unsigned long long* device_live_nnz) {
  CleanupScratchFrame scratch;
  auto* total = device_live_nnz;
  if (read_live_nnz && !total) total = scratch.get<unsigned long long>(7, 1);
  if (total) throw_if_cuda_error(cudaMemset(total, 0, sizeof(*total)), "zero live nnz");
  live_stats(lp.A,
      plan.keep_row_mask, plan.keep_col_mask, stats.row_nnz, stats.empty_row_mask,
      stats.singleton_row_mask, stats.singleton_row_col, stats.singleton_row_val, total);
  live_stats(lp.AT,
      plan.keep_col_mask, plan.keep_row_mask, stats.col_nnz, stats.empty_col_mask,
      stats.singleton_column_mask, stats.singleton_column_row, stats.singleton_column_val, nullptr);
  throw_if_cuda_error(cudaGetLastError(), "refresh live statistics");
  if (read_live_nnz) {
    unsigned long long count = 0;
    throw_if_cuda_error(cudaMemcpy(&count, total, sizeof(count), cudaMemcpyDeviceToHost),
                        "read live nnz");
    stats.live_nnz = static_cast<NnzOffset>(count);
  }
  stats.live_masks = true;
  stats.row_nnz_valid = stats.row_nnz != nullptr;
  stats.col_nnz_valid = stats.col_nnz != nullptr;
  stats.absolute_value_epigraph_screen_valid = false;
}

namespace {
__global__ void incremental_deleted_rows(const DeviceCsrMatrix a,
    const std::uint8_t* old_rows, const std::uint8_t* old_cols,
    const std::uint8_t* new_rows, const std::uint8_t* new_cols,
    std::int32_t* col_degree, std::int32_t* touched_cols,
    unsigned long long* removed, std::int32_t* invalid) {
  const int row = int(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= a.rows) return;
  if (!old_rows[row]) {
    if (new_rows[row]) atomicExch(invalid, 1);
    return;
  }
  if (new_rows[row]) return;
  unsigned long long count = 0;
  for (NnzOffset p = a.rowPtr[row]; p < a.rowPtr[row + 1]; ++p) {
    const int col = a.colVal[p];
    if (!old_cols[col]) continue;
    ++count;
    if (new_cols[col]) {
      atomicSub(col_degree + col, 1);
      atomicExch(touched_cols + col, 1);
    }
  }
  if (count) atomicAdd(removed, count);
}

__global__ void incremental_deleted_cols(const DeviceCsrMatrix at,
    const std::uint8_t* old_cols, const std::uint8_t* new_rows,
    const std::uint8_t* new_cols, std::int32_t* row_degree,
    std::int32_t* touched_rows, unsigned long long* removed,
    std::int32_t* invalid) {
  const int col = int(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (col >= at.rows) return;
  if (!old_cols[col]) {
    if (new_cols[col]) atomicExch(invalid, 1);
    return;
  }
  if (new_cols[col]) return;
  unsigned long long count = 0;
  for (NnzOffset p = at.rowPtr[col]; p < at.rowPtr[col + 1]; ++p) {
    const int row = at.colVal[p];
    if (!new_rows[row]) continue;
    ++count;
    atomicSub(row_degree + row, 1);
    atomicExch(touched_rows + row, 1);
  }
  if (count) atomicAdd(removed, count);
}

__global__ void finalize_incremental_stats(const DeviceCsrMatrix a,
    const DeviceCsrMatrix at, const std::uint8_t* keep_rows,
    const std::uint8_t* keep_cols, const std::int32_t* touched_rows,
    const std::int32_t* touched_cols, PresolveStatsGpu stats,
    std::int32_t* invalid) {
  const int index = int(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (index < a.rows && (!keep_rows[index] || touched_rows[index])) {
    const bool active = keep_rows[index] != 0;
    const int degree = active ? stats.row_nnz[index] : 0;
    if (degree < 0) atomicExch(invalid, 1);
    stats.row_nnz[index] = degree;
    stats.empty_row_mask[index] = active && degree == 0;
    stats.singleton_row_mask[index] = active && degree == 1;
    int support = -1;
    double value = 0.0;
    if (active && degree == 1) {
      for (NnzOffset p = a.rowPtr[index]; p < a.rowPtr[index + 1]; ++p) {
        if (!keep_cols[a.colVal[p]]) continue;
        if (support >= 0) atomicExch(invalid, 1);
        support = a.colVal[p];
        value = a.nzVal[p];
      }
      if (support < 0) atomicExch(invalid, 1);
    }
    stats.singleton_row_col[index] = support;
    stats.singleton_row_val[index] = value;
  }
  if (index < at.rows && (!keep_cols[index] || touched_cols[index])) {
    const bool active = keep_cols[index] != 0;
    const int degree = active ? stats.col_nnz[index] : 0;
    if (degree < 0) atomicExch(invalid, 1);
    stats.col_nnz[index] = degree;
    stats.empty_col_mask[index] = active && degree == 0;
    stats.singleton_column_mask[index] = active && degree == 1;
    int support = -1;
    double value = 0.0;
    if (active && degree == 1) {
      for (NnzOffset p = at.rowPtr[index]; p < at.rowPtr[index + 1]; ++p) {
        if (!keep_rows[at.colVal[p]]) continue;
        if (support >= 0) atomicExch(invalid, 1);
        support = at.colVal[p];
        value = at.nzVal[p];
      }
      if (support < 0) atomicExch(invalid, 1);
    }
    stats.singleton_column_row[index] = support;
    stats.singleton_column_val[index] = value;
  }
}
__global__ void subtract_removed_live_nnz(unsigned long long* live_nnz,
    const unsigned long long* removed, std::int32_t* invalid) {
  if (*invalid) return;
  if (*removed > *live_nnz) {
    *invalid = 1;
    return;
  }
  *live_nnz -= *removed;
}
} // namespace

bool update_live_stats_from_deletions(PresolveStatsGpu& stats, const LPInfoGpu& lp,
    const PresolvePlanGpu& plan, const std::uint8_t* old_rows,
    const std::uint8_t* old_cols, std::int32_t* touched_rows,
    std::int32_t* touched_cols, bool read_live_nnz,
    unsigned long long* device_live_nnz) {
  if (!stats.live_masks || !stats.row_nnz || !stats.col_nnz ||
      !stats.empty_row_mask || !stats.empty_col_mask ||
      !stats.singleton_row_mask || !stats.singleton_column_mask ||
      !stats.singleton_row_col || !stats.singleton_column_row ||
      !stats.singleton_row_val || !stats.singleton_column_val ||
      !old_rows || !old_cols || (!read_live_nnz && !device_live_nnz)) return false;
  const int m = lp.A.rows;
  const int n = lp.A.cols;
  if (m) throw_if_cuda_error(cudaMemset(touched_rows, 0, sizeof(int) * std::size_t(m)),
                             "clear touched rows");
  if (n) throw_if_cuda_error(cudaMemset(touched_cols, 0, sizeof(int) * std::size_t(n)),
                             "clear touched cols");
  CleanupScratchFrame scratch;
  auto* removed = scratch.get<unsigned long long>(7, 1);
  auto* invalid = scratch.get<std::int32_t>(6, 1);
  throw_if_cuda_error(cudaMemset(removed, 0, sizeof(*removed)), "clear removed nnz");
  throw_if_cuda_error(cudaMemset(invalid, 0, sizeof(*invalid)), "clear incremental invalid");
  if (m) incremental_deleted_rows<<<(m + 255) / 256, 256>>>(
      lp.A, old_rows, old_cols, plan.keep_row_mask, plan.keep_col_mask,
      stats.col_nnz, touched_cols, removed, invalid);
  if (n) incremental_deleted_cols<<<(n + 255) / 256, 256>>>(
      lp.AT, old_cols, plan.keep_row_mask, plan.keep_col_mask,
      stats.row_nnz, touched_rows, removed, invalid);
  const int count = std::max(m, n);
  if (count) finalize_incremental_stats<<<(count + 255) / 256, 256>>>(
      lp.A, lp.AT, plan.keep_row_mask, plan.keep_col_mask,
      touched_rows, touched_cols, stats, invalid);
  if (device_live_nnz) subtract_removed_live_nnz<<<1, 1>>>(device_live_nnz, removed, invalid);
  throw_if_cuda_error(cudaGetLastError(), "update live statistics from deletions");
  int invalid_host = 0;
  throw_if_cuda_error(cudaMemcpy(&invalid_host, invalid, sizeof(invalid_host),
                                cudaMemcpyDeviceToHost), "read incremental invalid");
  if (invalid_host) return false;
  if (read_live_nnz) {
    unsigned long long count_host = 0;
    throw_if_cuda_error(cudaMemcpy(&count_host, device_live_nnz ? device_live_nnz : removed,
                                  sizeof(count_host), cudaMemcpyDeviceToHost),
                        "read incremental live nnz");
    if (device_live_nnz) {
      stats.live_nnz = static_cast<NnzOffset>(count_host);
    } else {
      if (count_host > static_cast<unsigned long long>(stats.live_nnz)) return false;
      stats.live_nnz -= static_cast<NnzOffset>(count_host);
    }
  }
  stats.row_nnz_valid = true;
  stats.col_nnz_valid = true;
  stats.absolute_value_epigraph_screen_valid = false;
  return true;
}

void compute_row_nnz(std::int32_t* row_nnz, const DeviceCsrMatrix& A_csr) {
  if (A_csr.rows == 0) {
    return;
  }
  const int blocks = (A_csr.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_compute_row_nnz<<<blocks, GPU_PRESOLVE_THREADS>>>(row_nnz, A_csr.rowPtr, A_csr.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_compute_row_nnz");
}

void compute_col_nnz(std::int32_t* col_nnz, const DeviceCsrMatrix& AT_csr) {
  compute_row_nnz(col_nnz, AT_csr);
}

void compute_singleton_row_support(std::int32_t* singleton_column,
                                   double* singleton_val,
                                   const std::int32_t* row_nnz,
                                   const DeviceCsrMatrix& A_csr) {
  if (A_csr.rows == 0) {
    return;
  }
  const int blocks = (A_csr.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_singleton_row_support<<<blocks, GPU_PRESOLVE_THREADS>>>(
      singleton_column,
      singleton_val,
      row_nnz,
      A_csr.rowPtr,
      A_csr.colVal,
      A_csr.nzVal,
      A_csr.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_singleton_row_support");
}

void compute_singleton_column_support(std::int32_t* singleton_row,
                                   double* singleton_val,
                                   const std::int32_t* col_nnz,
                                   const DeviceCsrMatrix& AT_csr) {
  compute_singleton_row_support(singleton_row, singleton_val, col_nnz, AT_csr);
}

void compute_row_activity_bounds(double* row_min,
                                 double* row_max,
                                 const DeviceCsrMatrix& A_csr,
                                 const double* l,
                                 const double* u) {
  if (A_csr.rows == 0) {
    return;
  }
  const int blocks = (A_csr.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_compute_row_activity_bounds<<<blocks, GPU_PRESOLVE_THREADS>>>(
      row_min,
      row_max,
      A_csr.rowPtr,
      A_csr.colVal,
      A_csr.nzVal,
      l,
      u,
      A_csr.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_compute_row_activity_bounds");
}

void compute_row_activity_summary(double* row_min_fin,
                                  double* row_max_fin,
                                  std::int32_t* row_min_neg_inf_count,
                                  std::int32_t* row_max_pos_inf_count,
                                  const DeviceCsrMatrix& A_csr,
                                  const double* l,
                                  const double* u,
                                  double zero_tol,
                                  const std::uint8_t* keep_row,
                                  const std::uint8_t* keep_col,
                                  const std::int32_t* live_degrees) {
  if (A_csr.rows == 0) {
    return;
  }
  const bool adaptive = adaptive_row_dispatch_enabled();
  const int blocks = (A_csr.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_compute_row_activity_summary<<<blocks, GPU_PRESOLVE_THREADS>>>(
      row_min_fin,
      row_max_fin,
      row_min_neg_inf_count,
      row_max_pos_inf_count,
      A_csr.rowPtr,
      A_csr.colVal,
      A_csr.nzVal,
      l,
      u,
      zero_tol,
      A_csr.rows, adaptive ? ADAPTIVE_SHORT_ROW : 0, keep_row, keep_col, live_degrees);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_compute_row_activity_summary");
  if (adaptive) {
    const auto& groups = prepare_adaptive_row_groups(A_csr, live_degrees);
    const int warp_blocks = std::min(1024, (A_csr.rows + 7) / 8);
    cooperative_activity_summary<32><<<warp_blocks, 256>>>(
        row_min_fin, row_max_fin, row_min_neg_inf_count, row_max_pos_inf_count,
        A_csr.rowPtr, A_csr.colVal, A_csr.nzVal, l, u, zero_tol,
        groups.medium, groups.counts, keep_row, keep_col, live_degrees);
    cooperative_activity_summary<256><<<std::min(256, A_csr.rows), 256>>>(
        row_min_fin, row_max_fin, row_min_neg_inf_count, row_max_pos_inf_count,
        A_csr.rowPtr, A_csr.colVal, A_csr.nzVal, l, u, zero_tol,
        groups.long_rows, groups.counts + 1, keep_row, keep_col, live_degrees);
    throw_if_cuda_error(cudaGetLastError(), "cooperative activity summary");
  }

}

void compute_col_max_abs(double* col_max_abs, const DeviceCsrMatrix& AT_csr,
                         const std::uint8_t* keep_row) {
  if (AT_csr.rows == 0) {
    return;
  }
  const int blocks = (AT_csr.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_compute_col_max_abs<<<blocks, GPU_PRESOLVE_THREADS>>>(
      col_max_abs,
      AT_csr.rowPtr,
      AT_csr.nzVal,
      AT_csr.rows, adaptive_extended_enabled() ? ADAPTIVE_SHORT_ROW : 0, AT_csr.colVal, keep_row);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_compute_col_max_abs");
  if (adaptive_extended_enabled()) {
    const auto& groups = prepare_adaptive_row_groups(AT_csr);
    cooperative_col_max<32><<<std::min(1024, (AT_csr.rows+7)/8), 256>>>(
        col_max_abs, AT_csr.rowPtr, AT_csr.nzVal, groups.medium, groups.counts, AT_csr.colVal, keep_row);
    cooperative_col_max<256><<<std::min(256, AT_csr.rows), 256>>>(
        col_max_abs, AT_csr.rowPtr, AT_csr.nzVal, groups.long_rows, groups.counts+1, AT_csr.colVal, keep_row);
    throw_if_cuda_error(cudaGetLastError(), "cooperative col max");
  }
}

}  // namespace gpu_presolver::presolve
