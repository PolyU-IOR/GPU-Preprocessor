#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/live_matrix.cuh"
#include "gpu_presolver/presolve/cleanup_support.cuh"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/rules/rule_singleton_columns.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include "gpu_presolver/presolve/gpu_presolve_kernels.hpp"

#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;
constexpr std::uint8_t SINGLETON_COL_INEQ_NO_ACTION = 0;
constexpr std::uint8_t SINGLETON_COL_INEQ_ELIMINATE = 1;
constexpr std::uint8_t SINGLETON_COL_INEQ_TIGHTEN_LHS_TO_RHS = 2;
constexpr std::uint8_t SINGLETON_COL_INEQ_TIGHTEN_RHS_TO_LHS = 3;
constexpr std::int32_t SINGLETON_ACTIVITY_LONG_ROW_THRESHOLD = 1024;

// Immutable decisions computed against the input plan. Prefer deleting the
// whole row, then break ties by column ID; no candidate mutates the plan.
enum class SingletonAction : std::uint8_t {
  None = SINGLETON_COL_INEQ_NO_ACTION,
  SetRowSide = SINGLETON_COL_INEQ_ELIMINATE,
  TightenLhs = SINGLETON_COL_INEQ_TIGHTEN_LHS_TO_RHS,
  TightenRhs = SINGLETON_COL_INEQ_TIGHTEN_RHS_TO_LHS,
  EliminateFree,
  EliminateFreeAbove,
  EliminateFreeBelow,
};

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

template <class T>
cudaError_t singleton_columns_cuda_malloc(T** pointer, std::size_t bytes) {
  return cudaMallocAsync(reinterpret_cast<void**>(pointer), bytes, nullptr);
}

cudaError_t singleton_columns_cuda_free(void* pointer) {
  return pointer == nullptr ? cudaSuccess : cudaFreeAsync(pointer, nullptr);
}

template <typename T>
void free_device_pointer(T*& pointer) noexcept {
  T* allocation = pointer;
  pointer = nullptr;
  if (allocation != nullptr) {
    singleton_columns_cuda_free(allocation);
  }
}

template <typename T>
void allocate_device_pointer(T*& pointer,
                             std::size_t count,
                             const char* context) {
  pointer = nullptr;
  void* allocation = nullptr;
  const cudaError_t status = singleton_columns_cuda_malloc(&allocation, sizeof(T) * count);
  if (status != cudaSuccess) {
    if (allocation != nullptr) {
      singleton_columns_cuda_free(allocation);
    }
    throw_if_cuda_error(status, context);
  }
  pointer = static_cast<T*>(allocation);
}

struct SingletonColumnsWorkspace {
  std::int32_t* status_flag = nullptr;
  std::int32_t* row_owner = nullptr;
  std::uint8_t* col_delete = nullptr;
  std::uint8_t* row_delete = nullptr;
  std::uint8_t* row_lhs_change = nullptr;
  std::uint8_t* row_rhs_change = nullptr;
  SingletonAction* actions = nullptr;
  double* chosen_side = nullptr;
  double* c_new = nullptr;
  double* AL_new = nullptr;
  double* AU_new = nullptr;
  double* obj_contrib = nullptr;
  double* activity_finite_min = nullptr;
  double* activity_finite_max = nullptr;
  std::int32_t* activity_min_neg_inf = nullptr;
  std::int32_t* activity_min_pos_inf = nullptr;
  std::int32_t* activity_max_neg_inf = nullptr;
  std::int32_t* activity_max_pos_inf = nullptr;
  std::uint8_t* activity_invalid = nullptr;
  std::uint8_t* activity_row_needed = nullptr;
  std::int32_t* activity_long_rows = nullptr;
  std::int32_t* activity_long_row_count = nullptr;
  std::size_t row_capacity = 0;
  std::size_t col_capacity = 0;

  SingletonColumnsWorkspace() = default;
  SingletonColumnsWorkspace(const SingletonColumnsWorkspace&) = delete;
  SingletonColumnsWorkspace& operator=(const SingletonColumnsWorkspace&) = delete;

  void release_fixed() noexcept {
    free_device_pointer(status_flag);
    free_device_pointer(activity_long_row_count);
  }

  void release_rows() noexcept {
    row_capacity = 0;
    free_device_pointer(row_owner);
    free_device_pointer(row_delete);
    free_device_pointer(row_lhs_change);
    free_device_pointer(row_rhs_change);
    free_device_pointer(AL_new);
    free_device_pointer(AU_new);
    free_device_pointer(activity_finite_min);
    free_device_pointer(activity_finite_max);
    free_device_pointer(activity_min_neg_inf);
    free_device_pointer(activity_min_pos_inf);
    free_device_pointer(activity_max_neg_inf);
    free_device_pointer(activity_max_pos_inf);
    free_device_pointer(activity_invalid);
    free_device_pointer(activity_row_needed);
    free_device_pointer(activity_long_rows);
  }

  void release_cols() noexcept {
    col_capacity = 0;
    free_device_pointer(col_delete);
    free_device_pointer(actions);
    free_device_pointer(chosen_side);
    free_device_pointer(c_new);
    free_device_pointer(obj_contrib);
  }

  void release() noexcept {
    release_fixed();
    release_rows();
    release_cols();
  }

  void ensure(std::int32_t m, std::int32_t n) {
    if (status_flag == nullptr || activity_long_row_count == nullptr) {
      release_fixed();
      try {
        allocate_device_pointer(status_flag, 3, "cudaMalloc singleton workspace status");
        allocate_device_pointer(activity_long_row_count,
                                1,
                                "cudaMalloc singleton workspace long-row count");
      } catch (...) {
        release_fixed();
        throw;
      }
    }
    const std::size_t required_rows = static_cast<std::size_t>(m);
    if (required_rows > row_capacity) {
      release_rows();
      try {
        allocate_device_pointer(row_owner,
                                required_rows,
                                "cudaMalloc singleton workspace row_owner");
        allocate_device_pointer(row_delete,
                                required_rows,
                                "cudaMalloc singleton workspace row_delete");
        allocate_device_pointer(row_lhs_change,
                                required_rows,
                                "cudaMalloc singleton workspace row_lhs_change");
        allocate_device_pointer(row_rhs_change,
                                required_rows,
                                "cudaMalloc singleton workspace row_rhs_change");
        allocate_device_pointer(AL_new,
                                required_rows,
                                "cudaMalloc singleton workspace AL_new");
        allocate_device_pointer(AU_new,
                                required_rows,
                                "cudaMalloc singleton workspace AU_new");
        allocate_device_pointer(activity_finite_min,
                                required_rows,
                                "cudaMalloc singleton workspace activity finite min");
        allocate_device_pointer(activity_finite_max,
                                required_rows,
                                "cudaMalloc singleton workspace activity finite max");
        allocate_device_pointer(activity_min_neg_inf,
                                required_rows,
                                "cudaMalloc singleton workspace activity min -inf");
        allocate_device_pointer(activity_min_pos_inf,
                                required_rows,
                                "cudaMalloc singleton workspace activity min +inf");
        allocate_device_pointer(activity_max_neg_inf,
                                required_rows,
                                "cudaMalloc singleton workspace activity max -inf");
        allocate_device_pointer(activity_max_pos_inf,
                                required_rows,
                                "cudaMalloc singleton workspace activity max +inf");
        allocate_device_pointer(activity_invalid,
                                required_rows,
                                "cudaMalloc singleton workspace activity invalid");
        allocate_device_pointer(activity_row_needed,
                                required_rows,
                                "cudaMalloc singleton workspace activity rows");
        allocate_device_pointer(activity_long_rows,
                                required_rows,
                                "cudaMalloc singleton workspace long rows");
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
        allocate_device_pointer(col_delete,
                                required_cols,
                                "cudaMalloc singleton workspace col_delete");
        allocate_device_pointer(actions,
                                required_cols,
                                "cudaMalloc singleton workspace actions");
        allocate_device_pointer(chosen_side,
                                required_cols,
                                "cudaMalloc singleton workspace chosen_side");
        allocate_device_pointer(c_new,
                                required_cols,
                                "cudaMalloc singleton workspace c_new");
        allocate_device_pointer(obj_contrib,
                                required_cols,
                                "cudaMalloc singleton workspace objective contribution");
        col_capacity = required_cols;
      } catch (...) {
        release_cols();
        throw;
      }
    }
  }

  ~SingletonColumnsWorkspace() { release(); }
};

struct DeviceScratchBuffer {
  void* data = nullptr;
  std::size_t capacity = 0;

  DeviceScratchBuffer() = default;
  DeviceScratchBuffer(const DeviceScratchBuffer&) = delete;
  DeviceScratchBuffer& operator=(const DeviceScratchBuffer&) = delete;

  void release() noexcept {
    capacity = 0;
    void* allocation = data;
    data = nullptr;
    if (allocation != nullptr) {
      singleton_columns_cuda_free(allocation);
    }
  }

  void ensure(std::size_t bytes, const char* context) {
    if (bytes <= capacity) {
      return;
    }
    release();
    void* allocation = nullptr;
    const cudaError_t status = singleton_columns_cuda_malloc(&allocation, bytes);
    if (status != cudaSuccess) {
      if (allocation != nullptr) {
        singleton_columns_cuda_free(allocation);
      }
      data = nullptr;
      capacity = 0;
      throw_if_cuda_error(status, context);
    }
    data = allocation;
    capacity = bytes;
  }

  ~DeviceScratchBuffer() { release(); }
};

struct SingletonColumnsTlsCache {
  int owner_device = -1;
  SingletonColumnsWorkspace workspace;
  DeviceScratchBuffer inclusive_scratch;
  DeviceScratchBuffer reduce_scratch;
  DeviceScratchBuffer reduce_result_scratch;

  SingletonColumnsTlsCache() = default;
  SingletonColumnsTlsCache(const SingletonColumnsTlsCache&) = delete;
  SingletonColumnsTlsCache& operator=(const SingletonColumnsTlsCache&) = delete;

  void release_allocations() noexcept {
    workspace.release();
    inclusive_scratch.release();
    reduce_scratch.release();
    reduce_result_scratch.release();
  }

  void release() noexcept {
    const int allocation_device = owner_device;
    owner_device = -1;

    int current_device = -1;
    const bool have_current_device = cudaGetDevice(&current_device) == cudaSuccess;
    bool restore_current_device = false;
    if (allocation_device >= 0 &&
        (!have_current_device || current_device != allocation_device)) {
      restore_current_device = cudaSetDevice(allocation_device) == cudaSuccess &&
                               have_current_device;
    }

    release_allocations();

    if (restore_current_device) {
      cudaSetDevice(current_device);
    }
  }

  void bind_to_device(int device) noexcept {
    if (owner_device == device) {
      return;
    }
    if (owner_device >= 0) {
      release();
    }
    owner_device = device;
  }

  ~SingletonColumnsTlsCache() { release(); }
};

SingletonColumnsTlsCache& singleton_columns_tls_cache_storage() noexcept {
  thread_local SingletonColumnsTlsCache cache;
  return cache;
}

SingletonColumnsTlsCache& singleton_columns_tls_cache() {
  int current_device = -1;
  throw_if_cuda_error(cudaGetDevice(&current_device),
                      "cudaGetDevice singleton workspace");
  SingletonColumnsTlsCache& cache = singleton_columns_tls_cache_storage();
  cache.bind_to_device(current_device);
  return cache;
}

SingletonColumnsWorkspace& singleton_columns_workspace() {
  return singleton_columns_tls_cache().workspace;
}

DeviceScratchBuffer& singleton_columns_inclusive_scratch() {
  return singleton_columns_tls_cache().inclusive_scratch;
}

DeviceScratchBuffer& singleton_columns_reduce_scratch() {
  return singleton_columns_tls_cache().reduce_scratch;
}

DeviceScratchBuffer& singleton_columns_reduce_result_scratch() {
  return singleton_columns_tls_cache().reduce_result_scratch;
}

template <typename T>
void inclusive_scan_i32(T* values, std::int32_t n, const char* context) {
  if (n <= 0) {
    return;
  }
  DeviceScratchBuffer& scratch = singleton_columns_inclusive_scratch();
  void* temp_storage = nullptr;
  std::size_t temp_bytes = 0;
  throw_if_cuda_error(cub::DeviceScan::InclusiveSum(temp_storage, temp_bytes, values, values, n),
                      context);
  scratch.ensure(temp_bytes, context);
  throw_if_cuda_error(cub::DeviceScan::InclusiveSum(scratch.data, temp_bytes, values, values, n),
                      context);
}

double sum_device_double(const double* values, std::int32_t n, const char* context) {
  if (n <= 0) {
    return 0.0;
  }
  DeviceScratchBuffer& scratch = singleton_columns_reduce_scratch();
  DeviceScratchBuffer& result_scratch = singleton_columns_reduce_result_scratch();
  void* temp_storage = nullptr;
  std::size_t temp_bytes = 0;
  double result = 0.0;
  result_scratch.ensure(sizeof(double), context);
  double* result_device = static_cast<double*>(result_scratch.data);
  throw_if_cuda_error(cub::DeviceReduce::Sum(temp_storage, temp_bytes, values, result_device, n),
                      context);
  scratch.ensure(temp_bytes, context);
  throw_if_cuda_error(cub::DeviceReduce::Sum(scratch.data, temp_bytes, values, result_device, n),
                      context);
  throw_if_cuda_error(cudaMemcpy(&result, result_device, sizeof(double), cudaMemcpyDeviceToHost),
                      context);
  return result;
}

// Both singleton recovery formats are constructed on the GPU. Resident plans
// retain the segment; only the explicit legacy backend downloads its arrays.
void allocate_singleton_tape_segment(PostsolveTapeGpu& segment, NnzOffset records,
                                     NnzOffset indices, NnzOffset values,
                                     const char* context) {
  const auto allocate = [&](auto** pointer, NnzOffset count) {
    throw_if_cuda_error(singleton_columns_cuda_malloc(
        pointer, checked_nnz_bytes(count, sizeof(**pointer))), context);
  };
  allocate(&segment.types, records);
  allocate(&segment.dual_modes, records);
  allocate(&segment.index_starts, records + 1);
  allocate(&segment.value_starts, records + 1);
  allocate(&segment.indices, indices);
  allocate(&segment.vals, values);
  throw_if_cuda_error(cudaMemsetAsync(segment.index_starts, 0, sizeof(NnzOffset), nullptr),
                      context);
  throw_if_cuda_error(cudaMemsetAsync(segment.value_starts, 0, sizeof(NnzOffset), nullptr),
                      context);
  segment.record_count = segment.record_capacity = records;
  segment.index_count = segment.index_capacity = indices;
  segment.value_count = segment.value_capacity = values;
}

void append_singleton_tape_segment(PostsolveTape &tape, PostsolveTapeGpu &segment,
                                   const char *context) {
  if (tape.resident) {
    append_device_recovery_segment(tape, std::move(segment));
    return;
  }
  const std::size_t old_records = tape.types.size();
  const std::size_t old_indices = tape.resident ? 0 : tape.indices.size();
  const std::size_t old_values = tape.resident ? 0 : tape.vals.size();
  const NnzOffset records = segment.record_count;
  tape.types.resize(old_records + records);
  tape.dual_modes.resize(old_records + records);
  tape.index_starts.resize(old_records + records + 1);
  tape.value_starts.resize(old_records + records + 1);
  tape.indices.resize(old_indices + segment.index_count);
  tape.vals.resize(old_values + segment.value_count);
  const auto download = [&](auto* destination, const auto* source, NnzOffset count) {
    throw_if_cuda_error(cudaMemcpy(destination, source,
        checked_nnz_bytes(count, sizeof(*source)), cudaMemcpyDeviceToHost), context);
  };
  download(tape.types.data() + old_records, segment.types, records);
  download(tape.dual_modes.data() + old_records, segment.dual_modes, records);
  download(tape.index_starts.data() + old_records + 1, segment.index_starts + 1, records);
  download(tape.value_starts.data() + old_records + 1, segment.value_starts + 1, records);
  download(tape.indices.data() + old_indices, segment.indices, segment.index_count);
  download(tape.vals.data() + old_values, segment.vals, segment.value_count);
}

__global__ void _kernel_singleton_eq_to_ineq_counts(
    std::int32_t* selected_scan, const std::uint8_t* lhs_change,
    const std::uint8_t* rhs_change, std::int32_t m) {
  const NnzOffset row = NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < m) selected_scan[row] = lhs_change[row] != 0 || rhs_change[row] != 0;
}

__global__ void _kernel_singleton_pack_eq_to_ineq_tape(
    std::int32_t* types, std::uint8_t* dual_modes,
    NnzOffset* index_starts, NnzOffset* value_starts,
    std::int32_t* indices, double* vals, const std::int32_t* selected_scan,
    NnzOffset index_base, NnzOffset value_base, std::int32_t m) {
  const NnzOffset row = NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= m || selected_scan[row] == (row == 0 ? 0 : selected_scan[row - 1])) return;
  const NnzOffset record = selected_scan[row] - 1;
  types[record] = static_cast<std::int32_t>(PostsolveReductionType::EqToIneq);
  dual_modes[record] = static_cast<std::uint8_t>(PostsolveDualMode::Minimal);
  index_starts[record + 1] = index_base + record + 1;
  value_starts[record + 1] = value_base + record + 1;
  indices[record] = static_cast<std::int32_t>(row);
  vals[record] = 0.0;
}

void append_eq_to_ineq_tape_from_device(PostsolveTape& tape,
                                        const std::uint8_t* row_lhs_change,
                                        const std::uint8_t* row_rhs_change,
                                        std::int32_t m,
                                        const char* context) {
  if (m <= 0) return;
  DeviceScratchBuffer scan_storage;
  scan_storage.ensure(checked_nnz_bytes(m, sizeof(std::int32_t)), context);
  auto* selected_scan = static_cast<std::int32_t*>(scan_storage.data);
  const int blocks = static_cast<int>((NnzOffset(m) + GPU_PRESOLVE_THREADS - 1) /
                                      GPU_PRESOLVE_THREADS);
  _kernel_singleton_eq_to_ineq_counts<<<blocks, GPU_PRESOLVE_THREADS>>>(
      selected_scan, row_lhs_change, row_rhs_change, m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_singleton_eq_to_ineq_counts");
  inclusive_scan_i32(selected_scan, m, context);
  std::int32_t records = 0;
  throw_if_cuda_error(cudaMemcpy(&records, selected_scan + m - 1,
                                 sizeof(records), cudaMemcpyDeviceToHost), context);
  if (records == 0) return;
  PostsolveTapeGpu packed;
  allocate_singleton_tape_segment(packed, records, records, records, context);
  _kernel_singleton_pack_eq_to_ineq_tape<<<blocks, GPU_PRESOLVE_THREADS>>>(
      packed.types, packed.dual_modes, packed.index_starts, packed.value_starts, packed.indices,
      packed.vals, selected_scan, static_cast<NnzOffset>(tape.resident ? 0 : tape.indices.size()),
      static_cast<NnzOffset>(tape.resident ? 0 : tape.vals.size()), m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_singleton_pack_eq_to_ineq_tape");
  append_singleton_tape_segment(tape, packed, context);
}

__global__ void _kernel_fill_i32(std::int32_t* values, std::int32_t value, std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < n) {
    values[i] = value;
  }
}

__device__ bool _singleton_column_direct_unbounded(double cj,
                                                double a,
                                                double lhs,
                                                double rhs,
                                                double lb,
                                                double ub,
                                                double zero_tol) {
  return ((cj > zero_tol && a > zero_tol && !isfinite(lhs) && !isfinite(lb)) ||
          (cj > zero_tol && a < -zero_tol && !isfinite(rhs) && !isfinite(lb)) ||
          (cj < -zero_tol && a < -zero_tol && !isfinite(lhs) && !isfinite(ub)) ||
          (cj < -zero_tol && a > zero_tol && !isfinite(rhs) && !isfinite(ub)));
}

__device__ bool _singleton_column_eq_free_from_above(double implied_ub, double ub, double tol) {
  return !isfinite(ub) || implied_ub <= ub + tol;
}

__device__ bool _singleton_column_eq_free_from_below(double implied_lb, double lb, double tol) {
  return !isfinite(lb) || implied_lb >= lb - tol;
}

__device__ void _apply_singleton_column_eq_one_sided_row_update(double* AL,
                                                             double* AU,
                                                             std::int32_t row,
                                                             double a,
                                                             double bound_val,
                                                             bool keep_lower_part) {
  const double shifted_rhs = AU[row] - a * bound_val;
  if (keep_lower_part) {
    AL[row] = shifted_rhs;
    AU[row] = INFINITY;
  } else {
    AL[row] = -INFINITY;
    AU[row] = shifted_rhs;
  }
}

__global__ void _kernel_singleton_row_activity_cache(
    double* finite_min,
    double* finite_max,
    std::int32_t* min_neg_inf,
    std::int32_t* min_pos_inf,
    std::int32_t* max_neg_inf,
    std::int32_t* max_pos_inf,
    std::uint8_t* invalid,
    const std::uint8_t* activity_row_needed,
    const std::uint8_t* keep_row,
    const std::uint8_t* keep_col,
    const double* l,
    const double* u,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    std::int32_t* long_rows,
    std::int32_t* long_row_count,
    std::int32_t m, const std::int32_t* live_row_nnz) {
  const std::int32_t row =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }
  if (activity_row_needed[row] == std::uint8_t{0}) {
    return;
  }

  double row_finite_min = 0.0;
  double row_finite_max = 0.0;
  std::int32_t row_min_neg_inf = 0;
  std::int32_t row_min_pos_inf = 0;
  std::int32_t row_max_neg_inf = 0;
  std::int32_t row_max_pos_inf = 0;
  bool row_invalid = false;
  if ((live_row_nnz ? live_row_nnz[row] : row_ptr[row + 1] - row_ptr[row]) > SINGLETON_ACTIVITY_LONG_ROW_THRESHOLD) {
    const std::int32_t slot = atomicAdd(long_row_count, 1);
    long_rows[slot] = row;
    return;
  }
  if (keep_row[row] != std::uint8_t{0}) {
    for (NnzOffset p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
      const std::int32_t col = col_val[p];
      if (keep_col[col] == std::uint8_t{0}) {
        continue;
      }
      const double a = nz_val[p];
      const double term_min = a >= 0.0 ? a * l[col] : a * u[col];
      const double term_max = a >= 0.0 ? a * u[col] : a * l[col];
      if (isnan(term_min) || isnan(term_max)) {
        row_invalid = true;
        continue;
      }
      if (isinf(term_min)) {
        if (term_min < 0.0) {
          ++row_min_neg_inf;
        } else {
          ++row_min_pos_inf;
        }
      } else {
        row_finite_min += term_min;
      }
      if (isinf(term_max)) {
        if (term_max < 0.0) {
          ++row_max_neg_inf;
        } else {
          ++row_max_pos_inf;
        }
      } else {
        row_finite_max += term_max;
      }
    }
  }
  finite_min[row] = row_finite_min;
  finite_max[row] = row_finite_max;
  min_neg_inf[row] = row_min_neg_inf;
  min_pos_inf[row] = row_min_pos_inf;
  max_neg_inf[row] = row_max_neg_inf;
  max_pos_inf[row] = row_max_pos_inf;
  invalid[row] = row_invalid ? std::uint8_t{1} : std::uint8_t{0};
}

__global__ void _kernel_mark_singleton_activity_rows(
    std::uint8_t* activity_row_needed,
    const std::uint8_t* keep_row,
    const std::uint8_t* keep_col,
    const std::uint8_t* singleton_column_mask,
    const std::int32_t* singleton_column_row,
    std::int32_t m,
    std::int32_t n) {
  const std::int32_t col =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (col >= n || keep_col[col] == std::uint8_t{0} ||
      singleton_column_mask[col] == std::uint8_t{0}) {
    return;
  }
  const std::int32_t row = singleton_column_row[col];
  if (row >= 0 && row < m && keep_row[row] != std::uint8_t{0}) {
    activity_row_needed[row] = std::uint8_t{1};
  }
}

__global__ void _kernel_singleton_long_row_activity_cache(
    double* finite_min,
    double* finite_max,
    std::int32_t* min_neg_inf,
    std::int32_t* min_pos_inf,
    std::int32_t* max_neg_inf,
    std::int32_t* max_pos_inf,
    std::uint8_t* invalid,
    const std::uint8_t* keep_row,
    const std::uint8_t* keep_col,
    const double* l,
    const double* u,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    const std::int32_t* long_rows,
    std::int32_t long_row_count, const int* live_degrees) {
  const std::int32_t long_index = static_cast<std::int32_t>(blockIdx.x);
  if (long_index >= long_row_count) {
    return;
  }
  const std::int32_t row = long_rows[long_index];
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
  __shared__ double shared_finite_min[GPU_PRESOLVE_THREADS];
  __shared__ double shared_finite_max[GPU_PRESOLVE_THREADS];
  __shared__ std::int32_t shared_min_neg_inf[GPU_PRESOLVE_THREADS];
  __shared__ std::int32_t shared_min_pos_inf[GPU_PRESOLVE_THREADS];
  __shared__ std::int32_t shared_max_neg_inf[GPU_PRESOLVE_THREADS];
  __shared__ std::int32_t shared_max_pos_inf[GPU_PRESOLVE_THREADS];
  __shared__ std::int32_t shared_invalid[GPU_PRESOLVE_THREADS];

  __shared__ LiveRowStorage<GPU_PRESOLVE_THREADS> live_storage;
  double thread_finite_min = 0.0;
  double thread_finite_max = 0.0;
  std::int32_t thread_min_neg_inf = 0;
  std::int32_t thread_min_pos_inf = 0;
  std::int32_t thread_max_neg_inf = 0;
  std::int32_t thread_max_pos_inf = 0;
  std::int32_t thread_invalid = 0;
  if (keep_row[row] != std::uint8_t{0}) {
    const auto* mask = live_degrees && live_degrees[row] != row_ptr[row+1]-row_ptr[row]
        ? keep_col : nullptr;
    for_each_live_entry<GPU_PRESOLVE_THREADS>(row_ptr[row], row_ptr[row+1], col_val,
                                             mask, live_storage, [&](NnzOffset p) {
      const std::int32_t col = col_val[p];
      if (keep_col[col] == std::uint8_t{0}) {
        return;
      }
      const double a = nz_val[p];
      const double term_min = a >= 0.0 ? a * l[col] : a * u[col];
      const double term_max = a >= 0.0 ? a * u[col] : a * l[col];
      if (isnan(term_min) || isnan(term_max)) {
        thread_invalid = 1;
        return;
      }
      if (isinf(term_min)) {
        term_min < 0.0 ? ++thread_min_neg_inf : ++thread_min_pos_inf;
      } else {
        thread_finite_min += term_min;
      }
      if (isinf(term_max)) {
        term_max < 0.0 ? ++thread_max_neg_inf : ++thread_max_pos_inf;
      } else {
        thread_finite_max += term_max;
      }
    });
  }
  shared_finite_min[tid] = thread_finite_min;
  shared_finite_max[tid] = thread_finite_max;
  shared_min_neg_inf[tid] = thread_min_neg_inf;
  shared_min_pos_inf[tid] = thread_min_pos_inf;
  shared_max_neg_inf[tid] = thread_max_neg_inf;
  shared_max_pos_inf[tid] = thread_max_pos_inf;
  shared_invalid[tid] = thread_invalid;
  __syncthreads();

  for (NnzOffset stride = GPU_PRESOLVE_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_finite_min[tid] += shared_finite_min[tid + stride];
      shared_finite_max[tid] += shared_finite_max[tid + stride];
      shared_min_neg_inf[tid] += shared_min_neg_inf[tid + stride];
      shared_min_pos_inf[tid] += shared_min_pos_inf[tid + stride];
      shared_max_neg_inf[tid] += shared_max_neg_inf[tid + stride];
      shared_max_pos_inf[tid] += shared_max_pos_inf[tid + stride];
      shared_invalid[tid] |= shared_invalid[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    finite_min[row] = shared_finite_min[0];
    finite_max[row] = shared_finite_max[0];
    min_neg_inf[row] = shared_min_neg_inf[0];
    min_pos_inf[row] = shared_min_pos_inf[0];
    max_neg_inf[row] = shared_max_neg_inf[0];
    max_pos_inf[row] = shared_max_pos_inf[0];
    invalid[row] = shared_invalid[0] != 0 ? std::uint8_t{1} : std::uint8_t{0};
  }
}

__device__ bool _singleton_column_activity_bounds_from_cache(
    std::int32_t row,
    std::int32_t excluded_col,
    double excluded_val,
    const double* l,
    const double* u,
    const double* finite_min,
    const double* finite_max,
    const std::int32_t* min_neg_inf,
    const std::int32_t* min_pos_inf,
    const std::int32_t* max_neg_inf,
    const std::int32_t* max_pos_inf,
    const std::uint8_t* invalid,
    double* rest_min,
    double* rest_max) {
  if (invalid[row] != std::uint8_t{0}) {
    return false;
  }

  const double term_min =
      excluded_val >= 0.0 ? excluded_val * l[excluded_col]
                          : excluded_val * u[excluded_col];
  const double term_max =
      excluded_val >= 0.0 ? excluded_val * u[excluded_col]
                          : excluded_val * l[excluded_col];
  if (isnan(term_min) || isnan(term_max)) {
    return false;
  }

  std::int32_t min_neg = min_neg_inf[row];
  std::int32_t min_pos = min_pos_inf[row];
  std::int32_t max_neg = max_neg_inf[row];
  std::int32_t max_pos = max_pos_inf[row];
  double min_value = finite_min[row];
  double max_value = finite_max[row];
  if (isinf(term_min)) {
    if (term_min < 0.0) {
      --min_neg;
    } else {
      --min_pos;
    }
  } else {
    min_value -= term_min;
  }
  if (isinf(term_max)) {
    if (term_max < 0.0) {
      --max_neg;
    } else {
      --max_pos;
    }
  } else {
    max_value -= term_max;
  }
  if (min_neg < 0 || min_pos < 0 || max_neg < 0 || max_pos < 0 ||
      (min_neg > 0 && min_pos > 0) || (max_neg > 0 && max_pos > 0)) {
    return false;
  }
  *rest_min = min_neg > 0 ? -INFINITY : (min_pos > 0 ? INFINITY : min_value);
  *rest_max = max_neg > 0 ? -INFINITY : (max_pos > 0 ? INFINITY : max_value);
  return !isnan(*rest_min) && !isnan(*rest_max);
}

__device__ bool _singleton_column_implied_free_from_above(double a,
                                                       double lhs,
                                                       double rhs,
                                                       double ub,
                                                       double rest_min,
                                                       double rest_max,
                                                       double tol) {
  if (!isfinite(ub)) {
    return true;
  }
  double implied_ub = INFINITY;
  if (a > 0.0 && isfinite(rhs)) {
    implied_ub = (rhs - rest_min) / a;
  } else if (a < 0.0 && isfinite(lhs)) {
    implied_ub = (lhs - rest_max) / a;
  }
  return implied_ub <= ub + tol;
}

__device__ bool _singleton_column_implied_free_from_below(double a,
                                                       double lhs,
                                                       double rhs,
                                                       double lb,
                                                       double rest_min,
                                                       double rest_max,
                                                       double tol) {
  if (!isfinite(lb)) {
    return true;
  }
  double implied_lb = -INFINITY;
  if (a > 0.0 && isfinite(lhs)) {
    implied_lb = (lhs - rest_max) / a;
  } else if (a < 0.0 && isfinite(rhs)) {
    implied_lb = (rhs - rest_min) / a;
  }
  return implied_lb >= lb - tol;
}

__device__ double _singleton_column_active_side(double cj,
                                             double a,
                                             double lhs,
                                             double rhs,
                                             double zero_tol) {
  if ((cj > zero_tol && a > 0.0) || (cj < -zero_tol && a < 0.0)) {
    return lhs;
  }
  if ((cj > zero_tol && a < 0.0) || (cj < -zero_tol && a > 0.0)) {
    return rhs;
  }
  return isfinite(lhs) ? lhs : rhs;
}

__device__ std::uint8_t _singleton_column_ineq_action(double cj,
                                                   double a,
                                                   double lhs,
                                                   double rhs,
                                                   bool impl_free_from_above,
                                                   bool impl_free_from_below,
                                                   double zero_tol,
                                                   double* action_side) {
  if (impl_free_from_above && impl_free_from_below) {
    *action_side = _singleton_column_active_side(cj, a, lhs, rhs, zero_tol);
    if (isfinite(*action_side)) {
      return SINGLETON_COL_INEQ_ELIMINATE;
    }
    return SINGLETON_COL_INEQ_NO_ACTION;
  }
  const bool tighten_lhs_to_rhs =
      ((cj < -zero_tol && a > 0.0 && impl_free_from_above) ||
       (cj > zero_tol && a < 0.0 && impl_free_from_below)) &&
      isfinite(rhs);
  if (tighten_lhs_to_rhs) {
    *action_side = rhs;
    return SINGLETON_COL_INEQ_TIGHTEN_LHS_TO_RHS;
  }
  const bool tighten_rhs_to_lhs =
      ((cj > zero_tol && a > 0.0 && impl_free_from_below) ||
       (cj < -zero_tol && a < 0.0 && impl_free_from_above)) &&
      isfinite(lhs);
  if (tighten_rhs_to_lhs) {
    *action_side = lhs;
    return SINGLETON_COL_INEQ_TIGHTEN_RHS_TO_LHS;
  }
  *action_side = 0.0;
  return SINGLETON_COL_INEQ_NO_ACTION;
}

__global__ void _kernel_propose_singleton_column_actions(std::int32_t* status_flag,
                                                SingletonAction* actions,
                                                double* chosen_side,
                                                std::int32_t* row_owner,
                                                const std::uint8_t* keep_row,
                                                const std::uint8_t* keep_col,
                                                const std::uint8_t* singleton_mask,
                                                const std::int32_t* support_row,
                                                const double* support_val,
                                                const double* c_cur,
                                                const double* l_cur,
                                                const double* u_cur,
                                                const double* AL_cur,
                                                const double* AU_cur,
                                                const double* activity_finite_min,
                                                const double* activity_finite_max,
                                                const std::int32_t* activity_min_neg_inf,
                                                const std::int32_t* activity_min_pos_inf,
                                                const std::int32_t* activity_max_neg_inf,
                                                const std::int32_t* activity_max_pos_inf,
                                                const std::uint8_t* activity_invalid,
                                                const NnzOffset* row_ptr,
                                                const std::int32_t* col_val,
                                                const double* nz_val,
                                                double tol,
                                                double zero_tol,
                                                std::int32_t m,
                                                std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j < n) {
    if (keep_col[j] == std::uint8_t{0} || singleton_mask[j] == std::uint8_t{0}) {
      return;
    }
    const std::int32_t row = support_row[j];
    if (row < 0 || row >= m || keep_row[row] == std::uint8_t{0}) {
      return;
    }
    const double a = support_val[j];
    if (fabs(a) <= zero_tol) {
      return;
    }

    const NnzOffset row_start = row_ptr[row];
    const NnzOffset row_stop = row_ptr[row + 1];
    std::int32_t live_row_nnz = 0;
    for (NnzOffset p = row_start; p < row_stop; ++p) {
      const std::int32_t col = col_val[p];
      if (keep_col[col] != std::uint8_t{0}) {
        ++live_row_nnz;
        if (live_row_nnz > 1) {
          break;
        }
      }
    }
    if (live_row_nnz <= 1) {
      return;
    }

    const double lhs = AL_cur[row];
    const double rhs = AU_cur[row];
    const double cj = c_cur[j];
    if (_singleton_column_direct_unbounded(cj, a, lhs, rhs, l_cur[j], u_cur[j], zero_tol)) {
      atomicMax(&status_flag[0], 1);
      return;
    }

    double rest_min = 0.0;
    double rest_max = 0.0;
    if (!_singleton_column_activity_bounds_from_cache(
            row, j, a, l_cur, u_cur, activity_finite_min, activity_finite_max,
            activity_min_neg_inf, activity_min_pos_inf, activity_max_neg_inf,
            activity_max_pos_inf, activity_invalid, &rest_min, &rest_max)) {
      return;
    }

    const bool is_eq_row = isfinite(lhs) && isfinite(rhs) && fabs(lhs - rhs) <= tol;
    if (is_eq_row) {
      const double x1 = (rhs - rest_min) / a;
      const double x2 = (rhs - rest_max) / a;
      const double implied_lb = fmin(x1, x2);
      const double implied_ub = fmax(x1, x2);
      const bool impl_free_from_above = _singleton_column_eq_free_from_above(implied_ub, u_cur[j], tol);
      const bool impl_free_from_below = _singleton_column_eq_free_from_below(implied_lb, l_cur[j], tol);
      if (!(impl_free_from_above || impl_free_from_below)) {
        return;
      }
      actions[j] = impl_free_from_above && impl_free_from_below
          ? SingletonAction::EliminateFree
          : (impl_free_from_above ? SingletonAction::EliminateFreeAbove
                                 : SingletonAction::EliminateFreeBelow);
      chosen_side[j] = rhs;
    } else {
      const bool impl_free_from_above =
          _singleton_column_implied_free_from_above(a, lhs, rhs, u_cur[j], rest_min, rest_max, tol);
      const bool impl_free_from_below =
          _singleton_column_implied_free_from_below(a, lhs, rhs, l_cur[j], rest_min, rest_max, tol);
      double action_side = 0.0;
      const auto action = _singleton_column_ineq_action(
          cj, a, lhs, rhs, impl_free_from_above, impl_free_from_below, zero_tol, &action_side);
      if (action == SINGLETON_COL_INEQ_NO_ACTION) {
        return;
      }
      actions[j] = static_cast<SingletonAction>(action);
      chosen_side[j] = action_side;
    }
    // A one-sided elimination can let dual inference recreate the equality,
    // producing a long chain of full-support recovery records. Prefer a free
    // pivot that removes the row. Negative keys rank those pivots first without
    // extra workspace; staging decodes the winning column after this kernel.
    const int key = actions[j] == SingletonAction::EliminateFree ? j - INT_MAX : j;
    atomicMin(&row_owner[row], key);
  }
}

// Stage only selected rows. All reads are from the unmodified input plan;
// recovery packing and objective gathering complete before the common commit.
__global__ void _kernel_stage_singleton_column_actions(
    std::int32_t* status_flag,
    std::uint8_t* row_delete, std::uint8_t* row_lhs_change, std::uint8_t* row_rhs_change,
    std::uint8_t* col_delete, double* AL_new, double* AU_new, double* obj_contrib,
    std::int32_t* row_owner, const SingletonAction* actions, const double* chosen_side,
    const double* support_val, const double* c_cur, const double* l_cur, const double* u_cur,
    const double* AL_cur, const double* AU_cur, double tol, std::int32_t m, std::int32_t n) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  // An unbounded candidate suppresses the whole batch, including every owner.
  // status_flag[0] is immutable during this kernel.
  if (row >= m || status_flag[0] != 0) return;
  const std::int32_t j = row_owner[row] < 0 ? row_owner[row] + INT_MAX : row_owner[row];
  if (j < 0 || j >= n) return;
  row_owner[row] = j;

  const SingletonAction action = actions[j];
  const double lhs = AL_cur[row];
  const double rhs = AU_cur[row];
  AL_new[row] = lhs;
  AU_new[row] = rhs;
  if (action == SingletonAction::EliminateFree ||
      action == SingletonAction::EliminateFreeAbove ||
      action == SingletonAction::EliminateFreeBelow) {
    const double a = support_val[j];
    // Keep the original multiply/divide and boundary-shift expressions.
    obj_contrib[j] = c_cur[j] * rhs / a;
    if (action == SingletonAction::EliminateFree) {
      row_delete[row] = std::uint8_t{1};
    } else if (action == SingletonAction::EliminateFreeAbove) {
      _apply_singleton_column_eq_one_sided_row_update(AL_new, AU_new, row, a, l_cur[j], a < 0.0);
    } else {
      _apply_singleton_column_eq_one_sided_row_update(AL_new, AU_new, row, a, u_cur[j], a > 0.0);
    }
    col_delete[j] = std::uint8_t{1};
    atomicMax(&status_flag[1], 1);
    return;
  }

  if (action == SingletonAction::SetRowSide) {
    const double side = chosen_side[j];
    if (!isfinite(lhs) || fabs(lhs - side) > tol) row_lhs_change[row] = std::uint8_t{1};
    if (!isfinite(rhs) || fabs(rhs - side) > tol) row_rhs_change[row] = std::uint8_t{1};
    AL_new[row] = side;
    AU_new[row] = side;
  } else if (action == SingletonAction::TightenLhs) {
    row_lhs_change[row] = std::uint8_t{1};
    AL_new[row] = rhs;
  } else if (action == SingletonAction::TightenRhs) {
    row_rhs_change[row] = std::uint8_t{1};
    AU_new[row] = lhs;
  }
  atomicMax(&status_flag[2], 1);
}

__global__ void _kernel_apply_singleton_objective_updates_deterministic(
    double* c_new,
    const std::uint8_t* keep_col,
    const std::uint8_t* col_delete,
    const std::int32_t* row_owner,
    const double* support_val,
    const double* c_cur,
    const NnzOffset* at_row_ptr,
    const std::int32_t* at_col_val,
    const double* at_nz_val,
    std::int32_t n, int max_length) {
  const std::int32_t col =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (col >= n || (max_length && at_row_ptr[col+1]-at_row_ptr[col]>max_length) || keep_col[col] == std::uint8_t{0} ||
      col_delete[col] != std::uint8_t{0}) {
    return;
  }

  double updated = c_cur[col];
  for (NnzOffset p = at_row_ptr[col]; p < at_row_ptr[col + 1]; ++p) {
    const std::int32_t row = at_col_val[p];
    const std::int32_t owner = row_owner[row];
    if (owner < 0 || owner >= n || col_delete[owner] == std::uint8_t{0}) {
      continue;
    }
    updated += -(c_cur[owner] * at_nz_val[p] / support_val[owner]);
  }
  c_new[col] = updated;
}

template<int Width>
__global__ void grouped_singleton_objective_updates(
    double* output,const std::uint8_t* keep,const std::uint8_t* deleted,const int* owner,
    const double* support,const double* cost,const NnzOffset* ptr,const int* indices,
    const double* values,int n,const int* columns,const int* count) {
  const int lane=Width==32 ? threadIdx.x&31 : threadIdx.x;
  const int group=Width==32 ? blockIdx.x*8+threadIdx.x/32 : blockIdx.x;
  const NnzOffset stride=Width==32 ? gridDim.x*8 : gridDim.x;
  const NnzOffset base=Width==32 ? (threadIdx.x/32)*32 : 0;
  __shared__ double contribution[256];
  __shared__ unsigned char valid[256];
  for(NnzOffset q=group;q<*count;q+=stride) {
    const int col=columns[q];
    if(!keep[col] || deleted[col]) continue;
    double updated=lane==0 ? cost[col] : 0;
    for(NnzOffset chunk=ptr[col];chunk<ptr[col+1];chunk+=Width) {
      const NnzOffset p=chunk+lane;
      valid[threadIdx.x]=0;
      if(p<ptr[col+1]) {
        const int selected=owner[indices[p]];
        if(selected>=0 && selected<n && deleted[selected]) {
          valid[threadIdx.x]=1;
          contribution[threadIdx.x]=-(cost[selected]*values[p]/support[selected]);
        }
      }
      if constexpr(Width==32) __syncwarp(); else __syncthreads();
      // Preserve every addition in CSR order; parallelize only the gather/divide.
      if(lane==0) for(int j=0;j<Width && chunk+j<ptr[col+1];++j)
        if(valid[base+j]) updated+=contribution[base+j];
      if constexpr(Width==32) __syncwarp(); else __syncthreads();
    }
    if(lane==0) output[col]=updated;
    if constexpr(Width!=32) __syncthreads();
  }
}

__global__ void _kernel_count_singleton_column_support(
    std::int32_t* counts,
    const std::uint8_t* keep_row,
    const std::uint8_t* keep_col,
    const std::uint8_t* singleton_mask,
    const std::int32_t* support_row,
    std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (j < n && keep_col[j] != std::uint8_t{0} && singleton_mask[j] != std::uint8_t{0}) {
    const std::int32_t row = support_row[j];
    if (row >= 0 && keep_row[row] != std::uint8_t{0}) {
      atomicAdd(&counts[0], 1);
    }
  }
}

__global__ void _kernel_commit_singleton_column_actions(std::uint8_t* keep_row,
                                             std::uint8_t* keep_col,
                                             double* c_cur,
                                             double* AL_cur,
                                             double* AU_cur,
                                             const std::uint8_t* row_delete,
                                             const std::uint8_t* col_delete,
                                             const double* c_new,
                                             const double* AL_new,
                                             const double* AU_new,
                                             const std::int32_t* row_owner,
                                             bool update_cost,
                                             std::int32_t m,
                                             std::int32_t n) {
  const NnzOffset q = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (q < m) {
    if (row_delete[q] != std::uint8_t{0}) {
      keep_row[q] = std::uint8_t{0};
    }
    if (row_owner[q] >= 0 && row_owner[q] < n) {
      AL_cur[q] = AL_new[q];
      AU_cur[q] = AU_new[q];
    }
  }
  if (q < n) {
    if (col_delete[q] != std::uint8_t{0}) {
      keep_col[q] = std::uint8_t{0};
    }
    if (update_cost && keep_col[q] != std::uint8_t{0}) {
      c_cur[q] = c_new[q];
    }
  }
}

__global__ void _kernel_singleton_subcol_tape_counts(std::int32_t* selected_scan,
    NnzOffset* support_scan,
    const std::uint8_t* col_delete,
    const std::int32_t* row_owner,
    const std::uint8_t* keep_col,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    double zero_tol,
    std::int32_t* long_rows,
    std::int32_t* long_row_count,
    std::int32_t m,
    std::int32_t n) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row < m) {
    const std::int32_t col = row_owner[row];
    const bool selected = col >= 0 && col < n &&
                          col_delete[col] != std::uint8_t{0};
    selected_scan[row] = selected ? 1 : 0;
    std::int32_t support_count = 0;
    double pivot = 0.0;
    if (selected) {
      if (row_ptr[row + 1] - row_ptr[row] > SINGLETON_ACTIVITY_LONG_ROW_THRESHOLD) {
        const std::int32_t slot = atomicAdd(long_row_count, 1);
        long_rows[slot] = row;
        support_scan[row] = 0;
        return;
      }
      for (NnzOffset p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
        const std::int32_t row_col = col_val[p];
        if (row_col == col) pivot = nz_val[p];
        if (row_col != col && keep_col[row_col] != std::uint8_t{0}) {
          ++support_count;
        }
      }
    }
    // Match the old host-side record filter before scanning output positions.
    if (std::fabs(pivot) <= zero_tol) {
      selected_scan[row] = 0;
      support_count = 0;
    }
    support_scan[row] = support_count;
  }
}

__global__ void _kernel_singleton_subcol_long_tape_counts(
    std::int32_t* selected_scan,
    NnzOffset* support_scan,
    const std::uint8_t* keep_col,
    const std::int32_t* row_owner,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    double zero_tol,
    const std::int32_t* long_rows,
    std::int32_t long_row_count) {
  const std::int32_t long_index = static_cast<std::int32_t>(blockIdx.x);
  if (long_index >= long_row_count) return;
  const std::int32_t row = long_rows[long_index];
  const std::int32_t col = row_owner[row];
  NnzOffset count = 0;
  NnzOffset pivot_pos = -1;
  for (NnzOffset p = row_ptr[row] + static_cast<NnzOffset>(threadIdx.x);
       p < row_ptr[row + 1]; p += static_cast<std::int32_t>(blockDim.x)) {
    const std::int32_t row_col = col_val[p];
    if (row_col == col) pivot_pos = p;
    count += row_col != col && keep_col[row_col] != std::uint8_t{0} ? 1 : 0;
  }
  __shared__ NnzOffset counts[GPU_PRESOLVE_THREADS];
  __shared__ NnzOffset pivot_positions[GPU_PRESOLVE_THREADS];
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
  counts[tid] = count;
  pivot_positions[tid] = pivot_pos;
  __syncthreads();
  for (NnzOffset stride = GPU_PRESOLVE_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      counts[tid] += counts[tid + stride];
      pivot_positions[tid] = pivot_positions[tid] > pivot_positions[tid + stride]
          ? pivot_positions[tid] : pivot_positions[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const double pivot = pivot_positions[0] < 0 ? 0.0 : nz_val[pivot_positions[0]];
    const bool valid = !(std::fabs(pivot) <= zero_tol);
    selected_scan[row] = valid ? 1 : 0;
    support_scan[row] = valid ? counts[0] : 0;
  }
}

// Final SubCol layout: indices = [col, row, support_count, support columns...],
// vals = [pivot, side, lower, upper, cost, row_deleted, support coefficients...].
// Prefix sums preserve ascending row order and the original CSR support order.
struct SubcolTapeOutput {
  std::int32_t* types;
  NnzOffset* index_starts;
  NnzOffset* value_starts;
  std::uint8_t* dual_modes;
  std::int32_t* indices;
  double* vals;
  NnzOffset index_base;
  NnzOffset value_base;
};

__device__ void write_subcol_header(SubcolTapeOutput output,
                                    NnzOffset record,
                                    NnzOffset support_start,
                                    NnzOffset support_stop,
                                    std::int32_t col,
                                    std::int32_t row,
                                    double side, double lower, double upper,
                                    double cost, std::uint8_t row_deleted) {
  const NnzOffset i = 3 * record + support_start;
  const NnzOffset v = 6 * record + support_start;
  output.types[record] = static_cast<std::int32_t>(PostsolveReductionType::SubCol);
  output.dual_modes[record] = static_cast<std::uint8_t>(PostsolveDualMode::Minimal);
  output.index_starts[record + 1] = output.index_base + 3 * (record + 1) + support_stop;
  output.value_starts[record + 1] = output.value_base + 6 * (record + 1) + support_stop;
  output.indices[i] = col;
  output.indices[i + 1] = row;
  output.indices[i + 2] = static_cast<std::int32_t>(support_stop - support_start);
  output.vals[v + 1] = side;
  output.vals[v + 2] = lower;
  output.vals[v + 3] = upper;
  output.vals[v + 4] = cost;
  output.vals[v + 5] = row_deleted != std::uint8_t{0} ? 1.0 : 0.0;
}

__global__ void _kernel_singleton_pack_subcol_tape(SubcolTapeOutput output,
    const std::int32_t* selected_scan,
    const NnzOffset* support_scan,
    const std::uint8_t* col_delete,
    const std::uint8_t* row_delete,
    const std::uint8_t* keep_col,
    const std::int32_t* row_owner,
    const double* chosen_side,
    const double* c,
    const double* l,
    const double* u,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    std::int32_t long_row_threshold,
    std::int32_t m,
    std::int32_t n) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }
  const std::int32_t col = row_owner[row];
  if (col < 0 || col >= n || col_delete[col] == std::uint8_t{0}) {
    return;
  }
  if (row_ptr[row + 1] - row_ptr[row] > long_row_threshold) {
    return;
  }
  const std::int32_t prev_selected = row == 0 ? 0 : selected_scan[row - 1];
  const std::int32_t record = selected_scan[row] - 1;
  if (selected_scan[row] == prev_selected || record < 0) {
    return;
  }

  const NnzOffset support_start = row == 0 ? 0 : support_scan[row - 1];
  const NnzOffset support_stop = support_scan[row];
  write_subcol_header(output, record, support_start, support_stop, col, row,
                      chosen_side[col], l[col], u[col], c[col], row_delete[row]);

  double pivot = 0.0;
  NnzOffset out = support_start;
  for (NnzOffset p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
    const std::int32_t row_col = col_val[p];
    const double coeff = nz_val[p];
    if (row_col == col) {
      pivot = coeff;
    } else if (keep_col[row_col] != std::uint8_t{0} && out < support_stop) {
      output.indices[3 * NnzOffset(record) + 3 + out] = row_col;
      output.vals[6 * NnzOffset(record) + 6 + out] = coeff;
      ++out;
    }
  }
  output.vals[6 * NnzOffset(record) + support_start] = pivot;
}

__global__ void _kernel_singleton_pack_long_subcol_tape(SubcolTapeOutput output,
    const std::int32_t* selected_scan,
    const NnzOffset* support_scan,
    const std::uint8_t* row_delete,
    const std::uint8_t* keep_col,
    const std::int32_t* row_owner,
    const double* chosen_side,
    const double* c,
    const double* l,
    const double* u,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    const std::int32_t* long_rows,
    std::int32_t long_row_count) {
  const std::int32_t long_index = static_cast<std::int32_t>(blockIdx.x);
  if (long_index >= long_row_count) return;
  const std::int32_t row = long_rows[long_index];
  const std::int32_t col = row_owner[row];
  if (selected_scan[row] == (row == 0 ? 0 : selected_scan[row - 1])) return;
  const std::int32_t record = selected_scan[row] - 1;
  const NnzOffset support_start = row == 0 ? 0 : support_scan[row - 1];
  const NnzOffset support_stop = support_scan[row];
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
  if (tid == 0) {
    write_subcol_header(output, record, support_start, support_stop, col, row,
                        chosen_side[col], l[col], u[col], c[col], row_delete[row]);
  }
  __shared__ std::int32_t scan[GPU_PRESOLVE_THREADS];
  __shared__ NnzOffset base;
  __shared__ double pivot;
  if (tid == 0) {
    base = 0;
    pivot = 0.0;
  }
  __syncthreads();
  for (NnzOffset chunk = row_ptr[row]; chunk < row_ptr[row + 1];
       chunk += GPU_PRESOLVE_THREADS) {
    const NnzOffset p = chunk + tid;
    std::int32_t row_col = -1;
    double coeff = 0.0;
    if (p < row_ptr[row + 1]) {
      row_col = col_val[p];
      coeff = nz_val[p];
      if (row_col == col) pivot = coeff;
    }
    const bool copy = row_col >= 0 && row_col != col &&
                      keep_col[row_col] != std::uint8_t{0};
    scan[tid] = copy ? 1 : 0;
    __syncthreads();
    for (int offset = 1; offset < GPU_PRESOLVE_THREADS; offset <<= 1) {
      const std::int32_t value = scan[tid] + (tid >= offset ? scan[tid - offset] : 0);
      __syncthreads();
      scan[tid] = value;
      __syncthreads();
    }
    if (copy) {
      const NnzOffset out = support_start + base + scan[tid] - 1;
      if (out < support_stop) {
        output.indices[3 * NnzOffset(record) + 3 + out] = row_col;
        output.vals[6 * NnzOffset(record) + 6 + out] = coeff;
      }
    }
    const std::int32_t chunk_count = scan[GPU_PRESOLVE_THREADS - 1];
    __syncthreads();
    if (tid == 0) base += chunk_count;
    __syncthreads();
  }
  if (tid == 0) output.vals[6 * NnzOffset(record) + support_start] = pivot;
}

void append_subcol_tape_compacted_from_device(PostsolveTape& tape,
                                              const std::uint8_t* col_delete,
                                              const std::uint8_t* row_delete,
                                              const std::uint8_t* keep_col,
                                              const std::int32_t* row_owner,
                                              const double* chosen_side,
                                              const double* c,
                                              const double* l,
                                              const double* u,
                                              const DeviceCsrMatrix& A,
                                              double zero_tol,
                                              const char* context) {
  const std::int32_t m = A.rows;
  const std::int32_t n = A.cols;
  if (m <= 0 || n <= 0) {
    return;
  }

  std::int32_t* selected_scan = nullptr;
  NnzOffset* support_scan = nullptr;
  std::int32_t* long_rows = nullptr;
  std::int32_t* long_row_count = nullptr;
  throw_if_cuda_error(singleton_columns_cuda_malloc(&selected_scan, sizeof(std::int32_t) * static_cast<std::size_t>(m)),
                      context);
  throw_if_cuda_error(singleton_columns_cuda_malloc(&support_scan, sizeof(NnzOffset) * static_cast<std::size_t>(m)),
                      context);
  throw_if_cuda_error(singleton_columns_cuda_malloc(&long_rows, sizeof(std::int32_t) * static_cast<std::size_t>(m)),
                      context);
  throw_if_cuda_error(singleton_columns_cuda_malloc(&long_row_count, sizeof(std::int32_t)), context);
  throw_if_cuda_error(cudaMemset(long_row_count, 0, sizeof(std::int32_t)), context);

  const int blocks_m = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_singleton_subcol_tape_counts<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
      selected_scan,
      support_scan,
      col_delete,
      row_owner,
      keep_col,
      A.rowPtr,
      A.colVal,
      A.nzVal,
      zero_tol,
      long_rows,
      long_row_count,
      m,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_singleton_subcol_tape_counts");
  std::int32_t long_row_count_host = 0;
  throw_if_cuda_error(cudaMemcpy(&long_row_count_host, long_row_count,
                                 sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      context);
  if (long_row_count_host > 0) {
    _kernel_singleton_subcol_long_tape_counts
        <<<long_row_count_host, GPU_PRESOLVE_THREADS>>>(
            selected_scan, support_scan, keep_col, row_owner, A.rowPtr, A.colVal,
            A.nzVal, zero_tol,
            long_rows, long_row_count_host);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_singleton_subcol_long_tape_counts");
  }
  inclusive_scan_i32(selected_scan, m, context);
  inclusive_scan_i32(support_scan, m, context);

  std::int32_t record_count = 0;
  NnzOffset support_nnz = 0;
  throw_if_cuda_error(cudaMemcpy(&record_count, selected_scan + m - 1,
                                 sizeof(record_count), cudaMemcpyDeviceToHost), context);
  throw_if_cuda_error(cudaMemcpy(&support_nnz, support_scan + m - 1,
                                 sizeof(support_nnz), cudaMemcpyDeviceToHost), context);
  if (record_count <= 0) {
    singleton_columns_cuda_free(selected_scan);
    singleton_columns_cuda_free(support_scan);
    singleton_columns_cuda_free(long_rows);
    singleton_columns_cuda_free(long_row_count);
    return;
  }

  const NnzOffset records = record_count;
  const NnzOffset index_count = support_nnz + 3 * records;
  const NnzOffset value_count = support_nnz + 6 * records;
  const std::size_t old_indices = tape.resident ? 0 : tape.indices.size();
  const std::size_t old_values = tape.resident ? 0 : tape.vals.size();

  // Only final tape storage exists on either host or device. In particular, no
  // per-record host vectors or intermediate host support/metadata arrays remain.
  PostsolveTapeGpu packed;
  allocate_singleton_tape_segment(packed, records, index_count, value_count, context);
  const SubcolTapeOutput output{packed.types, packed.index_starts,
      packed.value_starts, packed.dual_modes, packed.indices, packed.vals,
      static_cast<NnzOffset>(old_indices), static_cast<NnzOffset>(old_values)};

  _kernel_singleton_pack_subcol_tape<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
      output, selected_scan, support_scan, col_delete, row_delete, keep_col,
      row_owner, chosen_side, c, l, u, A.rowPtr, A.colVal, A.nzVal,
      SINGLETON_ACTIVITY_LONG_ROW_THRESHOLD, m, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_singleton_pack_subcol_tape");
  if (long_row_count_host > 0) {
    _kernel_singleton_pack_long_subcol_tape
        <<<long_row_count_host, GPU_PRESOLVE_THREADS>>>(
            output, selected_scan, support_scan, row_delete, keep_col, row_owner,
            chosen_side, c, l, u, A.rowPtr, A.colVal, A.nzVal,
            long_rows, long_row_count_host);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_singleton_pack_long_subcol_tape");
  }

  // Transfer the finished segment to the recovery backend in chronological
  // order, before any original bounds, costs, or masks are changed.
  append_singleton_tape_segment(tape, packed, context);

  singleton_columns_cuda_free(selected_scan);
  singleton_columns_cuda_free(support_scan);
  singleton_columns_cuda_free(long_rows);
  singleton_columns_cuda_free(long_row_count);
}

}  // namespace

void apply_rule_singleton_columns(PresolvePlanGpu& plan,
                                    const LPInfoGpu& lp,
                                    const PresolveStatsGpu& stats,
                                    const PresolveParams& pparams,
                                    bool skip_precheck) {
  if (plan.has_infeasible || plan.has_unbounded || !pparams.enable_singleton_columns) {
    return;
  }
  const std::int32_t m = lp.A.rows;
  const std::int32_t n = lp.A.cols;
  if (m == 0 || n == 0) {
    return;
  }

  const int blocks_n = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  if (!skip_precheck) {
    std::int32_t* singleton_counts_device = nullptr;
    throw_if_cuda_error(singleton_columns_cuda_malloc(&singleton_counts_device, sizeof(std::int32_t)),
                        "cudaMalloc singleton_columns counts");
    throw_if_cuda_error(cudaMemset(singleton_counts_device, 0, sizeof(std::int32_t)),
                        "cudaMemset singleton_columns counts");
    _kernel_count_singleton_column_support<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
        singleton_counts_device,
        plan.keep_row_mask,
        plan.keep_col_mask,
        stats.singleton_column_mask,
        stats.singleton_column_row,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_count_singleton_column_support");
    std::int32_t singleton_counts[1] = {0};
    throw_if_cuda_error(cudaMemcpy(singleton_counts,
                                   singleton_counts_device,
                                   sizeof(singleton_counts),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy singleton_columns counts");
    singleton_columns_cuda_free(singleton_counts_device);
    const std::int32_t singleton_count = singleton_counts[0];
    if (singleton_count == 0) {
      return;
    }
  }

  SingletonColumnsWorkspace& workspace = singleton_columns_workspace();
  workspace.ensure(m, n);
  std::int32_t* status_flag = workspace.status_flag;
  std::int32_t* row_owner = workspace.row_owner;
  std::uint8_t* col_delete = workspace.col_delete;
  std::uint8_t* row_delete = workspace.row_delete;
  std::uint8_t* row_lhs_change = workspace.row_lhs_change;
  std::uint8_t* row_rhs_change = workspace.row_rhs_change;
  SingletonAction* actions = workspace.actions;
  double* chosen_side = workspace.chosen_side;
  double* c_new = workspace.c_new;
  double* AL_new = workspace.AL_new;
  double* AU_new = workspace.AU_new;
  double* obj_contrib = workspace.obj_contrib;
  double* activity_finite_min = workspace.activity_finite_min;
  double* activity_finite_max = workspace.activity_finite_max;
  std::int32_t* activity_min_neg_inf = workspace.activity_min_neg_inf;
  std::int32_t* activity_min_pos_inf = workspace.activity_min_pos_inf;
  std::int32_t* activity_max_neg_inf = workspace.activity_max_neg_inf;
  std::int32_t* activity_max_pos_inf = workspace.activity_max_pos_inf;
  std::uint8_t* activity_invalid = workspace.activity_invalid;
  std::uint8_t* activity_row_needed = workspace.activity_row_needed;
  std::int32_t* activity_long_rows = workspace.activity_long_rows;
  std::int32_t* activity_long_row_count = workspace.activity_long_row_count;
  throw_if_cuda_error(cudaMemset(activity_long_row_count, 0, sizeof(std::int32_t)),
                      "cudaMemset singleton_columns activity_long_row_count");
  throw_if_cuda_error(cudaMemset(activity_row_needed, 0, static_cast<std::size_t>(m)),
                      "cudaMemset singleton_columns activity_row_needed");
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t) * 3), "cudaMemset singleton_columns status_flag");
  throw_if_cuda_error(cudaMemset(col_delete, 0, static_cast<std::size_t>(n)), "cudaMemset singleton_columns col_delete");
  throw_if_cuda_error(cudaMemset(row_delete, 0, static_cast<std::size_t>(m)), "cudaMemset singleton_columns row_delete");
  throw_if_cuda_error(cudaMemset(row_lhs_change, 0, static_cast<std::size_t>(m)), "cudaMemset singleton_columns row_lhs_change");
  throw_if_cuda_error(cudaMemset(row_rhs_change, 0, static_cast<std::size_t>(m)), "cudaMemset singleton_columns row_rhs_change");
  throw_if_cuda_error(cudaMemset(obj_contrib, 0, sizeof(double) * static_cast<std::size_t>(n)), "cudaMemset singleton_columns obj_contrib");
  const int blocks_m = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_fill_i32<<<blocks_m, GPU_PRESOLVE_THREADS>>>(row_owner, INT_MAX, m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_fill_i32 singleton row_owner");

  _kernel_mark_singleton_activity_rows<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      activity_row_needed,
      plan.keep_row_mask,
      plan.keep_col_mask,
      stats.singleton_column_mask,
      stats.singleton_column_row,
      m,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_mark_singleton_activity_rows");

  // Compute each row's activity once.  A column singleton can then exclude its
  // own contribution in O(1), avoiding repeated full scans when many singleton
  // columns share one very long row.
  _kernel_singleton_row_activity_cache<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
      activity_finite_min,
      activity_finite_max,
      activity_min_neg_inf,
      activity_min_pos_inf,
      activity_max_neg_inf,
      activity_max_pos_inf,
      activity_invalid,
      activity_row_needed,
      plan.keep_row_mask,
      plan.keep_col_mask,
      plan.new_l,
      plan.new_u,
      lp.A.rowPtr,
      lp.A.colVal,
      lp.A.nzVal,
      activity_long_rows,
      activity_long_row_count,
      m, stats.live_masks ? stats.row_nnz : nullptr);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_singleton_row_activity_cache");
  std::int32_t activity_long_row_count_host = 0;
  throw_if_cuda_error(cudaMemcpy(&activity_long_row_count_host,
                                 activity_long_row_count,
                                 sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy singleton_columns activity_long_row_count");
  if (activity_long_row_count_host > 0) {
    _kernel_singleton_long_row_activity_cache
        <<<activity_long_row_count_host, GPU_PRESOLVE_THREADS>>>(
            activity_finite_min,
            activity_finite_max,
            activity_min_neg_inf,
            activity_min_pos_inf,
            activity_max_neg_inf,
            activity_max_pos_inf,
            activity_invalid,
            plan.keep_row_mask,
            plan.keep_col_mask,
            plan.new_l,
            plan.new_u,
            lp.A.rowPtr,
            lp.A.colVal,
            lp.A.nzVal,
            activity_long_rows,
            activity_long_row_count_host, stats.live_masks ? stats.row_nnz : nullptr);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_singleton_long_row_activity_cache");
  }

  // 1. Generate immutable actions and choose one owner per row.
  _kernel_propose_singleton_column_actions<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      status_flag,
      actions,
      chosen_side,
      row_owner,
      plan.keep_row_mask,
      plan.keep_col_mask,
      stats.singleton_column_mask,
      stats.singleton_column_row,
      stats.singleton_column_val,
      plan.new_c,
      plan.new_l,
      plan.new_u,
      plan.new_AL,
      plan.new_AU,
      activity_finite_min,
      activity_finite_max,
      activity_min_neg_inf,
      activity_min_pos_inf,
      activity_max_neg_inf,
      activity_max_pos_inf,
      activity_invalid,
      lp.A.rowPtr,
      lp.A.colVal,
      lp.A.nzVal,
      pparams.bound_tol,
      pparams.zero_tol,
      m,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_propose_singleton_column_actions");

  // 2. Stage selected actions without a host round trip between proposal and
  // staging. An unbounded proposal gates staging on device.
  _kernel_stage_singleton_column_actions<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
      status_flag, row_delete, row_lhs_change, row_rhs_change, col_delete,
      AL_new, AU_new, obj_contrib, row_owner, actions, chosen_side,
      stats.singleton_column_val, plan.new_c, plan.new_l, plan.new_u,
      plan.new_AL, plan.new_AU, pparams.bound_tol, m, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_stage_singleton_column_actions");
  std::int32_t status[3] = {0, 0, 0};
  throw_if_cuda_error(cudaMemcpy(status, status_flag, sizeof(status), cudaMemcpyDeviceToHost),
                      "cudaMemcpy singleton_columns staged status");
  if (status[0] != 0) {
    plan.has_unbounded = true;
  } else if (status[1] != 0 || status[2] != 0) {
    if (status[1] != 0) {
      _kernel_apply_singleton_objective_updates_deterministic
          <<<blocks_n, GPU_PRESOLVE_THREADS>>>(
              c_new,
              plan.keep_col_mask,
              col_delete,
              row_owner,
              stats.singleton_column_val,
              plan.new_c,
              lp.AT.rowPtr,
              lp.AT.colVal,
              lp.AT.nzVal,
              n, adaptive_gather_enabled() ? ADAPTIVE_SHORT_ROW : 0);
      if(adaptive_gather_enabled()) {
        const auto& groups=prepare_adaptive_row_groups(lp.AT);
        grouped_singleton_objective_updates<32><<<std::min(1024,(n+7)/8),256>>>(
            c_new,plan.keep_col_mask,col_delete,row_owner,stats.singleton_column_val,
            plan.new_c,lp.AT.rowPtr,lp.AT.colVal,lp.AT.nzVal,n,groups.medium,groups.counts);
        grouped_singleton_objective_updates<256><<<std::min(256,n),256>>>(
            c_new,plan.keep_col_mask,col_delete,row_owner,stats.singleton_column_val,
            plan.new_c,lp.AT.rowPtr,lp.AT.colVal,lp.AT.nzVal,n,groups.long_rows,groups.counts+1);
      }
      throw_if_cuda_error(cudaGetLastError(),
                          "_kernel_apply_singleton_objective_updates_deterministic");
    }
    const std::int32_t max_mn = m > n ? m : n;
    const int blocks_mn = (max_mn + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    if (status[1] != 0 && pparams.record_postsolve_tape) {
      append_subcol_tape_compacted_from_device(plan.tape,
                                               col_delete,
                                               row_delete,
                                               plan.keep_col_mask,
                                               row_owner,
                                               chosen_side,
                                               plan.new_c,
                                               plan.new_l,
                                               plan.new_u,
                                               lp.A,
                                               pparams.zero_tol,
                                               "cudaMemcpy singleton compact SubCol tape");
    }
    if (status[2] != 0 &&
        pparams.record_postsolve_tape) {
      append_eq_to_ineq_tape_from_device(plan.tape,
                                         row_lhs_change,
                                         row_rhs_change,
                                         m,
                                         "cudaMemcpy singleton eq-to-ineq tape");
    }
    // 3. Commit once, after recovery has captured the original data.
    _kernel_commit_singleton_column_actions<<<blocks_mn, GPU_PRESOLVE_THREADS>>>(
        plan.keep_row_mask,
        plan.keep_col_mask,
        plan.new_c,
        plan.new_AL,
        plan.new_AU,
        row_delete,
        col_delete,
        c_new,
        AL_new,
        AU_new,
        row_owner,
        status[1] != 0,
        m,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_commit_singleton_column_actions");
    throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_singleton_columns synchronize");
    const double obj_delta = sum_cleanup_domain(obj_contrib, n, plan.rule_entry_keep_col,
        [](const double* values, int count) {
          return sum_device_double(values, count, "cudaMemcpy singleton_columns obj_contrib");
        });
    plan.obj_constant_delta += obj_delta;
    plan.has_row_action = true;
    plan.has_col_action = plan.has_col_action || status[1] != 0;
    plan.has_change = true;
  }

}

void release_singleton_columns_workspace() noexcept {
  singleton_columns_tls_cache_storage().release();
}

}  // namespace gpu_presolver::presolve
