#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/rules/rule_doubleton_equations.hpp"

#include "doubleton_long_rows.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include "gpu_presolver/presolve/gpu_presolve_kernels.hpp"
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;
constexpr int DOUBLETON_EQUATION_BATCH_MATCHING_ROUNDS = 8;
constexpr double DOUBLETON_EQUATION_MAX_RATIO_PIVOT = 1.0e3;

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

template <class T>
cudaError_t doubleton_cuda_malloc(T** pointer, std::size_t bytes) {
  return cudaMallocAsync(reinterpret_cast<void**>(pointer), bytes, nullptr);
}

cudaError_t doubleton_cuda_free(void* pointer) {
  return pointer == nullptr ? cudaSuccess : cudaFreeAsync(pointer, nullptr);
}

// Thrust's default temporary allocator uses synchronous cudaMalloc/cudaFree.
// Keep its temporary storage ordered on the same default stream as the
// doubleton kernels so releasing sort workspace cannot force the host to wait.
struct DoubletonThrustAsyncAllocator {
  using value_type = char;

  char* allocate(std::ptrdiff_t bytes) {
    char* pointer = nullptr;
    throw_if_cuda_error(
        cudaMallocAsync(reinterpret_cast<void**>(&pointer),
                        static_cast<std::size_t>(bytes),
                        nullptr),
        "cudaMallocAsync doubleton thrust temporary storage");
    return pointer;
  }

  void deallocate(char* pointer, std::size_t) {
    throw_if_cuda_error(
        cudaFreeAsync(pointer, nullptr),
        "cudaFreeAsync doubleton thrust temporary storage");
  }
};

auto doubleton_thrust_policy() {
  return thrust::cuda::par_nosync(DoubletonThrustAsyncAllocator{}).on(nullptr);
}

void append_postsolve_record(PostsolveTape& tape,
                             PostsolveReductionType type,
                             const std::vector<std::int32_t>& indices,
                             const std::vector<double>& vals,
                             PostsolveDualMode dual_mode) {
  tape.types.push_back(static_cast<std::int32_t>(type));
  tape.indices.insert(tape.indices.end(), indices.begin(), indices.end());
  tape.vals.insert(tape.vals.end(), vals.begin(), vals.end());
  tape.index_starts.push_back(static_cast<NnzOffset>(tape.indices.size()));
  tape.value_starts.push_back(static_cast<NnzOffset>(tape.vals.size()));
  tape.dual_modes.push_back(static_cast<std::uint8_t>(dual_mode));
}

template <class T>
std::vector<T> copy_device_vector(const T* device, std::int32_t count, const char* context) {
  std::vector<T> values(static_cast<std::size_t>(count));
  if (count > 0) {
    throw_if_cuda_error(cudaMemcpy(values.data(),
                                   device,
                                   sizeof(T) * static_cast<std::size_t>(count),
                                   cudaMemcpyDeviceToHost),
                        context);
  }
  return values;
}

__global__ void _kernel_doubleton_sort_csr_row_keys(unsigned long long* keys,
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

__global__ void _kernel_doubleton_unpack_csr_row_keys(std::int32_t* col_val,
                                                      const unsigned long long* keys,
                                                      NnzOffset nnz) {
  const NnzOffset p = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (p < nnz) {
    col_val[p] = static_cast<std::int32_t>(keys[p] & 0xffffffffULL);
  }
}

void sort_csr_rows_by_col_thrust(DeviceCsrMatrix& matrix, const char* context) {
  if (csr_rows_strictly_sorted(matrix)) return;
  if (matrix.nnz <= 0) {
    return;
  }
  unsigned long long* sort_keys = nullptr;
  throw_if_cuda_error(doubleton_cuda_malloc(&sort_keys, sizeof(unsigned long long) * static_cast<std::size_t>(matrix.nnz)),
                      context);
  const int blocks_rows = (matrix.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_doubleton_sort_csr_row_keys<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
      sort_keys, matrix.rowPtr, matrix.colVal, matrix.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_sort_csr_row_keys");
  thrust::sort_by_key(doubleton_thrust_policy(),
                      thrust::device_pointer_cast(sort_keys),
                      thrust::device_pointer_cast(sort_keys + matrix.nnz),
                      thrust::device_pointer_cast(matrix.nzVal));
  throw_if_cuda_error(cudaGetLastError(), "thrust doubleton sort CSR rows");
  const int blocks_nnz = (matrix.nnz + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_doubleton_unpack_csr_row_keys<<<blocks_nnz, GPU_PRESOLVE_THREADS>>>(
      matrix.colVal, sort_keys, matrix.nnz);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_unpack_csr_row_keys");
  doubleton_cuda_free(sort_keys);
}

__device__ bool _is_integral_ratio_doubleton_equations(double num, double den, double tol) {
  if (fabs(den) <= tol) {
    return false;
  }
  const double ratio = fabs(num / den);
  if (!isfinite(ratio)) {
    return false;
  }
  return fabs(ratio - nearbyint(ratio)) <= tol;
}

__device__ bool _is_acceptable_doubleton_equations_pivot(double keep_val, double elim_val) {
  if (fabs(elim_val) <= 0.0) {
    return false;
  }
  const double pivot_ratio = fabs(keep_val / elim_val);
  return pivot_ratio <= DOUBLETON_EQUATION_MAX_RATIO_PIVOT &&
         pivot_ratio >= 1.0 / DOUBLETON_EQUATION_MAX_RATIO_PIVOT;
}

__device__ void _choose_doubleton_equations_columns(std::int32_t col1,
                                             double val1,
                                             std::int32_t col2,
                                             double val2,
                                             const std::int32_t* col_nnz,
                                             double tol,
                                             std::int32_t* elim_col,
                                             double* elim_val,
                                             std::int32_t* keep_col,
                                             double* keep_val) {
  const bool integral12 = _is_integral_ratio_doubleton_equations(val1, val2, tol);
  const bool integral21 = _is_integral_ratio_doubleton_equations(val2, val1, tol);

  if (col_nnz[col1] == 1 && col_nnz[col2] != 1) {
    *elim_col = col1;
    *elim_val = val1;
    *keep_col = col2;
    *keep_val = val2;
  } else if (col_nnz[col1] != 1 && col_nnz[col2] == 1) {
    *elim_col = col2;
    *elim_val = val2;
    *keep_col = col1;
    *keep_val = val1;
  } else if (integral12 && !integral21) {
    *elim_col = col2;
    *elim_val = val2;
    *keep_col = col1;
    *keep_val = val1;
  } else if (integral21 && !integral12) {
    *elim_col = col1;
    *elim_val = val1;
    *keep_col = col2;
    *keep_val = val2;
  } else if (col_nnz[col1] < col_nnz[col2]) {
    *elim_col = col1;
    *elim_val = val1;
    *keep_col = col2;
    *keep_val = val2;
  } else {
    *elim_col = col2;
    *elim_val = val2;
    *keep_col = col1;
    *keep_val = val1;
  }
}

__device__ std::int32_t _doubleton_equations_fill_in_proxy_device(const NnzOffset* row_ptr,
                                                           const std::int32_t* col_val,
                                                           std::int32_t keep_col,
                                                           std::int32_t elim_col) {
  std::int32_t fill_in = -1;
  NnzOffset jj = row_ptr[keep_col];
  const NnzOffset keep_stop = row_ptr[keep_col + 1];
  NnzOffset kk = row_ptr[elim_col];
  const NnzOffset elim_stop = row_ptr[elim_col + 1];
  if (keep_stop <= jj || elim_stop <= kk) {
    return fill_in;
  }
  while (jj < keep_stop && kk < elim_stop) {
    const std::int32_t keep_row = col_val[jj];
    const std::int32_t elim_row = col_val[kk];
    if (keep_row == elim_row) {
      ++jj;
      ++kk;
    } else if (elim_row < keep_row) {
      ++kk;
      ++fill_in;
    } else {
      ++jj;
    }
  }
  fill_in += elim_stop - kk;
  return fill_in;
}

__global__ void _kernel_doubleton_equations_live_col_nnz(std::int32_t* col_nnz,
                                                  const NnzOffset* AT_row_ptr,
                                                  const std::int32_t* AT_col_val,
                                                  const double* AT_nz_val,
                                                  const std::uint8_t* keep_row,
                                                  const std::uint8_t* keep_col,
                                                  double zero_tol,
                                                  std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (col < n) {
    if (keep_col[col] == std::uint8_t{0}) {
      col_nnz[col] = 0;
      return;
    }
    std::int32_t count = 0;
    const NnzOffset start_idx = AT_row_ptr[col];
    const NnzOffset stop_idx = AT_row_ptr[col + 1];
    for (NnzOffset p = start_idx; p < stop_idx; ++p) {
      const std::int32_t row = AT_col_val[p];
      const double a = AT_nz_val[p];
      if (keep_row[row] != std::uint8_t{0} && fabs(a) > zero_tol) {
        ++count;
      }
    }
    col_nnz[col] = count;
  }
}

__global__ void _kernel_select_doubleton_equations_target_row_direct(std::int32_t* target_row_ref,
                                                              std::int32_t* target_meta_i32,
                                                              double* target_meta_f64,
                                                              const NnzOffset* row_ptr,
                                                              const std::int32_t* col_val,
                                                              const double* nz_val,
                                                              const std::uint8_t* keep_row,
                                                              const std::uint8_t* keep_col_mask,
                                                              const std::int32_t* col_nnz,
                                                              const double* AL,
                                                              const double* AU,
                                                              const double* l,
                                                              const double* u,
                                                              const double* c,
                                                              const NnzOffset* AT_row_ptr,
                                                              const std::int32_t* AT_col_val,
                                                              double zero_tol,
                                                              double tol,
                                                              double ratio_tol,
                                                              std::int32_t max_fill_in_proxy,
                                                              std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < m && keep_row[i] != std::uint8_t{0}) {
    const double lhs = AL[i];
    const double rhs = AU[i];
    if (!isfinite(lhs) || !isfinite(rhs) || fabs(lhs - rhs) > tol) {
      return;
    }
    const NnzOffset row_start = row_ptr[i];
    const NnzOffset row_stop = row_ptr[i + 1];
    std::int32_t live_count = 0;
    std::int32_t col1 = -1;
    std::int32_t col2 = -1;
    double val1 = 0.0;
    double val2 = 0.0;
    for (NnzOffset p = row_start; p < row_stop; ++p) {
      const std::int32_t col_j = col_val[p];
      const double a = nz_val[p];
      if (keep_col_mask[col_j] != std::uint8_t{0} && fabs(a) > zero_tol) {
        ++live_count;
        if (live_count == 1) {
          col1 = col_j;
          val1 = a;
        } else if (live_count == 2) {
          col2 = col_j;
          val2 = a;
        } else {
          return;
        }
      }
    }
    if (live_count != 2) {
      return;
    }

    std::int32_t elim_col = -1;
    std::int32_t keep_col = -1;
    double elim_val = 0.0;
    double keep_val = 0.0;
    _choose_doubleton_equations_columns(col1, val1, col2, val2, col_nnz, ratio_tol,
                                 &elim_col, &elim_val, &keep_col, &keep_val);
    if (!_is_acceptable_doubleton_equations_pivot(keep_val, elim_val)) {
      return;
    }
    const std::int32_t keep_nnz = col_nnz[keep_col];
    const std::int32_t elim_nnz = col_nnz[elim_col];
    if (elim_nnz > keep_nnz + max_fill_in_proxy + 1) {
      return;
    }
    const std::int32_t fill_in =
        _doubleton_equations_fill_in_proxy_device(AT_row_ptr, AT_col_val, keep_col, elim_col);
    if (fill_in > max_fill_in_proxy) {
      return;
    }

    const std::int32_t old = atomicMin(target_row_ref, i);
    if (i < old) {
      target_meta_i32[0] = i;
      target_meta_i32[1] = elim_col;
      target_meta_i32[2] = keep_col;
      target_meta_f64[0] = elim_val;
      target_meta_f64[1] = keep_val;
      target_meta_f64[2] = rhs;
      target_meta_f64[3] = l[elim_col];
      target_meta_f64[4] = u[elim_col];
      target_meta_f64[5] = c[elim_col];
      target_meta_f64[6] = l[keep_col];
      target_meta_f64[7] = u[keep_col];
    }
  }
}

__global__ void _kernel_doubleton_equations_apply(std::int32_t* status,
                                           double* l,
                                           double* u,
                                           double* c,
                                           double* AL,
                                           double* AU,
                                           std::uint8_t* keep_row_mask,
                                           std::uint8_t* keep_col_mask,
                                           double* obj_delta,
                                           const std::int32_t* meta_i32,
                                           const double* meta_f64,
                                           const NnzOffset* AT_row_ptr,
                                           const std::int32_t* AT_col_val,
                                           const double* AT_nz_val,
                                           double bound_tol,
                                           std::int32_t m) {
  const std::int32_t target_row = meta_i32[0];
  const std::int32_t elim_col = meta_i32[1];
  const std::int32_t keep_col = meta_i32[2];
  const double elim_val = meta_f64[0];
  const double keep_val = meta_f64[1];
  const double rhs = meta_f64[2];
  const double old_elim_l = meta_f64[3];
  const double old_elim_u = meta_f64[4];
  const double elim_obj = meta_f64[5];
  const double old_keep_l = meta_f64[6];
  const double old_keep_u = meta_f64[7];
  const double alpha = -keep_val / elim_val;
  const double beta = rhs / elim_val;

  double mapped_l = -INFINITY;
  double mapped_u = INFINITY;
  if (isfinite(old_elim_l)) {
    const double bound_val = (old_elim_l - beta) / alpha;
    if (alpha > 0.0) {
      mapped_l = bound_val;
    } else {
      mapped_u = bound_val;
    }
  }
  if (isfinite(old_elim_u)) {
    const double bound_val = (old_elim_u - beta) / alpha;
    if (alpha > 0.0) {
      mapped_u = bound_val;
    } else {
      mapped_l = bound_val;
    }
  }
  if (mapped_l > mapped_u) {
    const double tmp = mapped_l;
    mapped_l = mapped_u;
    mapped_u = tmp;
  }
  const double keep_l_new = isfinite(mapped_l) ? fmax(old_keep_l, mapped_l) : old_keep_l;
  const double keep_u_new = isfinite(mapped_u) ? fmin(old_keep_u, mapped_u) : old_keep_u;
  if (keep_l_new > keep_u_new + bound_tol) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      status[0] = 1;
    }
    return;
  }

  const std::int32_t t = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (t < m) {
    double shift = 0.0;
    const NnzOffset start = AT_row_ptr[elim_col];
    const NnzOffset stop = AT_row_ptr[elim_col + 1];
    for (NnzOffset p = start; p < stop; ++p) {
      if (AT_col_val[p] == t) {
        shift = AT_nz_val[p] * beta;
        break;
      }
    }
    AL[t] -= shift;
    AU[t] -= shift;
  }
  if (t == 0) {
    l[keep_col] = keep_l_new;
    u[keep_col] = keep_u_new;
    c[keep_col] += elim_obj * alpha;
    keep_row_mask[target_row] = std::uint8_t{0};
    keep_col_mask[elim_col] = std::uint8_t{0};
    atomicAdd(obj_delta, elim_obj * beta);
    status[1] = 1;
  }
}

__global__ void _kernel_doubleton_equations_row_counts(NnzOffset* row_nnz_new,
                                                const NnzOffset* row_ptr,
                                                const std::int32_t* col_val,
                                                const double* nz_val,
                                                const std::int32_t* meta_i32,
                                                const double* meta_f64,
                                                double zero_tol,
                                                std::int32_t m) {
  const std::int32_t r = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (r >= m) {
    return;
  }
  const std::int32_t target_row = meta_i32[0];
  const std::int32_t elim_col = meta_i32[1];
  const std::int32_t keep_col = meta_i32[2];
  const double alpha = -meta_f64[1] / meta_f64[0];

  const NnzOffset first = row_ptr[r];
  const NnzOffset last = row_ptr[r + 1];
  NnzOffset len = last - first;
  if (r == target_row) {
    row_nnz_new[r] = len;
    return;
  }

  bool elim_present = false;
  bool keep_present = false;
  double are = 0.0;
  double old_keep = 0.0;
  for (NnzOffset p = first; p < last; ++p) {
    const std::int32_t col = col_val[p];
    const double val = nz_val[p];
    if (col == elim_col) {
      elim_present = true;
      are = val;
    } else if (col == keep_col) {
      keep_present = true;
      old_keep = val;
    }
  }

  const double new_keep = old_keep + alpha * are;
  if (elim_present && !keep_present && fabs(new_keep) > zero_tol) {
    ++len;
  } else if (keep_present && fabs(new_keep) <= zero_tol) {
    --len;
  }
  row_nnz_new[r] = len > 0 ? len : 0;
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

template <typename T>
void inclusive_scan_i32(T* values, std::int32_t n, const char* context) {
  if (n <= 0) {
    return;
  }
  void* temp_storage = nullptr;
  std::size_t temp_bytes = 0;
  throw_if_cuda_error(cub::DeviceScan::InclusiveSum(
                          temp_storage, temp_bytes, values, values, n),
                      context);
  throw_if_cuda_error(doubleton_cuda_malloc(&temp_storage, temp_bytes), context);
  throw_if_cuda_error(cub::DeviceScan::InclusiveSum(
                          temp_storage, temp_bytes, values, values, n),
                      context);
  doubleton_cuda_free(temp_storage);
}

__global__ void _kernel_fill_i32(std::int32_t* data, std::int32_t value, std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < n) {
    data[i] = value;
  }
}

__global__ void _kernel_u8_to_i32(std::int32_t* out, const std::uint8_t* in, std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < n) {
    out[i] = in[i] != std::uint8_t{0} ? 1 : 0;
  }
}

__global__ void _kernel_scatter_mask_indices(std::int32_t* indices,
                                             const std::uint8_t* mask,
                                             const std::int32_t* prefix,
                                             std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < n && mask[i] != std::uint8_t{0}) {
    indices[prefix[i] - 1] = i;
  }
}

struct DoubletonBatchTapeRecord {
  std::int32_t elim_col;
  std::int32_t keep_col;
  std::int32_t row;
  double elim_val;
  double keep_val;
  double old_row_au;
  double old_elim_l;
  double old_elim_u;
  double old_keep_l;
  double old_keep_u;
  double new_keep_l;
  double new_keep_u;
  double old_elim_c;
};

__global__ void capture_scalar_recovery(DoubletonBatchTapeRecord *out, const int *ids,
                                        const double *v, const double *l, const double *u) {
  *out = DoubletonBatchTapeRecord{ids[1], ids[2], ids[0], v[0],      v[1],      v[2], v[3],
                                  v[4],   v[6],   v[7],   l[ids[2]], u[ids[2]], v[5]};
}

struct DoubletonBatchTapeBuffer {
  DoubletonBatchTapeRecord* records = nullptr;
  std::int32_t count = 0;
  std::int32_t capacity = 0;

  DoubletonBatchTapeBuffer() = default;
  DoubletonBatchTapeBuffer(const DoubletonBatchTapeBuffer&) = delete;
  DoubletonBatchTapeBuffer& operator=(const DoubletonBatchTapeBuffer&) = delete;

  ~DoubletonBatchTapeBuffer() {
    doubleton_cuda_free(records);
  }

  DoubletonBatchTapeRecord* append_storage(std::int32_t additional_count) {
    if (additional_count <= 0) {
      return records == nullptr ? nullptr : records + count;
    }
    const std::int64_t required =
        static_cast<std::int64_t>(count) + additional_count;
    if (required > INT_MAX) {
      throw std::runtime_error("doubleton batch tape exceeds Int32 capacity");
    }
    if (required > capacity) {
      std::int64_t grown_capacity = capacity > 0
          ? 2 * static_cast<std::int64_t>(capacity)
          : static_cast<std::int64_t>(additional_count);
      if (grown_capacity < required) {
        grown_capacity = required;
      }
      if (grown_capacity > INT_MAX) {
        grown_capacity = required;
      }
      DoubletonBatchTapeRecord* grown_records = nullptr;
      throw_if_cuda_error(
          doubleton_cuda_malloc(
              &grown_records,
              sizeof(DoubletonBatchTapeRecord) *
                  static_cast<std::size_t>(grown_capacity)),
          "cudaMalloc doubleton accumulated batch tape");
      if (count > 0) {
        const cudaError_t copy_status =
            cudaMemcpyAsync(grown_records,
                            records,
                            sizeof(DoubletonBatchTapeRecord) *
                                static_cast<std::size_t>(count),
                            cudaMemcpyDeviceToDevice,
                            nullptr);
        if (copy_status != cudaSuccess) {
          doubleton_cuda_free(grown_records);
          throw_if_cuda_error(
              copy_status,
              "cudaMemcpy grow doubleton accumulated batch tape");
        }
      }
      doubleton_cuda_free(records);
      records = grown_records;
      capacity = static_cast<std::int32_t>(grown_capacity);
    }
    DoubletonBatchTapeRecord* output = records + count;
    count += additional_count;
    return output;
  }
};

void append_doubleton_batch_tape_records(
    PostsolveTape& tape,
    const std::vector<DoubletonBatchTapeRecord>& records) {
  tape.types.reserve(tape.types.size() + records.size());
  tape.indices.reserve(tape.indices.size() + 4 * records.size());
  tape.vals.reserve(tape.vals.size() + 12 * records.size());
  tape.index_starts.reserve(tape.index_starts.size() + records.size());
  tape.value_starts.reserve(tape.value_starts.size() + records.size());
  tape.dual_modes.reserve(tape.dual_modes.size() + records.size());
  for (const DoubletonBatchTapeRecord& record : records) {
    append_postsolve_record(
        tape,
        PostsolveReductionType::DoubletonEquation,
        std::vector<std::int32_t>{record.elim_col, record.keep_col, record.row, 0},
        std::vector<double>{record.elim_val,
                            record.keep_val,
                            record.old_row_au,
                            record.old_elim_l,
                            record.old_elim_u,
                            record.old_keep_l,
                            record.old_keep_u,
                            record.new_keep_l,
                            record.new_keep_u,
                            record.old_elim_c,
                            0.0,
                            0.0},
        PostsolveDualMode::Minimal);
  }
}

__global__ void _kernel_build_doubleton_postsolve_tape_gpu(
    std::int32_t* types,
    NnzOffset* index_starts,
    NnzOffset* value_starts,
    std::uint8_t* dual_modes,
    std::int32_t* indices,
    double* vals,
    const DoubletonBatchTapeRecord* records,
    std::int32_t count) {
  const std::int32_t i =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i > count) {
    return;
  }
  index_starts[i] = 4 * i;
  value_starts[i] = 12 * i;
  if (i == count) {
    return;
  }

  const DoubletonBatchTapeRecord record = records[i];
  types[i] = static_cast<std::int32_t>(PostsolveReductionType::DoubletonEquation);
  dual_modes[i] = static_cast<std::uint8_t>(PostsolveDualMode::Minimal);
  const NnzOffset idx = 4 * i;
  indices[idx] = record.elim_col;
  indices[idx + 1] = record.keep_col;
  indices[idx + 2] = record.row;
  indices[idx + 3] = 0;
  const std::int32_t val = 12 * i;
  vals[val] = record.elim_val;
  vals[val + 1] = record.keep_val;
  vals[val + 2] = record.old_row_au;
  vals[val + 3] = record.old_elim_l;
  vals[val + 4] = record.old_elim_u;
  vals[val + 5] = record.old_keep_l;
  vals[val + 6] = record.old_keep_u;
  vals[val + 7] = record.new_keep_l;
  vals[val + 8] = record.new_keep_u;
  vals[val + 9] = record.old_elim_c;
  vals[val + 10] = 0.0;
  vals[val + 11] = 0.0;
}

void build_doubleton_postsolve_tape_gpu(
    PostsolveTapeGpu& tape,
    const DoubletonBatchTapeRecord* records,
    std::int32_t count) {
  if (count <= 0) {
    return;
  }
  if (count > INT_MAX / 12) {
    throw std::runtime_error("doubleton GPU postsolve tape exceeds Int32 capacity");
  }
  tape.reset();
  tape.record_count = count;
  tape.index_count = 4 * count;
  tape.value_count = 12 * count;
  tape.record_capacity = count;
  tape.index_capacity = tape.index_count;
  tape.value_capacity = tape.value_count;
  throw_if_cuda_error(
      doubleton_cuda_malloc(
          &tape.types,
          sizeof(std::int32_t) * static_cast<std::size_t>(count)),
      "cudaMalloc doubleton GPU tape types");
  throw_if_cuda_error(
      doubleton_cuda_malloc(
          &tape.index_starts,
          sizeof(NnzOffset) * static_cast<std::size_t>(count + 1)),
      "cudaMalloc doubleton GPU tape index starts");
  throw_if_cuda_error(
      doubleton_cuda_malloc(
          &tape.value_starts,
          sizeof(NnzOffset) * static_cast<std::size_t>(count + 1)),
      "cudaMalloc doubleton GPU tape value starts");
  throw_if_cuda_error(
      doubleton_cuda_malloc(&tape.dual_modes, static_cast<std::size_t>(count)),
      "cudaMalloc doubleton GPU tape dual modes");
  throw_if_cuda_error(
      doubleton_cuda_malloc(
          &tape.indices,
          sizeof(std::int32_t) *
              static_cast<std::size_t>(tape.index_count)),
      "cudaMalloc doubleton GPU tape indices");
  throw_if_cuda_error(
      doubleton_cuda_malloc(
          &tape.vals,
          sizeof(double) * static_cast<std::size_t>(tape.value_count)),
      "cudaMalloc doubleton GPU tape values");
  const int blocks =
      (count + 1 + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_build_doubleton_postsolve_tape_gpu<<<blocks,
                                               GPU_PRESOLVE_THREADS>>>(
      tape.types, tape.index_starts, tape.value_starts, tape.dual_modes,
      tape.indices, tape.vals, records, count);
  throw_if_cuda_error(cudaGetLastError(),
                      "_kernel_build_doubleton_postsolve_tape_gpu");
}

__global__ void _kernel_capture_doubleton_batch_tape(
    DoubletonBatchTapeRecord* records,
    const std::int32_t* selected_rows,
    const std::int32_t* candidate_keep_col,
    const std::int32_t* candidate_elim_col,
    const double* candidate_keep_val,
    const double* candidate_elim_val,
    const double* l,
    const double* u,
    const double* c,
    const double* au,
    std::int32_t selected_count) {
  const std::int32_t i =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i >= selected_count) {
    return;
  }
  const std::int32_t row = selected_rows[i];
  const std::int32_t keep_col = candidate_keep_col[row];
  const std::int32_t elim_col = candidate_elim_col[row];
  records[i] = DoubletonBatchTapeRecord{elim_col,
                                         keep_col,
                                         row,
                                         candidate_elim_val[row],
                                         candidate_keep_val[row],
                                         au[row],
                                         l[elim_col],
                                         u[elim_col],
                                         l[keep_col],
                                         u[keep_col],
                                         0.0,
                                         0.0,
                                         c[elim_col]};
}

__global__ void _kernel_pack_doubleton_batch_tape(
    DoubletonBatchTapeRecord* packed_records,
    const DoubletonBatchTapeRecord* selected_records,
    const std::uint8_t* subst_pair_mask,
    const std::int32_t* prefix,
    const double* l,
    const double* u,
    std::int32_t selected_count) {
  const std::int32_t i =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i >= selected_count || subst_pair_mask[i] == std::uint8_t{0}) {
    return;
  }
  DoubletonBatchTapeRecord record = selected_records[i];
  record.new_keep_l = l[record.keep_col];
  record.new_keep_u = u[record.keep_col];
  packed_records[prefix[i] - 1] = record;
}

__device__ std::int32_t _doubleton_equations_next_live_col_row(const NnzOffset* row_ptr,
                                                        const std::int32_t* col_val,
                                                        const double* nz_val,
                                                        const std::uint8_t* keep_row,
                                                        double zero_tol,
                                                        std::int32_t col,
                                                        std::int32_t* pos) {
  const NnzOffset stop = row_ptr[col + 1];
  while (*pos < stop) {
    const std::int32_t row = col_val[*pos];
    const double val = nz_val[*pos];
    if (keep_row[row] != std::uint8_t{0} && fabs(val) > zero_tol) {
      return row;
    }
    ++(*pos);
  }
  return INT_MAX;
}

__device__ std::int32_t _doubleton_equations_fill_in_proxy_device_live(const NnzOffset* row_ptr,
                                                                const std::int32_t* col_val,
                                                                const double* nz_val,
                                                                const std::uint8_t* keep_row,
                                                                double zero_tol,
                                                                std::int32_t keep_col,
                                                                std::int32_t elim_col) {
  std::int32_t fill_in = -1;
  NnzOffset jj = row_ptr[keep_col];
  const NnzOffset keep_stop = row_ptr[keep_col + 1];
  NnzOffset kk = row_ptr[elim_col];
  const NnzOffset elim_stop = row_ptr[elim_col + 1];
  if (keep_stop <= jj || elim_stop <= kk) {
    return fill_in;
  }

  while (jj < keep_stop && kk < elim_stop) {
    std::int32_t jj_live = jj;
    std::int32_t kk_live = kk;
    const std::int32_t keep_row_j =
        _doubleton_equations_next_live_col_row(row_ptr, col_val, nz_val, keep_row, zero_tol, keep_col, &jj_live);
    const std::int32_t elim_row_k =
        _doubleton_equations_next_live_col_row(row_ptr, col_val, nz_val, keep_row, zero_tol, elim_col, &kk_live);
    if (keep_row_j == INT_MAX || elim_row_k == INT_MAX) {
      break;
    }
    if (keep_row_j == elim_row_k) {
      jj = jj_live + 1;
      kk = kk_live + 1;
    } else if (elim_row_k < keep_row_j) {
      kk = kk_live + 1;
      ++fill_in;
    } else {
      jj = jj_live + 1;
    }
  }
  while (kk < elim_stop) {
    std::int32_t kk_live = kk;
    const std::int32_t elim_row_k =
        _doubleton_equations_next_live_col_row(row_ptr, col_val, nz_val, keep_row, zero_tol, elim_col, &kk_live);
    if (elim_row_k == INT_MAX) {
      break;
    }
    ++fill_in;
    kk = kk_live + 1;
  }
  return fill_in;
}

__global__ void _kernel_doubleton_equations_candidates(std::uint8_t* candidate_mask,
                                                std::int32_t* candidate_elim_col,
                                                double* candidate_elim_val,
                                                std::int32_t* candidate_keep_col,
                                                double* candidate_keep_val,
                                                const NnzOffset* row_ptr,
                                                const std::int32_t* col_val,
                                                const double* nz_val,
                                                const std::uint8_t* keep_row,
                                                const std::uint8_t* keep_col_mask,
                                                const std::int32_t* col_nnz,
                                                const double* AL,
                                                const double* AU,
                                                double zero_tol,
                                                double tol,
                                                double ratio_tol,
                                                std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i >= m || keep_row[i] == std::uint8_t{0}) {
    return;
  }
  const double lhs = AL[i];
  const double rhs = AU[i];
  if (!isfinite(lhs) || !isfinite(rhs) || fabs(lhs - rhs) > tol) {
    return;
  }

  std::int32_t live_count = 0;
  std::int32_t col1 = -1;
  std::int32_t col2 = -1;
  double val1 = 0.0;
  double val2 = 0.0;
  for (NnzOffset p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
    const std::int32_t col = col_val[p];
    const double val = nz_val[p];
    if (keep_col_mask[col] != std::uint8_t{0} && fabs(val) > zero_tol) {
      ++live_count;
      if (live_count == 1) {
        col1 = col;
        val1 = val;
      } else if (live_count == 2) {
        col2 = col;
        val2 = val;
      } else {
        return;
      }
    }
  }
  if (live_count != 2) {
    return;
  }

  std::int32_t elim_col = -1;
  std::int32_t keep_col = -1;
  double elim_val = 0.0;
  double keep_val = 0.0;
  _choose_doubleton_equations_columns(col1, val1, col2, val2, col_nnz, ratio_tol,
                               &elim_col, &elim_val, &keep_col, &keep_val);
  if (!_is_acceptable_doubleton_equations_pivot(keep_val, elim_val)) {
    return;
  }
  candidate_mask[i] = std::uint8_t{1};
  candidate_elim_col[i] = elim_col;
  candidate_elim_val[i] = elim_val;
  candidate_keep_col[i] = keep_col;
  candidate_keep_val[i] = keep_val;
}

__global__ void _kernel_mark_doubleton_equations_batch_acceptable(std::uint8_t* acceptable_mask,
                                                           const std::uint8_t* candidate_mask,
                                                           const std::int32_t* candidate_keep_col,
                                                           const std::int32_t* candidate_elim_col,
                                                           const std::int32_t* col_nnz,
                                                           const NnzOffset* AT_row_ptr,
                                                           const std::int32_t* AT_col_val,
                                                           const double* AT_nz_val,
                                                           const std::uint8_t* keep_row,
                                                           double zero_tol,
                                                           std::int32_t max_fill_in_proxy,
                                                           std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < m && candidate_mask[i] != std::uint8_t{0}) {
    const std::int32_t keep_col = candidate_keep_col[i];
    const std::int32_t elim_col = candidate_elim_col[i];
    const std::int32_t keep_nnz = col_nnz[keep_col];
    const std::int32_t elim_nnz = col_nnz[elim_col];
    if (elim_nnz > keep_nnz + max_fill_in_proxy + 1) {
      return;
    }
    const std::int32_t fill_in = _doubleton_equations_fill_in_proxy_device_live(
        AT_row_ptr, AT_col_val, AT_nz_val, keep_row, zero_tol, keep_col, elim_col);
    if (fill_in <= max_fill_in_proxy) {
      acceptable_mask[i] = std::uint8_t{1};
    }
  }
}

__global__ void _kernel_claim_doubleton_equations_batch_columns(std::int32_t* col_owner,
                                                         const std::uint8_t* active_mask,
                                                         const std::uint8_t* blocked_col,
                                                         const std::int32_t* candidate_keep_col,
                                                         const std::int32_t* candidate_elim_col,
                                                         std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < m && active_mask[i] != std::uint8_t{0}) {
    const std::int32_t keep_col = candidate_keep_col[i];
    const std::int32_t elim_col = candidate_elim_col[i];
    if (blocked_col[keep_col] == std::uint8_t{0} && blocked_col[elim_col] == std::uint8_t{0}) {
      atomicMin(&col_owner[keep_col], i);
      atomicMin(&col_owner[elim_col], i);
    }
  }
}

__global__ void _kernel_select_doubleton_equations_batch_rows(std::uint8_t* selected_mask,
                                                       std::uint8_t* active_mask,
                                                       std::uint8_t* blocked_col,
                                                       const std::int32_t* candidate_keep_col,
                                                       const std::int32_t* candidate_elim_col,
                                                       const std::int32_t* col_owner,
                                                       std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < m && active_mask[i] != std::uint8_t{0}) {
    const std::int32_t keep_col = candidate_keep_col[i];
    const std::int32_t elim_col = candidate_elim_col[i];
    if (blocked_col[keep_col] == std::uint8_t{0} &&
        blocked_col[elim_col] == std::uint8_t{0} &&
        col_owner[keep_col] == i &&
        col_owner[elim_col] == i) {
      selected_mask[i] = std::uint8_t{1};
      active_mask[i] = std::uint8_t{0};
      blocked_col[keep_col] = std::uint8_t{1};
      blocked_col[elim_col] = std::uint8_t{1};
    }
  }
}

__global__ void _kernel_apply_doubleton_equations_batch(std::int32_t* status,
                                                 double* l,
                                                 double* u,
                                                 double* c,
                                                 double* AL,
                                                 double* AU,
                                                 std::uint8_t* keep_row_mask,
                                                 std::uint8_t* keep_col_mask,
                                                 std::uint8_t* subst_pair_mask,
                                                 std::uint8_t* fixed_pair_mask,
                                                 double* row_shift,
                                                 double* obj_delta,
                                                 double* fixed_obj_delta,
                                                 const std::int32_t* selected_rows,
                                                 const std::int32_t* candidate_keep_col,
                                                 const std::int32_t* candidate_elim_col,
                                                 const double* candidate_keep_val,
                                                 const double* candidate_elim_val,
                                                 const NnzOffset* AT_row_ptr,
                                                 const std::int32_t* AT_col_val,
                                                 const double* AT_nz_val,
                                                 const double* old_l,
                                                 const double* old_u,
                                                 const double* old_c,
                                                 const double* AU_old,
                                                 double bound_tol,
                                                 std::int32_t k) {
  const std::int32_t t = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (t >= k) {
    return;
  }
  const std::int32_t row = selected_rows[t];
  const std::int32_t keep = candidate_keep_col[row];
  const std::int32_t elim = candidate_elim_col[row];
  const double keep_val = candidate_keep_val[row];
  const double elim_val = candidate_elim_val[row];
  const double rhs = AU_old[row];
  const double alpha = -keep_val / elim_val;
  const double beta = rhs / elim_val;

  double mapped_l = -INFINITY;
  double mapped_u = INFINITY;
  if (isfinite(old_l[elim])) {
    const double bound_val = (old_l[elim] - beta) / alpha;
    if (alpha > 0.0) {
      mapped_l = bound_val;
    } else {
      mapped_u = bound_val;
    }
  }
  if (isfinite(old_u[elim])) {
    const double bound_val = (old_u[elim] - beta) / alpha;
    if (alpha > 0.0) {
      mapped_u = bound_val;
    } else {
      mapped_l = bound_val;
    }
  }
  if (mapped_l > mapped_u) {
    const double tmp = mapped_l;
    mapped_l = mapped_u;
    mapped_u = tmp;
  }
  const double keep_l_new = isfinite(mapped_l) ? fmax(old_l[keep], mapped_l) : old_l[keep];
  const double keep_u_new = isfinite(mapped_u) ? fmin(old_u[keep], mapped_u) : old_u[keep];
  if (keep_l_new > keep_u_new + bound_tol) {
    status[0] = 1;
    return;
  }

  const bool keep_fixed = isfinite(keep_l_new) && isfinite(keep_u_new) && keep_u_new <= keep_l_new + bound_tol;
  if (keep_fixed) {
    const double fixed_at = 0.5 * (keep_l_new + keep_u_new);
    l[keep] = fixed_at;
    u[keep] = fixed_at;
    keep_col_mask[keep] = std::uint8_t{0};
    fixed_pair_mask[t] = std::uint8_t{1};
    atomicAdd(fixed_obj_delta, old_c[keep] * fixed_at);
  } else {
    l[keep] = keep_l_new;
    u[keep] = keep_u_new;
    c[keep] += old_c[elim] * alpha;
    keep_row_mask[row] = std::uint8_t{0};
    keep_col_mask[elim] = std::uint8_t{0};
    subst_pair_mask[t] = std::uint8_t{1};
    atomicAdd(obj_delta, old_c[elim] * beta);
  }
  status[1] = 1;
}

__global__ void _kernel_apply_row_shift(double* AL, double* AU, const double* row_shift, std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < m) {
    AL[i] -= row_shift[i];
    AU[i] -= row_shift[i];
  }
}

__global__ void _kernel_doubleton_equations_batch_entry_counts(NnzOffset* entry_counts,
                                                        const std::int32_t* selected_rows,
                                                        const std::int32_t* candidate_elim_col,
                                                        const NnzOffset* AT_row_ptr,
                                                        const std::int32_t* AT_col_val,
                                                        const double* AT_nz_val,
                                                        const std::uint8_t* keep_row_mask,
                                                        double zero_tol,
                                                        std::int32_t k) {
  const std::int32_t t = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (t >= k) {
    return;
  }
  const std::int32_t row = selected_rows[t];
  const std::int32_t elim = candidate_elim_col[row];
  std::int32_t count = 0;
  for (NnzOffset p = AT_row_ptr[elim]; p < AT_row_ptr[elim + 1]; ++p) {
    const std::int32_t support_row = AT_col_val[p];
    const double a = AT_nz_val[p];
    if (support_row != row && keep_row_mask[support_row] != std::uint8_t{0} && fabs(a) > zero_tol) {
      ++count;
    }
  }
  entry_counts[t] = count;
}

__global__ void _kernel_scatter_selected_rows_by_pair_mask(std::int32_t* out_rows,
                                                           const std::int32_t* selected_rows,
                                                           const std::uint8_t* pair_mask,
                                                           const std::int32_t* prefix,
                                                           std::int32_t k) {
  const std::int32_t t = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (t < k && pair_mask[t] != std::uint8_t{0}) {
    out_rows[prefix[t] - 1] = selected_rows[t];
  }
}

__global__ void _kernel_accumulate_doubleton_equations_fixed_row_shift(double* row_shift,
                                                                const std::int32_t* selected_rows,
                                                                const std::uint8_t* fixed_pair_mask,
                                                                const std::int32_t* candidate_keep_col,
                                                                const double* fixed_values_by_col,
                                                                const NnzOffset* AT_row_ptr,
                                                                const std::int32_t* AT_col_val,
                                                                const double* AT_nz_val,
                                                                const std::uint8_t* keep_row_mask,
                                                                double zero_tol,
                                                                std::int32_t k) {
  const std::int32_t t = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (t >= k || fixed_pair_mask[t] == std::uint8_t{0}) {
    return;
  }
  const std::int32_t row = selected_rows[t];
  const std::int32_t keep = candidate_keep_col[row];
  const double fixed_at = fixed_values_by_col[keep];
  for (NnzOffset p = AT_row_ptr[keep]; p < AT_row_ptr[keep + 1]; ++p) {
    const std::int32_t support_row = AT_col_val[p];
    const double a = AT_nz_val[p];
    if (keep_row_mask[support_row] != std::uint8_t{0} && fabs(a) > zero_tol) {
      atomicAdd(&row_shift[support_row], a * fixed_at);
    }
  }
}

__global__ void _kernel_doubleton_equations_batch_build_delta(long long* delta_keys,
                                                       double* delta_vals,
                                                       double* row_shift,
                                                       const NnzOffset* entry_starts,
                                                       const std::int32_t* selected_rows,
                                                       const std::int32_t* candidate_keep_col,
                                                       const std::int32_t* candidate_elim_col,
                                                       const double* candidate_keep_val,
                                                       const double* candidate_elim_val,
                                                       const double* AU_old,
                                                       const NnzOffset* AT_row_ptr,
                                                       const std::int32_t* AT_col_val,
                                                       const double* AT_nz_val,
                                                       const std::uint8_t* keep_row_mask,
                                                       double zero_tol,
                                                       std::int32_t n,
                                                       std::int32_t k) {
  const std::int32_t t = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (t >= k) {
    return;
  }
  const std::int32_t row = selected_rows[t];
  const std::int32_t keep = candidate_keep_col[row];
  const std::int32_t elim = candidate_elim_col[row];
  const double alpha = -candidate_keep_val[row] / candidate_elim_val[row];
  const double beta = AU_old[row] / candidate_elim_val[row];
  NnzOffset write = entry_starts[t];
  for (NnzOffset p = AT_row_ptr[elim]; p < AT_row_ptr[elim + 1]; ++p) {
    const std::int32_t support_row = AT_col_val[p];
    const double a = AT_nz_val[p];
    if (support_row != row && keep_row_mask[support_row] != std::uint8_t{0} && fabs(a) > zero_tol) {
      delta_keys[write] = static_cast<long long>(support_row) * static_cast<long long>(n) +
                          static_cast<long long>(keep);
      delta_vals[write] = alpha * a;
      atomicAdd(&row_shift[support_row], a * beta);
      ++write;
    }
  }
}

__global__ void _kernel_delta_row_counts(NnzOffset* row_counts,
                                         const long long* delta_keys,
                                         NnzOffset unique_count,
                                         std::int32_t n) {
  const std::int32_t t = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (t < unique_count) {
    const std::int32_t row = static_cast<std::int32_t>(delta_keys[t] / static_cast<long long>(n));
    atomicAdd(&row_counts[row], 1);
  }
}

__global__ void _kernel_doubleton_equations_merged_row_counts(NnzOffset* row_nnz_new,
                                                       const NnzOffset* row_ptr,
                                                       const std::int32_t* col_val,
                                                       const double* nz_val,
                                                       const NnzOffset* delta_row_ptr,
                                                       const long long* delta_keys,
                                                       const double* delta_vals,
                                                       double zero_tol,
                                                       std::int32_t m,
                                                       std::int32_t n, int long_threshold = 0,
                                                       int* long_rows = nullptr, int* long_count = nullptr) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }
  const NnzOffset p_start = row_ptr[row];
  const NnzOffset p_stop = row_ptr[row + 1];
  const NnzOffset q_start = delta_row_ptr[row];
  const NnzOffset q_stop = delta_row_ptr[row + 1];
  if (static_cast<std::int64_t>(p_stop - p_start) + (q_stop - q_start) >= long_threshold && long_rows != nullptr) {
    long_rows[atomicAdd(long_count, 1)] = row;
    return;
  }
  std::int32_t count = 0;
  NnzOffset p = p_start;
  NnzOffset q = q_start;
  while (p < p_stop || q < q_stop) {
    const std::int32_t source_col = (p < p_stop) ? col_val[p] : INT_MAX;
    const std::int32_t delta_col =
        (q < q_stop) ? static_cast<std::int32_t>(delta_keys[q] % static_cast<long long>(n)) : INT_MAX;
    const std::int32_t next_col = source_col < delta_col ? source_col : delta_col;
    double merged = 0.0;
    while (p < p_stop && col_val[p] == next_col) {
      merged += nz_val[p];
      ++p;
    }
    while (q < q_stop &&
           static_cast<std::int32_t>(delta_keys[q] % static_cast<long long>(n)) == next_col) {
      merged += delta_vals[q];
      ++q;
    }
    if (fabs(merged) > zero_tol) {
      ++count;
    }
  }
  row_nnz_new[row] = count;
}

__global__ void _kernel_doubleton_equations_copy_merged_rows(std::int32_t* col_val_new,
                                                      double* nz_val_new,
                                                      const NnzOffset* row_ptr_new,
                                                      const NnzOffset* row_ptr,
                                                      const std::int32_t* col_val,
                                                      const double* nz_val,
                                                      const NnzOffset* delta_row_ptr,
                                                      const long long* delta_keys,
                                                      const double* delta_vals,
                                                      double zero_tol,
                                                      std::int32_t m,
                                                      std::int32_t n, int long_threshold = 0) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }
  const NnzOffset p_start = row_ptr[row];
  const NnzOffset p_stop = row_ptr[row + 1];
  const NnzOffset q_start = delta_row_ptr[row];
  const NnzOffset q_stop = delta_row_ptr[row + 1];
  if (long_threshold > 0 && static_cast<std::int64_t>(p_stop - p_start) + (q_stop - q_start) >= long_threshold) return;
  NnzOffset write = row_ptr_new[row];
  NnzOffset p = p_start;
  NnzOffset q = q_start;
  while (p < p_stop || q < q_stop) {
    const std::int32_t source_col = (p < p_stop) ? col_val[p] : INT_MAX;
    const std::int32_t delta_col =
        (q < q_stop) ? static_cast<std::int32_t>(delta_keys[q] % static_cast<long long>(n)) : INT_MAX;
    const std::int32_t next_col = source_col < delta_col ? source_col : delta_col;
    double merged = 0.0;
    while (p < p_stop && col_val[p] == next_col) {
      merged += nz_val[p];
      ++p;
    }
    while (q < q_stop &&
           static_cast<std::int32_t>(delta_keys[q] % static_cast<long long>(n)) == next_col) {
      merged += delta_vals[q];
      ++q;
    }
    if (fabs(merged) > zero_tol) {
      col_val_new[write] = next_col;
      nz_val_new[write] = merged;
      ++write;
    }
  }
}

DeviceCsrMatrix build_doubleton_batch_new_A(const DeviceCsrMatrix& source,
                                            const DeviceCsrMatrix& AT_source,
                                            const PresolvePlanGpu& plan,
                                            const std::int32_t* selected_rows,
                                            const std::int32_t* candidate_keep_col,
                                            const std::int32_t* candidate_elim_col,
                                            const double* candidate_keep_val,
                                            const double* candidate_elim_val,
                                            double* row_shift,
                                            double zero_tol,
                                            std::int32_t selected_count,
                                            bool* matrix_changed) {
  *matrix_changed = false;
  DeviceCsrMatrix out;
  if (selected_count <= 0) {
    return out;
  }
  const int blocks_selected = (selected_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const int blocks_rows = (source.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  NnzOffset* entry_counts = nullptr;
  NnzOffset* entry_prefix = nullptr;
  NnzOffset* entry_starts = nullptr;
  long long* delta_keys = nullptr;
  double* delta_vals = nullptr;
  NnzOffset* delta_row_counts = nullptr;
  NnzOffset* delta_row_ptr = nullptr;
  NnzOffset* row_counts_new = nullptr;
  // Keep the serial implementation available as a numerical/performance oracle.
  const bool parallel_long_rows = runtime::options().doubleton_long_rows;
  int* long_rows = nullptr;
  int* long_count = nullptr;
  NnzOffset* segment_counts = nullptr;


  throw_if_cuda_error(doubleton_cuda_malloc(&entry_counts, sizeof(NnzOffset) * static_cast<std::size_t>(selected_count)),
                      "cudaMalloc doubleton batch entry_counts");
  throw_if_cuda_error(doubleton_cuda_malloc(&entry_prefix, sizeof(NnzOffset) * static_cast<std::size_t>(selected_count)),
                      "cudaMalloc doubleton batch entry_prefix");
  _kernel_doubleton_equations_batch_entry_counts<<<blocks_selected, GPU_PRESOLVE_THREADS>>>(
      entry_counts, selected_rows, candidate_elim_col, AT_source.rowPtr, AT_source.colVal, AT_source.nzVal,
      plan.keep_row_mask, zero_tol, selected_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_equations_batch_entry_counts");
  throw_if_cuda_error(cudaMemcpy(entry_prefix, entry_counts,
                                 sizeof(NnzOffset) * static_cast<std::size_t>(selected_count),
                                 cudaMemcpyDeviceToDevice),
                      "cudaMemcpy doubleton batch entry_prefix");
  inclusive_scan_i32(entry_prefix, selected_count, "cub inclusive scan doubleton batch entries");
  NnzOffset total_entries = 0;
  throw_if_cuda_error(cudaMemcpy(&total_entries, entry_prefix + selected_count - 1, sizeof(NnzOffset),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy doubleton batch total_entries");
  if (total_entries <= 0) {
    doubleton_cuda_free(entry_counts);
    doubleton_cuda_free(entry_prefix);
    return out;
  }

  // The prefix-to-row-pointer kernel also writes the terminal sentinel.
  throw_if_cuda_error(doubleton_cuda_malloc(&entry_starts, sizeof(NnzOffset) * (static_cast<std::size_t>(selected_count) + 1)),
                      "cudaMalloc doubleton batch entry_starts");
  _kernel_row_ptr_from_prefix<<<blocks_selected, GPU_PRESOLVE_THREADS>>>(entry_starts, entry_prefix, selected_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_row_ptr_from_prefix batch entry starts");
  throw_if_cuda_error(doubleton_cuda_malloc(&delta_keys, sizeof(long long) * static_cast<std::size_t>(total_entries)),
                      "cudaMalloc doubleton batch delta_keys");
  throw_if_cuda_error(doubleton_cuda_malloc(&delta_vals, sizeof(double) * static_cast<std::size_t>(total_entries)),
                      "cudaMalloc doubleton batch delta_vals");
  _kernel_doubleton_equations_batch_build_delta<<<blocks_selected, GPU_PRESOLVE_THREADS>>>(
      delta_keys, delta_vals, row_shift, entry_starts, selected_rows, candidate_keep_col, candidate_elim_col,
      candidate_keep_val, candidate_elim_val, plan.new_AU, AT_source.rowPtr, AT_source.colVal, AT_source.nzVal,
      plan.keep_row_mask, zero_tol, source.cols, selected_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_equations_batch_build_delta");
  // Keep the deterministic row-major ordering used by the paper path.  This
  // preserves the floating-point merge order for duplicate (row, col) deltas.
  thrust::sort_by_key(doubleton_thrust_policy(),
                      thrust::device_pointer_cast(delta_keys),
                      thrust::device_pointer_cast(delta_keys + total_entries),
                      thrust::device_pointer_cast(delta_vals));
  throw_if_cuda_error(cudaGetLastError(), "thrust sort doubleton delta keys");

  throw_if_cuda_error(doubleton_cuda_malloc(&delta_row_counts, sizeof(NnzOffset) * static_cast<std::size_t>(source.rows)),
                      "cudaMalloc doubleton batch delta_row_counts");
  throw_if_cuda_error(cudaMemset(delta_row_counts, 0, sizeof(NnzOffset) * static_cast<std::size_t>(source.rows)),
                      "cudaMemset doubleton batch delta_row_counts");
  const int blocks_delta = (total_entries + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_delta_row_counts<<<blocks_delta, GPU_PRESOLVE_THREADS>>>(
      delta_row_counts, delta_keys, total_entries, source.cols);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_delta_row_counts");
  inclusive_scan_i32(delta_row_counts, source.rows, "cub inclusive scan doubleton delta rows");
  throw_if_cuda_error(doubleton_cuda_malloc(&delta_row_ptr, sizeof(NnzOffset) * static_cast<std::size_t>(source.rows + 1)),
                      "cudaMalloc doubleton batch delta_row_ptr");
  _kernel_row_ptr_from_prefix<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(delta_row_ptr, delta_row_counts, source.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_row_ptr_from_prefix delta rows");
  throw_if_cuda_error(doubleton_cuda_malloc(&row_counts_new, sizeof(NnzOffset) * static_cast<std::size_t>(source.rows)),
                      "cudaMalloc doubleton batch row_counts_new");
  const int long_threshold = parallel_long_rows ? detail::DOUBLETON_LONG_ROW_THRESHOLD : 0;
  const int long_capacity = parallel_long_rows ? static_cast<int>(std::min<std::int64_t>(
      source.rows, (static_cast<std::int64_t>(source.nnz) + total_entries) / long_threshold)) : 0;
  const int long_blocks = std::min(long_capacity, 256);
  if (long_capacity > 0) {
    throw_if_cuda_error(doubleton_cuda_malloc(&long_rows, sizeof(int) * long_capacity), "doubleton long rows");
    throw_if_cuda_error(doubleton_cuda_malloc(&long_count, sizeof(int)), "doubleton long count");
    throw_if_cuda_error(cudaMemsetAsync(long_count, 0, sizeof(int)), "doubleton reset long count");
    throw_if_cuda_error(doubleton_cuda_malloc(&segment_counts,
        sizeof(NnzOffset) * static_cast<std::size_t>(long_capacity) * detail::DOUBLETON_MERGE_THREADS),
        "doubleton segment counts");
  }
  _kernel_doubleton_equations_merged_row_counts<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
      row_counts_new, source.rowPtr, source.colVal, source.nzVal, delta_row_ptr, delta_keys, delta_vals,
      zero_tol, source.rows, source.cols, long_threshold, long_rows, long_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_equations_merged_row_counts");
  if (long_blocks > 0) {
    detail::doubleton_merge_long_rows<false><<<long_blocks, detail::DOUBLETON_MERGE_THREADS>>>(
        source.rowPtr, source.colVal, source.nzVal, delta_row_ptr, delta_keys, delta_vals,
        long_rows, long_count, source.cols, zero_tol, row_counts_new, segment_counts);
    throw_if_cuda_error(cudaGetLastError(), "doubleton count long rows");
  }
  inclusive_scan_i32(row_counts_new, source.rows, "cub inclusive scan doubleton merged rows");
  NnzOffset nnz_new = 0;
  throw_if_cuda_error(cudaMemcpy(&nnz_new, row_counts_new + source.rows - 1, sizeof(NnzOffset),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy doubleton batch nnz_new");

  out.rows = source.rows;
  out.cols = source.cols;
  out.nnz = nnz_new;
  throw_if_cuda_error(doubleton_cuda_malloc(&out.rowPtr, sizeof(NnzOffset) * static_cast<std::size_t>(source.rows + 1)),
                      "cudaMalloc doubleton batch rowPtr_new");
  _kernel_row_ptr_from_prefix<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(out.rowPtr, row_counts_new, source.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_row_ptr_from_prefix merged rows");
  if (nnz_new > 0) {
    throw_if_cuda_error(doubleton_cuda_malloc(&out.colVal, sizeof(std::int32_t) * static_cast<std::size_t>(nnz_new)),
                        "cudaMalloc doubleton batch colVal_new");
    throw_if_cuda_error(doubleton_cuda_malloc(&out.nzVal, sizeof(double) * static_cast<std::size_t>(nnz_new)),
                        "cudaMalloc doubleton batch nzVal_new");
    _kernel_doubleton_equations_copy_merged_rows<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
        out.colVal, out.nzVal, out.rowPtr, source.rowPtr, source.colVal, source.nzVal, delta_row_ptr,
        delta_keys, delta_vals, zero_tol, source.rows, source.cols, long_threshold);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_equations_copy_merged_rows");
    if (long_blocks > 0) {
      detail::doubleton_merge_long_rows<true><<<long_blocks, detail::DOUBLETON_MERGE_THREADS>>>(
          source.rowPtr, source.colVal, source.nzVal, delta_row_ptr, delta_keys, delta_vals,
          long_rows, long_count, source.cols, zero_tol, nullptr, segment_counts,
          out.rowPtr, out.colVal, out.nzVal);
      throw_if_cuda_error(cudaGetLastError(), "doubleton copy long rows");
    }
  }
  *matrix_changed = true;
  doubleton_cuda_free(long_rows); doubleton_cuda_free(long_count); doubleton_cuda_free(segment_counts);

  doubleton_cuda_free(entry_counts); doubleton_cuda_free(entry_prefix); doubleton_cuda_free(entry_starts); doubleton_cuda_free(delta_keys); doubleton_cuda_free(delta_vals);
  doubleton_cuda_free(delta_row_counts); doubleton_cuda_free(delta_row_ptr); doubleton_cuda_free(row_counts_new);
  return out;
}

void count_mask_to_device(std::uint8_t* mask,
                          std::int32_t n,
                          std::int32_t* prefix,
                          std::int32_t* count_device) {
  if (n == 0) {
    throw_if_cuda_error(cudaMemsetAsync(count_device, 0, sizeof(std::int32_t), nullptr),
                        "cudaMemset empty mask count");
    return;
  }
  const int blocks = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_u8_to_i32<<<blocks, GPU_PRESOLVE_THREADS>>>(prefix, mask, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_u8_to_i32");
  inclusive_scan_i32(prefix, n, "cub inclusive scan doubleton mask count");
  throw_if_cuda_error(cudaMemcpyAsync(count_device,
                                      prefix + n - 1,
                                      sizeof(std::int32_t),
                                      cudaMemcpyDeviceToDevice,
                                      nullptr),
                      "cudaMemcpy mask count to batch summary");
}

bool apply_rule_doubleton_equations_batch_once(PresolvePlanGpu& plan,
                                        const LPInfoGpu& lp,
                                        const DeviceCsrMatrix& AT_source,
                                        const PresolveParams& pparams,
                                        DoubletonBatchTapeBuffer* tape_buffer,
                                        double* obj_delta_device,
                                        double* fixed_obj_delta_device) {
  const std::int32_t m = lp.A.rows;
  const std::int32_t n = lp.A.cols;
  if (m == 0 || n == 0) {
    return false;
  }
  const DeviceCsrMatrix& A_source = plan.has_new_A ? plan.new_A : lp.A;
  const int blocks_rows = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const int blocks_cols = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const double ratio_tol = fmax(1.0e-9, pparams.bound_tol);
  const bool profile_batch = runtime::options().profile_doubleton;

  std::int32_t* live_col_nnz = nullptr;
  std::uint8_t* candidate_mask = nullptr;
  std::uint8_t* acceptable_mask = nullptr;
  std::uint8_t* active_mask = nullptr;
  std::uint8_t* selected_mask = nullptr;
  std::uint8_t* blocked_col = nullptr;
  std::int32_t* candidate_elim_col = nullptr;
  std::int32_t* candidate_keep_col = nullptr;
  double* candidate_elim_val = nullptr;
  double* candidate_keep_val = nullptr;
  std::int32_t* col_owner = nullptr;
  std::int32_t* prefix = nullptr;
  std::int32_t* selected_rows = nullptr;
  std::int32_t* subst_rows = nullptr;
  DoubletonBatchTapeRecord* selected_tape_records = nullptr;
  std::int32_t* status = nullptr;
  std::uint8_t* subst_pair_mask = nullptr;
  std::uint8_t* fixed_pair_mask = nullptr;
  double* row_shift = nullptr;

  throw_if_cuda_error(doubleton_cuda_malloc(&live_col_nnz, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      "cudaMalloc batch live_col_nnz");
  throw_if_cuda_error(doubleton_cuda_malloc(&candidate_mask, static_cast<std::size_t>(m)), "cudaMalloc candidate_mask");
  throw_if_cuda_error(doubleton_cuda_malloc(&acceptable_mask, static_cast<std::size_t>(m)), "cudaMalloc acceptable_mask");
  throw_if_cuda_error(doubleton_cuda_malloc(&active_mask, static_cast<std::size_t>(m)), "cudaMalloc active_mask");
  throw_if_cuda_error(doubleton_cuda_malloc(&selected_mask, static_cast<std::size_t>(m)), "cudaMalloc selected_mask");
  throw_if_cuda_error(doubleton_cuda_malloc(&blocked_col, static_cast<std::size_t>(n)), "cudaMalloc blocked_col");
  throw_if_cuda_error(doubleton_cuda_malloc(&candidate_elim_col, sizeof(std::int32_t) * static_cast<std::size_t>(m)),
                      "cudaMalloc candidate_elim_col");
  throw_if_cuda_error(doubleton_cuda_malloc(&candidate_keep_col, sizeof(std::int32_t) * static_cast<std::size_t>(m)),
                      "cudaMalloc candidate_keep_col");
  throw_if_cuda_error(doubleton_cuda_malloc(&candidate_elim_val, sizeof(double) * static_cast<std::size_t>(m)),
                      "cudaMalloc candidate_elim_val");
  throw_if_cuda_error(doubleton_cuda_malloc(&candidate_keep_val, sizeof(double) * static_cast<std::size_t>(m)),
                      "cudaMalloc candidate_keep_val");
  throw_if_cuda_error(doubleton_cuda_malloc(&col_owner, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      "cudaMalloc col_owner");
  throw_if_cuda_error(doubleton_cuda_malloc(&prefix, sizeof(std::int32_t) * static_cast<std::size_t>(m)),
                      "cudaMalloc batch prefix");
  throw_if_cuda_error(doubleton_cuda_malloc(&status, sizeof(std::int32_t) * 4), "cudaMalloc batch status");
  throw_if_cuda_error(doubleton_cuda_malloc(&subst_pair_mask, static_cast<std::size_t>(m)), "cudaMalloc subst_pair_mask");
  throw_if_cuda_error(doubleton_cuda_malloc(&fixed_pair_mask, static_cast<std::size_t>(m)), "cudaMalloc fixed_pair_mask");
  throw_if_cuda_error(doubleton_cuda_malloc(&row_shift, sizeof(double) * static_cast<std::size_t>(m)),
                      "cudaMalloc batch row_shift");
  throw_if_cuda_error(cudaMemset(candidate_mask, 0, static_cast<std::size_t>(m)), "cudaMemset candidate_mask");
  throw_if_cuda_error(cudaMemset(acceptable_mask, 0, static_cast<std::size_t>(m)), "cudaMemset acceptable_mask");
  throw_if_cuda_error(cudaMemset(selected_mask, 0, static_cast<std::size_t>(m)), "cudaMemset selected_mask");
  throw_if_cuda_error(cudaMemset(blocked_col, 0, static_cast<std::size_t>(n)), "cudaMemset blocked_col");
  throw_if_cuda_error(cudaMemset(status, 0, sizeof(std::int32_t) * 4), "cudaMemset batch status");
  throw_if_cuda_error(cudaMemset(subst_pair_mask, 0, static_cast<std::size_t>(m)), "cudaMemset subst_pair_mask");
  throw_if_cuda_error(cudaMemset(fixed_pair_mask, 0, static_cast<std::size_t>(m)), "cudaMemset fixed_pair_mask");
  throw_if_cuda_error(cudaMemset(row_shift, 0, sizeof(double) * static_cast<std::size_t>(m)), "cudaMemset row_shift");

  _kernel_doubleton_equations_live_col_nnz<<<blocks_cols, GPU_PRESOLVE_THREADS>>>(
      live_col_nnz, AT_source.rowPtr, AT_source.colVal, AT_source.nzVal,
      plan.keep_row_mask, plan.keep_col_mask, pparams.zero_tol, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_equations_live_col_nnz batch");

  _kernel_doubleton_equations_candidates<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
      candidate_mask, candidate_elim_col, candidate_elim_val, candidate_keep_col, candidate_keep_val,
      A_source.rowPtr, A_source.colVal, A_source.nzVal, plan.keep_row_mask, plan.keep_col_mask,
      live_col_nnz, plan.new_AL, plan.new_AU, pparams.zero_tol, pparams.bound_tol, ratio_tol, m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_equations_candidates");

  _kernel_mark_doubleton_equations_batch_acceptable<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
      acceptable_mask, candidate_mask, candidate_keep_col, candidate_elim_col, live_col_nnz,
      AT_source.rowPtr, AT_source.colVal, AT_source.nzVal, plan.keep_row_mask, pparams.zero_tol,
      static_cast<std::int32_t>(pparams.doubleton_equations_max_fill_in_proxy), m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_mark_doubleton_equations_batch_acceptable");

  count_mask_to_device(acceptable_mask, m, prefix, status);

  throw_if_cuda_error(cudaMemcpy(active_mask, acceptable_mask, static_cast<std::size_t>(m), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy active_mask");
  for (int round = 0; round < DOUBLETON_EQUATION_BATCH_MATCHING_ROUNDS; ++round) {
    _kernel_fill_i32<<<blocks_cols, GPU_PRESOLVE_THREADS>>>(col_owner, INT_MAX, n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_fill_i32 col_owner");
    _kernel_claim_doubleton_equations_batch_columns<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
        col_owner, active_mask, blocked_col, candidate_keep_col, candidate_elim_col, m);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_claim_doubleton_equations_batch_columns");
    _kernel_select_doubleton_equations_batch_rows<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
        selected_mask, active_mask, blocked_col, candidate_keep_col, candidate_elim_col, col_owner, m);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_select_doubleton_equations_batch_rows");
  }

  count_mask_to_device(selected_mask, m, prefix, status + 1);
  std::int32_t selection_counts[2] = {0, 0};
  throw_if_cuda_error(cudaMemcpy(selection_counts,
                                 status,
                                 sizeof(selection_counts),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy doubleton batch selection counts");
  const std::int32_t acceptable_count = selection_counts[0];
  const std::int32_t selected_count = selection_counts[1];
  const double selected_ratio =
      acceptable_count == 0 ? 0.0 : static_cast<double>(selected_count) / static_cast<double>(acceptable_count);
  if (profile_batch) {
    std::cout << ">>> [doubleton_equations batch C++] acceptable=" << acceptable_count
              << " selected=" << selected_count
              << " ratio=" << selected_ratio << "\n";
  }
  if (selected_count == 0 ||
      (pparams.doubleton_equations_scan &&
       (selected_count < pparams.doubleton_equations_min_selected_per_batch ||
        selected_ratio < pparams.doubleton_equations_min_selected_ratio))) {
    doubleton_cuda_free(live_col_nnz); doubleton_cuda_free(candidate_mask); doubleton_cuda_free(acceptable_mask); doubleton_cuda_free(active_mask);
    doubleton_cuda_free(selected_mask); doubleton_cuda_free(blocked_col); doubleton_cuda_free(candidate_elim_col); doubleton_cuda_free(candidate_keep_col);
    doubleton_cuda_free(candidate_elim_val); doubleton_cuda_free(candidate_keep_val); doubleton_cuda_free(col_owner); doubleton_cuda_free(prefix);
    doubleton_cuda_free(status); doubleton_cuda_free(subst_pair_mask); doubleton_cuda_free(fixed_pair_mask);
    doubleton_cuda_free(row_shift);
    return false;
  }

  throw_if_cuda_error(cudaMemsetAsync(status, 0, sizeof(std::int32_t) * 4, nullptr),
                      "cudaMemset batch outcome status");

  throw_if_cuda_error(doubleton_cuda_malloc(&selected_rows, sizeof(std::int32_t) * static_cast<std::size_t>(selected_count)),
                      "cudaMalloc selected_rows");
  _kernel_scatter_mask_indices<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(selected_rows, selected_mask, prefix, m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_scatter_mask_indices selected");

  const int blocks_selected = (selected_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  if (pparams.record_postsolve_tape) {
    throw_if_cuda_error(
        doubleton_cuda_malloc(&selected_tape_records,
                   sizeof(DoubletonBatchTapeRecord) * static_cast<std::size_t>(selected_count)),
        "cudaMalloc doubleton batch selected tape records");
    _kernel_capture_doubleton_batch_tape<<<blocks_selected, GPU_PRESOLVE_THREADS>>>(
        selected_tape_records,
        selected_rows,
        candidate_keep_col,
        candidate_elim_col,
        candidate_keep_val,
        candidate_elim_val,
        plan.new_l,
        plan.new_u,
        plan.new_c,
        plan.new_AU,
        selected_count);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_capture_doubleton_batch_tape");
  }
  _kernel_apply_doubleton_equations_batch<<<blocks_selected, GPU_PRESOLVE_THREADS>>>(
      status, plan.new_l, plan.new_u, plan.new_c, plan.new_AL, plan.new_AU, plan.keep_row_mask,
      plan.keep_col_mask, subst_pair_mask, fixed_pair_mask, row_shift, obj_delta_device,
      fixed_obj_delta_device, selected_rows, candidate_keep_col, candidate_elim_col,
      candidate_keep_val, candidate_elim_val, AT_source.rowPtr, AT_source.colVal, AT_source.nzVal,
      plan.new_l, plan.new_u, plan.new_c, plan.new_AU, pparams.bound_tol, selected_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_apply_doubleton_equations_batch");
  // Produce all post-apply scalar decisions on the device, then cross the
  // host boundary once.  Keep the substitution prefix last because it is
  // consumed by the scatter and tape-pack kernels below.
  count_mask_to_device(fixed_pair_mask, selected_count, prefix, status + 3);
  count_mask_to_device(subst_pair_mask, selected_count, prefix, status + 2);
  std::int32_t status_host[4] = {0, 0, 0, 0};
  throw_if_cuda_error(cudaMemcpy(status_host, status, sizeof(status_host), cudaMemcpyDeviceToHost),
                      "cudaMemcpy doubleton batch outcome summary");
  bool changed = false;
  if (status_host[0] != 0) {
    plan.has_infeasible = true;
  } else if (status_host[1] != 0) {
    const std::int32_t subst_count = status_host[2];
    if (subst_count > 0) {
      throw_if_cuda_error(doubleton_cuda_malloc(&subst_rows, sizeof(std::int32_t) * static_cast<std::size_t>(subst_count)),
                          "cudaMalloc subst_rows");
      _kernel_scatter_selected_rows_by_pair_mask<<<blocks_selected, GPU_PRESOLVE_THREADS>>>(
          subst_rows, selected_rows, subst_pair_mask, prefix, selected_count);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_scatter_selected_rows_by_pair_mask");
      if (pparams.record_postsolve_tape) {
        if (tape_buffer == nullptr) {
          throw std::runtime_error("missing doubleton accumulated batch tape");
        }
        DoubletonBatchTapeRecord* packed_tape_records =
            tape_buffer->append_storage(subst_count);
        _kernel_pack_doubleton_batch_tape<<<blocks_selected, GPU_PRESOLVE_THREADS>>>(
            packed_tape_records,
            selected_tape_records,
            subst_pair_mask,
            prefix,
            plan.new_l,
            plan.new_u,
            selected_count);
        throw_if_cuda_error(cudaGetLastError(), "_kernel_pack_doubleton_batch_tape");
      }
    }
    const std::int32_t fixed_count = status_host[3];
    if (profile_batch) {
      std::cout << ">>> [doubleton_equations batch C++] subst=" << subst_count
                << " fixed=" << fixed_count << "\n";
    }
    if (fixed_count > 0) {
      _kernel_accumulate_doubleton_equations_fixed_row_shift<<<blocks_selected, GPU_PRESOLVE_THREADS>>>(
          row_shift, selected_rows, fixed_pair_mask, candidate_keep_col, plan.new_l, AT_source.rowPtr,
          AT_source.colVal, AT_source.nzVal, plan.keep_row_mask, pparams.zero_tol, selected_count);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_accumulate_doubleton_equations_fixed_row_shift");
    }
    const DeviceCsrMatrix source_A = plan.has_new_A ? plan.new_A : lp.A;
    bool matrix_changed = false;
    if (subst_count > 0) {
      DeviceCsrMatrix rewritten_A = build_doubleton_batch_new_A(
          source_A,
          AT_source,
          plan,
          subst_rows,
          candidate_keep_col,
          candidate_elim_col,
          candidate_keep_val,
          candidate_elim_val,
          row_shift,
          pparams.zero_tol,
          subst_count,
          &matrix_changed);
      if (matrix_changed) {
        if (plan.has_new_A) {
          doubleton_cuda_free(plan.new_A.rowPtr);
          doubleton_cuda_free(plan.new_A.colVal);
          doubleton_cuda_free(plan.new_A.nzVal);
        }
        plan.new_A = rewritten_A;
        plan.has_new_A = true;
      }
    }
    _kernel_apply_row_shift<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(plan.new_AL, plan.new_AU, row_shift, m);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_apply_row_shift batch");
    plan.has_row_action = subst_count > 0;
    plan.has_col_action = true;
    plan.has_change = true;
    changed = true;
  }

  doubleton_cuda_free(live_col_nnz); doubleton_cuda_free(candidate_mask); doubleton_cuda_free(acceptable_mask); doubleton_cuda_free(active_mask);
  doubleton_cuda_free(selected_mask); doubleton_cuda_free(blocked_col); doubleton_cuda_free(candidate_elim_col); doubleton_cuda_free(candidate_keep_col);
  doubleton_cuda_free(candidate_elim_val); doubleton_cuda_free(candidate_keep_val); doubleton_cuda_free(col_owner); doubleton_cuda_free(prefix);
  doubleton_cuda_free(selected_rows); doubleton_cuda_free(subst_rows); doubleton_cuda_free(status); doubleton_cuda_free(subst_pair_mask); doubleton_cuda_free(fixed_pair_mask);
  doubleton_cuda_free(selected_tape_records);
  doubleton_cuda_free(row_shift);
  return changed;
}

bool apply_rule_doubleton_equations_batch(PresolvePlanGpu& plan,
                                   const LPInfoGpu& lp,
                                   const PresolveParams& pparams) {
  DeviceCsrMatrix AT_source = lp.AT;
  sort_csr_rows_by_col_thrust(AT_source, "doubleton sort input AT rows");
  DoubletonBatchTapeBuffer tape_buffer;
  double* batch_obj_deltas_device = nullptr;
  throw_if_cuda_error(
      doubleton_cuda_malloc(&batch_obj_deltas_device, sizeof(double) * 2),
      "cudaMalloc doubleton batch objective deltas");
  throw_if_cuda_error(
      cudaMemsetAsync(batch_obj_deltas_device, 0, sizeof(double) * 2, nullptr),
      "cudaMemset doubleton batch objective deltas");

  // Process one independent batch. Later batches belong to later scheduler
  // executions; do not transpose the rewritten matrix for an unused next batch.
  const bool changed = apply_rule_doubleton_equations_batch_once(
      plan,
      lp,
      AT_source,
      pparams,
      pparams.record_postsolve_tape ? &tape_buffer : nullptr,
      batch_obj_deltas_device,
      batch_obj_deltas_device + 1);

  if (pparams.record_postsolve_tape && tape_buffer.count > 0) {
    if (plan.tape.resident) {
      PostsolveTapeGpu segment;
      build_doubleton_postsolve_tape_gpu(segment, tape_buffer.records, tape_buffer.count);
      append_device_recovery_segment(plan.tape, std::move(segment));
    } else {
      const std::vector<DoubletonBatchTapeRecord> tape_records = copy_device_vector(
          tape_buffer.records, tape_buffer.count, "cudaMemcpy doubleton batch tape");
      append_doubleton_batch_tape_records(plan.tape, tape_records);
    }
  }
  if (changed && !plan.has_infeasible && !plan.has_unbounded) {
    const std::vector<double> batch_obj_deltas =
        copy_device_vector(batch_obj_deltas_device, 2, "cudaMemcpy doubleton batch objective deltas");
    plan.obj_constant_delta += batch_obj_deltas[0];
    plan.obj_constant_delta += batch_obj_deltas[1];
  }
  doubleton_cuda_free(batch_obj_deltas_device);
  return changed;
}

__global__ void _kernel_copy_doubleton_equations_rows(std::int32_t* col_val_new,
                                               double* nz_val_new,
                                               const NnzOffset* row_ptr_new,
                                               const NnzOffset* row_ptr,
                                               const std::int32_t* col_val,
                                               const double* nz_val,
                                               const std::int32_t* meta_i32,
                                               const double* meta_f64,
                                               double zero_tol,
                                               std::int32_t m) {
  const std::int32_t r = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (r >= m) {
    return;
  }
  const std::int32_t target_row = meta_i32[0];
  const std::int32_t elim_col = meta_i32[1];
  const std::int32_t keep_col = meta_i32[2];
  const double alpha = -meta_f64[1] / meta_f64[0];

  const NnzOffset first = row_ptr[r];
  const NnzOffset last = row_ptr[r + 1];
  NnzOffset write = row_ptr_new[r];
  if (first >= last) {
    return;
  }
  if (r == target_row) {
    for (NnzOffset p = first; p < last; ++p) {
      col_val_new[write] = col_val[p];
      nz_val_new[write] = nz_val[p];
      ++write;
    }
    return;
  }

  bool elim_present = false;
  bool keep_present = false;
  double are = 0.0;
  double old_keep = 0.0;
  for (NnzOffset p = first; p < last; ++p) {
    const std::int32_t col = col_val[p];
    const double val = nz_val[p];
    if (col == elim_col) {
      elim_present = true;
      are = val;
    } else if (col == keep_col) {
      keep_present = true;
      old_keep = val;
    }
  }

  const double new_keep = old_keep + alpha * are;
  const bool insert_keep = elim_present && !keep_present && fabs(new_keep) > zero_tol;
  bool inserted = !insert_keep;
  for (NnzOffset p = first; p < last; ++p) {
    const std::int32_t col = col_val[p];
    const double val = nz_val[p];
    if (insert_keep && !inserted && col > keep_col) {
      col_val_new[write] = keep_col;
      nz_val_new[write] = new_keep;
      ++write;
      inserted = true;
    }
    if (col == keep_col) {
      if (fabs(new_keep) > zero_tol) {
        col_val_new[write] = keep_col;
        nz_val_new[write] = new_keep;
        ++write;
      }
    } else {
      col_val_new[write] = col;
      nz_val_new[write] = val;
      ++write;
    }
  }
  if (insert_keep && !inserted) {
    col_val_new[write] = keep_col;
    nz_val_new[write] = new_keep;
  }
}

DeviceCsrMatrix build_doubleton_scalar_new_A(const DeviceCsrMatrix& source,
                                             const std::int32_t* meta_i32,
                                             const double* meta_f64,
                                             double zero_tol) {
  DeviceCsrMatrix out;
  out.rows = source.rows;
  out.cols = source.cols;
  if (source.rows == 0) {
    return out;
  }

  NnzOffset* row_counts = nullptr;
  throw_if_cuda_error(doubleton_cuda_malloc(&row_counts, sizeof(NnzOffset) * static_cast<std::size_t>(source.rows)),
                      "cudaMalloc doubleton row_counts");
  const int blocks = (source.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_doubleton_equations_row_counts<<<blocks, GPU_PRESOLVE_THREADS>>>(
      row_counts, source.rowPtr, source.colVal, source.nzVal, meta_i32, meta_f64, zero_tol, source.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_equations_row_counts");

  inclusive_scan_i32(row_counts, source.rows, "cub inclusive scan doubleton scalar rows");

  NnzOffset nnz_new = 0;
  throw_if_cuda_error(cudaMemcpy(&nnz_new, row_counts + source.rows - 1, sizeof(NnzOffset),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy doubleton nnz_new");
  out.nnz = nnz_new;
  throw_if_cuda_error(doubleton_cuda_malloc(&out.rowPtr, sizeof(NnzOffset) * static_cast<std::size_t>(source.rows + 1)),
                      "cudaMalloc doubleton rowPtr_new");
  _kernel_row_ptr_from_prefix<<<blocks, GPU_PRESOLVE_THREADS>>>(out.rowPtr, row_counts, source.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_row_ptr_from_prefix doubleton");
  if (nnz_new > 0) {
    throw_if_cuda_error(doubleton_cuda_malloc(&out.colVal, sizeof(std::int32_t) * static_cast<std::size_t>(nnz_new)),
                        "cudaMalloc doubleton colVal_new");
    throw_if_cuda_error(doubleton_cuda_malloc(&out.nzVal, sizeof(double) * static_cast<std::size_t>(nnz_new)),
                        "cudaMalloc doubleton nzVal_new");
    _kernel_copy_doubleton_equations_rows<<<blocks, GPU_PRESOLVE_THREADS>>>(
        out.colVal,
        out.nzVal,
        out.rowPtr,
        source.rowPtr,
        source.colVal,
        source.nzVal,
        meta_i32,
        meta_f64,
        zero_tol,
        source.rows);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_copy_doubleton_equations_rows");
  }
  doubleton_cuda_free(row_counts);
  return out;
}

}  // namespace

void apply_rule_doubleton_equations(PresolvePlanGpu& plan,
                             const LPInfoGpu& lp,
                             const PresolveStatsGpu& stats,
                             const PresolveParams& pparams) {
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }
  if (pparams.doubleton_equations_scan) {
    (void)apply_rule_doubleton_equations_batch(plan, lp, pparams);
    return;
  }
  const std::int32_t m = lp.A.rows;
  const std::int32_t n = lp.A.cols;
  if (m == 0 || n == 0) {
    return;
  }

  std::int32_t* live_col_nnz = nullptr;
  std::int32_t* target_row_ref = nullptr;
  std::int32_t* meta_i32 = nullptr;
  double* meta_f64 = nullptr;
  std::int32_t* status = nullptr;
  double* obj_delta_device = nullptr;
  throw_if_cuda_error(doubleton_cuda_malloc(&live_col_nnz, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      "cudaMalloc doubleton live_col_nnz");
  throw_if_cuda_error(doubleton_cuda_malloc(&target_row_ref, sizeof(std::int32_t)), "cudaMalloc doubleton target_row_ref");
  throw_if_cuda_error(doubleton_cuda_malloc(&meta_i32, sizeof(std::int32_t) * 3), "cudaMalloc doubleton meta_i32");
  throw_if_cuda_error(doubleton_cuda_malloc(&meta_f64, sizeof(double) * 8), "cudaMalloc doubleton meta_f64");
  throw_if_cuda_error(doubleton_cuda_malloc(&status, sizeof(std::int32_t) * 2), "cudaMalloc doubleton status");
  throw_if_cuda_error(doubleton_cuda_malloc(&obj_delta_device, sizeof(double)), "cudaMalloc doubleton obj_delta");
  throw_if_cuda_error(cudaMemset(status, 0, sizeof(std::int32_t) * 2), "cudaMemset doubleton status");
  throw_if_cuda_error(cudaMemset(obj_delta_device, 0, sizeof(double)), "cudaMemset doubleton obj_delta");

  const int blocks_cols = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_doubleton_equations_live_col_nnz<<<blocks_cols, GPU_PRESOLVE_THREADS>>>(
      live_col_nnz,
      lp.AT.rowPtr,
      lp.AT.colVal,
      lp.AT.nzVal,
      plan.keep_row_mask,
      plan.keep_col_mask,
      pparams.zero_tol,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_equations_live_col_nnz");

  const std::int32_t target_init = m;
  throw_if_cuda_error(cudaMemcpy(target_row_ref, &target_init, sizeof(std::int32_t), cudaMemcpyHostToDevice),
                      "cudaMemcpy doubleton target init");
  const int blocks_rows = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const double ratio_tol = fmax(1.0e-9, pparams.bound_tol);
  _kernel_select_doubleton_equations_target_row_direct<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
      target_row_ref,
      meta_i32,
      meta_f64,
      (plan.has_new_A ? plan.new_A : lp.A).rowPtr,
      (plan.has_new_A ? plan.new_A : lp.A).colVal,
      (plan.has_new_A ? plan.new_A : lp.A).nzVal,
      plan.keep_row_mask,
      plan.keep_col_mask,
      live_col_nnz,
      plan.new_AL,
      plan.new_AU,
      plan.new_l,
      plan.new_u,
      plan.new_c,
      lp.AT.rowPtr,
      lp.AT.colVal,
      pparams.zero_tol,
      pparams.bound_tol,
      ratio_tol,
      static_cast<std::int32_t>(pparams.doubleton_equations_max_fill_in_proxy),
      m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_select_doubleton_equations_target_row_direct");

  std::int32_t target_row = m;
  throw_if_cuda_error(cudaMemcpy(&target_row, target_row_ref, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy doubleton target");
  if (target_row >= m) {
    doubleton_cuda_free(live_col_nnz);
    doubleton_cuda_free(target_row_ref);
    doubleton_cuda_free(meta_i32);
    doubleton_cuda_free(meta_f64);
    doubleton_cuda_free(status);
    doubleton_cuda_free(obj_delta_device);
    return;
  }

  _kernel_doubleton_equations_apply<<<blocks_rows, GPU_PRESOLVE_THREADS>>>(
      status,
      plan.new_l,
      plan.new_u,
      plan.new_c,
      plan.new_AL,
      plan.new_AU,
      plan.keep_row_mask,
      plan.keep_col_mask,
      obj_delta_device,
      meta_i32,
      meta_f64,
      lp.AT.rowPtr,
      lp.AT.colVal,
      lp.AT.nzVal,
      pparams.bound_tol,
      m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_doubleton_equations_apply");
  throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_doubleton_equations synchronize");

  std::int32_t status_host[2] = {0, 0};
  throw_if_cuda_error(cudaMemcpy(status_host, status, sizeof(status_host), cudaMemcpyDeviceToHost),
                      "cudaMemcpy doubleton status");
  if (status_host[0] != 0) {
    plan.has_infeasible = true;
  } else if (status_host[1] != 0) {
    if (pparams.record_postsolve_tape) {
      if (plan.tape.resident) {
        DoubletonBatchTapeRecord *one = nullptr;
        throw_if_cuda_error(doubleton_cuda_malloc(&one, sizeof(DoubletonBatchTapeRecord)),
                            "allocate scalar recovery");
        capture_scalar_recovery<<<1, 1>>>(one, meta_i32, meta_f64, plan.new_l, plan.new_u);
        throw_if_cuda_error(cudaGetLastError(), "capture scalar recovery");
        PostsolveTapeGpu segment;
        build_doubleton_postsolve_tape_gpu(segment, one, 1);
        append_device_recovery_segment(plan.tape, std::move(segment));
        doubleton_cuda_free(one);
      } else {

        std::int32_t meta_i32_host[3] = {0, 0, 0};
        double meta_f64_host[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        throw_if_cuda_error(
            cudaMemcpy(meta_i32_host, meta_i32, sizeof(meta_i32_host), cudaMemcpyDeviceToHost),
            "cudaMemcpy doubleton meta_i32 for tape");
        throw_if_cuda_error(
            cudaMemcpy(meta_f64_host, meta_f64, sizeof(meta_f64_host), cudaMemcpyDeviceToHost),
            "cudaMemcpy doubleton meta_f64 for tape");
        const std::int32_t elim_col = meta_i32_host[1];
        const std::int32_t keep_col = meta_i32_host[2];
        double new_keep_l = 0.0;
        double new_keep_u = 0.0;
        throw_if_cuda_error(
            cudaMemcpy(&new_keep_l, plan.new_l + keep_col, sizeof(double), cudaMemcpyDeviceToHost),
            "cudaMemcpy doubleton new keep l for tape");
        throw_if_cuda_error(
            cudaMemcpy(&new_keep_u, plan.new_u + keep_col, sizeof(double), cudaMemcpyDeviceToHost),
            "cudaMemcpy doubleton new keep u for tape");
        append_postsolve_record(plan.tape, PostsolveReductionType::DoubletonEquation,
                                std::vector<std::int32_t>{elim_col, keep_col, meta_i32_host[0], 0},
                                std::vector<double>{meta_f64_host[0], meta_f64_host[1],
                                                    meta_f64_host[2], meta_f64_host[3],
                                                    meta_f64_host[4], meta_f64_host[6],
                                                    meta_f64_host[7], new_keep_l, new_keep_u,
                                                    meta_f64_host[5], 0.0, 0.0},
                                PostsolveDualMode::Minimal);
      }
    }
    const DeviceCsrMatrix source_A = plan.has_new_A ? plan.new_A : lp.A;
    const bool free_old_new_A = plan.has_new_A;
    DeviceCsrMatrix rewritten_A = build_doubleton_scalar_new_A(source_A, meta_i32, meta_f64, pparams.zero_tol);
    if (plan.has_new_A) {
      doubleton_cuda_free(plan.new_A.rowPtr);
      doubleton_cuda_free(plan.new_A.colVal);
      doubleton_cuda_free(plan.new_A.nzVal);
    }
    (void)free_old_new_A;
    plan.new_A = rewritten_A;
    plan.has_new_A = true;
    double obj_delta = 0.0;
    throw_if_cuda_error(cudaMemcpy(&obj_delta, obj_delta_device, sizeof(double), cudaMemcpyDeviceToHost),
                        "cudaMemcpy doubleton obj_delta");
    plan.obj_constant_delta += obj_delta;
    plan.has_row_action = true;
    plan.has_col_action = true;
    plan.has_change = true;
  }

  doubleton_cuda_free(live_col_nnz);
  doubleton_cuda_free(target_row_ref);
  doubleton_cuda_free(meta_i32);
  doubleton_cuda_free(meta_f64);
  doubleton_cuda_free(status);
  doubleton_cuda_free(obj_delta_device);
}

}  // namespace gpu_presolver::presolve
