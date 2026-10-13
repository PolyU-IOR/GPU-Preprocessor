#include "gpu_presolver/presolve/transpose64.hpp"
#include "gpu_presolver/offset.hpp"
#include "folding_internal.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gpu_presolver::folding {
namespace {

constexpr int kFoldingThreads = 256;

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

void synchronize_for_timing(const char* context) {
  throw_if_cuda_error(cudaDeviceSynchronize(), context);
}

double seconds_since(std::chrono::steady_clock::time_point start,
                     std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double>(stop - start).count();
}

__global__ void kernel_count_color(std::int32_t* color_count,
                                   const std::int32_t* color_id,
                                   std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    atomicAdd(color_count + (color_id[i] - 1), 1);
  }
}

// Independent entries keep their original indices and arithmetic; a warp
// cooperates on each row to avoid serial emission on very long rows.
__global__ void kernel_emit_triplet_warp(std::int64_t* entry_key,
                                    double* entry_value,
                                    const NnzOffset* row_ptr,
                                    const std::int32_t* col_idx,
                                    const double* nz_value,
                                    const std::int32_t* row_color_id,
                                    const std::int32_t* col_color_id,
                                    const std::int32_t* row_color_count,
                                    std::int32_t num_row,
                                    std::int32_t num_col_color) {
  const std::int32_t row = static_cast<std::int32_t>((std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32);
  const std::int32_t lane = threadIdx.x % 32;
  if (row < num_row) {
    const std::int32_t row_id = row_color_id[row];
    const double row_weight = 1.0 / static_cast<double>(row_color_count[row_id - 1]);
    for (std::int64_t idx = static_cast<std::int64_t>(row_ptr[row]) + lane;
         idx < row_ptr[row + 1]; idx += 32) {
      const std::int32_t col_id = col_color_id[col_idx[idx]];
      entry_key[idx] =
          static_cast<std::int64_t>(row_id - 1) * static_cast<std::int64_t>(num_col_color) +
          static_cast<std::int64_t>(col_id - 1);
      entry_value[idx] = nz_value[idx] * row_weight;
    }
  }
}

__global__ void kernel_mark_nonzero(NnzOffset* keep,
                                    const double* value,
                                    double tolerance,
                                    NnzOffset n) {
  const NnzOffset i = NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    keep[i] = fabs(value[i]) > tolerance ? 1 : 0;
  }
}

__global__ void kernel_write_triplet(std::int32_t* row_idx,
                                     std::int32_t* col_idx,
                                     double* nz_val,
                                     const std::int64_t* unique_key,
                                     const double* unique_sum,
                                     const NnzOffset* keep,
                                     const NnzOffset* keep_prefix,
                                     NnzOffset unique_count,
                                     std::int32_t num_col_color) {
  const NnzOffset i = NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < unique_count && keep[i] != 0) {
    const NnzOffset out = keep_prefix[i] - 1;
    const std::int64_t key = unique_key[i];
    row_idx[out] = static_cast<std::int32_t>(key / static_cast<std::int64_t>(num_col_color));
    col_idx[out] = static_cast<std::int32_t>(key % static_cast<std::int64_t>(num_col_color));
    nz_val[out] = unique_sum[i];
  }
}

__global__ void kernel_make_color_member_keys(
    std::uint64_t* keys,
    const std::int32_t* color_id,
    std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    const std::uint64_t color = static_cast<std::uint32_t>(color_id[i] - 1);
    keys[i] = (color << 32) | static_cast<std::uint32_t>(i);
  }
}

// Initial coloring uses a tolerance, so constancy must be checked against the
// actual final members. Each bit marks a nonconstant input field; only integer
// atomics are used. Member lists retain the original-index ordering for fallback.
__global__ void kernel_check_color_constants(
    unsigned int* nonconstant,
    const std::uint64_t* sorted_member_keys,
    const std::int32_t* color_offsets,
    const double* values,
    const double* lower,
    const double* upper,
    std::int32_t n) {
  const std::int32_t p = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= n) return;
  const std::uint64_t key = sorted_member_keys[p];
  const std::int32_t color = static_cast<std::int32_t>(key >> 32);
  const std::uint32_t member = static_cast<std::uint32_t>(key);
  const std::uint32_t first = static_cast<std::uint32_t>(sorted_member_keys[color_offsets[color]]);
  unsigned int mask = 0;
  if (values && values[member] != values[first]) mask |= 1U;
  if (lower[member] != lower[first]) mask |= 2U;
  if (upper[member] != upper[first]) mask |= 4U;
  if (mask) atomicOr(nonconstant + color, mask);
}

// Return true only if replacing the original left-to-right sum by count*value
// is exact in FP64. Strip significand trailing zeros and bound its integer
// product by 2^53: every partial sum is then representable. A finite final
// product also excludes overflow. Zeros are normalized to the original +0 sum.
__device__ bool exact_constant_sum(double value, std::int32_t count, double& sum) {
  if (value == 0.0) { sum = 0.0; return true; }
  if (isinf(value)) { sum = value; return true; }
  if (!isfinite(value) || count <= 0) return false;
  const unsigned long long bits = static_cast<unsigned long long>(__double_as_longlong(value)) & 0x7fffffffffffffffULL;
  unsigned long long mantissa = bits & 0xfffffffffffffULL;
  if (bits >> 52) mantissa |= 1ULL << 52;
  mantissa >>= (__ffsll(static_cast<long long>(mantissa)) - 1);
  if (mantissa > (1ULL << 53) / static_cast<unsigned int>(count)) return false;
  const double product = value * static_cast<double>(count);
  if (!isfinite(product)) return false;
  sum = product;
  return true;
}

__global__ void kernel_reduce_color_values_constant(
    double* output,
    const double* values,
    const std::uint64_t* sorted_member_keys,
    const std::int32_t* color_offsets,
    const std::int32_t* color_counts,
    std::int32_t num_colors,
    bool average,
    const unsigned int* nonconstant) {
  const std::int32_t color =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (color < num_colors) {
    const std::int32_t begin = color_offsets[color];
    const std::int32_t count = color_counts[color];
    double sum = 0.0;
    const std::uint32_t first = static_cast<std::uint32_t>(sorted_member_keys[begin]);
    if (!(nonconstant[color] & 1U) && exact_constant_sum(values[first], count, sum)) {
      output[color] = average ? sum / static_cast<double>(count) : sum;
      return;
    }
    for (std::int32_t p = begin; p < begin + count; ++p) {
      const std::uint32_t member =
          static_cast<std::uint32_t>(sorted_member_keys[p]);
      sum += values[member];
    }
    output[color] = average ? sum / static_cast<double>(count) : sum;
  }
}

__global__ void kernel_reduce_color_bounds_constant(
    double* lower_output,
    double* upper_output,
    const double* lower,
    const double* upper,
    const std::uint64_t* sorted_member_keys,
    const std::int32_t* color_offsets,
    const std::int32_t* color_counts,
    std::int32_t num_colors,
    const unsigned int* nonconstant) {
  const std::int32_t color =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (color < num_colors) {
    const std::int32_t begin = color_offsets[color];
    const std::int32_t count = color_counts[color];
    double lower_sum = 0.0;
    double upper_sum = 0.0;
    bool lower_exact = false, upper_exact = false;
    const std::uint32_t first = static_cast<std::uint32_t>(sorted_member_keys[begin]);
    lower_exact = !(nonconstant[color] & 2U) && exact_constant_sum(lower[first], count, lower_sum);
    upper_exact = !(nonconstant[color] & 4U) && exact_constant_sum(upper[first], count, upper_sum);
    if (lower_exact && upper_exact) {
      lower_output[color] = lower_sum / static_cast<double>(count);
      upper_output[color] = upper_sum / static_cast<double>(count);
      return;
    }
    for (std::int32_t p = begin; p < begin + count; ++p) {
      const std::uint32_t member = static_cast<std::uint32_t>(sorted_member_keys[p]);
      if (!lower_exact) lower_sum += lower[member];
      if (!upper_exact) upper_sum += upper[member];
    }
    lower_output[color] = lower_sum / static_cast<double>(count);
    upper_output[color] = upper_sum / static_cast<double>(count);
  }
}

__global__ void sorted_coo_row_ptr(const int* rows, NnzOffset nnz, int m, NnzOffset* ptr) {
  for (NnzOffset row = NnzOffset(blockIdx.x)*blockDim.x+threadIdx.x; row <= m;
       row += NnzOffset(gridDim.x)*blockDim.x) {
    NnzOffset lo=0, hi=nnz;
    while(lo<hi) { NnzOffset mid=lo+(hi-lo)/2; if(rows[mid]<row) lo=mid+1; else hi=mid; }
    ptr[row]=lo;
  }
}

presolve::DeviceCsrMatrix transpose_csr_gpu(const presolve::DeviceCsrMatrix& A) {
  return gpu_presolver::presolve::transpose_csr64(A);
}

}  // namespace

presolve::LPInfoGpu reduce_size(FoldingWorkspace& workspace,
                                const presolve::LPInfoGpu& model,
                                double tolerance,
                                FoldingReduceProfile* profile) {
  const auto total_start = std::chrono::steady_clock::now();
  const std::int32_t num_row = workspace.num_row;
  const std::int32_t num_col = workspace.num_col;
  const std::int32_t num_row_color = workspace.num_row_color;
  const std::int32_t num_col_color = workspace.num_col_color;

  std::int32_t* row_color_count = nullptr;
  std::int32_t* col_color_count = nullptr;
  auto reserve = [&](std::int32_t*& pointer, std::int32_t& capacity, std::int32_t count) {
    if (capacity >= count) return;
    std::int32_t* next = nullptr;
    throw_if_cuda_error(cudaMalloc(&next, sizeof(std::int32_t) * static_cast<std::size_t>(count)), "cudaMalloc shared color count");
    cudaFree(pointer);
    pointer = next;
    capacity = count;
  };
  reserve(workspace.row_color_count, workspace.row_count_capacity, num_row_color);
  reserve(workspace.col_color_count, workspace.col_count_capacity, num_col_color);
  row_color_count = workspace.row_color_count;
  col_color_count = workspace.col_color_count;
  throw_if_cuda_error(cudaMemset(row_color_count, 0, sizeof(std::int32_t) * static_cast<std::size_t>(num_row_color)),
                      "cudaMemset row_color_count");
  throw_if_cuda_error(cudaMemset(col_color_count, 0, sizeof(std::int32_t) * static_cast<std::size_t>(num_col_color)),
                      "cudaMemset col_color_count");

  const int row_blocks = (num_row + kFoldingThreads - 1) / kFoldingThreads;
  const int col_blocks = (num_col + kFoldingThreads - 1) / kFoldingThreads;
  const auto count_color_start = std::chrono::steady_clock::now();
  kernel_count_color<<<row_blocks, kFoldingThreads>>>(row_color_count, workspace.row_color_id, num_row);
  kernel_count_color<<<col_blocks, kFoldingThreads>>>(col_color_count, workspace.col_color_id, num_col);
  throw_if_cuda_error(cudaGetLastError(), "kernel_count_color");
  synchronize_for_timing("cudaDeviceSynchronize kernel_count_color");
  if (profile != nullptr) {
    profile->count_color_seconds +=
        seconds_since(count_color_start, std::chrono::steady_clock::now());
  }

  // Floating-point atomicAdd makes the folded model depend on block arrival
  // order.  Build one deterministic, original-index-ordered member list per
  // color and reuse it for cost and bound reductions.
  std::uint64_t* row_member_keys = nullptr;
  std::uint64_t* col_member_keys = nullptr;
  std::int32_t* row_color_offsets = nullptr;
  std::int32_t* col_color_offsets = nullptr;
  throw_if_cuda_error(cudaMalloc(
                          &row_member_keys,
                          sizeof(std::uint64_t) *
                              static_cast<std::size_t>(num_row)),
                      "cudaMalloc row_member_keys");
  throw_if_cuda_error(cudaMalloc(
                          &col_member_keys,
                          sizeof(std::uint64_t) *
                              static_cast<std::size_t>(num_col)),
                      "cudaMalloc col_member_keys");
  throw_if_cuda_error(cudaMalloc(
                          &row_color_offsets,
                          sizeof(std::int32_t) *
                              static_cast<std::size_t>(num_row_color)),
                      "cudaMalloc row_color_offsets");
  throw_if_cuda_error(cudaMalloc(
                          &col_color_offsets,
                          sizeof(std::int32_t) *
                              static_cast<std::size_t>(num_col_color)),
                      "cudaMalloc col_color_offsets");
  kernel_make_color_member_keys<<<row_blocks, kFoldingThreads>>>(
      row_member_keys, workspace.row_color_id, num_row);
  kernel_make_color_member_keys<<<col_blocks, kFoldingThreads>>>(
      col_member_keys, workspace.col_color_id, num_col);
  throw_if_cuda_error(cudaGetLastError(), "kernel_make_color_member_keys");
  thrust::sort(thrust::device_pointer_cast(row_member_keys),
               thrust::device_pointer_cast(row_member_keys + num_row));
  thrust::sort(thrust::device_pointer_cast(col_member_keys),
               thrust::device_pointer_cast(col_member_keys + num_col));
  thrust::exclusive_scan(
      thrust::device_pointer_cast(row_color_count),
      thrust::device_pointer_cast(row_color_count + num_row_color),
      thrust::device_pointer_cast(row_color_offsets));
  thrust::exclusive_scan(
      thrust::device_pointer_cast(col_color_count),
      thrust::device_pointer_cast(col_color_count + num_col_color),
      thrust::device_pointer_cast(col_color_offsets));
  synchronize_for_timing("cudaDeviceSynchronize ordered color members");

  unsigned int* row_nonconstant = nullptr;
  unsigned int* col_nonconstant = nullptr;
  throw_if_cuda_error(cudaMalloc(&row_nonconstant, sizeof(unsigned int) * static_cast<std::size_t>(num_row_color)), "cudaMalloc row_nonconstant");
  throw_if_cuda_error(cudaMalloc(&col_nonconstant, sizeof(unsigned int) * static_cast<std::size_t>(num_col_color)), "cudaMalloc col_nonconstant");
  throw_if_cuda_error(cudaMemset(row_nonconstant, 0, sizeof(unsigned int) * static_cast<std::size_t>(num_row_color)), "cudaMemset row_nonconstant");
  throw_if_cuda_error(cudaMemset(col_nonconstant, 0, sizeof(unsigned int) * static_cast<std::size_t>(num_col_color)), "cudaMemset col_nonconstant");
  kernel_check_color_constants<<<row_blocks, kFoldingThreads>>>(row_nonconstant, row_member_keys, row_color_offsets, nullptr, model.AL, model.AU, num_row);
  kernel_check_color_constants<<<col_blocks, kFoldingThreads>>>(col_nonconstant, col_member_keys, col_color_offsets, model.c, model.l, model.u, num_col);
  throw_if_cuda_error(cudaGetLastError(), "kernel_check_color_constants");
  synchronize_for_timing("cudaDeviceSynchronize constant checks");

  std::int64_t* entry_key = nullptr;
  double* entry_value = nullptr;
  if (model.A.nnz > 0) {
    throw_if_cuda_error(cudaMalloc(&entry_key, sizeof(std::int64_t) * static_cast<std::size_t>(model.A.nnz)),
                        "cudaMalloc entry_key");
    throw_if_cuda_error(cudaMalloc(&entry_value, sizeof(double) * static_cast<std::size_t>(model.A.nnz)),
                        "cudaMalloc entry_value");
    const auto emit_triplet_start = std::chrono::steady_clock::now();
    kernel_emit_triplet_warp<<<static_cast<unsigned>((static_cast<std::int64_t>(num_row) * 32 + kFoldingThreads - 1) / kFoldingThreads), kFoldingThreads>>>(entry_key,
                                                         entry_value,
                                                         model.A.rowPtr,
                                                         model.A.colVal,
                                                         model.A.nzVal,
                                                         workspace.row_color_id,
                                                         workspace.col_color_id,
                                                         row_color_count,
                                                         num_row,
                                                         num_col_color);

    throw_if_cuda_error(cudaGetLastError(), "kernel_emit_triplet");
    synchronize_for_timing("cudaDeviceSynchronize kernel_emit_triplet");
    if (profile != nullptr) {
      profile->emit_triplet_seconds +=
          seconds_since(emit_triplet_start, std::chrono::steady_clock::now());
    }
  }

  presolve::LPInfoGpu reduced{};
  reduced.A.rows = num_row_color;
  reduced.A.cols = num_col_color;

  if (model.A.nnz > 0) {
    thrust::device_ptr<std::int64_t> key_ptr(entry_key);
    thrust::device_ptr<double> value_ptr(entry_value);
    const auto sort_reduce_start = std::chrono::steady_clock::now();
    thrust::sort_by_key(key_ptr, key_ptr + model.A.nnz, value_ptr);

    std::int64_t* unique_key = nullptr;
    double* unique_sum = nullptr;
    throw_if_cuda_error(cudaMalloc(&unique_key, sizeof(std::int64_t) * static_cast<std::size_t>(model.A.nnz)),
                        "cudaMalloc unique_key");
    throw_if_cuda_error(cudaMalloc(&unique_sum, sizeof(double) * static_cast<std::size_t>(model.A.nnz)),
                        "cudaMalloc unique_sum");
    auto reduced_end = thrust::reduce_by_key(key_ptr,
                                             key_ptr + model.A.nnz,
                                             value_ptr,
                                             thrust::device_pointer_cast(unique_key),
                                             thrust::device_pointer_cast(unique_sum));
    synchronize_for_timing("cudaDeviceSynchronize sort_reduce");
    const NnzOffset unique_count =
        static_cast<NnzOffset>(reduced_end.first - thrust::device_pointer_cast(unique_key));
    if (profile != nullptr) {
      profile->sort_reduce_seconds +=
          seconds_since(sort_reduce_start, std::chrono::steady_clock::now());
      profile->unique_entries = unique_count;
    }

    NnzOffset* keep = nullptr;
    NnzOffset* keep_prefix = nullptr;
    if (unique_count > 0) {
      throw_if_cuda_error(cudaMalloc(&keep, sizeof(NnzOffset) * static_cast<std::size_t>(unique_count)),
                          "cudaMalloc keep");
      throw_if_cuda_error(cudaMalloc(&keep_prefix, sizeof(NnzOffset) * static_cast<std::size_t>(unique_count)),
                          "cudaMalloc keep_prefix");
      const int unique_blocks = (unique_count + kFoldingThreads - 1) / kFoldingThreads;
      kernel_mark_nonzero<<<unique_blocks, kFoldingThreads>>>(keep, unique_sum, tolerance, unique_count);
      throw_if_cuda_error(cudaGetLastError(), "kernel_mark_nonzero");
      thrust::inclusive_scan(thrust::device_pointer_cast(keep),
                             thrust::device_pointer_cast(keep + unique_count),
                             thrust::device_pointer_cast(keep_prefix));
      synchronize_for_timing("cudaDeviceSynchronize reduce write_triplet prep");

      NnzOffset keep_num = 0;
      throw_if_cuda_error(cudaMemcpy(&keep_num,
                                     keep_prefix + (unique_count - 1),
                                     sizeof(NnzOffset),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy keep_num");
      reduced.A.nnz = keep_num;
      if (profile != nullptr) {
        profile->kept_entries = keep_num;
      }

      throw_if_cuda_error(cudaMalloc(&reduced.A.rowPtr,
                                     sizeof(NnzOffset) * static_cast<std::size_t>(num_row_color + 1)),
                          "cudaMalloc reduced rowPtr");
      if (keep_num > 0) {
        std::int32_t* row_idx = nullptr;
        throw_if_cuda_error(cudaMalloc(&row_idx, sizeof(std::int32_t) * static_cast<std::size_t>(keep_num)),
                            "cudaMalloc row_idx");
        throw_if_cuda_error(cudaMalloc(&reduced.A.colVal, sizeof(std::int32_t) * static_cast<std::size_t>(keep_num)),
                            "cudaMalloc reduced colVal");
        throw_if_cuda_error(cudaMalloc(&reduced.A.nzVal, sizeof(double) * static_cast<std::size_t>(keep_num)),
                            "cudaMalloc reduced nzVal");
        kernel_write_triplet<<<unique_blocks, kFoldingThreads>>>(row_idx,
                                                                 reduced.A.colVal,
                                                                 reduced.A.nzVal,
                                                                 unique_key,
                                                                 unique_sum,
                                                                 keep,
                                                                 keep_prefix,
                                                                 unique_count,
                                                                 num_col_color);
        throw_if_cuda_error(cudaGetLastError(), "kernel_write_triplet");
        synchronize_for_timing("cudaDeviceSynchronize kernel_write_triplet");

        sorted_coo_row_ptr<<<(num_row_color + 256) / 256, 256>>>(
            row_idx, keep_num, num_row_color, reduced.A.rowPtr);
        throw_if_cuda_error(cudaGetLastError(), "sorted_coo_row_ptr");
        cudaFree(row_idx);
      } else {
        throw_if_cuda_error(cudaMemset(reduced.A.rowPtr,
                                       0,
                                       sizeof(NnzOffset) * static_cast<std::size_t>(num_row_color + 1)),
                            "cudaMemset reduced rowPtr");
      }
      cudaFree(keep);
      cudaFree(keep_prefix);
    } else {
      reduced.A.nnz = 0;
      throw_if_cuda_error(cudaMalloc(&reduced.A.rowPtr,
                                     sizeof(NnzOffset) * static_cast<std::size_t>(num_row_color + 1)),
                          "cudaMalloc reduced rowPtr");
      throw_if_cuda_error(cudaMemset(reduced.A.rowPtr,
                                     0,
                                     sizeof(NnzOffset) * static_cast<std::size_t>(num_row_color + 1)),
                          "cudaMemset reduced rowPtr");
    }
    cudaFree(unique_key);
    cudaFree(unique_sum);
  } else {
    reduced.A.nnz = 0;
    throw_if_cuda_error(cudaMalloc(&reduced.A.rowPtr,
                                   sizeof(NnzOffset) * static_cast<std::size_t>(num_row_color + 1)),
                        "cudaMalloc reduced rowPtr");
    throw_if_cuda_error(cudaMemset(reduced.A.rowPtr,
                                   0,
                                   sizeof(NnzOffset) * static_cast<std::size_t>(num_row_color + 1)),
                        "cudaMemset reduced rowPtr");
  }

  const auto transpose_start = std::chrono::steady_clock::now();
  reduced.AT = transpose_csr_gpu(reduced.A);
  synchronize_for_timing("cudaDeviceSynchronize transpose");
  if (profile != nullptr) {
    profile->transpose_seconds +=
        seconds_since(transpose_start, std::chrono::steady_clock::now());
  }

  throw_if_cuda_error(cudaMalloc(&reduced.c, sizeof(double) * static_cast<std::size_t>(num_col_color)),
                      "cudaMalloc reduced c");
  throw_if_cuda_error(cudaMemset(reduced.c, 0, sizeof(double) * static_cast<std::size_t>(num_col_color)),
                      "cudaMemset reduced c");
  const auto sum_cost_start = std::chrono::steady_clock::now();
  const int col_color_blocks =
      (num_col_color + kFoldingThreads - 1) / kFoldingThreads;
  kernel_reduce_color_values_constant<<<col_color_blocks, kFoldingThreads>>>(
      reduced.c, model.c, col_member_keys, col_color_offsets,
      col_color_count, num_col_color, false, col_nonconstant);

  throw_if_cuda_error(cudaGetLastError(),
                      "kernel_reduce_color_values_constant cost");
  synchronize_for_timing(
      "cudaDeviceSynchronize kernel_reduce_color_values_constant cost");
  if (profile != nullptr) {
    profile->sum_cost_seconds +=
        seconds_since(sum_cost_start, std::chrono::steady_clock::now());
  }

  const auto row_bounds_start = std::chrono::steady_clock::now();
  throw_if_cuda_error(cudaMalloc(&reduced.AL, sizeof(double) * static_cast<std::size_t>(num_row_color)),
                      "cudaMalloc reduced AL");
  throw_if_cuda_error(cudaMalloc(&reduced.AU, sizeof(double) * static_cast<std::size_t>(num_row_color)),
                      "cudaMalloc reduced AU");
  const int row_color_blocks = (num_row_color + kFoldingThreads - 1) / kFoldingThreads;
  kernel_reduce_color_bounds_constant<<<row_color_blocks, kFoldingThreads>>>(
      reduced.AL, reduced.AU, model.AL, model.AU, row_member_keys,
      row_color_offsets, row_color_count, num_row_color, row_nonconstant);

  throw_if_cuda_error(cudaGetLastError(),
                      "kernel_reduce_color_bounds_constant rows");
  synchronize_for_timing("cudaDeviceSynchronize row bounds");
  if (profile != nullptr) {
    profile->row_bounds_seconds +=
        seconds_since(row_bounds_start, std::chrono::steady_clock::now());
  }

  const auto col_bounds_start = std::chrono::steady_clock::now();
  throw_if_cuda_error(cudaMalloc(&reduced.l, sizeof(double) * static_cast<std::size_t>(num_col_color)),
                      "cudaMalloc reduced l");
  throw_if_cuda_error(cudaMalloc(&reduced.u, sizeof(double) * static_cast<std::size_t>(num_col_color)),
                      "cudaMalloc reduced u");
  kernel_reduce_color_bounds_constant<<<col_color_blocks, kFoldingThreads>>>(
      reduced.l, reduced.u, model.l, model.u, col_member_keys,
      col_color_offsets, col_color_count, num_col_color, col_nonconstant);

  throw_if_cuda_error(cudaGetLastError(),
                      "kernel_reduce_color_bounds_constant cols");
  synchronize_for_timing("cudaDeviceSynchronize col bounds");
  if (profile != nullptr) {
    profile->col_bounds_seconds +=
        seconds_since(col_bounds_start, std::chrono::steady_clock::now());
  }

  reduced.obj_constant = model.obj_constant;
  reduced.AT_leading_slack = 0;
  reduced.AT_slack_after = nullptr;

  cudaFree(row_member_keys);
  cudaFree(col_member_keys);
  cudaFree(row_color_offsets);
  cudaFree(col_color_offsets);
  cudaFree(row_nonconstant);
  cudaFree(col_nonconstant);
  cudaFree(entry_key);
  cudaFree(entry_value);
  if (profile != nullptr) {
    profile->total_seconds =
        seconds_since(total_start, std::chrono::steady_clock::now());
  }
  return reduced;
}

}  // namespace gpu_presolver::folding
