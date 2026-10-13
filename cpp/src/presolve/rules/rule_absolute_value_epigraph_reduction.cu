#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/rules/rule_absolute_value_epigraph_reduction.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;
constexpr double ABSOLUTE_VALUE_EPIGRAPH_SUB_TOL = 1.0e-12;
constexpr std::int32_t STRUCTURAL_OUTER_UNASSIGNED = -1;
constexpr std::int32_t STRUCTURAL_OUTER_BOUND_ROLE = -2;
constexpr std::int32_t STRUCTURAL_OUTER_BLOCK_ROLE = -3;

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

__global__ void _kernel_fill_i32(std::int32_t* values, std::int32_t value, std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < n) {
    values[i] = value;
  }
}

__global__ void _kernel_inclusive_scan_i32_serial(NnzOffset* values, std::int32_t n) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    NnzOffset running = 0;
    for (std::int32_t i = 0; i < n; ++i) {
      running += values[i];
      values[i] = running;
    }
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

__global__ void _kernel_structural_check_csr_sorted(std::int32_t* status_flag,
                                                    const NnzOffset* row_ptr,
                                                    const std::int32_t* col_val,
                                                    std::int32_t rows) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= rows) {
    return;
  }
  for (NnzOffset p = row_ptr[row] + 1; p < row_ptr[row + 1]; ++p) {
    if (col_val[p - 1] >= col_val[p]) {
      atomicMax(status_flag, 1);
      return;
    }
  }
}

__global__ void _kernel_structural_build_csr_sort_keys(unsigned long long* keys,
                                                       const NnzOffset* row_ptr,
                                                       const std::int32_t* col_val,
                                                       std::int32_t rows) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= rows) {
    return;
  }
  const unsigned long long row_key =
      static_cast<unsigned long long>(static_cast<std::uint32_t>(row)) << 32;
  for (NnzOffset p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
    keys[p] = row_key |
              static_cast<unsigned long long>(static_cast<std::uint32_t>(col_val[p]));
  }
}

__global__ void _kernel_structural_unpack_csr_sort_keys(std::int32_t* col_val,
                                                        const unsigned long long* keys,
                                                        NnzOffset nnz) {
  const NnzOffset p = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (p < nnz) {
    col_val[p] = static_cast<std::int32_t>(keys[p] & 0xffffffffULL);
  }
}

bool ensure_structural_csr_rows_sorted_unique(DeviceCsrMatrix& matrix, const char* context) {
  if (matrix.rows <= 0 || matrix.nnz <= 0) {
    return true;
  }

  std::int32_t* status_flag = nullptr;
  throw_if_cuda_error(cudaMalloc(&status_flag, sizeof(std::int32_t)), context);
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)), context);
  const int row_blocks = (matrix.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_structural_check_csr_sorted<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag, matrix.rowPtr, matrix.colVal, matrix.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_check_csr_sorted");
  std::int32_t status = 0;
  throw_if_cuda_error(
      cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
      "cudaMemcpy structural CSR sorted status");
  cudaFree(status_flag);
  if (status == 0) {
    return true;
  }

  unsigned long long* keys = nullptr;
  throw_if_cuda_error(
      cudaMalloc(&keys, sizeof(unsigned long long) * static_cast<std::size_t>(matrix.nnz)),
      context);
  _kernel_structural_build_csr_sort_keys<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
      keys, matrix.rowPtr, matrix.colVal, matrix.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_build_csr_sort_keys");
  thrust::sort_by_key(thrust::device_pointer_cast(keys),
                      thrust::device_pointer_cast(keys + matrix.nnz),
                      thrust::device_pointer_cast(matrix.nzVal));
  throw_if_cuda_error(cudaGetLastError(), "thrust structural CSR row sort");
  const int nnz_blocks = (matrix.nnz + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_structural_unpack_csr_sort_keys<<<nnz_blocks, GPU_PRESOLVE_THREADS>>>(
      matrix.colVal, keys, matrix.nnz);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_unpack_csr_sort_keys");
  throw_if_cuda_error(cudaDeviceSynchronize(), context);
  cudaFree(keys);

  // A structural substitution must never create parallel columns.  Sorting
  // makes any collision adjacent, so one final linear check can reject it
  // without the quadratic all-pairs scan that is prohibitive on dense Hans
  // rows.
  throw_if_cuda_error(cudaMalloc(&status_flag, sizeof(std::int32_t)), context);
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)), context);
  _kernel_structural_check_csr_sorted<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag, matrix.rowPtr, matrix.colVal, matrix.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_check_csr_sorted after sort");
  status = 0;
  throw_if_cuda_error(
      cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
      "cudaMemcpy structural CSR post-sort status");
  cudaFree(status_flag);
  return status == 0;
}

// These structural rules delete unbounded epigraph rows.  A tolerance-based
// coefficient match is not a valid certificate: an arbitrarily large primal
// value can amplify even a tiny coefficient mismatch.  Require the defining
// identities exactly and use tolerances only for conservative rejection of
// unrelated near-zero entries.
__device__ bool _structural_exact(double lhs, double rhs) {
  return lhs == rhs;
}

__device__ std::int32_t _sls_outer_map_col_device(const std::int32_t* free_to_bound,
                                                  std::int32_t col) {
  const std::int32_t mapped = free_to_bound[col];
  return mapped >= 0 ? mapped : col;
}

__device__ bool _sls_outer_pair_signature_device(std::int32_t c1a,
                                                 std::int32_t c2a,
                                                 double v1a,
                                                 double v2a,
                                                 std::int32_t c1b,
                                                 std::int32_t c2b,
                                                 double v1b,
                                                 double v2b,
                                                 std::int32_t c1c,
                                                 std::int32_t c2c,
                                                 double v1c,
                                                 double v2c) {
  if (c1a != c1b || c1a != c1c || c2a != c2b || c2a != c2c) {
    return false;
  }
  return _structural_exact(v1a, 1.0) && _structural_exact(v2a, -1.0) &&
         _structural_exact(v1b, 1.0) && _structural_exact(v2b, 1.0) &&
         _structural_exact(v1c, -1.0) && _structural_exact(v2c, 1.0);
}

__device__ bool _sls_extract_l1_pair_device(std::int32_t ep1_col1,
                                            std::int32_t ep1_col2,
                                            double ep1_val1,
                                            double ep1_val2,
                                            double ep2_val1,
                                            double ep2_val2,
                                            std::int32_t* q_col,
                                            std::int32_t* e_col) {
  if (!_structural_exact(ep1_val1, -1.0) ||
      !_structural_exact(ep1_val2, 1.0) ||
      !_structural_exact(ep2_val1, 1.0) ||
      !_structural_exact(ep2_val2, 1.0)) {
    return false;
  }
  *e_col = ep1_col1;
  *q_col = ep1_col2;
  return *q_col != *e_col;
}

__device__ bool _sls_extract_l1_split_pair_device(std::int32_t c1a,
                                                  std::int32_t c2a,
                                                  double v1a,
                                                  double v2a,
                                                  std::int32_t c1b,
                                                  std::int32_t c2b,
                                                  double v1b,
                                                  double v2b,
                                                  std::int32_t* t_col,
                                                  std::int32_t* e_col) {
  if (c1a != c1b || c2a != c2b) {
    return false;
  }
  const bool first_is_t = isfinite(v1a) && v1a > 0.0 &&
                          _structural_exact(v1a, v1b) &&
                          _structural_exact(v2a, -v1a) &&
                          _structural_exact(v2b, v1b);
  if (first_is_t) {
    *t_col = c1a;
    *e_col = c2a;
    return *t_col != *e_col;
  }
  const bool second_is_t = isfinite(v2a) && v2a > 0.0 &&
                           _structural_exact(v2a, v2b) &&
                           _structural_exact(v1a, -v2a) &&
                           _structural_exact(v1b, v2b);
  if (second_is_t) {
    *t_col = c2a;
    *e_col = c1a;
    return *t_col != *e_col;
  }
  return false;
}

__device__ bool _sls_extract_l1_orientation_device(std::int32_t c1a,
                                                   std::int32_t c2a,
                                                   double v1a,
                                                   double v2a,
                                                   double v1b,
                                                   double v2b,
                                                   std::int32_t* t_col,
                                                   std::int32_t* e_col,
                                                   double* rho) {
  if (!_sls_extract_l1_split_pair_device(c1a, c2a, v1a, v2a, c1a, c2a, v1b, v2b, t_col, e_col)) {
    return false;
  }
  if (*t_col == c1a && *e_col == c2a) {
    *rho = fabs(v2a / v1a);
  } else if (*t_col == c2a && *e_col == c1a) {
    *rho = fabs(v1a / v2a);
  } else {
    *rho = 0.0;
  }
  return isfinite(*rho) && *rho > 0.0;
}

__global__ void _kernel_structural_row_metadata(std::uint8_t* eq_row,
                                                std::uint8_t* eq_zero_two_nnz,
                                                std::uint8_t* lower_two_nnz,
                                                std::uint8_t* zero_lower_two_nnz,
                                                std::int32_t* raw_col1,
                                                std::int32_t* raw_col2,
                                                double* raw_val1,
                                                double* raw_val2,
                                                std::int32_t* zl_col1,
                                                std::int32_t* zl_col2,
                                                double* zl_val1,
                                                double* zl_val2,
                                                const NnzOffset* row_ptr,
                                                const std::int32_t* col_val,
                                                const double* nz_val,
                                                const double* AL,
                                                const double* AU,
                                                std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }

  const double al = AL[row];
  const double au = AU[row];
  const NnzOffset lo = row_ptr[row];
  const NnzOffset hi = row_ptr[row + 1];
  const NnzOffset nnz = hi - lo;
  const bool is_eq = isfinite(al) && isfinite(au) && al == au;

  eq_row[row] = is_eq ? std::uint8_t{1} : std::uint8_t{0};
  eq_zero_two_nnz[row] = std::uint8_t{0};
  lower_two_nnz[row] = std::uint8_t{0};
  zero_lower_two_nnz[row] = std::uint8_t{0};
  raw_col1[row] = -1;
  raw_col2[row] = -1;
  raw_val1[row] = 0.0;
  raw_val2[row] = 0.0;
  zl_col1[row] = -1;
  zl_col2[row] = -1;
  zl_val1[row] = 0.0;
  zl_val2[row] = 0.0;

  if (nnz != 2) {
    return;
  }

  std::int32_t c1 = col_val[lo];
  std::int32_t c2 = col_val[lo + 1];
  double v1 = nz_val[lo];
  double v2 = nz_val[lo + 1];
  if (c2 < c1) {
    const std::int32_t tc = c1;
    c1 = c2;
    c2 = tc;
    const double tv = v1;
    v1 = v2;
    v2 = tv;
  }

  raw_col1[row] = c1;
  raw_col2[row] = c2;
  raw_val1[row] = v1;
  raw_val2[row] = v2;

  if (is_eq && al == 0.0) {
    eq_zero_two_nnz[row] = std::uint8_t{1};
  }
  if (isfinite(al) && isinf(au) && au > 0.0) {
    lower_two_nnz[row] = std::uint8_t{1};
  }

  const bool is_zero_lower = al == 0.0 && isinf(au) && au > 0.0;
  const bool is_zero_upper = isinf(al) && al < 0.0 && au == 0.0;
  if (is_zero_lower || is_zero_upper) {
    const double scale = is_zero_lower ? 1.0 : -1.0;
    zero_lower_two_nnz[row] = std::uint8_t{1};
    zl_col1[row] = c1;
    zl_col2[row] = c2;
    zl_val1[row] = scale * v1;
    zl_val2[row] = scale * v2;
  }
}

__global__ void _kernel_structural_outer_count_pairs(std::int32_t* status_flag,
                                                     std::int32_t* pair_count,
                                                     std::int32_t* start_row_out,
                                                     const std::uint8_t* zero_lower_two_nnz,
                                                     const std::int32_t* zl_col1,
                                                     const std::int32_t* zl_col2,
                                                     const double* zl_val1,
                                                     const double* zl_val2,
                                                     std::int32_t m) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    std::int32_t row = 0;
    std::int32_t count = 0;
    while (row + 2 < m) {
      const bool ok =
          zero_lower_two_nnz[row] != std::uint8_t{0} &&
          zero_lower_two_nnz[row + 1] != std::uint8_t{0} &&
          zero_lower_two_nnz[row + 2] != std::uint8_t{0} &&
          _sls_outer_pair_signature_device(
              zl_col1[row], zl_col2[row], zl_val1[row], zl_val2[row],
              zl_col1[row + 1], zl_col2[row + 1], zl_val1[row + 1], zl_val2[row + 1],
              zl_col1[row + 2], zl_col2[row + 2], zl_val1[row + 2], zl_val2[row + 2]);
      if (!ok) {
        break;
      }
      ++count;
      row += 3;
    }

    pair_count[0] = count;
    start_row_out[0] = row;
    if (count == 0 || row >= m || ((m - row) % 8) != 0) {
      status_flag[0] = 1;
    }
  }
}

__global__ void _kernel_structural_outer_extract_pairs(std::int32_t* bound_cols,
                                                       std::int32_t* free_cols,
                                                       const std::int32_t* zl_col1,
                                                       const std::int32_t* zl_col2,
                                                       std::int32_t pair_count) {
  const NnzOffset idx = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (idx < pair_count) {
    const std::int32_t row = 3 * idx;
    bound_cols[idx] = zl_col1[row];
    free_cols[idx] = zl_col2[row];
  }
}

__global__ void _kernel_absolute_value_epigraph_split_extract_pairs(std::int32_t* status_flag,
                                                          std::int32_t* t_cols,
                                                          std::int32_t* e_cols,
                                                          const std::uint8_t* eq_row,
                                                          const std::uint8_t* zero_lower_two_nnz,
                                                          const std::int32_t* zl_col1,
                                                          const std::int32_t* zl_col2,
                                                          const double* zl_val1,
                                                          const double* zl_val2,
                                                          std::int32_t nblocks) {
  const NnzOffset idx = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (idx >= nblocks) {
    return;
  }

  const std::int32_t eq = 3 * idx;
  const std::int32_t r1 = eq + 1;
  const std::int32_t r2 = eq + 2;
  if (eq_row[eq] == std::uint8_t{0} ||
      zero_lower_two_nnz[r1] == std::uint8_t{0} ||
      zero_lower_two_nnz[r2] == std::uint8_t{0}) {
    atomicMax(status_flag, 1);
    return;
  }

  std::int32_t t_col = -1;
  std::int32_t e_col = -1;
  if (!_sls_extract_l1_split_pair_device(
          zl_col1[r1], zl_col2[r1], zl_val1[r1], zl_val2[r1],
          zl_col1[r2], zl_col2[r2], zl_val1[r2], zl_val2[r2],
          &t_col, &e_col)) {
    atomicMax(status_flag, 1);
    return;
  }
  t_cols[idx] = t_col;
  e_cols[idx] = e_col;
}

__global__ void _kernel_absolute_value_epigraph_split_claim_roles(std::int32_t* status_flag,
                                                        std::int32_t* role_owner,
                                                        const std::int32_t* t_cols,
                                                        const std::int32_t* e_cols,
                                                        std::int32_t nblocks,
                                                        std::int32_t n) {
  const NnzOffset idx = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (idx >= nblocks) {
    return;
  }

  const std::int32_t t_col = t_cols[idx];
  const std::int32_t e_col = e_cols[idx];
  if (t_col < 0 || t_col >= n || e_col < 0 || e_col >= n || t_col == e_col) {
    atomicMax(status_flag, 1);
    return;
  }

  const std::int32_t t_role = 2 * idx;
  const std::int32_t e_role = t_role + 1;
  if (atomicCAS(role_owner + t_col, -1, t_role) != -1) {
    atomicMax(status_flag, 1);
  }
  if (atomicCAS(role_owner + e_col, -1, e_role) != -1) {
    atomicMax(status_flag, 1);
  }
}

__global__ void _kernel_absolute_value_epigraph_split_validate_blocks(
    std::int32_t* status_flag,
    std::uint8_t* preserve_bound_row,
    const std::uint8_t* keep_row,
    const std::uint8_t* keep_col,
    const double* c,
    const double* l,
    const double* u,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    const NnzOffset* at_row_ptr,
    const std::int32_t* at_col_val,
    const double* at_nz_val,
    const std::int32_t* t_cols,
    const std::int32_t* e_cols,
    std::uint8_t residual_bound_mode,
    double residual_bound_as_free_min,
    std::int32_t nblocks) {
  const NnzOffset idx = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (idx >= nblocks) {
    return;
  }

  preserve_bound_row[idx] = std::uint8_t{0};
  const std::int32_t t_col = t_cols[idx];
  const std::int32_t e_col = e_cols[idx];
  const std::int32_t eq = 3 * idx;
  const std::int32_t r1 = eq + 1;
  const std::int32_t r2 = eq + 2;

  const double l_t = l[t_col];
  const double u_t = u[t_col];
  const double l_e = l[e_col];
  const double u_e = u[e_col];
  const double c_t = c[t_col];
  const double c_e = c[e_col];

  const bool t_bounds_valid = l_t == 0.0 &&
                              isinf(u_t) && u_t > 0.0;
  const bool e_bounds_valid = !isnan(l_e) && !isnan(u_e) &&
                              !(isinf(l_e) && l_e > 0.0) &&
                              !(isinf(u_e) && u_e < 0.0) && l_e <= u_e;
  const bool e_is_truly_free = isinf(l_e) && l_e < 0.0 &&
                               isinf(u_e) && u_e > 0.0;
  const bool preserve_finite_mode =
      residual_bound_mode ==
      static_cast<std::uint8_t>(AbsoluteValueEpigraphResidualBoundMode::PreserveFinite);
  const bool strict_free_mode =
      residual_bound_mode ==
      static_cast<std::uint8_t>(AbsoluteValueEpigraphResidualBoundMode::StrictFreeOnly);
  const bool legacy_mode =
      residual_bound_mode ==
      static_cast<std::uint8_t>(AbsoluteValueEpigraphResidualBoundMode::LegacyWideAsFree);
  const bool legacy_wide_as_free =
      legacy_mode &&
      isfinite(residual_bound_as_free_min) && residual_bound_as_free_min > 0.0 &&
      l_e <= -residual_bound_as_free_min && u_e >= residual_bound_as_free_min;
  const bool residual_mode_valid = preserve_finite_mode || strict_free_mode || legacy_mode;
  const bool residual_bounds_supported =
      e_is_truly_free || preserve_finite_mode || legacy_wide_as_free;
  bool transformed_costs_valid = isfinite(c_t) && isfinite(c_e);
  if (transformed_costs_valid) {
    transformed_costs_valid = isfinite(c_t + c_e) && isfinite(c_t - c_e);
  }
  if (!t_bounds_valid || !e_bounds_valid || !transformed_costs_valid ||
      keep_row[eq] == std::uint8_t{0} ||
      keep_row[r1] == std::uint8_t{0} || keep_row[r2] == std::uint8_t{0} ||
      keep_col[t_col] == std::uint8_t{0} || keep_col[e_col] == std::uint8_t{0} ||
      !residual_mode_valid || !residual_bounds_supported) {
    atomicMax(status_flag, 1);
    return;
  }

  // The specialized fast fill below is exact only when t occurs in the two
  // epigraph rows and e occurs once in the retained equality plus once in
  // each epigraph row.  These exact AT certificates also reject duplicate
  // entries and all cross-block t/e role reuse.
  // Recheck the A-side equality row as well as AT.  The public LP structure
  // carries both matrices; relying only on AT for the allocation certificate
  // could under/over-fill new_A if a malformed caller supplied inconsistent
  // matrices.  Strictly increasing columns also preserve the sorted-CSR
  // invariant needed by later sparse merge rules.
  std::int32_t a_t_eq_count = 0;
  std::int32_t a_e_eq_count = 0;
  std::int32_t previous_col = -1;
  bool incidence_valid = true;
  for (NnzOffset p = row_ptr[eq]; p < row_ptr[eq + 1]; ++p) {
    const std::int32_t col = col_val[p];
    if (col <= previous_col || !isfinite(nz_val[p]) || nz_val[p] == 0.0) {
      incidence_valid = false;
    }
    previous_col = col;
    if (col == t_col) {
      ++a_t_eq_count;
    }
    if (col == e_col) {
      ++a_e_eq_count;
    }
  }
  incidence_valid = incidence_valid && a_t_eq_count == 0 && a_e_eq_count == 1;

  std::int32_t t_r1_count = 0;
  std::int32_t t_r2_count = 0;
  for (NnzOffset p = at_row_ptr[t_col]; p < at_row_ptr[t_col + 1]; ++p) {
    if (fabs(at_nz_val[p]) <= ABSOLUTE_VALUE_EPIGRAPH_SUB_TOL) {
      incidence_valid = false;
      continue;
    }
    const std::int32_t row = at_col_val[p];
    if (row == r1) {
      ++t_r1_count;
    } else if (row == r2) {
      ++t_r2_count;
    } else {
      incidence_valid = false;
    }
  }

  std::int32_t e_eq_count = 0;
  std::int32_t e_r1_count = 0;
  std::int32_t e_r2_count = 0;
  for (NnzOffset p = at_row_ptr[e_col]; p < at_row_ptr[e_col + 1]; ++p) {
    if (fabs(at_nz_val[p]) <= ABSOLUTE_VALUE_EPIGRAPH_SUB_TOL) {
      incidence_valid = false;
      continue;
    }
    const std::int32_t row = at_col_val[p];
    if (row == eq) {
      ++e_eq_count;
    } else if (row == r1) {
      ++e_r1_count;
    } else if (row == r2) {
      ++e_r2_count;
    } else {
      incidence_valid = false;
    }
  }
  incidence_valid = incidence_valid && t_r1_count == 1 && t_r2_count == 1 &&
                    e_eq_count == 1 && e_r1_count == 1 && e_r2_count == 1;
  if (!incidence_valid) {
    atomicMax(status_flag, 1);
    return;
  }

  preserve_bound_row[idx] =
      (preserve_finite_mode && !e_is_truly_free) ? std::uint8_t{1} : std::uint8_t{0};
}

__global__ void _kernel_absolute_value_epigraph_split_apply_plan(
    std::uint8_t* keep_row,
    double* new_c,
    double* new_l,
    double* new_u,
    double* new_AL,
    double* new_AU,
    const std::uint8_t* preserve_bound_row,
    const std::int32_t* t_cols,
    const std::int32_t* e_cols,
    std::int32_t nblocks) {
  const NnzOffset idx = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (idx >= nblocks) {
    return;
  }

  const std::int32_t t_col = t_cols[idx];
  const std::int32_t e_col = e_cols[idx];
  const std::int32_t r1 = 3 * idx + 1;
  const std::int32_t r2 = r1 + 1;
  const double c_t = new_c[t_col];
  const double c_e = new_c[e_col];
  const double l_e = new_l[e_col];
  const double u_e = new_u[e_col];

  if (preserve_bound_row[idx] != std::uint8_t{0}) {
    new_AL[r1] = l_e;
    new_AU[r1] = u_e;
  } else {
    keep_row[r1] = std::uint8_t{0};
  }
  keep_row[r2] = std::uint8_t{0};
  new_c[t_col] = c_t + c_e;
  new_c[e_col] = c_t - c_e;
  new_l[t_col] = 0.0;
  new_l[e_col] = 0.0;
  new_u[t_col] = INFINITY;
  new_u[e_col] = INFINITY;
}

__global__ void _kernel_absolute_value_epigraph_split_count_rows(NnzOffset* row_nnz_new,
                                                       const NnzOffset* row_ptr,
                                                       const std::uint8_t* preserve_bound_row,
                                                       std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }
  if ((row % 3) == 0) {
    row_nnz_new[row] = (row_ptr[row + 1] - row_ptr[row]) + 1;
  } else if ((row % 3) == 1 &&
             preserve_bound_row[row / 3] != std::uint8_t{0}) {
    row_nnz_new[row] = 2;
  } else {
    row_nnz_new[row] = 0;
  }
}

__global__ void _kernel_absolute_value_epigraph_split_fill(std::int32_t* col_val_new,
                                                 double* nz_val_new,
                                                 const NnzOffset* row_ptr_new,
                                                 const NnzOffset* row_ptr_org,
                                                 const std::int32_t* col_val_org,
                                                 const double* nz_val_org,
                                                 const std::int32_t* t_cols,
                                                 const std::int32_t* e_cols,
                                                 const std::uint8_t* preserve_bound_row,
                                                 std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }

  const std::int32_t block_idx = row / 3;
  const std::int32_t t_col = t_cols[block_idx];
  const std::int32_t e_col = e_cols[block_idx];
  NnzOffset write_ptr = row_ptr_new[row];
  if ((row % 3) == 1) {
    if (preserve_bound_row[block_idx] != std::uint8_t{0}) {
      const bool t_first = t_col < e_col;
      col_val_new[write_ptr] = t_first ? t_col : e_col;
      nz_val_new[write_ptr] = t_first ? 1.0 : -1.0;
      col_val_new[write_ptr + 1] = t_first ? e_col : t_col;
      nz_val_new[write_ptr + 1] = t_first ? -1.0 : 1.0;
    }
    return;
  }
  if ((row % 3) != 0) {
    return;
  }
  double e_value = 0.0;
  for (NnzOffset p = row_ptr_org[row]; p < row_ptr_org[row + 1]; ++p) {
    if (col_val_org[p] == e_col) {
      e_value = nz_val_org[p];
      break;
    }
  }

  const std::int32_t first_col = t_col < e_col ? t_col : e_col;
  const std::int32_t second_col = t_col < e_col ? e_col : t_col;
  const double first_value = t_col < e_col ? e_value : -e_value;
  const double second_value = t_col < e_col ? -e_value : e_value;
  bool wrote_first = false;
  bool wrote_second = false;
  for (NnzOffset p = row_ptr_org[row]; p < row_ptr_org[row + 1]; ++p) {
    const std::int32_t col = col_val_org[p];
    const double val = nz_val_org[p];
    if (col == e_col) {
      continue;
    }
    if (!wrote_first && first_col < col) {
      col_val_new[write_ptr] = first_col;
      nz_val_new[write_ptr] = first_value;
      ++write_ptr;
      wrote_first = true;
    }
    if (!wrote_second && second_col < col) {
      col_val_new[write_ptr] = second_col;
      nz_val_new[write_ptr] = second_value;
      ++write_ptr;
      wrote_second = true;
    }
    col_val_new[write_ptr] = col;
    nz_val_new[write_ptr] = val;
    ++write_ptr;
  }
  if (!wrote_first) {
    col_val_new[write_ptr] = first_col;
    nz_val_new[write_ptr] = first_value;
    ++write_ptr;
  }
  if (!wrote_second) {
    col_val_new[write_ptr] = second_col;
    nz_val_new[write_ptr] = second_value;
  }
}

__global__ void _kernel_structural_graph_extract_blocks(std::int32_t* status_flag,
                                                        std::int32_t* coupling_rows,
                                                        std::int32_t* t_cols,
                                                        std::int32_t* e_cols,
                                                        double* rhos,
                                                        const std::uint8_t* eq_row,
                                                        const std::uint8_t* zero_lower_two_nnz,
                                                        const std::int32_t* zl_col1,
                                                        const std::int32_t* zl_col2,
                                                        const double* zl_val1,
                                                        const double* zl_val2,
                                                        const NnzOffset* row_ptr,
                                                        const std::int32_t* col_val,
                                                        const double* nz_val,
                                                        const NnzOffset* at_row_ptr,
                                                        const std::int32_t* at_col_val,
                                                        const double* c,
                                                        const double* l,
                                                        const double* u,
                                                        std::int32_t n,
                                                        std::int32_t nblocks) {
  const NnzOffset idx = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (idx >= nblocks) {
    return;
  }

  const std::int32_t eq = 4 * idx;
  const std::int32_t r1 = eq + 1;
  const std::int32_t r2 = eq + 2;
  const std::int32_t r3 = eq + 3;
  if (eq_row[eq] == std::uint8_t{0} ||
      zero_lower_two_nnz[r1] == std::uint8_t{0} ||
      zero_lower_two_nnz[r2] == std::uint8_t{0} ||
      zero_lower_two_nnz[r3] == std::uint8_t{0}) {
    atomicMax(status_flag, 1);
    return;
  }

  std::int32_t t_col = -1;
  std::int32_t e_col = -1;
  double rho = 0.0;
  if (!_sls_extract_l1_orientation_device(
          zl_col1[r1], zl_col2[r1], zl_val1[r1], zl_val2[r1],
          zl_val1[r2], zl_val2[r2], &t_col, &e_col, &rho)) {
    atomicMax(status_flag, 1);
    return;
  }

  double coeff_t = 0.0;
  double coeff_s = 0.0;
  if (zl_col1[r3] == t_col) {
    coeff_t = zl_val1[r3];
    coeff_s = zl_val2[r3];
  } else if (zl_col2[r3] == t_col) {
    coeff_t = zl_val2[r3];
    coeff_s = zl_val1[r3];
  } else {
    atomicMax(status_flag, 1);
    return;
  }
  if (!(coeff_t < -ABSOLUTE_VALUE_EPIGRAPH_SUB_TOL && coeff_s > ABSOLUTE_VALUE_EPIGRAPH_SUB_TOL)) {
    atomicMax(status_flag, 1);
    return;
  }

  // Complete the three-row substitution certificate during block extraction.
  // Removing the fourth row and its auxiliary cannot change these incidences,
  // so no second CSR/CSC read is needed after the targeted dual fix.
  if (t_col < 0 || t_col >= n || e_col < 0 || e_col >= n || t_col == e_col) {
    atomicMax(status_flag, 1);
    return;
  }
  const bool valid_bounds = isinf(l[e_col]) && l[e_col] < 0.0 &&
                            isinf(u[e_col]) && u[e_col] > 0.0 &&
                            l[t_col] == 0.0 &&
                            isinf(u[t_col]) && u[t_col] > 0.0 &&
                            isfinite(c[t_col]) && c[t_col] >= 0.0 &&
                            isfinite(c[e_col]);
  bool transformed_costs_valid = valid_bounds && isfinite(rho) && rho > 0.0;
  if (transformed_costs_valid) {
    const double scaled_e_cost = c[e_col] / rho;
    transformed_costs_valid = isfinite(scaled_e_cost) &&
                              isfinite(c[t_col] + scaled_e_cost) &&
                              isfinite(c[t_col] - scaled_e_cost);
  }
  if (!valid_bounds || !transformed_costs_valid) {
    atomicMax(status_flag, 1);
    return;
  }

  std::int32_t e_count = 0;
  std::int32_t t_count = 0;
  std::int32_t previous_col = -1;
  bool source_row_valid = true;
  const NnzOffset row_start = row_ptr[eq];
  const NnzOffset row_stop = row_ptr[eq + 1];
  for (NnzOffset p = row_start; p < row_stop; ++p) {
    const std::int32_t col = col_val[p];
    if (col <= previous_col || !isfinite(nz_val[p]) || nz_val[p] == 0.0) {
      source_row_valid = false;
    }
    previous_col = col;
    e_count += col == e_col ? 1 : 0;
    t_count += col == t_col ? 1 : 0;
  }

  std::int32_t seen_eq = 0;
  std::int32_t seen_r1 = 0;
  std::int32_t seen_r2 = 0;
  bool at_valid = true;
  for (NnzOffset p = at_row_ptr[e_col]; p < at_row_ptr[e_col + 1]; ++p) {
    const std::int32_t row = at_col_val[p];
    if (row == eq) {
      ++seen_eq;
    } else if (row == r1) {
      ++seen_r1;
    } else if (row == r2) {
      ++seen_r2;
    } else {
      at_valid = false;
    }
  }

  if (!(source_row_valid && e_count == 1 && t_count == 0 && at_valid &&
        seen_eq == 1 && seen_r1 == 1 && seen_r2 == 1)) {
    atomicMax(status_flag, 1);
    return;
  }

  coupling_rows[idx] = eq;
  t_cols[idx] = t_col;
  e_cols[idx] = e_col;
  rhos[idx] = rho;
}

__global__ void _kernel_structural_graph_mark_extra_rows(std::int32_t* status_flag,
                                                         std::uint8_t* block_bad,
                                                         std::int32_t* row_to_block,
                                                         std::int32_t* row_slack_col,
                                                         double* row_factor,
                                                         std::int32_t* role_owner,
                                                         const NnzOffset* row_ptr,
                                                         const std::int32_t* col_val,
                                                         const double* nz_val,
                                                         const NnzOffset* at_row_ptr,
                                                         const std::int32_t* at_col_val,
                                                         const double* AL,
                                                         const double* AU,
                                                         const std::int32_t* coupling_rows,
                                                         const std::int32_t* t_cols,
                                                         std::int32_t block_count,
                                                         std::int32_t n) {
  const NnzOffset idx = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (idx >= block_count) {
    return;
  }

  const std::int32_t t_col = t_cols[idx];
  const std::int32_t r1 = coupling_rows[idx] + 1;
  const std::int32_t r2 = coupling_rows[idx] + 2;
  for (NnzOffset p = at_row_ptr[t_col]; p < at_row_ptr[t_col + 1]; ++p) {
    const std::int32_t row = at_col_val[p];
    if (row == r1 || row == r2) {
      continue;
    }

    bool valid_row = AL[row] == 0.0 && isinf(AU[row]) && AU[row] > 0.0;
    const NnzOffset row_start = row_ptr[row];
    const NnzOffset row_stop = row_ptr[row + 1];
    valid_row = valid_row && ((row_stop - row_start) == 2);

    double coeff_t = 0.0;
    double coeff_s = 0.0;
    std::int32_t slack_col = -1;
    if (valid_row) {
      const std::int32_t c1 = col_val[row_start];
      const std::int32_t c2 = col_val[row_start + 1];
      const double v1 = nz_val[row_start];
      const double v2 = nz_val[row_start + 1];
      valid_row = c1 != c2 && isfinite(v1) && isfinite(v2);
      if (valid_row && c1 == t_col) {
        coeff_t = v1;
        coeff_s = v2;
        slack_col = c2;
      } else if (valid_row && c2 == t_col) {
        coeff_t = v2;
        coeff_s = v1;
        slack_col = c1;
      } else {
        valid_row = false;
      }
    }
    valid_row = valid_row && slack_col >= 0 && slack_col < n &&
                coeff_t < -ABSOLUTE_VALUE_EPIGRAPH_SUB_TOL &&
                coeff_s > ABSOLUTE_VALUE_EPIGRAPH_SUB_TOL;
    const double factor = valid_row ? -coeff_t / coeff_s : 0.0;
    valid_row = valid_row && isfinite(factor) && factor > 0.0;
    if (!valid_row) {
      block_bad[idx] = std::uint8_t{1};
      atomicMax(status_flag, 1);
      continue;
    }

    const std::int32_t existing = atomicCAS(row_to_block + row, -1, idx);
    if (existing != -1 && existing != idx) {
      block_bad[idx] = std::uint8_t{1};
      atomicMax(status_flag, 1);
      continue;
    }
    row_slack_col[row] = slack_col;
    row_factor[row] = factor;
    // -2 marks an auxiliary; another block may claim the same column, but a
    // t/e role may not also be an auxiliary.
    const std::int32_t owner = atomicCAS(role_owner + slack_col, -1, -2);
    if (owner != -1 && owner != -2) {
      block_bad[idx] = std::uint8_t{1};
      atomicMax(status_flag, 1);
    }
  }
}

__global__ void _kernel_structural_graph_validate_required_rows(
    std::int32_t* status_flag,
    std::uint8_t* block_bad,
    const std::int32_t* row_to_block,
    const std::int32_t* row_slack_col,
    const double* row_factor,
    const std::uint8_t* row_removable,
    const std::int32_t* coupling_rows,
    std::int32_t block_count,
    std::int32_t n) {
  const NnzOffset idx = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (idx >= block_count) {
    return;
  }
  const std::int32_t required_row = coupling_rows[idx] + 3;
  const std::int32_t slack_col = row_slack_col[required_row];
  const double factor = row_factor[required_row];
  if (row_to_block[required_row] != idx || slack_col < 0 || slack_col >= n ||
      !isfinite(factor) || factor <= 0.0 ||
      row_removable[required_row] == std::uint8_t{0}) {
    block_bad[idx] = std::uint8_t{1};
    atomicMax(status_flag, 1);
  }
}

__global__ void _kernel_structural_graph_dual_fix_auxiliaries(std::int32_t* status_flag,
                                                            std::uint8_t* row_removable,
                                                            const std::int32_t* role_owner,
                                                            const std::int32_t* row_to_block,
                                                            const std::int32_t* row_slack_col,
                                                            const NnzOffset* at_row_ptr,
                                                            const std::int32_t* at_col_val,
                                                            const double* at_nz_val,
                                                            const double* AL,
                                                            const double* AU,
                                                            const double* c,
                                                            const double* l,
                                                            const double* u,
                                                            std::int32_t m,
                                                            std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (col >= n || role_owner[col] != -2) {
    return;
  }

  // This is the zero-cost/no-up-lock case of dual_fix, restricted to the
  // auxiliaries already identified by the four-row pattern.  Every incident
  // row must be a certified lower-only s-beta*t row.  Its column is read once
  // even if thousands of blocks share it.
  bool valid = c[col] == 0.0 &&
               !isnan(l[col]) && !(isinf(l[col]) && l[col] > 0.0) &&
               isinf(u[col]) && u[col] > 0.0;
  for (NnzOffset p = at_row_ptr[col]; valid && p < at_row_ptr[col + 1]; ++p) {
    const std::int32_t row = at_col_val[p];
    valid = row >= 0 && row < m && row_to_block[row] >= 0 &&
            row_slack_col[row] == col &&
            at_nz_val[p] > ABSOLUTE_VALUE_EPIGRAPH_SUB_TOL &&
            AL[row] == 0.0 && isinf(AU[row]) && AU[row] > 0.0;
    if (valid) {
      row_removable[row] = std::uint8_t{1};
    }
  }
  if (!valid) {
    atomicMax(status_flag, 1);
    return;
  }
}

__global__ void _kernel_structural_graph_delete_auxiliaries(
    std::uint8_t* keep_col,
    const std::int32_t* role_owner,
    std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (col < n && role_owner[col] == -2) {
    keep_col[col] = std::uint8_t{0};
  }
}

__global__ void _kernel_structural_graph_apply_plan(std::int32_t* status_flag,
                                                    std::uint8_t* keep_row,
                                                    double* new_c,
                                                    double* new_l,
                                                    double* new_u,
                                                    const std::uint8_t* block_bad,
                                                    const std::uint8_t* row_removable,
                                                    const NnzOffset* at_row_ptr,
                                                    const std::int32_t* at_col_val,
                                                    const double* c,
                                                    const double* l,
                                                    const double* u,
                                                    const std::int32_t* coupling_rows,
                                                    const std::int32_t* t_cols,
                                                    const std::int32_t* e_cols,
                                                    const double* rhos,
                                                    std::int32_t block_count) {
  const NnzOffset idx = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (idx >= block_count) {
    return;
  }
  if (block_bad[idx] != std::uint8_t{0}) {
    atomicMax(status_flag, 1);
    return;
  }

  const std::int32_t eq = coupling_rows[idx];
  const std::int32_t r1 = eq + 1;
  const std::int32_t r2 = eq + 2;
  const std::int32_t t_col = t_cols[idx];
  const std::int32_t e_col = e_cols[idx];
  const double rho = rhos[idx];

  keep_row[r1] = std::uint8_t{0};
  keep_row[r2] = std::uint8_t{0};
  for (NnzOffset p = at_row_ptr[t_col]; p < at_row_ptr[t_col + 1]; ++p) {
    const std::int32_t row = at_col_val[p];
    if (row == r1 || row == r2) {
      continue;
    }
    if (row_removable[row] != std::uint8_t{0}) {
      keep_row[row] = std::uint8_t{0};
    } else {
      atomicMax(status_flag, 1);
      return;
    }
  }

  const double inv_rho = 1.0 / rho;
  const double c_t_old = c[t_col];
  const double c_e_old = c[e_col];
  new_c[t_col] = c_t_old + c_e_old * inv_rho;
  new_c[e_col] = c_t_old - c_e_old * inv_rho;
  new_l[t_col] = 0.0;
  new_l[e_col] = 0.0;
  new_u[t_col] = INFINITY;
  new_u[e_col] = INFINITY;
  (void)l;
  (void)u;
}

__global__ void _kernel_structural_graph_count_rows(NnzOffset* row_nnz_new,
                                                    const std::uint8_t* coupling_keep,
                                                    const NnzOffset* row_ptr,
                                                    std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }
  row_nnz_new[row] = coupling_keep[row] != std::uint8_t{0} ? (row_ptr[row + 1] - row_ptr[row]) + 1 : 0;
}

__global__ void _kernel_structural_graph_build_rewrite_maps(std::uint8_t* coupling_keep,
                                                            std::int32_t* row_to_block,
                                                            const std::int32_t* coupling_rows,
                                                            std::int32_t block_count) {
  const NnzOffset idx = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (idx < block_count) {
    const std::int32_t row = coupling_rows[idx];
    coupling_keep[row] = std::uint8_t{1};
    row_to_block[row] = idx;
  }
}

__global__ void _kernel_structural_graph_fill(std::int32_t* col_val_new,
                                              double* nz_val_new,
                                              const NnzOffset* row_ptr_new,
                                              const NnzOffset* row_ptr_org,
                                              const std::int32_t* col_val_org,
                                              const double* nz_val_org,
                                              const std::uint8_t* coupling_keep,
                                              const std::int32_t* row_to_block,
                                              const std::int32_t* t_cols,
                                              const std::int32_t* e_cols,
                                              const double* rhos,
                                              std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= m || coupling_keep[row] == std::uint8_t{0}) {
    return;
  }
  const std::int32_t block_idx = row_to_block[row];
  const std::int32_t t_col = t_cols[block_idx];
  const std::int32_t e_col = e_cols[block_idx];
  const double rho = rhos[block_idx];
  NnzOffset write_ptr = row_ptr_new[row];

  double e_value = 0.0;
  for (NnzOffset p = row_ptr_org[row]; p < row_ptr_org[row + 1]; ++p) {
    if (col_val_org[p] == e_col) {
      e_value = nz_val_org[p] / rho;
      break;
    }
  }

  const std::int32_t first_col = t_col < e_col ? t_col : e_col;
  const std::int32_t second_col = t_col < e_col ? e_col : t_col;
  const double first_value = t_col < e_col ? e_value : -e_value;
  const double second_value = t_col < e_col ? -e_value : e_value;
  bool wrote_first = false;
  bool wrote_second = false;
  for (NnzOffset p = row_ptr_org[row]; p < row_ptr_org[row + 1]; ++p) {
    const std::int32_t col = col_val_org[p];
    if (col == e_col) {
      continue;
    }
    if (!wrote_first && first_col < col) {
      col_val_new[write_ptr] = first_col;
      nz_val_new[write_ptr] = first_value;
      ++write_ptr;
      wrote_first = true;
    }
    if (!wrote_second && second_col < col) {
      col_val_new[write_ptr] = second_col;
      nz_val_new[write_ptr] = second_value;
      ++write_ptr;
      wrote_second = true;
    }
    col_val_new[write_ptr] = col;
    nz_val_new[write_ptr] = nz_val_org[p];
    ++write_ptr;
  }
  if (!wrote_first) {
    col_val_new[write_ptr] = first_col;
    nz_val_new[write_ptr] = first_value;
    ++write_ptr;
  }
  if (!wrote_second) {
    col_val_new[write_ptr] = second_col;
    nz_val_new[write_ptr] = second_value;
  }
}

__global__ void _kernel_structural_outer_validate_pairs_and_build_free_to_bound(
    std::int32_t* status_flag,
    std::int32_t* free_to_bound,
    const std::int32_t* bound_cols,
    const std::int32_t* free_cols,
    const double* c,
    double* accumulated_cost,
    const double* l,
    const double* u,
    std::int32_t pair_count,
    std::int32_t n) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  // The objective update below deliberately permits several distinct free
  // columns to map to one bound column.  The reverse is not a substitution:
  // one free column cannot be owned by two pairs, and no column can be both a
  // substitution source and destination in the same structural step.
  for (std::int32_t i = 0; i < pair_count; ++i) {
    const std::int32_t bound_col = bound_cols[i];
    const std::int32_t free_col = free_cols[i];
    if (bound_col < 0 || bound_col >= n || free_col < 0 || free_col >= n ||
        bound_col == free_col || l[bound_col] != 0.0 ||
        !isfinite(u[bound_col]) || u[bound_col] < 0.0 ||
        !isinf(l[free_col]) || l[free_col] >= 0.0 ||
        !isinf(u[free_col]) || u[free_col] <= 0.0 ||
        !isfinite(c[bound_col]) || !isfinite(c[free_col])) {
      status_flag[0] = 1;
      return;
    }
    const double next_cost = accumulated_cost[bound_col] + c[free_col];
    if (!isfinite(next_cost)) {
      status_flag[0] = 1;
      return;
    }
    accumulated_cost[bound_col] = next_cost;
    if (free_to_bound[free_col] != STRUCTURAL_OUTER_UNASSIGNED) {
      status_flag[0] = 1;
      return;
    }
    free_to_bound[free_col] = bound_col;
  }

  for (std::int32_t i = 0; i < pair_count; ++i) {
    const std::int32_t bound_col = bound_cols[i];
    if (free_to_bound[bound_col] >= 0) {
      status_flag[0] = 1;
      return;
    }
    // Negative role markers are ignored by _sls_outer_map_col_device, so the
    // same array can cheaply validate the later q/e/s roles without another
    // allocation or a quadratic pair scan.
    free_to_bound[bound_col] = STRUCTURAL_OUTER_BOUND_ROLE;
  }
}

__global__ void _kernel_structural_outer_extract_blocks(std::int32_t* status_flag,
                                                        std::int32_t* q_cols,
                                                        std::int32_t* e_cols,
                                                        std::int32_t* s_cols,
                                                        double* alphas,
                                                        std::int32_t* local_x_cols,
                                                        const std::int32_t* free_to_bound,
                                                        const std::uint8_t* eq_row,
                                                        const std::uint8_t* eq_zero_two_nnz,
                                                        const std::uint8_t* lower_two_nnz,
                                                        const std::uint8_t* zero_lower_two_nnz,
                                                        const std::int32_t* raw_col1,
                                                        const std::int32_t* raw_col2,
                                                        const double* raw_val1,
                                                        const double* raw_val2,
                                                        const std::int32_t* zl_col1,
                                                        const std::int32_t* zl_col2,
                                                        const double* zl_val1,
                                                        const double* zl_val2,
                                                        std::int32_t start_row,
                                                        std::int32_t block_count) {
  const NnzOffset idx = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (idx >= block_count) {
    return;
  }

  const std::int32_t dense_row = start_row + idx * 8;
  const std::int32_t ep1 = dense_row + 1;
  const std::int32_t ep2 = dense_row + 2;
  const std::int32_t link = dense_row + 3;

  if (eq_row[dense_row] == std::uint8_t{0} ||
      zero_lower_two_nnz[ep1] == std::uint8_t{0} ||
      zero_lower_two_nnz[ep2] == std::uint8_t{0} ||
      eq_zero_two_nnz[link] == std::uint8_t{0}) {
    atomicMax(status_flag, 1);
    return;
  }

  std::int32_t q_col = -1;
  std::int32_t e_col = -1;
  if (!_sls_extract_l1_pair_device(
          zl_col1[ep1], zl_col2[ep1], zl_val1[ep1], zl_val2[ep1],
          zl_val1[ep2], zl_val2[ep2], &q_col, &e_col)) {
    atomicMax(status_flag, 1);
    return;
  }

  const std::int32_t link_c1 = raw_col1[link];
  const std::int32_t link_c2 = raw_col2[link];
  const double link_v1 = raw_val1[link];
  const double link_v2 = raw_val2[link];
  std::int32_t s_col = -1;
  double alpha = 0.0;
  if (link_c1 == q_col) {
    s_col = link_c2;
    alpha = link_v1;
    if (!_structural_exact(link_v2, -1.0)) {
      atomicMax(status_flag, 1);
      return;
    }
  } else if (link_c2 == q_col) {
    s_col = link_c1;
    alpha = link_v2;
    if (!_structural_exact(link_v1, -1.0)) {
      atomicMax(status_flag, 1);
      return;
    }
  } else {
    atomicMax(status_flag, 1);
    return;
  }
  if (!isfinite(alpha) || alpha <= 0.0 || q_col == e_col) {
    atomicMax(status_flag, 1);
    return;
  }

  for (std::int32_t k = 0; k < 4; ++k) {
    const std::int32_t rr = link + 1 + k;
    if (lower_two_nnz[rr] == std::uint8_t{0}) {
      atomicMax(status_flag, 1);
      return;
    }
    const std::int32_t loc_c1 = raw_col1[rr];
    const std::int32_t loc_c2 = raw_col2[rr];
    const double loc_v1 = raw_val1[rr];
    const double loc_v2 = raw_val2[rr];
    std::int32_t x_col = -1;
    if (loc_c1 == s_col) {
      if (!_structural_exact(loc_v1, -1.0) || !_structural_exact(loc_v2, 1.0)) {
        atomicMax(status_flag, 1);
        return;
      }
      x_col = loc_c2;
    } else if (loc_c2 == s_col) {
      if (!_structural_exact(loc_v2, -1.0) || !_structural_exact(loc_v1, 1.0)) {
        atomicMax(status_flag, 1);
        return;
      }
      x_col = loc_c1;
    } else {
      atomicMax(status_flag, 1);
      return;
    }

    const std::int32_t mapped_x = _sls_outer_map_col_device(free_to_bound, x_col);
    if (mapped_x == q_col || mapped_x == e_col) {
      atomicMax(status_flag, 1);
      return;
    }
    local_x_cols[idx * 4 + k] = x_col;
  }

  q_cols[idx] = q_col;
  e_cols[idx] = e_col;
  s_cols[idx] = s_col;
  alphas[idx] = alpha;
}

__global__ void _kernel_structural_outer_mark_block_roles(
    std::int32_t* status_flag,
    std::int32_t* free_to_bound,
    const std::int32_t* q_cols,
    const std::int32_t* e_cols,
    const std::int32_t* s_cols,
    std::int32_t block_count,
    std::int32_t n) {
  const NnzOffset idx = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  const std::int32_t role_count = 3 * block_count;
  if (idx >= role_count) {
    return;
  }

  const std::int32_t block = idx / 3;
  const std::int32_t role_idx = idx % 3;
  const std::int32_t role =
      role_idx == 0 ? q_cols[block] : (role_idx == 1 ? e_cols[block] : s_cols[block]);
  if (role < 0 || role >= n) {
    atomicMax(status_flag, 1);
    return;
  }

  // For a valid match every q/e/s column has exactly one owner and is not an
  // outer endpoint.  On an invalid match the winning marker is irrelevant:
  // status prevents every plan mutation below.  Negative role markers leave
  // the actual free-to-bound mapping unchanged.
  const std::int32_t previous = atomicCAS(
      free_to_bound + role, STRUCTURAL_OUTER_UNASSIGNED, STRUCTURAL_OUTER_BLOCK_ROLE);
  if (previous != STRUCTURAL_OUTER_UNASSIGNED) {
    atomicMax(status_flag, 1);
  }
}

__global__ void _kernel_structural_outer_validate_local_x_roles(
    std::int32_t* status_flag,
    const std::int32_t* free_to_bound,
    const std::int32_t* local_x_cols,
    std::int32_t local_x_count,
    std::int32_t n) {
  const NnzOffset idx = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (idx >= local_x_count) {
    return;
  }

  // Original x columns may legitimately be outer endpoints, but a block's
  // auxiliary q/e/s column cannot simultaneously be an x column in any other
  // block: the latter use would otherwise escape that block's rewrite.
  const std::int32_t x_col = local_x_cols[idx];
  if (x_col < 0 || x_col >= n || free_to_bound[x_col] == STRUCTURAL_OUTER_BLOCK_ROLE) {
    atomicMax(status_flag, 1);
  }
}

__global__ void _kernel_structural_outer_pair_apply(std::uint8_t* keep_row,
                                                    std::uint8_t* keep_col,
                                                    double* new_c,
                                                    const double* c,
                                                    const std::int32_t* bound_cols,
                                                    const std::int32_t* free_cols,
                                                    std::int32_t pair_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  // Pair order is the canonical summation order.  A serial accumulation is
  // intentional: shared bound columns are legal, while atomicAdd would make
  // the floating-point result depend on scheduling.
  for (NnzOffset idx = 0; idx < pair_count; ++idx) {
    const std::int32_t bound_col = bound_cols[idx];
    const std::int32_t free_col = free_cols[idx];
    const std::int32_t row1 = 3 * idx;
    keep_row[row1] = std::uint8_t{0};
    keep_row[row1 + 1] = std::uint8_t{0};
    keep_row[row1 + 2] = std::uint8_t{0};
    keep_col[free_col] = std::uint8_t{0};
    new_c[bound_col] += c[free_col];
  }
}

__global__ void _kernel_structural_outer_block_validate(
    std::int32_t* status_flag,
    const double* c,
    const double* l,
    const double* u,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const std::int32_t* free_to_bound,
    const std::int32_t* q_cols,
    const std::int32_t* e_cols,
    const std::int32_t* s_cols,
    const double* alphas,
    const std::int32_t* local_x_cols,
    std::int32_t start_row,
    std::int32_t block_count,
    std::int32_t n) {
  const NnzOffset idx = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (idx >= block_count) {
    return;
  }

  const std::int32_t dense_row = start_row + idx * 8;
  const std::int32_t q_col = q_cols[idx];
  const std::int32_t e_col = e_cols[idx];
  const std::int32_t s_col = s_cols[idx];
  if (q_col < 0 || q_col >= n || e_col < 0 || e_col >= n || s_col < 0 || s_col >= n) {
    atomicMax(status_flag, 1);
    return;
  }

  const bool valid_bounds = isinf(l[e_col]) && l[e_col] < 0.0 &&
                            isinf(u[e_col]) && u[e_col] > 0.0 &&
                            l[q_col] == 0.0 && isinf(u[q_col]) && u[q_col] > 0.0 &&
                            l[s_col] == 0.0 && isinf(u[s_col]) && u[s_col] > 0.0 &&
                            isfinite(c[q_col]) && isfinite(c[e_col]) && isfinite(c[s_col]);
  const double alpha = alphas[idx];
  bool transformed_costs_valid = valid_bounds && isfinite(alpha) && alpha > 0.0;
  if (transformed_costs_valid) {
    const double linked_cost = alpha * c[s_col];
    transformed_costs_valid = isfinite(linked_cost) &&
                              isfinite(c[q_col] + c[e_col] + linked_cost) &&
                              isfinite(c[q_col] - c[e_col] + linked_cost);
  }
  if (!valid_bounds || !transformed_costs_valid) {
    atomicMax(status_flag, 1);
    return;
  }

  for (std::int32_t k = 0; k < 4; ++k) {
    const std::int32_t x_col = local_x_cols[idx * 4 + k];
    if (x_col < 0 || x_col >= n ||
        !(l[x_col] == 0.0 && isinf(u[x_col]) && u[x_col] > 0.0 &&
          isfinite(c[x_col]) && c[x_col] > ABSOLUTE_VALUE_EPIGRAPH_SUB_TOL)) {
      atomicMax(status_flag, 1);
      return;
    }
  }

  std::int32_t e_occurrences = 0;
  const NnzOffset row_start = row_ptr[dense_row];
  const NnzOffset row_stop = row_ptr[dense_row + 1];
  for (NnzOffset p = row_start; p < row_stop; ++p) {
    const std::int32_t col_p = col_val[p];
    // The specialized fill below rewrites exactly this block's e column in
    // its dense row.  Any q/e/s role owned by this or another block would
    // otherwise escape the global change of variables (and an s column would
    // subsequently be removed while its coefficient remained in the row).
    // Requiring exactly one own-e occurrence also certifies the +1 row-nnz
    // allocation used by _kernel_structural_outer_count_rows.
    if (free_to_bound[col_p] == STRUCTURAL_OUTER_BLOCK_ROLE) {
      if (col_p != e_col) {
        atomicMax(status_flag, 1);
        return;
      }
      ++e_occurrences;
    }
  }

  if (e_occurrences != 1) {
    atomicMax(status_flag, 1);
  }
}

__global__ void _kernel_structural_outer_block_apply(std::uint8_t* keep_row,
                                                     std::uint8_t* keep_col,
                                                     double* new_c,
                                                     double* new_l,
                                                     double* new_u,
                                                     const double* c,
                                                     const std::int32_t* q_cols,
                                                     const std::int32_t* e_cols,
                                                     const std::int32_t* s_cols,
                                                     const double* alphas,
                                                     std::int32_t start_row,
                                                     std::int32_t block_count) {
  const NnzOffset idx = static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (idx >= block_count) {
    return;
  }

  const std::int32_t dense_row = start_row + idx * 8;
  const std::int32_t ep1 = dense_row + 1;
  const std::int32_t ep2 = dense_row + 2;
  const std::int32_t link = dense_row + 3;
  const std::int32_t q_col = q_cols[idx];
  const std::int32_t e_col = e_cols[idx];
  const std::int32_t s_col = s_cols[idx];
  const double alpha = alphas[idx];

  keep_row[ep1] = std::uint8_t{0};
  keep_row[ep2] = std::uint8_t{0};
  keep_row[link] = std::uint8_t{0};
  keep_col[s_col] = std::uint8_t{0};
  const double c_e_old = c[e_col];
  const double c_q_old = c[q_col];
  const double c_s_old = c[s_col];
  new_c[q_col] = c_e_old + c_q_old + alpha * c_s_old;
  new_c[e_col] = -c_e_old + c_q_old + alpha * c_s_old;
  new_l[q_col] = 0.0;
  new_l[e_col] = 0.0;
  new_u[q_col] = INFINITY;
  new_u[e_col] = INFINITY;
}

__global__ void _kernel_structural_outer_count_rows(NnzOffset* row_nnz_new,
                                                    const NnzOffset* row_ptr,
                                                    std::int32_t start_row,
                                                    std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }
  if (row < start_row) {
    row_nnz_new[row] = 0;
  } else {
    const NnzOffset offset = row - start_row;
    const std::int32_t rem8 = offset % 8;
    if (rem8 == 0 || rem8 >= 4) {
      row_nnz_new[row] = (row_ptr[row + 1] - row_ptr[row]) + 1;
    } else {
      row_nnz_new[row] = 0;
    }
  }
}

__global__ void _kernel_structural_outer_fill(std::int32_t* col_val_new,
                                              double* nz_val_new,
                                              const NnzOffset* row_ptr_new,
                                              const NnzOffset* row_ptr_org,
                                              const std::int32_t* col_val_org,
                                              const double* nz_val_org,
                                              const std::int32_t* free_to_bound,
                                              std::int32_t start_row,
                                              const std::int32_t* q_cols,
                                              const std::int32_t* e_cols,
                                              const std::int32_t* s_cols,
                                              const double* alphas,
                                              std::int32_t m) {
  const std::int32_t row = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= m || row < start_row) {
    return;
  }

  const NnzOffset offset = row - start_row;
  const std::int32_t rem8 = offset % 8;
  if (!(rem8 == 0 || rem8 >= 4)) {
    return;
  }

  const std::int32_t block_idx = offset / 8;
  const std::int32_t q_col = q_cols[block_idx];
  const std::int32_t e_col = e_cols[block_idx];
  const std::int32_t s_col = s_cols[block_idx];
  const double alpha = alphas[block_idx];
  NnzOffset write_ptr = row_ptr_new[row];

  for (NnzOffset p = row_ptr_org[row]; p < row_ptr_org[row + 1]; ++p) {
    const std::int32_t col_org = col_val_org[p];
    const double val = nz_val_org[p];
    const std::int32_t mapped = _sls_outer_map_col_device(free_to_bound, col_org);
    if (rem8 == 0 && col_org == e_col) {
      const bool q_first = q_col < e_col;
      col_val_new[write_ptr] = q_first ? q_col : e_col;
      nz_val_new[write_ptr] = q_first ? val : -val;
      ++write_ptr;
      col_val_new[write_ptr] = q_first ? e_col : q_col;
      nz_val_new[write_ptr] = q_first ? -val : val;
      ++write_ptr;
    } else if (rem8 >= 4 && col_org == s_col) {
      const double coeff = val * alpha;
      const bool q_first = q_col < e_col;
      col_val_new[write_ptr] = q_first ? q_col : e_col;
      nz_val_new[write_ptr] = coeff;
      ++write_ptr;
      col_val_new[write_ptr] = q_first ? e_col : q_col;
      nz_val_new[write_ptr] = coeff;
      ++write_ptr;
    } else {
      col_val_new[write_ptr] = mapped;
      nz_val_new[write_ptr] = val;
      ++write_ptr;
    }
  }
}

DeviceCsrMatrix build_structural_outer_new_A(const DeviceCsrMatrix& source,
                                             const std::int32_t* free_to_bound,
                                             std::int32_t start_row,
                                             const std::int32_t* q_cols,
                                             const std::int32_t* e_cols,
                                             const std::int32_t* s_cols,
                                             const double* alphas,
                                             bool* rows_sorted_unique) {
  DeviceCsrMatrix out;
  *rows_sorted_unique = true;
  out.rows = source.rows;
  out.cols = source.cols;
  throw_if_cuda_error(cudaMalloc(&out.rowPtr, sizeof(NnzOffset) * static_cast<std::size_t>(out.rows + 1)),
                      "cudaMalloc structural outer rowPtr");

  if (out.rows == 0) {
    throw_if_cuda_error(cudaMemset(out.rowPtr, 0, sizeof(NnzOffset)), "cudaMemset structural outer empty rowPtr");
    return out;
  }

  NnzOffset* row_counts = nullptr;
  throw_if_cuda_error(cudaMalloc(&row_counts, sizeof(NnzOffset) * static_cast<std::size_t>(out.rows)),
                      "cudaMalloc structural outer row_counts");
  const int row_blocks = (out.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_structural_outer_count_rows<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
      row_counts, source.rowPtr, start_row, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_count_rows");
  _kernel_inclusive_scan_i32_serial<<<1, 1>>>(row_counts, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_inclusive_scan_i32_serial structural outer");
  throw_if_cuda_error(cudaMemcpy(&out.nnz, row_counts + out.rows - 1, sizeof(NnzOffset), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural outer nnz");
  _kernel_row_ptr_from_prefix<<<row_blocks, GPU_PRESOLVE_THREADS>>>(out.rowPtr, row_counts, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_row_ptr_from_prefix structural outer");

  if (out.nnz > 0) {
    throw_if_cuda_error(cudaMalloc(&out.colVal, sizeof(std::int32_t) * static_cast<std::size_t>(out.nnz)),
                        "cudaMalloc structural outer colVal");
    throw_if_cuda_error(cudaMalloc(&out.nzVal, sizeof(double) * static_cast<std::size_t>(out.nnz)),
                        "cudaMalloc structural outer nzVal");
    _kernel_structural_outer_fill<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
        out.colVal, out.nzVal, out.rowPtr, source.rowPtr, source.colVal, source.nzVal,
        free_to_bound, start_row, q_cols, e_cols, s_cols, alphas, out.rows);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_fill");
  }
  // Most generated rows are already ordered.  Mapping an outer free column to
  // an earlier bound column can, however, reorder an otherwise sorted dense
  // row.  Detect that cheap fast path first and invoke the parallel sort only
  // when the mapping actually disturbed column order.
  *rows_sorted_unique = ensure_structural_csr_rows_sorted_unique(
      out, "structural outer CSR row sort");
  cudaFree(row_counts);
  return out;
}

DeviceCsrMatrix build_absolute_value_epigraph_split_new_A(const DeviceCsrMatrix& source,
                                                const std::int32_t* t_cols,
                                                const std::int32_t* e_cols,
                                                const std::uint8_t* preserve_bound_row) {
  DeviceCsrMatrix out;
  out.rows = source.rows;
  out.cols = source.cols;
  throw_if_cuda_error(cudaMalloc(&out.rowPtr, sizeof(NnzOffset) * static_cast<std::size_t>(out.rows + 1)),
                      "cudaMalloc structural l1 split rowPtr");

  if (out.rows == 0) {
    throw_if_cuda_error(cudaMemset(out.rowPtr, 0, sizeof(NnzOffset)), "cudaMemset structural l1 split empty rowPtr");
    return out;
  }

  NnzOffset* row_counts = nullptr;
  throw_if_cuda_error(cudaMalloc(&row_counts, sizeof(NnzOffset) * static_cast<std::size_t>(out.rows)),
                      "cudaMalloc structural l1 split row_counts");
  const int row_blocks = (out.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_absolute_value_epigraph_split_count_rows<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
      row_counts, source.rowPtr, preserve_bound_row, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_absolute_value_epigraph_split_count_rows");
  _kernel_inclusive_scan_i32_serial<<<1, 1>>>(row_counts, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_inclusive_scan_i32_serial structural l1 split");
  throw_if_cuda_error(cudaMemcpy(&out.nnz, row_counts + out.rows - 1, sizeof(NnzOffset), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural l1 split nnz");
  _kernel_row_ptr_from_prefix<<<row_blocks, GPU_PRESOLVE_THREADS>>>(out.rowPtr, row_counts, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_row_ptr_from_prefix structural l1 split");

  if (out.nnz > 0) {
    throw_if_cuda_error(cudaMalloc(&out.colVal, sizeof(std::int32_t) * static_cast<std::size_t>(out.nnz)),
                        "cudaMalloc structural l1 split colVal");
    throw_if_cuda_error(cudaMalloc(&out.nzVal, sizeof(double) * static_cast<std::size_t>(out.nnz)),
                        "cudaMalloc structural l1 split nzVal");
    _kernel_absolute_value_epigraph_split_fill<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
        out.colVal, out.nzVal, out.rowPtr, source.rowPtr, source.colVal, source.nzVal,
        t_cols, e_cols, preserve_bound_row, out.rows);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_absolute_value_epigraph_split_fill");
  }
  cudaFree(row_counts);
  return out;
}

DeviceCsrMatrix build_structural_graph_new_A(const DeviceCsrMatrix& source,
                                             const std::int32_t* coupling_rows,
                                             const std::int32_t* t_cols,
                                             const std::int32_t* e_cols,
                                             const double* rhos,
                                             std::int32_t block_count) {
  DeviceCsrMatrix out;
  out.rows = source.rows;
  out.cols = source.cols;
  throw_if_cuda_error(cudaMalloc(&out.rowPtr, sizeof(NnzOffset) * static_cast<std::size_t>(out.rows + 1)),
                      "cudaMalloc structural graph rowPtr");
  if (out.rows == 0) {
    throw_if_cuda_error(cudaMemset(out.rowPtr, 0, sizeof(NnzOffset)), "cudaMemset structural graph empty rowPtr");
    return out;
  }

  std::uint8_t* coupling_keep = nullptr;
  std::int32_t* row_to_block = nullptr;
  NnzOffset* row_counts = nullptr;
  throw_if_cuda_error(cudaMalloc(&coupling_keep, static_cast<std::size_t>(out.rows)),
                      "cudaMalloc structural graph coupling_keep");
  throw_if_cuda_error(cudaMalloc(&row_to_block, sizeof(std::int32_t) * static_cast<std::size_t>(out.rows)),
                      "cudaMalloc structural graph row_to_block");
  throw_if_cuda_error(cudaMalloc(&row_counts, sizeof(NnzOffset) * static_cast<std::size_t>(out.rows)),
                      "cudaMalloc structural graph row_counts");
  throw_if_cuda_error(cudaMemset(coupling_keep, 0, static_cast<std::size_t>(out.rows)),
                      "cudaMemset structural graph coupling_keep");
  _kernel_fill_i32<<<(out.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS, GPU_PRESOLVE_THREADS>>>(
      row_to_block, -1, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_fill_i32 structural graph row_to_block");
  if (block_count > 0) {
    const int block_blocks = (block_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    _kernel_structural_graph_build_rewrite_maps<<<block_blocks, GPU_PRESOLVE_THREADS>>>(
        coupling_keep, row_to_block, coupling_rows, block_count);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_graph_build_rewrite_maps");
  }

  const int row_blocks = (out.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_structural_graph_count_rows<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
      row_counts, coupling_keep, source.rowPtr, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_graph_count_rows");
  _kernel_inclusive_scan_i32_serial<<<1, 1>>>(row_counts, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_inclusive_scan_i32_serial structural graph");
  throw_if_cuda_error(cudaMemcpy(&out.nnz, row_counts + out.rows - 1, sizeof(NnzOffset), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural graph nnz");
  _kernel_row_ptr_from_prefix<<<row_blocks, GPU_PRESOLVE_THREADS>>>(out.rowPtr, row_counts, out.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_row_ptr_from_prefix structural graph");

  if (out.nnz > 0) {
    throw_if_cuda_error(cudaMalloc(&out.colVal, sizeof(std::int32_t) * static_cast<std::size_t>(out.nnz)),
                        "cudaMalloc structural graph colVal");
    throw_if_cuda_error(cudaMalloc(&out.nzVal, sizeof(double) * static_cast<std::size_t>(out.nnz)),
                        "cudaMalloc structural graph nzVal");
    _kernel_structural_graph_fill<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
        out.colVal, out.nzVal, out.rowPtr, source.rowPtr, source.colVal, source.nzVal,
        coupling_keep, row_to_block, t_cols, e_cols, rhos, out.rows);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_graph_fill");
  }

  cudaFree(coupling_keep);
  cudaFree(row_to_block);
  cudaFree(row_counts);
  return out;
}

template <class T>
std::vector<T> copy_device_vector(const T* device, std::int32_t count, const char* context) {
  std::vector<T> values(static_cast<std::size_t>(count));
  if (count > 0) {
    throw_if_cuda_error(cudaMemcpy(values.data(), device, sizeof(T) * static_cast<std::size_t>(count),
                                   cudaMemcpyDeviceToHost),
                        context);
  }
  return values;
}

void record_l1_split_recovery(PresolvePlanGpu& plan,
                              const std::int32_t* t_cols,
                              const std::int32_t* e_cols,
                              const double* rhos,
                              std::int32_t count,
                              const char* pattern) {
  if (plan.tape.resident) {
    capture_absolute_value_splits_device(plan, t_cols, e_cols, rhos, count, pattern);
    return;
  }
  const std::vector<std::int32_t> t_h = copy_device_vector(t_cols, count, "cudaMemcpy structural recovery t_cols");
  const std::vector<std::int32_t> e_h = copy_device_vector(e_cols, count, "cudaMemcpy structural recovery e_cols");
  std::vector<double> rho_h(static_cast<std::size_t>(count), 1.0);
  if (rhos != nullptr) {
    rho_h = copy_device_vector(rhos, count, "cudaMemcpy structural recovery rhos");
  }
  plan.absolute_value_epigraph_recovery.pattern = pattern;
  plan.absolute_value_epigraph_recovery.splits.clear();
  plan.absolute_value_epigraph_recovery.outer_pairs.clear();
  plan.absolute_value_epigraph_recovery.linked_slacks.clear();
  plan.absolute_value_epigraph_recovery.max_slacks.clear();
  plan.absolute_value_epigraph_recovery.splits.reserve(static_cast<std::size_t>(count));
  for (std::int32_t i = 0; i < count; ++i) {
    plan.absolute_value_epigraph_recovery.splits.push_back({t_h[static_cast<std::size_t>(i)],
                                                       e_h[static_cast<std::size_t>(i)],
                                                       rho_h[static_cast<std::size_t>(i)]});
  }
  plan.has_absolute_value_epigraph_recovery = true;
}

bool try_apply_l1_split(PresolvePlanGpu& plan,
                        const LPInfoGpu& lp,
                        const DeviceCsrMatrix& source_A,
                        const std::uint8_t* eq_row,
                        const std::uint8_t* zero_lower_two_nnz,
                        const std::int32_t* zl_col1,
                        const std::int32_t* zl_col2,
                        const double* zl_val1,
                        const double* zl_val2,
                        AbsoluteValueEpigraphResidualBoundMode residual_bound_mode,
                        double residual_bound_as_free_min,
                        bool profile) {
  const std::int32_t m = source_A.rows;
  const std::int32_t n = source_A.cols;
  if (m < 3 || (m % 3) != 0 || plan.has_new_A ||
      lp.AT.rows != n || lp.AT.cols != m) {
    return false;
  }
  const std::int32_t nblocks = m / 3;
  const int blocks = (nblocks + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  std::int32_t* status_flag = nullptr;
  std::int32_t* t_cols = nullptr;
  std::int32_t* e_cols = nullptr;
  std::int32_t* role_owner = nullptr;
  std::uint8_t* preserve_bound_row = nullptr;
  throw_if_cuda_error(cudaMalloc(&status_flag, sizeof(std::int32_t)), "cudaMalloc structural l1 split status");
  throw_if_cuda_error(cudaMalloc(&t_cols, sizeof(std::int32_t) * static_cast<std::size_t>(nblocks)),
                      "cudaMalloc structural l1 split t_cols");
  throw_if_cuda_error(cudaMalloc(&e_cols, sizeof(std::int32_t) * static_cast<std::size_t>(nblocks)),
                      "cudaMalloc structural l1 split e_cols");
  throw_if_cuda_error(cudaMalloc(&role_owner, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      "cudaMalloc structural l1 split role_owner");
  throw_if_cuda_error(cudaMalloc(&preserve_bound_row, static_cast<std::size_t>(nblocks)),
                      "cudaMalloc structural l1 split preserve_bound_row");
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)), "cudaMemset structural l1 split status");

  _kernel_absolute_value_epigraph_split_extract_pairs<<<blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag, t_cols, e_cols, eq_row, zero_lower_two_nnz,
      zl_col1, zl_col2, zl_val1, zl_val2, nblocks);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_absolute_value_epigraph_split_extract_pairs");
  std::int32_t status = 0;
  throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural l1 split extract status");
  if (profile) {
    std::cerr << ">>> [absolute_value_epigraph C++] l1_split extract status=" << status
              << " nblocks=" << nblocks << "\n";
  }
  if (status != 0) {
    cudaFree(status_flag);
    cudaFree(t_cols);
    cudaFree(e_cols);
    cudaFree(role_owner);
    cudaFree(preserve_bound_row);
    return false;
  }

  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)),
                      "cudaMemset structural l1 split ownership status");
  _kernel_fill_i32<<<(n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS,
                     GPU_PRESOLVE_THREADS>>>(role_owner, -1, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_fill_i32 structural l1 split role_owner");
  _kernel_absolute_value_epigraph_split_claim_roles<<<blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag, role_owner, t_cols, e_cols, nblocks, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_absolute_value_epigraph_split_claim_roles");
  throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural l1 split ownership status");
  if (profile) {
    std::cerr << ">>> [absolute_value_epigraph C++] l1_split ownership status=" << status << "\n";
  }
  if (status != 0) {
    cudaFree(status_flag);
    cudaFree(t_cols);
    cudaFree(e_cols);
    cudaFree(role_owner);
    cudaFree(preserve_bound_row);
    return false;
  }

  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)),
                      "cudaMemset structural l1 split validation status");
  _kernel_absolute_value_epigraph_split_validate_blocks<<<blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag, preserve_bound_row, plan.keep_row_mask, plan.keep_col_mask,
      plan.new_c, plan.new_l, plan.new_u,
      source_A.rowPtr, source_A.colVal, source_A.nzVal,
      lp.AT.rowPtr, lp.AT.colVal, lp.AT.nzVal,
      t_cols, e_cols, static_cast<std::uint8_t>(residual_bound_mode),
      residual_bound_as_free_min, nblocks);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_absolute_value_epigraph_split_validate_blocks");
  throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural l1 split validation status");
  if (profile) {
    std::cerr << ">>> [absolute_value_epigraph C++] l1_split validation status=" << status << "\n";
  }
  if (status != 0) {
    cudaFree(status_flag);
    cudaFree(t_cols);
    cudaFree(e_cols);
    cudaFree(role_owner);
    cudaFree(preserve_bound_row);
    return false;
  }

  // Commit only after every block has passed the global role and incidence
  // certificate.  A rejected structural match therefore leaves the shared
  // presolve plan exactly as it was on entry.
  DeviceCsrMatrix rewritten_A = build_absolute_value_epigraph_split_new_A(
      source_A, t_cols, e_cols, preserve_bound_row);
  _kernel_absolute_value_epigraph_split_apply_plan<<<blocks, GPU_PRESOLVE_THREADS>>>(
      plan.keep_row_mask, plan.new_c, plan.new_l, plan.new_u,
      plan.new_AL, plan.new_AU, preserve_bound_row, t_cols, e_cols, nblocks);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_absolute_value_epigraph_split_apply_plan");
  throw_if_cuda_error(cudaDeviceSynchronize(), "structural l1 split apply synchronize");

  plan.new_A = rewritten_A;
  plan.has_new_A = true;
  plan.has_change = true;
  plan.has_col_action = true;
  plan.has_row_action = true;
  record_l1_split_recovery(plan, t_cols, e_cols, nullptr, nblocks, "l1_split_3row");
  if (profile) {
    std::cerr << ">>> [absolute_value_epigraph C++] l1_split changed=1 new_A_nnz=" << plan.new_A.nnz << "\n";
  }

  cudaFree(status_flag);
  cudaFree(t_cols);
  cudaFree(e_cols);
  cudaFree(role_owner);
  cudaFree(preserve_bound_row);
  return true;
}

bool try_apply_graph_l1(PresolvePlanGpu& plan,
                        const LPInfoGpu& lp,
                        const DeviceCsrMatrix& source_A,
                        const std::uint8_t* eq_row,
                        const std::uint8_t* zero_lower_two_nnz,
                        const std::int32_t* zl_col1,
                        const std::int32_t* zl_col2,
                        const double* zl_val1,
                        const double* zl_val2,
                        bool profile) {
  const std::int32_t m = source_A.rows;
  const std::int32_t n = source_A.cols;
  if (m < 4 || (m % 4) != 0 || plan.has_new_A ||
      lp.AT.rows != n || lp.AT.cols != m) {
    return false;
  }
  const std::int32_t block_count = m / 4;
  const int block_blocks = (block_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const int row_blocks = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;

  std::int32_t* status_flag = nullptr;
  std::int32_t* coupling_rows = nullptr;
  std::int32_t* t_cols = nullptr;
  std::int32_t* e_cols = nullptr;
  std::int32_t* role_owner = nullptr;
  double* rhos = nullptr;
  std::uint8_t* block_bad = nullptr;
  std::int32_t* row_to_block = nullptr;
  std::int32_t* row_slack_col = nullptr;
  double* row_factor = nullptr;
  std::uint8_t* row_removable = nullptr;

  throw_if_cuda_error(cudaMalloc(&status_flag, sizeof(std::int32_t)), "cudaMalloc structural graph status");
  throw_if_cuda_error(cudaMalloc(&coupling_rows, sizeof(std::int32_t) * static_cast<std::size_t>(block_count)),
                      "cudaMalloc structural graph coupling_rows");
  throw_if_cuda_error(cudaMalloc(&t_cols, sizeof(std::int32_t) * static_cast<std::size_t>(block_count)),
                      "cudaMalloc structural graph t_cols");
  throw_if_cuda_error(cudaMalloc(&e_cols, sizeof(std::int32_t) * static_cast<std::size_t>(block_count)),
                      "cudaMalloc structural graph e_cols");
  throw_if_cuda_error(cudaMalloc(&rhos, sizeof(double) * static_cast<std::size_t>(block_count)),
                      "cudaMalloc structural graph rhos");
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)), "cudaMemset structural graph status");

  _kernel_structural_graph_extract_blocks<<<block_blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag, coupling_rows, t_cols, e_cols, rhos, eq_row, zero_lower_two_nnz,
      zl_col1, zl_col2, zl_val1, zl_val2,
      source_A.rowPtr, source_A.colVal, source_A.nzVal,
      lp.AT.rowPtr, lp.AT.colVal,
      plan.new_c, plan.new_l, plan.new_u, n, block_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_graph_extract_blocks");
  std::int32_t status = 0;
  throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural graph extract status");
  if (profile) {
    std::cerr << ">>> [absolute_value_epigraph C++] graph extract status=" << status
              << " block_count=" << block_count << "\n";
  }
  if (status != 0) {
    cudaFree(status_flag);
    cudaFree(coupling_rows);
    cudaFree(t_cols);
    cudaFree(e_cols);
    cudaFree(rhos);
    return false;
  }

  // Claim every t/e role on device.  Besides catching cross-role reuse, this
  // keeps graph validation linear when a model contains many four-row blocks.
  throw_if_cuda_error(cudaMalloc(&role_owner, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      "cudaMalloc structural graph role_owner");
  _kernel_fill_i32<<<(n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS,
                     GPU_PRESOLVE_THREADS>>>(role_owner, -1, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_fill_i32 structural graph role_owner");
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)),
                      "cudaMemset structural graph ownership status");
  _kernel_absolute_value_epigraph_split_claim_roles<<<block_blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag, role_owner, t_cols, e_cols, block_count, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_graph_claim_roles");
  throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural graph ownership status");
  if (status != 0) {
    cudaFree(status_flag);
    cudaFree(coupling_rows);
    cudaFree(t_cols);
    cudaFree(e_cols);
    cudaFree(role_owner);
    cudaFree(rhos);
    return false;
  }

  throw_if_cuda_error(cudaMalloc(&block_bad, static_cast<std::size_t>(block_count)),
                      "cudaMalloc structural graph block_bad");
  throw_if_cuda_error(cudaMalloc(&row_to_block, sizeof(std::int32_t) * static_cast<std::size_t>(m)),
                      "cudaMalloc structural graph row_to_block");
  throw_if_cuda_error(cudaMalloc(&row_slack_col, sizeof(std::int32_t) * static_cast<std::size_t>(m)),
                      "cudaMalloc structural graph row_slack_col");
  throw_if_cuda_error(cudaMalloc(&row_factor, sizeof(double) * static_cast<std::size_t>(m)),
                      "cudaMalloc structural graph row_factor");
  throw_if_cuda_error(cudaMalloc(&row_removable, static_cast<std::size_t>(m)),
                      "cudaMalloc structural graph row_removable");
  throw_if_cuda_error(cudaMemset(block_bad, 0, static_cast<std::size_t>(block_count)),
                      "cudaMemset structural graph block_bad");
  _kernel_fill_i32<<<row_blocks, GPU_PRESOLVE_THREADS>>>(row_to_block, -1, m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_fill_i32 structural graph row_to_block");
  _kernel_fill_i32<<<row_blocks, GPU_PRESOLVE_THREADS>>>(row_slack_col, -1, m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_fill_i32 structural graph row_slack_col");
  throw_if_cuda_error(cudaMemset(row_factor, 0, sizeof(double) * static_cast<std::size_t>(m)),
                      "cudaMemset structural graph row_factor");
  throw_if_cuda_error(cudaMemset(row_removable, 0, static_cast<std::size_t>(m)),
                      "cudaMemset structural graph row_removable");
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)), "cudaMemset structural graph validate status");

  _kernel_structural_graph_mark_extra_rows<<<block_blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag, block_bad, row_to_block, row_slack_col, row_factor, role_owner,
      source_A.rowPtr, source_A.colVal, source_A.nzVal, lp.AT.rowPtr, lp.AT.colVal,
      plan.new_AL, plan.new_AU, coupling_rows, t_cols, block_count, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_graph_mark_extra_rows");
  const int col_blocks = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_structural_graph_dual_fix_auxiliaries<<<col_blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag, row_removable, role_owner, row_to_block, row_slack_col,
      lp.AT.rowPtr, lp.AT.colVal, lp.AT.nzVal, plan.new_AL, plan.new_AU,
      plan.new_c, plan.new_l, plan.new_u, m, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_graph_dual_fix_auxiliaries");
  _kernel_structural_graph_validate_required_rows<<<block_blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag, block_bad, row_to_block, row_slack_col, row_factor, row_removable,
      coupling_rows, block_count, n);
  throw_if_cuda_error(cudaGetLastError(),
                      "_kernel_structural_graph_validate_required_rows");
  throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural graph validate status");
  if (profile) {
    std::cerr << ">>> [absolute_value_epigraph C++] graph validate status=" << status << "\n";
  }
  if (status != 0) {
    cudaFree(status_flag);
    cudaFree(coupling_rows);
    cudaFree(t_cols);
    cudaFree(e_cols);
    cudaFree(role_owner);
    cudaFree(rhos);
    cudaFree(block_bad);
    cudaFree(row_to_block);
    cudaFree(row_slack_col);
    cudaFree(row_factor);
    cudaFree(row_removable);
    return false;
  }
  DeviceCsrMatrix rewritten_A = build_structural_graph_new_A(
      source_A, coupling_rows, t_cols, e_cols, rhos, block_count);
  _kernel_structural_graph_apply_plan<<<block_blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag, plan.keep_row_mask, plan.new_c, plan.new_l, plan.new_u,
      block_bad, row_removable, lp.AT.rowPtr, lp.AT.colVal,
      plan.new_c, plan.new_l, plan.new_u, coupling_rows, t_cols, e_cols, rhos, block_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_graph_apply_plan");
  _kernel_structural_graph_delete_auxiliaries<<<col_blocks, GPU_PRESOLVE_THREADS>>>(
      plan.keep_col_mask, role_owner, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_graph_delete_auxiliaries");
  throw_if_cuda_error(cudaDeviceSynchronize(), "structural graph apply synchronize");
  throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural graph apply status");
  if (profile) {
    std::cerr << ">>> [absolute_value_epigraph C++] graph apply status=" << status << "\n";
  }
  if (status != 0) {
    cudaFree(rewritten_A.rowPtr);
    cudaFree(rewritten_A.colVal);
    cudaFree(rewritten_A.nzVal);
    cudaFree(status_flag);
    cudaFree(coupling_rows);
    cudaFree(t_cols);
    cudaFree(e_cols);
    cudaFree(role_owner);
    cudaFree(rhos);
    cudaFree(block_bad);
    cudaFree(row_to_block);
    cudaFree(row_slack_col);
    cudaFree(row_factor);
    cudaFree(row_removable);
    return false;
  }

  plan.new_A = rewritten_A;
  plan.has_new_A = true;
  plan.has_change = true;
  plan.has_col_action = true;
  plan.has_row_action = true;
  record_l1_split_recovery(plan, t_cols, e_cols, rhos, block_count, "graph_l1_substitution");

  if (plan.tape.resident) {
    capture_absolute_value_max_slacks_device(plan, t_cols, row_to_block, row_slack_col,
                                             row_factor, row_removable, m, n);
  } else {
    const std::vector<std::int32_t> t_h =
        copy_device_vector(t_cols, block_count, "cudaMemcpy graph t_cols");

    const std::vector<std::int32_t> row_block_h = copy_device_vector(row_to_block, m, "cudaMemcpy graph row_to_block");
    const std::vector<std::int32_t> row_slack_h = copy_device_vector(row_slack_col, m, "cudaMemcpy graph row_slack_col");
    const std::vector<double> row_factor_h = copy_device_vector(row_factor, m, "cudaMemcpy graph row_factor");
    const std::vector<std::uint8_t> row_removable_h =
        copy_device_vector(row_removable, m, "cudaMemcpy graph row_removable");
    std::vector<std::int32_t> recovery_by_slack(static_cast<std::size_t>(n), -1);
    for (std::int32_t row = 0; row < m; ++row) {
      if (row_removable_h[static_cast<std::size_t>(row)] == std::uint8_t{0}) {
        continue;
      }
      const std::int32_t block_idx = row_block_h[static_cast<std::size_t>(row)];
      const std::int32_t slack_col = row_slack_h[static_cast<std::size_t>(row)];
      if (block_idx >= 0 && slack_col >= 0) {
        std::int32_t recovery_idx = recovery_by_slack[static_cast<std::size_t>(slack_col)];
        if (recovery_idx < 0) {
          plan.absolute_value_epigraph_recovery.max_slacks.push_back({slack_col, {}, {}});
          recovery_idx = static_cast<std::int32_t>(
              plan.absolute_value_epigraph_recovery.max_slacks.size() - 1);
          recovery_by_slack[static_cast<std::size_t>(slack_col)] = recovery_idx;
        }
        AbsoluteValueMaxSlackRecovery& recovery =
            plan.absolute_value_epigraph_recovery.max_slacks[static_cast<std::size_t>(recovery_idx)];
        recovery.t_cols.push_back(t_h[static_cast<std::size_t>(block_idx)]);
        recovery.factors.push_back(row_factor_h[static_cast<std::size_t>(row)]);
      }
    }

  }

  if (profile) {
    std::cerr << ">>> [absolute_value_epigraph C++] graph changed=1 new_A_nnz=" << plan.new_A.nnz
              << " max_slacks=" << (plan.absolute_value_epigraph_device
                  ? plan.absolute_value_epigraph_device->max_slacks.size()
                  : plan.absolute_value_epigraph_recovery.max_slacks.size()) << "\n";
  }

  cudaFree(status_flag);
  cudaFree(coupling_rows);
  cudaFree(t_cols);
  cudaFree(e_cols);
  cudaFree(role_owner);
  cudaFree(rhos);
  cudaFree(block_bad);
  cudaFree(row_to_block);
  cudaFree(row_slack_col);
  cudaFree(row_factor);
  cudaFree(row_removable);
  return true;
}

struct PrefixRow2 {
  std::uint8_t is_eq = 0;
  std::uint8_t zero_lower_two_nnz = 0;
  std::int32_t zl_col1 = -1;
  std::int32_t zl_col2 = -1;
  double zl_val1 = 0.0;
  double zl_val2 = 0.0;
};

__device__ PrefixRow2 _sls_prefix_row2_device(std::int32_t row,
                                              const NnzOffset* row_ptr,
                                              const std::int32_t* col_val,
                                              const double* nz_val,
                                              const double* AL,
                                              const double* AU) {
  PrefixRow2 out;
  const double al = AL[row];
  const double au = AU[row];
  const NnzOffset lo = row_ptr[row];
  const NnzOffset hi = row_ptr[row + 1];
  const bool is_eq = isfinite(al) && isfinite(au) && al == au;
  out.is_eq = is_eq ? std::uint8_t{1} : std::uint8_t{0};
  if (hi - lo != 2) {
    return out;
  }

  std::int32_t c1 = col_val[lo];
  std::int32_t c2 = col_val[lo + 1];
  double v1 = nz_val[lo];
  double v2 = nz_val[lo + 1];
  if (c2 < c1) {
    const std::int32_t tc = c1;
    c1 = c2;
    c2 = tc;
    const double tv = v1;
    v1 = v2;
    v2 = tv;
  }

  const bool is_zero_lower = al == 0.0 && isinf(au) && au > 0.0;
  const bool is_zero_upper = isinf(al) && al < 0.0 && au == 0.0;
  if (is_zero_lower || is_zero_upper) {
    const double scale = is_zero_lower ? 1.0 : -1.0;
    out.zero_lower_two_nnz = std::uint8_t{1};
    out.zl_col1 = c1;
    out.zl_col2 = c2;
    out.zl_val1 = scale * v1;
    out.zl_val2 = scale * v2;
  }
  return out;
}

__device__ bool _sls_prefix_has_l1_pair_device(const PrefixRow2& a, const PrefixRow2& b) {
  if (a.zl_col1 != b.zl_col1 || a.zl_col2 != b.zl_col2) {
    return false;
  }
  const bool first_is_t =
      isfinite(a.zl_val1) && a.zl_val1 > 0.0 &&
      _structural_exact(a.zl_val1, b.zl_val1) &&
      _structural_exact(a.zl_val2, -a.zl_val1) &&
      _structural_exact(b.zl_val2, b.zl_val1);
  const bool second_is_t =
      isfinite(a.zl_val2) && a.zl_val2 > 0.0 &&
      _structural_exact(a.zl_val2, b.zl_val2) &&
      _structural_exact(a.zl_val1, -a.zl_val2) &&
      _structural_exact(b.zl_val1, b.zl_val2);
  return first_is_t || second_is_t;
}

__device__ bool _sls_prefix_has_outer_pair_signature_device(const PrefixRow2& a,
                                                            const PrefixRow2& b,
                                                            const PrefixRow2& c) {
  if (a.zl_col1 != b.zl_col1 || a.zl_col1 != c.zl_col1 ||
      a.zl_col2 != b.zl_col2 || a.zl_col2 != c.zl_col2) {
    return false;
  }
  return _structural_exact(a.zl_val1, 1.0) && _structural_exact(a.zl_val2, -1.0) &&
         _structural_exact(b.zl_val1, 1.0) && _structural_exact(b.zl_val2, 1.0) &&
         _structural_exact(c.zl_val1, -1.0) && _structural_exact(c.zl_val2, 1.0);
}

__device__ bool _sls_prefix_possible_l1_split_device(const NnzOffset* row_ptr,
                                                     const std::int32_t* col_val,
                                                     const double* nz_val,
                                                     const double* AL,
                                                     const double* AU,
                                                     std::int32_t m) {
  if (m < 3 || (m % 3) != 0) {
    return false;
  }
  const PrefixRow2 eq = _sls_prefix_row2_device(0, row_ptr, col_val, nz_val, AL, AU);
  const PrefixRow2 r1 = _sls_prefix_row2_device(1, row_ptr, col_val, nz_val, AL, AU);
  const PrefixRow2 r2 = _sls_prefix_row2_device(2, row_ptr, col_val, nz_val, AL, AU);
  return eq.is_eq != 0 &&
         r1.zero_lower_two_nnz != 0 &&
         r2.zero_lower_two_nnz != 0 &&
         _sls_prefix_has_l1_pair_device(r1, r2);
}

__device__ bool _sls_prefix_possible_outer_pair_device(const NnzOffset* row_ptr,
                                                       const std::int32_t* col_val,
                                                       const double* nz_val,
                                                       const double* AL,
                                                       const double* AU,
                                                       std::int32_t m) {
  if (m < 11) {
    return false;
  }
  const PrefixRow2 r1 = _sls_prefix_row2_device(0, row_ptr, col_val, nz_val, AL, AU);
  const PrefixRow2 r2 = _sls_prefix_row2_device(1, row_ptr, col_val, nz_val, AL, AU);
  const PrefixRow2 r3 = _sls_prefix_row2_device(2, row_ptr, col_val, nz_val, AL, AU);
  return r1.zero_lower_two_nnz != 0 &&
         r2.zero_lower_two_nnz != 0 &&
         r3.zero_lower_two_nnz != 0 &&
         _sls_prefix_has_outer_pair_signature_device(r1, r2, r3);
}

__device__ bool _sls_prefix_possible_graph_device(const NnzOffset* row_ptr,
                                                  const std::int32_t* col_val,
                                                  const double* nz_val,
                                                  const double* AL,
                                                  const double* AU,
                                                  std::int32_t m) {
  if (m < 4 || (m % 4) != 0) {
    return false;
  }
  const PrefixRow2 eq = _sls_prefix_row2_device(0, row_ptr, col_val, nz_val, AL, AU);
  const PrefixRow2 r1 = _sls_prefix_row2_device(1, row_ptr, col_val, nz_val, AL, AU);
  const PrefixRow2 r2 = _sls_prefix_row2_device(2, row_ptr, col_val, nz_val, AL, AU);
  const PrefixRow2 r3 = _sls_prefix_row2_device(3, row_ptr, col_val, nz_val, AL, AU);
  const bool has_pos = r3.zl_val1 > ABSOLUTE_VALUE_EPIGRAPH_SUB_TOL || r3.zl_val2 > ABSOLUTE_VALUE_EPIGRAPH_SUB_TOL;
  const bool has_neg = r3.zl_val1 < -ABSOLUTE_VALUE_EPIGRAPH_SUB_TOL || r3.zl_val2 < -ABSOLUTE_VALUE_EPIGRAPH_SUB_TOL;
  const bool overlaps = r3.zl_col1 == r1.zl_col1 || r3.zl_col1 == r1.zl_col2 ||
                        r3.zl_col2 == r1.zl_col1 || r3.zl_col2 == r1.zl_col2;
  return eq.is_eq != 0 &&
         r1.zero_lower_two_nnz != 0 &&
         r2.zero_lower_two_nnz != 0 &&
         _sls_prefix_has_l1_pair_device(r1, r2) &&
         r3.zero_lower_two_nnz != 0 &&
         has_pos && has_neg && overlaps;
}

__global__ void _kernel_structural_prefix_auto_screen(std::uint8_t* screen_flag,
                                                      const NnzOffset* row_ptr,
                                                      const std::int32_t* col_val,
                                                      const double* nz_val,
                                                      const double* AL,
                                                      const double* AU,
                                                      std::int32_t m,
                                                      std::int32_t n) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const bool passed =
        m > 0 && n > 0 &&
        (_sls_prefix_possible_graph_device(row_ptr, col_val, nz_val, AL, AU, m) ||
         _sls_prefix_possible_outer_pair_device(row_ptr, col_val, nz_val, AL, AU, m) ||
         _sls_prefix_possible_l1_split_device(row_ptr, col_val, nz_val, AL, AU, m));
    screen_flag[0] = passed ? std::uint8_t{1} : std::uint8_t{0};
  }
}

}  // namespace

bool absolute_value_epigraph_prefix_screen_passes(const LPInfoGpu& lp) {
  if (lp.A.rows <= 0 || lp.A.cols <= 0) {
    return false;
  }
  std::uint8_t* screen_flag = nullptr;
  throw_if_cuda_error(cudaMalloc(&screen_flag, sizeof(std::uint8_t)), "cudaMalloc structural prefix screen");
  _kernel_structural_prefix_auto_screen<<<1, 1>>>(
      screen_flag, lp.A.rowPtr, lp.A.colVal, lp.A.nzVal, lp.AL, lp.AU, lp.A.rows, lp.A.cols);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_prefix_auto_screen");
  std::uint8_t host_flag = 0;
  throw_if_cuda_error(cudaMemcpy(&host_flag, screen_flag, sizeof(std::uint8_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural prefix screen");
  cudaFree(screen_flag);
  return host_flag != std::uint8_t{0};
}

void apply_rule_absolute_value_epigraph_reduction(PresolvePlanGpu& plan,
                                           const LPInfoGpu& lp,
                                           const PresolveStatsGpu& stats,
                                           const PresolveParams& pparams) {
  (void)stats;
  if (plan.has_infeasible || plan.has_unbounded || !pparams.enable_absolute_value_epigraph_reduction) {
    return;
  }

  const DeviceCsrMatrix& source_A = plan.has_new_A ? plan.new_A : lp.A;
  const std::int32_t m = source_A.rows;
  const std::int32_t n = source_A.cols;
  const bool profile = runtime::options().profile_absolute_value;
  if (m < 3 || n < 2) {
    if (profile) {
      std::cerr << ">>> [absolute_value_epigraph C++] skip small m=" << m << " n=" << n << "\n";
    }
    return;
  }

  const int row_blocks = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  std::uint8_t* eq_row = nullptr;
  std::uint8_t* eq_zero_two_nnz = nullptr;
  std::uint8_t* lower_two_nnz = nullptr;
  std::uint8_t* zero_lower_two_nnz = nullptr;
  std::int32_t* raw_col1 = nullptr;
  std::int32_t* raw_col2 = nullptr;
  double* raw_val1 = nullptr;
  double* raw_val2 = nullptr;
  std::int32_t* zl_col1 = nullptr;
  std::int32_t* zl_col2 = nullptr;
  double* zl_val1 = nullptr;
  double* zl_val2 = nullptr;
  std::int32_t* status_flag = nullptr;
  std::int32_t* pair_count_d = nullptr;
  std::int32_t* start_row_d = nullptr;
  std::int32_t* bound_cols = nullptr;
  std::int32_t* free_cols = nullptr;
  std::int32_t* free_to_bound = nullptr;
  double* pair_cost_after = nullptr;
  std::int32_t* q_cols = nullptr;
  std::int32_t* e_cols = nullptr;
  std::int32_t* s_cols = nullptr;
  double* alphas = nullptr;
  std::int32_t* local_x_cols = nullptr;
  std::int32_t status = 0;
  std::int32_t pair_count = 0;
  std::int32_t start_row = 0;

  throw_if_cuda_error(cudaMalloc(&eq_row, static_cast<std::size_t>(m)), "cudaMalloc structural eq_row");
  throw_if_cuda_error(cudaMalloc(&eq_zero_two_nnz, static_cast<std::size_t>(m)), "cudaMalloc structural eq_zero");
  throw_if_cuda_error(cudaMalloc(&lower_two_nnz, static_cast<std::size_t>(m)), "cudaMalloc structural lower_two");
  throw_if_cuda_error(cudaMalloc(&zero_lower_two_nnz, static_cast<std::size_t>(m)), "cudaMalloc structural zero_lower");
  throw_if_cuda_error(cudaMalloc(&raw_col1, sizeof(std::int32_t) * static_cast<std::size_t>(m)), "cudaMalloc structural raw_col1");
  throw_if_cuda_error(cudaMalloc(&raw_col2, sizeof(std::int32_t) * static_cast<std::size_t>(m)), "cudaMalloc structural raw_col2");
  throw_if_cuda_error(cudaMalloc(&raw_val1, sizeof(double) * static_cast<std::size_t>(m)), "cudaMalloc structural raw_val1");
  throw_if_cuda_error(cudaMalloc(&raw_val2, sizeof(double) * static_cast<std::size_t>(m)), "cudaMalloc structural raw_val2");
  throw_if_cuda_error(cudaMalloc(&zl_col1, sizeof(std::int32_t) * static_cast<std::size_t>(m)), "cudaMalloc structural zl_col1");
  throw_if_cuda_error(cudaMalloc(&zl_col2, sizeof(std::int32_t) * static_cast<std::size_t>(m)), "cudaMalloc structural zl_col2");
  throw_if_cuda_error(cudaMalloc(&zl_val1, sizeof(double) * static_cast<std::size_t>(m)), "cudaMalloc structural zl_val1");
  throw_if_cuda_error(cudaMalloc(&zl_val2, sizeof(double) * static_cast<std::size_t>(m)), "cudaMalloc structural zl_val2");
  throw_if_cuda_error(cudaMalloc(&status_flag, sizeof(std::int32_t)), "cudaMalloc structural status");
  throw_if_cuda_error(cudaMalloc(&pair_count_d, sizeof(std::int32_t)), "cudaMalloc structural pair_count");
  throw_if_cuda_error(cudaMalloc(&start_row_d, sizeof(std::int32_t)), "cudaMalloc structural start_row");
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)), "cudaMemset structural status");

  _kernel_structural_row_metadata<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
      eq_row, eq_zero_two_nnz, lower_two_nnz, zero_lower_two_nnz,
      raw_col1, raw_col2, raw_val1, raw_val2, zl_col1, zl_col2, zl_val1, zl_val2,
      source_A.rowPtr, source_A.colVal, source_A.nzVal, plan.new_AL, plan.new_AU, m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_row_metadata");

  if (try_apply_l1_split(plan, lp, source_A, eq_row, zero_lower_two_nnz,
                         zl_col1, zl_col2, zl_val1, zl_val2,
                         pparams.absolute_value_epigraph_residual_bound_mode,
                         pparams.absolute_value_epigraph_residual_bound_as_free_min, profile)) {
    goto cleanup;
  }

  _kernel_structural_outer_count_pairs<<<1, 1>>>(
      status_flag, pair_count_d, start_row_d, zero_lower_two_nnz,
      zl_col1, zl_col2, zl_val1, zl_val2, m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_count_pairs");

  throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural status count_pairs");
  throw_if_cuda_error(cudaMemcpy(&pair_count, pair_count_d, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural pair_count");
  throw_if_cuda_error(cudaMemcpy(&start_row, start_row_d, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy structural start_row");
  if (profile) {
    std::cerr << ">>> [absolute_value_epigraph C++] count_pairs status=" << status
              << " pair_count=" << pair_count << " start_row0=" << start_row
              << " m=" << m << " n=" << n << "\n";
  }
  if (status != 0 || pair_count <= 0) {
    (void)try_apply_graph_l1(plan, lp, source_A, eq_row, zero_lower_two_nnz,
                             zl_col1, zl_col2, zl_val1, zl_val2, profile);
    goto cleanup;
  }

  {
    const int pair_blocks = (pair_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    const std::int32_t block_count = (m - start_row) / 8;
    const int block_blocks = (block_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    const std::int32_t block_role_count = 3 * block_count;
    const int block_role_blocks =
        (block_role_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    const std::int32_t local_x_count = 4 * block_count;
    const int local_x_blocks =
        (local_x_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;

    throw_if_cuda_error(cudaMalloc(&bound_cols, sizeof(std::int32_t) * static_cast<std::size_t>(pair_count)),
                        "cudaMalloc structural bound_cols");
    throw_if_cuda_error(cudaMalloc(&free_cols, sizeof(std::int32_t) * static_cast<std::size_t>(pair_count)),
                        "cudaMalloc structural free_cols");
    _kernel_structural_outer_extract_pairs<<<pair_blocks, GPU_PRESOLVE_THREADS>>>(
        bound_cols, free_cols, zl_col1, zl_col2, pair_count);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_extract_pairs");

    throw_if_cuda_error(cudaMalloc(&free_to_bound, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                        "cudaMalloc structural free_to_bound");
    throw_if_cuda_error(cudaMalloc(&pair_cost_after, sizeof(double) * static_cast<std::size_t>(n)),
                        "cudaMalloc structural pair_cost_after");
    throw_if_cuda_error(cudaMemcpy(pair_cost_after, plan.new_c,
                                   sizeof(double) * static_cast<std::size_t>(n),
                                   cudaMemcpyDeviceToDevice),
                        "cudaMemcpy structural pair_cost_after");
    _kernel_fill_i32<<<(n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS, GPU_PRESOLVE_THREADS>>>(
        free_to_bound, -1, n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_fill_i32 structural free_to_bound");
    throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)),
                        "cudaMemset structural status validate_pairs");
    _kernel_structural_outer_validate_pairs_and_build_free_to_bound<<<1, 1>>>(
        status_flag, free_to_bound, bound_cols, free_cols,
        plan.new_c, pair_cost_after, plan.new_l, plan.new_u, pair_count, n);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_structural_outer_validate_pairs_and_build_free_to_bound");
    throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy structural status validate_pairs");
    cudaFree(pair_cost_after);
    pair_cost_after = nullptr;
    if (profile) {
      std::cerr << ">>> [absolute_value_epigraph C++] validate_pairs status=" << status << "\n";
    }
    if (status != 0) {
      (void)try_apply_graph_l1(plan, lp, source_A, eq_row, zero_lower_two_nnz,
                               zl_col1, zl_col2, zl_val1, zl_val2, profile);
      goto cleanup;
    }

    if (block_count <= 0) {
      if (profile) {
        std::cerr << ">>> [absolute_value_epigraph C++] skip block_count=" << block_count << "\n";
      }
      (void)try_apply_graph_l1(plan, lp, source_A, eq_row, zero_lower_two_nnz,
                               zl_col1, zl_col2, zl_val1, zl_val2, profile);
      goto cleanup;
    }
    throw_if_cuda_error(cudaMalloc(&q_cols, sizeof(std::int32_t) * static_cast<std::size_t>(block_count)),
                        "cudaMalloc structural q_cols");
    throw_if_cuda_error(cudaMalloc(&e_cols, sizeof(std::int32_t) * static_cast<std::size_t>(block_count)),
                        "cudaMalloc structural e_cols");
    throw_if_cuda_error(cudaMalloc(&s_cols, sizeof(std::int32_t) * static_cast<std::size_t>(block_count)),
                        "cudaMalloc structural s_cols");
    throw_if_cuda_error(cudaMalloc(&alphas, sizeof(double) * static_cast<std::size_t>(block_count)),
                        "cudaMalloc structural alphas");
    throw_if_cuda_error(cudaMalloc(&local_x_cols, sizeof(std::int32_t) * static_cast<std::size_t>(4 * block_count)),
                        "cudaMalloc structural local_x_cols");
    throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)), "cudaMemset structural status blocks");
    _kernel_structural_outer_extract_blocks<<<block_blocks, GPU_PRESOLVE_THREADS>>>(
        status_flag, q_cols, e_cols, s_cols, alphas, local_x_cols, free_to_bound,
        eq_row, eq_zero_two_nnz, lower_two_nnz, zero_lower_two_nnz,
        raw_col1, raw_col2, raw_val1, raw_val2, zl_col1, zl_col2, zl_val1, zl_val2,
        start_row, block_count);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_extract_blocks");
    throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy structural status extract_blocks");
    if (profile) {
      std::cerr << ">>> [absolute_value_epigraph C++] extract_blocks status=" << status
                << " block_count=" << block_count << "\n";
    }
    if (status != 0) {
      (void)try_apply_graph_l1(plan, lp, source_A, eq_row, zero_lower_two_nnz,
                               zl_col1, zl_col2, zl_val1, zl_val2, profile);
      goto cleanup;
    }

    // Complete all validation before mutating the plan.  Besides making an
    // invalid match fail closed, this also proves that the aliased c/new_c
    // reads in the two apply kernels cannot race with another variable role.
    throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)),
                        "cudaMemset structural status validate_outer");
    _kernel_structural_outer_mark_block_roles<<<block_role_blocks, GPU_PRESOLVE_THREADS>>>(
        status_flag, free_to_bound, q_cols, e_cols, s_cols, block_count, n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_mark_block_roles");
    // Kernel ordering in the same stream makes every q/e/s marker visible
    // before local-x ownership is checked.
    _kernel_structural_outer_validate_local_x_roles<<<local_x_blocks, GPU_PRESOLVE_THREADS>>>(
        status_flag, free_to_bound, local_x_cols, local_x_count, n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_validate_local_x_roles");
    _kernel_structural_outer_block_validate<<<block_blocks, GPU_PRESOLVE_THREADS>>>(
        status_flag, plan.new_c, plan.new_l, plan.new_u, source_A.rowPtr, source_A.colVal,
        free_to_bound, q_cols, e_cols, s_cols, alphas, local_x_cols,
        start_row, block_count, n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_block_validate");
    throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy structural status validate_outer");
    if (profile) {
      std::cerr << ">>> [absolute_value_epigraph C++] validate_outer status=" << status << "\n";
    }
    if (status != 0) {
      (void)try_apply_graph_l1(plan, lp, source_A, eq_row, zero_lower_two_nnz,
                               zl_col1, zl_col2, zl_val1, zl_val2, profile);
      goto cleanup;
    }

    // Construct and, only when needed, sort the replacement matrix before
    // mutating the shared plan.  Allocation/sort failures therefore cannot
    // leave masks, costs, or bounds partially committed.
    bool rewritten_rows_sorted_unique = false;
    DeviceCsrMatrix rewritten_A = build_structural_outer_new_A(
        source_A, free_to_bound, start_row, q_cols, e_cols, s_cols, alphas,
        &rewritten_rows_sorted_unique);
    if (!rewritten_rows_sorted_unique) {
      cudaFree(rewritten_A.rowPtr);
      cudaFree(rewritten_A.colVal);
      cudaFree(rewritten_A.nzVal);
      goto cleanup;
    }
    _kernel_structural_outer_pair_apply<<<1, 1>>>(
        plan.keep_row_mask, plan.keep_col_mask, plan.new_c,
        plan.new_c, bound_cols, free_cols, pair_count);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_pair_apply");
    _kernel_structural_outer_block_apply<<<block_blocks, GPU_PRESOLVE_THREADS>>>(
        plan.keep_row_mask, plan.keep_col_mask, plan.new_c, plan.new_l, plan.new_u,
        plan.new_c, q_cols, e_cols, s_cols, alphas, start_row, block_count);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_outer_block_apply");
    throw_if_cuda_error(cudaDeviceSynchronize(), "structural outer apply synchronize");
    if (profile) {
      std::cerr << ">>> [absolute_value_epigraph C++] apply complete\n";
    }

    plan.new_A = rewritten_A;
    plan.has_new_A = true;
    plan.has_change = true;
    plan.has_col_action = true;
    plan.has_row_action = true;
    record_l1_split_recovery(plan, q_cols, e_cols, nullptr, block_count, "outer_pair_linked_l1");
    if (plan.tape.resident) {
      capture_absolute_value_outer_device(plan, bound_cols, free_cols, pair_count,
                                          s_cols, q_cols, alphas, block_count);
    } else {
      const std::vector<std::int32_t> bound_h =
          copy_device_vector(bound_cols, pair_count, "cudaMemcpy structural outer bound_cols recovery");
      const std::vector<std::int32_t> free_h =
          copy_device_vector(free_cols, pair_count, "cudaMemcpy structural outer free_cols recovery");
      for (std::int32_t i = 0; i < pair_count; ++i) {
        plan.absolute_value_epigraph_recovery.outer_pairs.push_back(
            {bound_h[static_cast<std::size_t>(i)], free_h[static_cast<std::size_t>(i)]});
      }
      const std::vector<std::int32_t> s_h =
          copy_device_vector(s_cols, block_count, "cudaMemcpy structural outer s_cols recovery");
      const std::vector<std::int32_t> q_h =
          copy_device_vector(q_cols, block_count, "cudaMemcpy structural outer q_cols recovery");
      const std::vector<double> alpha_h =
          copy_device_vector(alphas, block_count, "cudaMemcpy structural outer alphas recovery");
      for (std::int32_t i = 0; i < block_count; ++i) {
        plan.absolute_value_epigraph_recovery.linked_slacks.push_back(
            {s_h[static_cast<std::size_t>(i)], q_h[static_cast<std::size_t>(i)],
             alpha_h[static_cast<std::size_t>(i)]});
      }
    }
    if (profile) {
      std::cerr << ">>> [absolute_value_epigraph C++] changed=1 new_A_nnz=" << plan.new_A.nnz << "\n";
    }
  }

cleanup:
  cudaFree(eq_row);
  cudaFree(eq_zero_two_nnz);
  cudaFree(lower_two_nnz);
  cudaFree(zero_lower_two_nnz);
  cudaFree(raw_col1);
  cudaFree(raw_col2);
  cudaFree(raw_val1);
  cudaFree(raw_val2);
  cudaFree(zl_col1);
  cudaFree(zl_col2);
  cudaFree(zl_val1);
  cudaFree(zl_val2);
  cudaFree(status_flag);
  cudaFree(pair_count_d);
  cudaFree(start_row_d);
  cudaFree(bound_cols);
  cudaFree(free_cols);
  cudaFree(free_to_bound);
  cudaFree(pair_cost_after);
  cudaFree(q_cols);
  cudaFree(e_cols);
  cudaFree(s_cols);
  cudaFree(alphas);
  cudaFree(local_x_cols);
}

}  // namespace gpu_presolver::presolve
