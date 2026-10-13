#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/rules/rule_column_domination.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;

enum ValidationFailure : std::int32_t {
  BAD_ROW_BOUNDS = 1 << 0,
  BAD_COL_DOMAIN = 1 << 1,
  BAD_OBJECTIVE = 1 << 2,
  BAD_MATRIX_VALUE = 1 << 3,
  MISSING_COST_ONE_WITNESS = 1 << 4,
};

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(status));
  }
}

__device__ bool exact_positive_infinity(double value) {
  return isinf(value) && value > 0.0;
}

__device__ bool exact_positive_integer(double value) {
  return isfinite(value) && value >= 1.0 && floor(value) == value;
}

__device__ NnzOffset sampled_index(NnzOffset sample,
                                      NnzOffset samples,
                                      NnzOffset count) {
  return static_cast<NnzOffset>(
      (static_cast<long long>(sample) * static_cast<long long>(count)) /
      static_cast<long long>(samples));
}

__global__ void quick_probe_rows(std::int32_t* failure,
                                 const NnzOffset* row_ptr,
                                 const std::int32_t* col_val,
                                 const double* c,
                                 const double* AL,
                                 const double* AU,
                                 std::int32_t rows,
                                 std::int32_t probes) {
  const std::int32_t sample =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (sample >= probes) {
    return;
  }
  const std::int32_t row = sampled_index(sample, probes, rows);
  if (AL[row] != 1.0 || !exact_positive_infinity(AU[row])) {
    atomicOr(failure, static_cast<std::int32_t>(BAD_ROW_BOUNDS));
    return;
  }
  bool has_witness = false;
  for (NnzOffset p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
    if (c[col_val[p]] == 1.0) {
      has_witness = true;
      break;
    }
  }
  if (!has_witness) {
    atomicOr(failure,
             static_cast<std::int32_t>(MISSING_COST_ONE_WITNESS));
  }
}

__global__ void quick_probe_cols(std::int32_t* failure,
                                 const double* c,
                                 const double* l,
                                 const double* u,
                                 std::int32_t cols,
                                 std::int32_t probes) {
  const std::int32_t sample =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (sample >= probes) {
    return;
  }
  const std::int32_t col = sampled_index(sample, probes, cols);
  if (l[col] != 0.0 || u[col] != 1.0) {
    atomicOr(failure, static_cast<std::int32_t>(BAD_COL_DOMAIN));
  }
  if (!exact_positive_integer(c[col])) {
    atomicOr(failure, static_cast<std::int32_t>(BAD_OBJECTIVE));
  }
}

__global__ void quick_probe_values(std::int32_t* failure,
                                   const double* values,
                                   NnzOffset nnz,
                                   std::int32_t probes) {
  const std::int32_t sample =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (sample >= probes) {
    return;
  }
  const NnzOffset p = sampled_index(sample, probes, nnz);
  if (values[p] != 1.0) {
    atomicOr(failure, static_cast<std::int32_t>(BAD_MATRIX_VALUE));
  }
}

__global__ void validate_rows_and_witnesses(std::int32_t* failure,
                                            const NnzOffset* row_ptr,
                                            const std::int32_t* col_val,
                                            const double* c,
                                            const double* AL,
                                            const double* AU,
                                            std::int32_t rows) {
  const std::int32_t row =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= rows) {
    return;
  }
  if (AL[row] != 1.0 || !exact_positive_infinity(AU[row])) {
    atomicOr(failure, static_cast<std::int32_t>(BAD_ROW_BOUNDS));
  }
  bool has_witness = false;
  for (NnzOffset p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
    if (c[col_val[p]] == 1.0) {
      has_witness = true;
      break;
    }
  }
  if (!has_witness) {
    atomicOr(failure,
             static_cast<std::int32_t>(MISSING_COST_ONE_WITNESS));
  }
}

__global__ void validate_columns(std::int32_t* failure,
                                 const double* c,
                                 const double* l,
                                 const double* u,
                                 std::int32_t cols) {
  const std::int32_t col =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (col >= cols) {
    return;
  }
  if (l[col] != 0.0 || u[col] != 1.0) {
    atomicOr(failure, static_cast<std::int32_t>(BAD_COL_DOMAIN));
  }
  if (!exact_positive_integer(c[col])) {
    atomicOr(failure, static_cast<std::int32_t>(BAD_OBJECTIVE));
  }
}

__global__ void validate_matrix_values(std::int32_t* failure,
                                       const double* values,
                                       NnzOffset nnz) {
  const NnzOffset p =
      static_cast<NnzOffset>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (p < nnz && values[p] != 1.0) {
    atomicOr(failure, static_cast<std::int32_t>(BAD_MATRIX_VALUE));
  }
}

__global__ void mark_dominated_columns(std::uint8_t* delete_col,
                                       std::int32_t* candidate_count,
                                       const NnzOffset* at_row_ptr,
                                       const double* c,
                                       std::int32_t cols) {
  const std::int32_t col =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (col >= cols) {
    return;
  }
  const std::int32_t degree = at_row_ptr[col + 1] - at_row_ptr[col];
  const bool dominated = c[col] > static_cast<double>(degree);
  delete_col[col] = dominated ? std::uint8_t{1} : std::uint8_t{0};
  if (dominated) {
    atomicAdd(candidate_count, 1);
  }
}

__global__ void apply_dominated_columns(std::uint8_t* keep_col,
                                        double* new_l,
                                        double* new_u,
                                        const std::uint8_t* delete_col,
                                        std::int32_t cols) {
  const std::int32_t col =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (col < cols && delete_col[col] != std::uint8_t{0}) {
    keep_col[col] = std::uint8_t{0};
    new_l[col] = 0.0;
    new_u[col] = 0.0;
  }
}

}  // namespace

bool quick_probe_column_domination(const LPInfoGpu& lp,
                                         const PresolveParams& params) {
  if (!params.enable_column_domination || lp.A.rows <= 0 ||
      lp.A.cols <
          std::max(1, params.column_domination_min_fixed_cols) ||
      lp.A.nnz <= 0 || lp.AT.rows != lp.A.cols ||
      lp.AT.cols != lp.A.rows || lp.AT.nnz != lp.A.nnz ||
      lp.A.rowPtr == nullptr || lp.A.colVal == nullptr ||
      lp.A.nzVal == nullptr || lp.AT.rowPtr == nullptr ||
      lp.AT.colVal == nullptr || lp.AT.nzVal == nullptr || lp.c == nullptr ||
      lp.AL == nullptr || lp.AU == nullptr || lp.l == nullptr ||
      lp.u == nullptr) {
    return false;
  }

  const std::int32_t row_probes = std::max(
      1, std::min(lp.A.rows, params.column_domination_probe_rows));
  const std::int32_t col_probes = std::max(
      1, std::min(lp.A.cols, params.column_domination_probe_cols));
  const std::int32_t nz_probes = std::max<NnzOffset>(
      1, std::min<NnzOffset>(lp.A.nnz, params.column_domination_probe_nnz));
  std::int32_t* failure = nullptr;
  throw_if_cuda_error(cudaMalloc(&failure, sizeof(std::int32_t)),
                      "cudaMalloc covering dominance probe status");
  try {
    throw_if_cuda_error(cudaMemset(failure, 0, sizeof(std::int32_t)),
                        "cudaMemset covering dominance probe status");
    quick_probe_rows<<<(row_probes + GPU_PRESOLVE_THREADS - 1) /
                           GPU_PRESOLVE_THREADS,
                       GPU_PRESOLVE_THREADS>>>(
        failure, lp.A.rowPtr, lp.A.colVal, lp.c, lp.AL, lp.AU, lp.A.rows,
        row_probes);
    throw_if_cuda_error(cudaGetLastError(), "quick_probe_rows");
    quick_probe_cols<<<(col_probes + GPU_PRESOLVE_THREADS - 1) /
                           GPU_PRESOLVE_THREADS,
                       GPU_PRESOLVE_THREADS>>>(
        failure, lp.c, lp.l, lp.u, lp.A.cols, col_probes);
    throw_if_cuda_error(cudaGetLastError(), "quick_probe_cols");
    quick_probe_values<<<(nz_probes + GPU_PRESOLVE_THREADS - 1) /
                             GPU_PRESOLVE_THREADS,
                         GPU_PRESOLVE_THREADS>>>(
        failure, lp.A.nzVal, lp.A.nnz, nz_probes);
    throw_if_cuda_error(cudaGetLastError(), "quick_probe_values A");
    quick_probe_values<<<(nz_probes + GPU_PRESOLVE_THREADS - 1) /
                             GPU_PRESOLVE_THREADS,
                         GPU_PRESOLVE_THREADS>>>(
        failure, lp.AT.nzVal, lp.AT.nnz, nz_probes);
    throw_if_cuda_error(cudaGetLastError(), "quick_probe_values AT");
    std::int32_t host_failure = 0;
    throw_if_cuda_error(cudaMemcpy(&host_failure, failure,
                                   sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy covering dominance probe status");
    cudaFree(failure);
    return host_failure == 0;
  } catch (...) {
    cudaFree(failure);
    throw;
  }
}

ColumnDominationAnalysisGpu analyze_column_domination(
    const LPInfoGpu& lp,
    const PresolveParams& params) {
  ColumnDominationAnalysisGpu analysis;
  if (!quick_probe_column_domination(lp, params)) {
    return analysis;
  }

  std::int32_t* failure = nullptr;
  std::int32_t* candidate_count = nullptr;
  try {
    throw_if_cuda_error(cudaMalloc(&failure, sizeof(std::int32_t)),
                        "cudaMalloc covering dominance status");
    throw_if_cuda_error(cudaMemset(failure, 0, sizeof(std::int32_t)),
                        "cudaMemset covering dominance status");
    const int row_blocks =
        (lp.A.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    const int col_blocks =
        (lp.A.cols + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    const int nz_blocks =
        (lp.A.nnz + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    validate_rows_and_witnesses<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
        failure, lp.A.rowPtr, lp.A.colVal, lp.c, lp.AL, lp.AU, lp.A.rows);
    throw_if_cuda_error(cudaGetLastError(),
                        "validate_rows_and_witnesses");
    validate_columns<<<col_blocks, GPU_PRESOLVE_THREADS>>>(
        failure, lp.c, lp.l, lp.u, lp.A.cols);
    throw_if_cuda_error(cudaGetLastError(), "validate_columns");
    validate_matrix_values<<<nz_blocks, GPU_PRESOLVE_THREADS>>>(
        failure, lp.A.nzVal, lp.A.nnz);
    throw_if_cuda_error(cudaGetLastError(), "validate_matrix_values A");
    validate_matrix_values<<<nz_blocks, GPU_PRESOLVE_THREADS>>>(
        failure, lp.AT.nzVal, lp.AT.nnz);
    throw_if_cuda_error(cudaGetLastError(), "validate_matrix_values AT");

    std::int32_t host_failure = 0;
    throw_if_cuda_error(cudaMemcpy(&host_failure, failure,
                                   sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy covering dominance status");
    if (host_failure != 0) {
      cudaFree(failure);
      return analysis;
    }

    throw_if_cuda_error(
        cudaMalloc(&analysis.delete_col,
                   static_cast<std::size_t>(lp.A.cols)),
        "cudaMalloc covering dominance delete mask");
    throw_if_cuda_error(cudaMalloc(&candidate_count, sizeof(std::int32_t)),
                        "cudaMalloc covering dominance candidate count");
    throw_if_cuda_error(cudaMemset(candidate_count, 0,
                                   sizeof(std::int32_t)),
                        "cudaMemset covering dominance candidate count");
    mark_dominated_columns<<<col_blocks, GPU_PRESOLVE_THREADS>>>(
        analysis.delete_col, candidate_count, lp.AT.rowPtr, lp.c, lp.A.cols);
    throw_if_cuda_error(cudaGetLastError(), "mark_dominated_columns");
    throw_if_cuda_error(cudaMemcpy(&analysis.candidate_count, candidate_count,
                                   sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy covering dominance candidate count");

    const std::int32_t min_candidates =
        std::max(1, params.column_domination_min_fixed_cols);
    const double candidate_ratio =
        static_cast<double>(analysis.candidate_count) /
        static_cast<double>(lp.A.cols);
    if (analysis.candidate_count < min_candidates ||
        candidate_ratio + 1.0e-15 <
            params.column_domination_min_candidate_ratio) {
      cudaFree(analysis.delete_col);
      analysis.delete_col = nullptr;
      analysis.candidate_count = 0;
    } else {
      analysis.applicable = true;
    }

    cudaFree(failure);
    cudaFree(candidate_count);
    return analysis;
  } catch (...) {
    cudaFree(failure);
    cudaFree(candidate_count);
    cudaFree(analysis.delete_col);
    analysis = ColumnDominationAnalysisGpu{};
    throw;
  }
}

void build_column_domination_plan(
    PresolvePlanGpu& plan,
    const LPInfoGpu& lp,
    const ColumnDominationAnalysisGpu& analysis,
    const PresolveParams&) {
  if (!analysis.applicable || analysis.delete_col == nullptr ||
      analysis.candidate_count <= 0 || plan.keep_col_mask == nullptr ||
      plan.new_l == nullptr || plan.new_u == nullptr) {
    return;
  }
  const int blocks =
      (lp.A.cols + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  apply_dominated_columns<<<blocks, GPU_PRESOLVE_THREADS>>>(
      plan.keep_col_mask, plan.new_l, plan.new_u, analysis.delete_col,
      lp.A.cols);
  throw_if_cuda_error(cudaGetLastError(), "apply_dominated_columns");
  throw_if_cuda_error(cudaDeviceSynchronize(),
                      "covering dominance plan synchronize");
  plan.has_change = true;
  plan.has_col_action = true;
  plan.has_column_domination_reduction = true;
}

void free_column_domination_analysis(
    ColumnDominationAnalysisGpu& analysis) {
  cudaFree(analysis.delete_col);
  analysis = ColumnDominationAnalysisGpu{};
}

}  // namespace gpu_presolver::presolve
