#pragma once
#include "gpu_presolver/folding/folding.hpp"
#include "gpu_presolver/presolve/cleanup_workspace.hpp"
#include "gpu_presolver/presolve/duplicate_rule_workspace.hpp"
#include "gpu_presolver/presolve/rules/rule_bound_strengthening.hpp"
#include "gpu_presolver/presolve/rules/rule_parallel_rows.hpp"
#include "gpu_presolver/presolve/rules/rule_parallel_columns.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace gpu_presolver::folding::detail {
namespace gp = gpu_presolver::presolve;
inline void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}
// Reserve the two initial recovery-map sizes before the run_presolve timer.
// Keep the async pool's freed blocks across the setup synchronization so the
// first recovery allocations can reuse them; restore the pool policy later.
class PreparedRecoveryAsyncPool {
public:
    PreparedRecoveryAsyncPool(std::int32_t rows, std::int32_t cols) {
        if (rows <= 0 && cols <= 0) return;
        int device = 0;
        throw_if_cuda_error(cudaGetDevice(&device), "recovery pool device");
        throw_if_cuda_error(cudaDeviceGetDefaultMemPool(&pool_, device), "recovery pool query");
        throw_if_cuda_error(cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrReleaseThreshold,
                                                   &old_threshold_), "recovery pool old threshold");
        std::uint64_t retain = std::numeric_limits<std::uint64_t>::max();
        throw_if_cuda_error(cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold,
                                                   &retain), "recovery pool retain");
        try {
            if (rows > 0) {
                throw_if_cuda_error(cudaMallocAsync(&row_temp_, sizeof(std::int32_t) *
                                                     static_cast<std::size_t>(rows), nullptr),
                                    "prewarm recovery rows");
            }
            if (cols > 0) {
                throw_if_cuda_error(cudaMallocAsync(&col_temp_, sizeof(std::int32_t) *
                                                     static_cast<std::size_t>(cols), nullptr),
                                    "prewarm recovery cols");
            }
            if (row_temp_) {
                throw_if_cuda_error(cudaFreeAsync(row_temp_, nullptr), "release prewarm rows");
                row_temp_ = nullptr;
            }
            if (col_temp_) {
                throw_if_cuda_error(cudaFreeAsync(col_temp_, nullptr), "release prewarm cols");
                col_temp_ = nullptr;
            }
        } catch (...) {
            if (row_temp_) cudaFreeAsync(row_temp_, nullptr);
            if (col_temp_) cudaFreeAsync(col_temp_, nullptr);
            cudaDeviceSynchronize();
            cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold, &old_threshold_);
            pool_ = nullptr;
            throw;
        }
    }
    ~PreparedRecoveryAsyncPool() {
        if (pool_) {
            cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold, &old_threshold_);
            cudaMemPoolTrimTo(pool_, 0);
        }
    }
    PreparedRecoveryAsyncPool(const PreparedRecoveryAsyncPool &) = delete;
    PreparedRecoveryAsyncPool &operator=(const PreparedRecoveryAsyncPool &) = delete;

private:
    cudaMemPool_t pool_ = nullptr;
    std::uint64_t old_threshold_ = 0;
    void *row_temp_ = nullptr;
    void *col_temp_ = nullptr;
};

// Shared by the public API and all standalone tools. The core timer remains
// inside presolve; these allocations and their release belong to wall time.
inline void run_prepared_presolve(FoldingPipelineSummary& summary,
                                  const gp::LPInfoGpu& input,
                                  const gp::PresolveParams& params,
                                  bool keep_reduced_lp) {
  if (!params.enable_presolve) {
    summary.presolve = run_optional_gpu_presolve(input, params, keep_reduced_lp);
    return;
  }
  using Clock = std::chrono::steady_clock;
  const auto setup_start = Clock::now();
  auto workspace = std::make_unique<gp::PreparedCleanupWorkspace>(input.A.rows, input.A.cols);
  if (workspace->enabled() && input.A.cols > 0) {
    workspace->reserve_reduction_scratch(gp::cleanup_reduction_temp_bytes(
        input.c, workspace->reduction_result(), input.A.cols));
    workspace->prepare_selection(input.c, input.A.cols);
  }
  auto recovery_pool = std::make_unique<PreparedRecoveryAsyncPool>(input.A.rows, input.A.cols);
  auto bounds = std::make_unique<gp::PreparedBoundStrengtheningWorkspace>(
      input.A.rows, input.A.cols, params.enable_bound_strengthening);
  const bool preverify = runtime::options().parallel_column_preverify;
  const auto row_bytes = params.enable_parallel_rows ? gp::parallel_rows_scratch_bytes(input.A.rows) : 0;
  const auto col_bytes = params.enable_parallel_columns
      ? gp::parallel_columns_scratch_bytes(input.A.rows, input.A.cols, preverify) : 0;
  auto duplicates = std::make_unique<gp::PreparedDuplicateRuleWorkspace>(std::max(row_bytes, col_bytes));
  throw_if_cuda_error(cudaDeviceSynchronize(), "preprocessing workspace setup");
  summary.workspace_setup_seconds = std::chrono::duration<double>(Clock::now() - setup_start).count();
  summary.presolve = run_optional_gpu_presolve(input, params, keep_reduced_lp);
  throw_if_cuda_error(cudaDeviceSynchronize(), "preprocessing presolve completion");
  const auto release_start = Clock::now();
  duplicates.reset();
  bounds.reset();
  workspace.reset();
  recovery_pool.reset();
  throw_if_cuda_error(cudaDeviceSynchronize(), "preprocessing workspace release");
  summary.workspace_release_seconds = std::chrono::duration<double>(Clock::now() - release_start).count();
}
}  // namespace gpu_presolver::folding::detail
