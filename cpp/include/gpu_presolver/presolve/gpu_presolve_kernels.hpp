#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

// Dispatch rows by stored length or supplied live degrees.
constexpr int ADAPTIVE_SHORT_ROW = 8;
constexpr int ADAPTIVE_LONG_ROW = 1024;
struct AdaptiveRowGroups {
  std::int32_t* medium = nullptr;
  std::int32_t* long_rows = nullptr;
  std::int32_t* counts = nullptr;
};
bool adaptive_row_dispatch_enabled();
bool adaptive_extended_enabled();
bool adaptive_reuse_enabled();
bool adaptive_sort_reuse_enabled();
bool adaptive_hash_scan_enabled();
bool adaptive_gather_enabled();
// True only after checking strict index order; duplicate coordinates fall back to sorting.
bool csr_rows_strictly_sorted(const DeviceCsrMatrix& matrix);
// Caching is enabled only inside an explicitly scoped, immutable matrix epoch.
// Standalone rule/kernel callers never cache by device-pointer identity alone.
class MatrixAnalysisScope {
 public:
  MatrixAnalysisScope();
  ~MatrixAnalysisScope();
  MatrixAnalysisScope(const MatrixAnalysisScope&) = delete;
  MatrixAnalysisScope& operator=(const MatrixAnalysisScope&) = delete;
 private:
  bool previous_;
};
void invalidate_matrix_analysis();
std::uint64_t matrix_analysis_epoch();
// Two cache slots cover A and AT. Consumers and invalidations use the default stream.
const AdaptiveRowGroups& prepare_adaptive_row_groups(const DeviceCsrMatrix& matrix,
                                                     const std::int32_t* live_degrees = nullptr);

// Fill allocated statistics from current masks, including singleton supports.
// Call after every mask change before consuming stats; bounds/shifts must already
// be committed to plan.new_*. Stored zeros count as entries. A/AT stay unchanged.
// Allocate the statistics required by the next rule before calling this helper.
// device_live_nnz, when supplied, receives the nnz total from the same row
// traversal. With read_live_nnz=false, leave stats.live_nnz unchanged and avoid
// a host synchronization; the caller can read device_live_nnz when needed.
void refresh_live_stats(PresolveStatsGpu& stats, const LPInfoGpu& lp,
                        const PresolvePlanGpu& plan, bool read_live_nnz = true,
                        unsigned long long* device_live_nnz = nullptr);

// Update the same statistics from monotonically deleted row/column masks.
// old_* masks describe the state represented by stats; touched_* are reusable
// device buffers with one int per row/column. Returns false if the delta cannot
// be certified, in which case the caller must do a full refresh.
// With device_live_nnz, subtract deletions from that device count. Setting
// read_live_nnz=false leaves the host stats.live_nnz cache untouched.
bool update_live_stats_from_deletions(PresolveStatsGpu& stats, const LPInfoGpu& lp,
    const PresolvePlanGpu& plan, const std::uint8_t* old_rows,
    const std::uint8_t* old_cols, std::int32_t* touched_rows,
    std::int32_t* touched_cols, bool read_live_nnz = true,
    unsigned long long* device_live_nnz = nullptr);

void compute_row_nnz(std::int32_t* row_nnz, const DeviceCsrMatrix& A_csr);

void compute_col_nnz(std::int32_t* col_nnz, const DeviceCsrMatrix& AT_csr);

void compute_singleton_row_support(std::int32_t* support_col,
                                   double* singleton_val,
                                   const std::int32_t* row_nnz,
                                   const DeviceCsrMatrix& A_csr);

void compute_singleton_column_support(std::int32_t* singleton_row,
                                      double* singleton_val,
                                      const std::int32_t* col_nnz,
                                      const DeviceCsrMatrix& AT_csr);

void compute_row_activity_bounds(double* row_min,
                                 double* row_max,
                                 const DeviceCsrMatrix& A_csr,
                                 const double* l,
                                 const double* u);

void compute_row_activity_summary(double* row_min_fin,
                                  double* row_max_fin,
                                  std::int32_t* row_min_neg_inf_count,
                                  std::int32_t* row_max_pos_inf_count,
                                  const DeviceCsrMatrix& A_csr,
                                  const double* l,
                                  const double* u,
                                  double zero_tol,
                                  const std::uint8_t* keep_row = nullptr,
                                  const std::uint8_t* keep_col = nullptr,
                                  const std::int32_t* live_degrees = nullptr);

void compute_col_max_abs(double* col_max_abs, const DeviceCsrMatrix& AT_csr,
                         const std::uint8_t* keep_row = nullptr);

}  // namespace gpu_presolver::presolve
