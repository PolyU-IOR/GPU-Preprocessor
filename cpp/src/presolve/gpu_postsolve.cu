#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;

__device__ double _postsolve_project_single_column_dual_from_original(double* x_org,
                                                                      double* z_org,
                                                                      const NnzOffset* AT_rowPtr,
                                                                      const std::int32_t* AT_colVal,
                                                                      const double* AT_nzVal,
                                                                      const double* y_org,
                                                                      const double* c,
                                                                      const double* l,
                                                                      const double* u,
                                                                      std::int32_t col,
                                                                      double tol);

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

template <class T>
T* copy_vector_to_device(const std::vector<T>& values, const char* context) {
  if (values.empty()) {
    return nullptr;
  }
  T* device = nullptr;
  throw_if_cuda_error(cudaMalloc(&device, sizeof(T) * values.size()), context);
  throw_if_cuda_error(cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice),
                      context);
  return device;
}

__global__ void _kernel_structural_restore_splits(double* x_org,
                                                  const std::int32_t* t_cols,
                                                  const std::int32_t* e_cols,
                                                  const double* rhos,
                                                  std::int32_t k) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i >= k) {
    return;
  }
  const std::int32_t t_idx = t_cols[i];
  const std::int32_t e_idx = e_cols[i];
  const double rho = rhos[i];
  const double t_new = x_org[t_idx];
  const double e_new = x_org[e_idx];
  x_org[t_idx] = t_new + e_new;
  x_org[e_idx] = (t_new - e_new) / rho;
}

__global__ void _kernel_structural_restore_outer_pairs(double* x_org,
                                                       const std::int32_t* bound_cols,
                                                       const std::int32_t* free_cols,
                                                       std::int32_t k) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < k) {
    x_org[free_cols[i]] = x_org[bound_cols[i]];
  }
}

__global__ void _kernel_structural_restore_linked_slacks(double* x_org,
                                                         const std::int32_t* slack_cols,
                                                         const std::int32_t* t_cols,
                                                         const double* factors,
                                                         std::int32_t k) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < k) {
    x_org[slack_cols[i]] = factors[i] * x_org[t_cols[i]];
  }
}

__global__ void _kernel_structural_restore_max_slack(double* x_org,
                                                     std::int32_t slack_col,
                                                     const std::int32_t* t_cols,
                                                     const double* factors,
                                                     std::int32_t count,
                                                     double lower_bound) {
  const int tid = threadIdx.x;
  const int lane_count = blockDim.x;
  double partial = lower_bound;
  for (std::int32_t j = tid; j < count; j += lane_count) {
    partial = fmax(partial, factors[j] * x_org[t_cols[j]]);
  }

  __shared__ double scratch[GPU_PRESOLVE_THREADS];
  scratch[tid] = partial;
  __syncthreads();

  for (NnzOffset offset = lane_count >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      scratch[tid] = fmax(scratch[tid], scratch[tid + offset]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    x_org[slack_col] = scratch[0];
  }
}
__global__ void _kernel_device_restore_max_slack(double *x_org, const std::int32_t *slack_id,
                                                 const std::int32_t *t_cols, const double *factors,
                                                 std::int32_t count, const double *original_l) {
  const int slack_col = slack_id[0];
  double lower_bound = original_l ? original_l[slack_col] : 0.0;
  if (!isfinite(lower_bound))
    lower_bound = 0.0;
  const int tid = threadIdx.x;
  const int lane_count = blockDim.x;
  double partial = lower_bound;
  for (std::int32_t j = tid; j < count; j += lane_count) {
    partial = fmax(partial, factors[j] * x_org[t_cols[j]]);
  }

  __shared__ double scratch[GPU_PRESOLVE_THREADS];
  scratch[tid] = partial;
  __syncthreads();

  for (NnzOffset offset = lane_count >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      scratch[tid] = fmax(scratch[tid], scratch[tid + offset]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    x_org[slack_col] = scratch[0];
  }
}

__global__ void _kernel_cache_variable_pair_root_values(
    double* plus_cache,
    double* minus_cache,
    const double* x_org,
    const std::int32_t* root_plus_cols,
    const std::int32_t* root_minus_cols,
    std::int32_t count) {
  const std::int32_t i =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < count) {
    plus_cache[i] = x_org[root_plus_cols[i]];
    minus_cache[i] = x_org[root_minus_cols[i]];
  }
}

__global__ void _kernel_restore_variable_pair_aggregation_members(
    double* x_org,
    const std::int32_t* plus_cols,
    const std::int32_t* minus_cols,
    const double* plus_shift,
    const double* minus_shift,
    const double* plus_cache,
    const double* minus_cache,
    std::int32_t count) {
  const std::int32_t i =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < count) {
    x_org[plus_cols[i]] = plus_shift[i] + plus_cache[i];
    x_org[minus_cols[i]] = minus_shift[i] + minus_cache[i];
  }
}

__global__ void _kernel_scatter_reduced_to_original(double* dst,
                                                    const double* src,
                                                    const std::int32_t* red2org,
                                                    std::int32_t n_red) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < n_red) {
    dst[red2org[i]] = src[i];
  }
}

__global__ void _kernel_mark_retrieved(std::uint8_t* mask,
                                       const std::int32_t* red2org,
                                       std::int32_t n_red) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < n_red) {
    mask[red2org[i]] = std::uint8_t{1};
  }
}

__global__ void _kernel_project_column_duals_from_original(double* x_org,
                                                           double* z_org,
                                                           const std::uint8_t* protected_cols,
                                                           const NnzOffset* AT_rowPtr,
                                                           const std::int32_t* AT_colVal,
                                                           const double* AT_nzVal,
                                                           const double* y_org,
                                                           const double* c,
                                                           const double* l,
                                                           const double* u,
                                                           std::int32_t n,
                                                           double tol) {
  const std::int32_t col = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (col >= n || (protected_cols != nullptr && protected_cols[col] != std::uint8_t{0})) {
    return;
  }
  _postsolve_project_single_column_dual_from_original(
      x_org, z_org, AT_rowPtr, AT_colVal, AT_nzVal, y_org, c, l, u, col, tol);
}

__global__ void _kernel_restore_fixed_values(double* x_org,
                                             const std::int32_t* fixed_idx,
                                             const double* fixed_val,
                                             std::int32_t fixed_count) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < fixed_count) {
    x_org[fixed_idx[i]] = fixed_val[i];
  }
}

__device__ double _choose_merged_source_value(double lo, double hi, double tol) {
  if (isfinite(lo) && isfinite(hi)) {
    return 0.5 * (lo + hi);
  }
  if (isfinite(lo)) {
    if (0.0 > lo + tol) {
      return 0.0;
    }
    return lo + fmax(1.0, fabs(lo) * fmax(tol, 1.0e-12));
  }
  if (isfinite(hi)) {
    if (0.0 < hi - tol) {
      return 0.0;
    }
    return hi - fmax(1.0, fabs(hi) * fmax(tol, 1.0e-12));
  }
  return 0.0;
}

__device__ double _parallel_column_aggregate_lower_bound(double target_l,
                                                      double source_l,
                                                      double source_u,
                                                      double ratio) {
  if (ratio > 0.0) {
    if (isfinite(target_l) && isfinite(source_l)) {
      return target_l + ratio * source_l;
    }
  } else if (isfinite(target_l) && isfinite(source_u)) {
    return target_l + ratio * source_u;
  }
  return -INFINITY;
}

__device__ double _parallel_column_aggregate_upper_bound(double target_u,
                                                      double source_l,
                                                      double source_u,
                                                      double ratio) {
  if (ratio > 0.0) {
    if (isfinite(target_u) && isfinite(source_u)) {
      return target_u + ratio * source_u;
    }
  } else if (isfinite(target_u) && isfinite(source_l)) {
    return target_u + ratio * source_l;
  }
  return INFINITY;
}

__device__ void _recover_parallel_column_primal(double y_val,
                                             double z_to,
                                             double ratio,
                                             double source_l,
                                             double source_u,
                                             double target_l,
                                             double target_u,
                                             double tol,
                                             double* x_from,
                                             double* x_to) {
  double lo = source_l;
  double hi = source_u;
  if (isfinite(target_u)) {
    const double bound = (y_val - target_u) / ratio;
    if (ratio > 0.0) {
      lo = fmax(lo, bound);
    } else {
      hi = fmin(hi, bound);
    }
  }
  if (isfinite(target_l)) {
    const double bound = (y_val - target_l) / ratio;
    if (ratio > 0.0) {
      hi = fmin(hi, bound);
    } else {
      lo = fmax(lo, bound);
    }
  }

  const double lower = _parallel_column_aggregate_lower_bound(target_l, source_l, source_u, ratio);
  const double upper = _parallel_column_aggregate_upper_bound(target_u, source_l, source_u, ratio);
  if (z_to > tol && isfinite(lower) && fabs(y_val - lower) <= tol) {
    *x_from = ratio > 0.0 ? source_l : source_u;
  } else if (z_to < -tol && isfinite(upper) && fabs(y_val - upper) <= tol) {
    *x_from = ratio > 0.0 ? source_u : source_l;
  } else {
    *x_from = _choose_merged_source_value(lo, hi, tol);
  }
  *x_to = y_val - ratio * (*x_from);
}

__device__ bool _postsolve_is_finite(double x) {
  return x > -INFINITY && x < INFINITY;
}

__device__ bool _postsolve_at_lower(double x, double l, double tol) {
  return _postsolve_is_finite(l) && x <= l + tol;
}

__device__ bool _postsolve_at_upper(double x, double u, double tol) {
  return _postsolve_is_finite(u) && x >= u - tol;
}

__device__ double _clamp_with_inf(double x, double lo, double hi) {
  double y = x;
  if (_postsolve_is_finite(lo)) {
    y = fmax(y, lo);
  }
  if (_postsolve_is_finite(hi)) {
    y = fmin(y, hi);
  }
  return y;
}

__device__ void _parallel_row_scaled_interval(double lower,
                                              double upper,
                                              double ratio,
                                              double* scaled_l,
                                              double* scaled_u) {
  const double raw_l = lower * ratio;
  const double raw_u = upper * ratio;
  if (ratio >= 0.0) {
    *scaled_l = raw_l;
    *scaled_u = raw_u;
  } else {
    *scaled_l = raw_u;
    *scaled_u = raw_l;
  }
}

__device__ bool _parallel_row_interval_contains(double outer_lo,
                                                double outer_hi,
                                                double inner_lo,
                                                double inner_hi,
                                                double tol) {
  return outer_lo <= inner_lo + tol && inner_hi <= outer_hi + tol;
}

__device__ double _parallel_row_activity(std::int32_t row,
                                         const NnzOffset* A_rowPtr,
                                         const std::int32_t* A_colVal,
                                         const double* A_nzVal,
                                         const double* x_org) {
  double activity = 0.0;
  for (NnzOffset p = A_rowPtr[row]; p < A_rowPtr[row + 1]; ++p) {
    activity += A_nzVal[p] * x_org[A_colVal[p]];
  }
  return activity;
}

enum : std::uint8_t {
  PARALLEL_ROW_STATE_ZERO = 0,
  PARALLEL_ROW_STATE_NONNEG = 1,
  PARALLEL_ROW_STATE_NONPOS = 2,
  PARALLEL_ROW_STATE_FREE = 3,
};

__device__ std::uint8_t _parallel_row_dual_state(double activity,
                                                 double lower,
                                                 double upper,
                                                 double tol) {
  const bool at_lower = _postsolve_is_finite(lower) && activity <= lower + tol;
  const bool at_upper = _postsolve_is_finite(upper) && activity >= upper - tol;
  if (at_lower && at_upper) {
    return PARALLEL_ROW_STATE_FREE;
  }
  if (at_lower) {
    return PARALLEL_ROW_STATE_NONNEG;
  }
  if (at_upper) {
    return PARALLEL_ROW_STATE_NONPOS;
  }
  return PARALLEL_ROW_STATE_ZERO;
}

__device__ void _parallel_row_state_interval(std::uint8_t state, double* lo, double* hi) {
  if (state == PARALLEL_ROW_STATE_FREE) {
    *lo = -INFINITY;
    *hi = INFINITY;
  } else if (state == PARALLEL_ROW_STATE_NONNEG) {
    *lo = 0.0;
    *hi = INFINITY;
  } else if (state == PARALLEL_ROW_STATE_NONPOS) {
    *lo = -INFINITY;
    *hi = 0.0;
  } else {
    *lo = 0.0;
    *hi = 0.0;
  }
}

__device__ void _parallel_row_deleted_dual_interval(double ybar_p,
                                                    double ratio,
                                                    std::uint8_t kept_state,
                                                    double* lo,
                                                    double* hi) {
  const double pivot = ratio * ybar_p;
  if (kept_state == PARALLEL_ROW_STATE_FREE) {
    *lo = -INFINITY;
    *hi = INFINITY;
  } else if (kept_state == PARALLEL_ROW_STATE_ZERO) {
    *lo = pivot;
    *hi = pivot;
  } else if (kept_state == PARALLEL_ROW_STATE_NONNEG) {
    if (ratio > 0.0) {
      *lo = -INFINITY;
      *hi = pivot;
    } else {
      *lo = pivot;
      *hi = INFINITY;
    }
  } else {
    if (ratio > 0.0) {
      *lo = pivot;
      *hi = INFINITY;
    } else {
      *lo = -INFINITY;
      *hi = pivot;
    }
  }
}

__device__ bool _parallel_row_doc_dual_split(double ybar_p,
                                             double ratio,
                                             double kept_old_AL,
                                             double kept_old_AU,
                                             double deleted_scaled_AL,
                                             double deleted_scaled_AU,
                                             double tol,
                                             double* yp,
                                             double* yd) {
  if (fabs(ybar_p) <= tol) {
    *yp = 0.0;
    *yd = 0.0;
    return true;
  }

  const double bar_l = fmax(kept_old_AL, deleted_scaled_AL);
  const double bar_u = fmin(kept_old_AU, deleted_scaled_AU);
  if (!(bar_l <= bar_u + tol)) {
    *yp = ybar_p;
    *yd = 0.0;
    return false;
  }

  const bool lower_tightened = bar_l > kept_old_AL + tol;
  const bool upper_tightened = bar_u < kept_old_AU - tol;
  if (lower_tightened && upper_tightened) {
    *yp = ybar_p;
    *yd = 0.0;
    return false;
  }
  if (!lower_tightened && !upper_tightened) {
    *yp = ybar_p;
    *yd = 0.0;
    return true;
  }

  if (ybar_p > 0.0) {
    if (lower_tightened) {
      *yp = 0.0;
      *yd = ratio * ybar_p;
    } else {
      *yp = ybar_p;
      *yd = 0.0;
    }
    return true;
  }

  if (lower_tightened) {
    *yp = ybar_p;
    *yd = 0.0;
  } else {
    *yp = 0.0;
    *yd = ratio * ybar_p;
  }
  return true;
}

__device__ void _doubleton_equations_recover_z_split(double x_keep,
                                              double z_keep_hat,
                                              double alpha,
                                              double old_keep_l,
                                              double old_keep_u,
                                              double new_keep_l,
                                              double new_keep_u,
                                              double lower_source,
                                              double upper_source,
                                              double tol,
                                              double* z_keep,
                                              double* z_elim) {
  constexpr double BOUND_UNKNOWN = 0.0;
  constexpr double BOUND_INHERITED = 1.0;
  constexpr double BOUND_INDUCED = 2.0;
  if (fabs(alpha) <= tol) {
    *z_keep = z_keep_hat;
    *z_elim = 0.0;
    return;
  }

  const bool at_lower = _postsolve_is_finite(new_keep_l) && x_keep <= new_keep_l + tol;
  const bool at_upper = _postsolve_is_finite(new_keep_u) && x_keep >= new_keep_u - tol;
  if (!at_lower && !at_upper) {
    *z_keep = 0.0;
    *z_elim = 0.0;
    return;
  }

  double source = at_lower ? lower_source : upper_source;
  if (source == BOUND_UNKNOWN) {
    if (at_lower) {
      source = (_postsolve_is_finite(old_keep_l) && fabs(new_keep_l - old_keep_l) <= tol)
                   ? BOUND_INHERITED
                   : BOUND_INDUCED;
    } else {
      source = (_postsolve_is_finite(old_keep_u) && fabs(new_keep_u - old_keep_u) <= tol)
                   ? BOUND_INHERITED
                   : BOUND_INDUCED;
    }
  }

  if (source == BOUND_INDUCED) {
    *z_keep = 0.0;
    *z_elim = z_keep_hat / alpha;
  } else {
    *z_keep = z_keep_hat;
    *z_elim = 0.0;
  }
}

__device__ double _doubleton_equations_deleted_row_dual_from_original(const double* y_org,
                                                               const NnzOffset* AT_rowPtr,
                                                               const std::int32_t* AT_colVal,
                                                               const double* AT_nzVal,
                                                               const double* c,
                                                               std::int32_t row,
                                                               std::int32_t elim_col,
                                                               double elim_coeff,
                                                               double z_elim) {
  double sum_other = 0.0;
  for (NnzOffset p = AT_rowPtr[elim_col]; p < AT_rowPtr[elim_col + 1]; ++p) {
    const std::int32_t prow = AT_colVal[p];
    if (prow != row) {
      sum_other += AT_nzVal[p] * y_org[prow];
    }
  }
  return (c[elim_col] - sum_other - z_elim) / elim_coeff;
}

__device__ double _doubleton_equations_deleted_row_dual_from_tape(const double* y_org,
                                                           const std::int32_t* tape_idx,
                                                           const double* tape_val,
                                                           NnzOffset idx0,
                                                           NnzOffset val0,
                                                           double elim_coeff,
                                                           double z_elim) {
  const std::int32_t support_count = tape_idx[idx0 + 3];
  double sum_other = 0.0;
  for (std::int32_t t = 0; t < support_count; ++t) {
    const std::int32_t row = tape_idx[idx0 + 4 + t];
    const double coeff = tape_val[val0 + 12 + t];
    sum_other += coeff * y_org[row];
  }
  const double elim_obj = tape_val[val0 + 9];
  return (elim_obj - sum_other - z_elim) / elim_coeff;
}

__device__ double _postsolve_compute_aty_col(const NnzOffset* AT_rowPtr,
                                             const std::int32_t* AT_colVal,
                                             const double* AT_nzVal,
                                             const double* y_org,
                                             std::int32_t col) {
  double aty = 0.0;
  for (NnzOffset p = AT_rowPtr[col]; p < AT_rowPtr[col + 1]; ++p) {
    aty += AT_nzVal[p] * y_org[AT_colVal[p]];
  }
  return aty;
}

__device__ double _postsolve_project_single_column_dual_from_original(double* x_org,
                                                                      double* z_org,
                                                                      const NnzOffset* AT_rowPtr,
                                                                      const std::int32_t* AT_colVal,
                                                                      const double* AT_nzVal,
                                                                      const double* y_org,
                                                                      const double* c,
                                                                      const double* l,
                                                                      const double* u,
                                                                      std::int32_t col,
                                                                      double tol) {
  const double r = c[col] - _postsolve_compute_aty_col(AT_rowPtr, AT_colVal, AT_nzVal, y_org, col);
  const double xj = x_org[col];
  const double lj = l[col];
  const double uj = u[col];
  const bool at_lower = _postsolve_at_lower(xj, lj, tol);
  const bool at_upper = _postsolve_at_upper(xj, uj, tol);
  double zj = 0.0;
  if (at_lower && at_upper) {
    zj = r;
  } else if (!at_lower && !at_upper) {
    zj = 0.0;
  } else if (at_lower) {
    zj = fmax(r, 0.0);
  } else {
    zj = fmin(r, 0.0);
  }
  const double dual_eps = fmin(tol, 1.0e-12);
  if (zj > dual_eps && !isfinite(lj)) {
    zj = 0.0;
  } else if (zj < -dual_eps && !isfinite(uj)) {
    zj = 0.0;
  }
  z_org[col] = zj;
  return zj;
}

__global__ void _kernel_replay_postsolve_tape(double* x_org,
                                              double* y_org,
                                              double* z_org,
                                              std::uint8_t* z_retrieved,
                                              const std::int32_t* reduction_types,
                                              const NnzOffset* idx_starts,
                                              const NnzOffset* val_starts,
                                              const std::int32_t* tape_idx,
                                              const double* tape_val,
                                              const NnzOffset* A_rowPtr,
                                              const std::int32_t* A_colVal,
                                              const double* A_nzVal,
                                              const NnzOffset* AT_rowPtr,
                                              const std::int32_t* AT_colVal,
                                              const double* AT_nzVal,
                                              const double* c,
                                              const double* AL,
                                              const double* AU,
                                              const double* l,
                                              const double* u,
                                              NnzOffset record_begin,
                                              NnzOffset record_end,
                                              std::int32_t has_original_model,
                                              std::int32_t replay_primal,
                                              std::int32_t replay_dual,
                                              std::int32_t parallel_row_doc_recovery,
                                              double tol) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (NnzOffset k = record_end - 1; k >= record_begin; --k) {
    const std::int32_t reduction_type = reduction_types[k];
    const NnzOffset idx0 = idx_starts[k];
    const NnzOffset idx1 = idx_starts[k + 1] - 1;
    const NnzOffset val0 = val_starts[k];
    const NnzOffset val1 = val_starts[k + 1] - 1;
    if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::ParallelColumn)) {
      const std::int32_t from_idx = tape_idx[idx0];
      const std::int32_t to_idx = tape_idx[idx0 + 1];
      const double ratio = tape_val[val0];
      const double lo = tape_val[val0 + 1];
      const double hi = tape_val[val0 + 2];
      const double to_l = tape_val[val0 + 3];
      const double to_u = tape_val[val0 + 4];
      if (replay_primal != 0) {
        const double y_val = x_org[to_idx];
        const double z_to = z_org != nullptr ? z_org[to_idx] : 0.0;
        double x_from = 0.0;
        double x_to = y_val;
        _recover_parallel_column_primal(y_val, z_to, ratio, lo, hi, to_l, to_u, tol, &x_from, &x_to);
        x_org[from_idx] = x_from;
        x_org[to_idx] = x_to;
      }
      if (replay_dual != 0 && z_org != nullptr) {
        z_org[from_idx] = ratio * z_org[to_idx];
        if (z_retrieved != nullptr) {
          z_retrieved[from_idx] = std::uint8_t{1};
        }
      }
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::DoubletonEquation)) {
      const std::int32_t elim_col = tape_idx[idx0];
      const std::int32_t keep_col = tape_idx[idx0 + 1];
      const std::int32_t row = tape_idx[idx0 + 2];
      const double elim_coeff = tape_val[val0];
      const double keep_coeff = tape_val[val0 + 1];
      const double rhs = tape_val[val0 + 2];
      const double elim_obj = tape_val[val0 + 9];
      const double alpha = -keep_coeff / elim_coeff;
      const double beta = rhs / elim_coeff;
      if (replay_primal != 0) {
        x_org[elim_col] = beta + alpha * x_org[keep_col];
      }
      if (replay_dual != 0 && z_org != nullptr && y_org != nullptr && has_original_model != 0) {
        const double old_keep_l = tape_val[val0 + 5];
        const double old_keep_u = tape_val[val0 + 6];
        const double new_keep_l = tape_val[val0 + 7];
        const double new_keep_u = tape_val[val0 + 8];
        const double lower_source = val1 >= val0 + 10 ? tape_val[val0 + 10] : 0.0;
        const double upper_source = val1 >= val0 + 11 ? tape_val[val0 + 11] : 0.0;
        double z_keep = 0.0;
        double z_elim = 0.0;
        _doubleton_equations_recover_z_split(x_org[keep_col],
                                      z_org[keep_col],
                                      alpha,
                                      old_keep_l,
                                      old_keep_u,
                                      new_keep_l,
                                      new_keep_u,
                                      lower_source,
                                      upper_source,
                                      tol,
                                      &z_keep,
                                      &z_elim);
        const bool has_support_payload = idx1 >= idx0 + 3 && val1 >= val0 + 11;
        y_org[row] = has_support_payload
                         ? _doubleton_equations_deleted_row_dual_from_tape(
                               y_org, tape_idx, tape_val, idx0, val0, elim_coeff, z_elim)
                         : _doubleton_equations_deleted_row_dual_from_original(
                               y_org, AT_rowPtr, AT_colVal, AT_nzVal, c, row, elim_col, elim_coeff, z_elim);
        z_org[keep_col] = z_keep;
        z_org[elim_col] = z_elim;
        if (z_retrieved != nullptr) {
          z_retrieved[keep_col] = std::uint8_t{1};
          z_retrieved[elim_col] = std::uint8_t{1};
        }
      } else if (replay_dual != 0) {
        if (z_org != nullptr) {
          z_org[elim_col] = 0.0;
          if (z_retrieved != nullptr) {
            z_retrieved[elim_col] = std::uint8_t{1};
          }
        }
        if (y_org != nullptr && row >= 0) {
          y_org[row] = elim_obj / elim_coeff;
        }
      }
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::SubCol)) {
      const std::int32_t elim_col = tape_idx[idx0];
      const std::int32_t row = tape_idx[idx0 + 1];
      const std::int32_t support_count = tape_idx[idx0 + 2];
      const double pivot_coeff = tape_val[val0];
      const double rhs = tape_val[val0 + 1];
      const double elim_obj = tape_val[val0 + 4];
      const bool row_deleted = tape_val[val0 + 5] != 0.0;
      double partial = 0.0;
      for (std::int32_t t = 0; t < support_count; ++t) {
        const std::int32_t col = tape_idx[idx0 + 3 + t];
        const double coeff = tape_val[val0 + 6 + t];
        partial += coeff * x_org[col];
      }
      if (replay_primal != 0) {
        x_org[elim_col] = (rhs - partial) / pivot_coeff;
      }
      if (replay_dual != 0 && y_org != nullptr && row >= 0) {
        if (row_deleted) {
          y_org[row] = elim_obj / pivot_coeff;
        } else {
          y_org[row] += elim_obj / pivot_coeff;
        }
      }
      if (replay_dual != 0 && z_org != nullptr) {
        z_org[elim_col] = row_deleted || y_org == nullptr ? 0.0 : elim_obj - pivot_coeff * y_org[row];
        if (z_retrieved != nullptr) {
          z_retrieved[elim_col] = std::uint8_t{1};
        }
      }
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::FixedCol)) {
      const std::int32_t col = tape_idx[idx0];
      if (replay_primal != 0) {
        x_org[col] = tape_val[val0];
      }
      if (replay_dual != 0 && z_org != nullptr) {
        double zj = tape_val[val0 + 1];
        for (NnzOffset pos = idx0 + 1; pos <= idx1; ++pos) {
          const std::int32_t row = tape_idx[pos];
          const double coeff = tape_val[val0 + 2 + (pos - (idx0 + 1))];
          if (y_org != nullptr) {
            zj -= coeff * y_org[row];
          }
        }
        z_org[col] = zj;
        if (z_retrieved != nullptr) {
          z_retrieved[col] = std::uint8_t{1};
        }
      }
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::FixedColInf)) {
      const bool fix_to_pos_inf = tape_idx[idx0] > 0;
      const std::int32_t col = tape_idx[idx0 + 1];
      const std::int32_t n_rows = static_cast<std::int32_t>(llround(tape_val[val0]));
      double extreme_val = tape_val[val0 + 1];
      NnzOffset idx_pos = idx0 + 2;
      NnzOffset val_pos = val0 + 2;
      if (replay_primal != 0) {
        for (std::int32_t r = 0; r < n_rows; ++r) {
          const NnzOffset row_len = tape_idx[idx_pos];
          double side = tape_val[val_pos];
          double coeff = 0.0;
          for (std::int32_t j = 0; j < row_len; ++j) {
            const std::int32_t row_col = tape_idx[idx_pos + 1 + j];
            const double row_coeff = tape_val[val_pos + 1 + j];
            if (row_col == col) {
              coeff = row_coeff;
            } else {
              side -= row_coeff * x_org[row_col];
            }
          }
          const double candidate = side / coeff;
          extreme_val = fix_to_pos_inf ? fmax(extreme_val, candidate) : fmin(extreme_val, candidate);
          idx_pos += row_len + 1;
          val_pos += row_len + 1;
        }
        x_org[col] = extreme_val;
      }
      if (replay_dual != 0 && z_org != nullptr) {
        z_org[col] = 0.0;
        if (z_retrieved != nullptr) {
          z_retrieved[col] = std::uint8_t{1};
        }
      }
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::FmeCol)) {
      const std::int32_t K = tape_idx[idx0];
      const std::int32_t elim_col = tape_idx[idx0 + 1];
      const std::int32_t L = static_cast<std::int32_t>(llround(tape_val[val0]));
      NnzOffset col_cursor = idx0 + 3 + K;
      NnzOffset vpos = val0 + 1;
      double lo = -INFINITY;
      double hi = INFINITY;
      bool has_lo = false;
      bool has_hi = false;
      if (replay_primal != 0) {
        for (std::int32_t s = 0; s < L; ++s) {
          const double orient = tape_val[vpos];
          const NnzOffset len = static_cast<NnzOffset>(llround(tape_val[vpos + 1]));
          double cand = tape_val[vpos + 2];
          for (std::int32_t t = 0; t < len; ++t) {
            const std::int32_t row_col = tape_idx[col_cursor++];
            const double coeff = tape_val[vpos + 3 + t];
            cand += coeff * x_org[row_col];
          }
          if (orient > 0.0) {
            lo = fmax(lo, cand);
            has_lo = true;
          } else {
            hi = fmin(hi, cand);
            has_hi = true;
          }
          vpos += 3 + len;
        }
        x_org[elim_col] = has_lo ? lo : (has_hi ? hi : 0.0);
      }
      if (replay_dual != 0 && z_org != nullptr) {
        z_org[elim_col] = 0.0;
        if (z_retrieved != nullptr) {
          z_retrieved[elim_col] = std::uint8_t{1};
        }
      }
      if (replay_dual != 0 && y_org != nullptr) {
        for (std::int32_t t = 0; t < K; ++t) {
          y_org[tape_idx[idx0 + 2 + t]] = 0.0;
        }
      }
    } else if (replay_dual != 0 &&
               reduction_type == static_cast<std::int32_t>(PostsolveReductionType::EqToIneq)) {
      if (y_org != nullptr) {
        const std::int32_t row = tape_idx[idx0];
        y_org[row] += tape_val[val0];
      }
    } else if (replay_dual != 0 &&
               reduction_type == static_cast<std::int32_t>(PostsolveReductionType::ParallelRow) &&
               idx1 >= idx0 + 1 && val1 >= val0 + 4) {
      const std::int32_t kept_row = tape_idx[idx0];
      const std::int32_t deleted_row = tape_idx[idx0 + 1];
      const double ratio = tape_val[val0];
      const double kept_old_AL = tape_val[val0 + 1];
      const double kept_old_AU = tape_val[val0 + 2];
      const double deleted_old_AL = tape_val[val0 + 3];
      const double deleted_old_AU = tape_val[val0 + 4];
      if (fabs(ratio) <= tol || y_org == nullptr) {
        continue;
      }
      double deleted_scaled_AL = 0.0;
      double deleted_scaled_AU = 0.0;
      _parallel_row_scaled_interval(deleted_old_AL, deleted_old_AU, ratio,
                                    &deleted_scaled_AL, &deleted_scaled_AU);
      if (_parallel_row_interval_contains(deleted_scaled_AL,
                                          deleted_scaled_AU,
                                          kept_old_AL,
                                          kept_old_AU,
                                          tol)) {
        y_org[deleted_row] = 0.0;
        continue;
      }
      if (has_original_model != 0) {
        const double alpha_p = _parallel_row_activity(kept_row, A_rowPtr, A_colVal, A_nzVal, x_org);
        const double alpha_d = _parallel_row_activity(deleted_row, A_rowPtr, A_colVal, A_nzVal, x_org);
        const std::uint8_t kept_state =
            _parallel_row_dual_state(alpha_p, kept_old_AL, kept_old_AU, tol);
        const std::uint8_t deleted_state =
            _parallel_row_dual_state(alpha_d, deleted_old_AL, deleted_old_AU, tol);
        double yd_lo = 0.0;
        double yd_hi = 0.0;
        double yp_lo = 0.0;
        double yp_hi = 0.0;
        _parallel_row_state_interval(deleted_state, &yd_lo, &yd_hi);
        _parallel_row_deleted_dual_interval(y_org[kept_row], ratio, kept_state, &yp_lo, &yp_hi);
        const double sd_lo = fmax(yd_lo, yp_lo);
        const double sd_hi = fmin(yd_hi, yp_hi);
        if (sd_lo <= sd_hi + tol) {
          const double yd_new = _clamp_with_inf(0.0, sd_lo, sd_hi);
          const double yp_new = y_org[kept_row] - yd_new / ratio;
          y_org[kept_row] = yp_new;
          y_org[deleted_row] = yd_new;
          continue;
        }

        if (parallel_row_doc_recovery != 0) {
          double yp_doc = y_org[kept_row];
          double yd_doc = 0.0;
          if (_parallel_row_doc_dual_split(y_org[kept_row],
                                           ratio,
                                           kept_old_AL,
                                           kept_old_AU,
                                           deleted_scaled_AL,
                                           deleted_scaled_AU,
                                           tol,
                                           &yp_doc,
                                           &yd_doc)) {
            y_org[kept_row] = yp_doc;
            y_org[deleted_row] = yd_doc;
          }
        }
      }
    } else if (replay_dual != 0 &&
               reduction_type == static_cast<std::int32_t>(PostsolveReductionType::DeletedRow) &&
               has_original_model != 0 && idx1 >= idx0 + 1 && val1 >= val0 + 2) {
      const std::int32_t row = tape_idx[idx0];
      const std::int32_t col = tape_idx[idx0 + 1];
      const double old_AL = tape_val[val0];
      const double old_AU = tape_val[val0 + 1];
      const double aij = tape_val[val0 + 2];
      if (fabs(aij) <= tol) {
        continue;
      }
      double gamma_j = c[col];
      for (NnzOffset p = AT_rowPtr[col]; p < AT_rowPtr[col + 1]; ++p) {
        const std::int32_t prow = AT_colVal[p];
        if (prow != row && y_org != nullptr) {
          gamma_j -= AT_nzVal[p] * y_org[prow];
        }
      }
      const double xj = x_org[col];
      const double zbar_j = z_org[col];
      const double row_activity = aij * xj;
      const bool row_at_lower = isfinite(old_AL) && row_activity <= old_AL + tol;
      const bool row_at_upper = isfinite(old_AU) && row_activity >= old_AU - tol;
      double iz_lo = -INFINITY;
      double iz_hi = INFINITY;
      if (!(row_at_lower && row_at_upper)) {
        if (row_at_lower) {
          if (aij > 0.0) {
            iz_hi = gamma_j;
          } else {
            iz_lo = gamma_j;
          }
        } else if (row_at_upper) {
          if (aij > 0.0) {
            iz_lo = gamma_j;
          } else {
            iz_hi = gamma_j;
          }
        } else {
          iz_lo = gamma_j;
          iz_hi = gamma_j;
        }
      }
      const bool at_lower = _postsolve_at_lower(xj, l[col], tol);
      const bool at_upper = _postsolve_at_upper(xj, u[col], tol);
      double z_lo = iz_lo;
      double z_hi = iz_hi;
      if (!at_lower && !at_upper) {
        if ((isfinite(iz_lo) && iz_lo > tol) || (isfinite(iz_hi) && iz_hi < -tol)) {
          z_lo = gamma_j;
          z_hi = gamma_j;
        } else {
          z_lo = 0.0;
          z_hi = 0.0;
        }
      } else if (at_lower && !at_upper) {
        z_lo = fmax(z_lo, 0.0);
      } else if (!at_lower && at_upper) {
        z_hi = fmin(z_hi, 0.0);
      }
      if (isfinite(z_lo) && isfinite(z_hi) && z_lo > z_hi + tol) {
        continue;
      }
      const double z_new = _clamp_with_inf(zbar_j, z_lo, z_hi);
      z_org[col] = z_new;
      if (y_org != nullptr) {
        y_org[row] = (gamma_j - z_new) / aij;
      }
      if (z_retrieved != nullptr) {
        z_retrieved[col] = std::uint8_t{1};
      }
    } else if (replay_dual != 0 &&
               reduction_type == static_cast<std::int32_t>(PostsolveReductionType::BoundChangeTheRow) &&
               has_original_model != 0 && idx1 >= idx0 + 1) {
      const std::int32_t col = tape_idx[idx0];
      const std::int32_t row = tape_idx[idx0 + 1];
      if (z_retrieved != nullptr && z_retrieved[col] == std::uint8_t{0}) {
        continue;
      }
      const double old_l = tape_val[val0];
      const double old_u = tape_val[val0 + 1];
      const double new_l = tape_val[val0 + 2];
      const double new_u = tape_val[val0 + 3];
      const bool lower_changed = new_l > old_l + tol;
      const bool upper_changed = new_u < old_u - tol;
      if (lower_changed == upper_changed) {
        continue;
      }
      const double implied_bound = lower_changed ? new_l : new_u;
      const double original_other_bound = lower_changed ? old_u : old_l;
      const bool original_other_is_lower = !lower_changed;
      const double xj = x_org[col];
      const double zj = z_org[col];
      if (_postsolve_is_finite(original_other_bound) && fabs(xj - original_other_bound) <= tol) {
        const bool sign_ok = original_other_is_lower ? (zj >= -tol) : (zj <= tol);
        if (sign_ok) {
          continue;
        }
      }
      if (fabs(xj - implied_bound) > tol) {
        continue;
      }
      double aij = 0.0;
      for (NnzOffset p = A_rowPtr[row]; p < A_rowPtr[row + 1]; ++p) {
        if (A_colVal[p] == col) {
          aij = A_nzVal[p];
          break;
        }
      }
      if (fabs(aij) <= tol) {
        continue;
      }
      const double delta_y = zj / aij;
      if (y_org != nullptr) {
        y_org[row] += delta_y;
      }
      for (NnzOffset p = A_rowPtr[row]; p < A_rowPtr[row + 1]; ++p) {
        const std::int32_t kcol = A_colVal[p];
        if (kcol == col || (z_retrieved != nullptr && z_retrieved[kcol] == std::uint8_t{0})) {
          continue;
        }
        z_org[kcol] -= (A_nzVal[p] / aij) * zj;
      }
      z_org[col] = 0.0;
      if (z_retrieved != nullptr) {
        z_retrieved[col] = std::uint8_t{1};
      }
    }
  }
}

void restore_splits(double* x_org, const std::vector<AbsoluteValueEpigraphSplitRecovery>& splits) {
  if (splits.empty()) {
    return;
  }
  std::vector<std::int32_t> t_cols;
  std::vector<std::int32_t> e_cols;
  std::vector<double> rhos;
  t_cols.reserve(splits.size());
  e_cols.reserve(splits.size());
  rhos.reserve(splits.size());
  for (const AbsoluteValueEpigraphSplitRecovery& split : splits) {
    t_cols.push_back(split.t_col);
    e_cols.push_back(split.e_col);
    rhos.push_back(split.rho);
  }
  std::int32_t* t_d = copy_vector_to_device(t_cols, "cudaMalloc/copy absolute_value_epigraph split t_cols");
  std::int32_t* e_d = copy_vector_to_device(e_cols, "cudaMalloc/copy absolute_value_epigraph split e_cols");
  double* rho_d = copy_vector_to_device(rhos, "cudaMalloc/copy absolute_value_epigraph split rhos");
  const std::int32_t k = static_cast<std::int32_t>(splits.size());
  const int blocks = (k + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_structural_restore_splits<<<blocks, GPU_PRESOLVE_THREADS>>>(x_org, t_d, e_d, rho_d, k);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_restore_splits");
  cudaFree(t_d);
  cudaFree(e_d);
  cudaFree(rho_d);
}

void restore_outer_pairs(double* x_org, const std::vector<AbsoluteValueOuterPairRecovery>& pairs) {
  if (pairs.empty()) {
    return;
  }
  std::vector<std::int32_t> bound_cols;
  std::vector<std::int32_t> free_cols;
  bound_cols.reserve(pairs.size());
  free_cols.reserve(pairs.size());
  for (const AbsoluteValueOuterPairRecovery& pair : pairs) {
    bound_cols.push_back(pair.bound_col);
    free_cols.push_back(pair.free_col);
  }
  std::int32_t* bound_d = copy_vector_to_device(bound_cols, "cudaMalloc/copy absolute_value_epigraph outer bound_cols");
  std::int32_t* free_d = copy_vector_to_device(free_cols, "cudaMalloc/copy absolute_value_epigraph outer free_cols");
  const std::int32_t k = static_cast<std::int32_t>(pairs.size());
  const int blocks = (k + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_structural_restore_outer_pairs<<<blocks, GPU_PRESOLVE_THREADS>>>(x_org, bound_d, free_d, k);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_restore_outer_pairs");
  cudaFree(bound_d);
  cudaFree(free_d);
}

void restore_linked_slacks(double* x_org, const std::vector<AbsoluteValueLinkedSlackRecovery>& slacks) {
  if (slacks.empty()) {
    return;
  }
  std::vector<std::int32_t> slack_cols;
  std::vector<std::int32_t> t_cols;
  std::vector<double> factors;
  slack_cols.reserve(slacks.size());
  t_cols.reserve(slacks.size());
  factors.reserve(slacks.size());
  for (const AbsoluteValueLinkedSlackRecovery& slack : slacks) {
    slack_cols.push_back(slack.slack_col);
    t_cols.push_back(slack.t_col);
    factors.push_back(slack.factor);
  }
  std::int32_t* slack_d = copy_vector_to_device(slack_cols, "cudaMalloc/copy absolute_value_epigraph linked slack_cols");
  std::int32_t* t_d = copy_vector_to_device(t_cols, "cudaMalloc/copy absolute_value_epigraph linked t_cols");
  double* factor_d = copy_vector_to_device(factors, "cudaMalloc/copy absolute_value_epigraph linked factors");
  const std::int32_t k = static_cast<std::int32_t>(slacks.size());
  const int blocks = (k + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_structural_restore_linked_slacks<<<blocks, GPU_PRESOLVE_THREADS>>>(x_org, slack_d, t_d, factor_d, k);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_restore_linked_slacks");
  cudaFree(slack_d);
  cudaFree(t_d);
  cudaFree(factor_d);
}

void restore_max_slacks(double* x_org,
                        const std::vector<AbsoluteValueMaxSlackRecovery>& slacks,
                        const double* original_l) {
  for (const AbsoluteValueMaxSlackRecovery& slack : slacks) {
    if (slack.t_cols.empty()) {
      continue;
    }
    double lower_bound = 0.0;
    if (original_l != nullptr) {
      throw_if_cuda_error(cudaMemcpy(&lower_bound, original_l + slack.slack_col, sizeof(double),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy absolute_value_epigraph max slack lower bound");
      if (!isfinite(lower_bound)) {
        lower_bound = 0.0;
      }
    }
    std::int32_t* t_d = copy_vector_to_device(slack.t_cols, "cudaMalloc/copy absolute_value_epigraph max t_cols");
    double* factor_d = copy_vector_to_device(slack.factors, "cudaMalloc/copy absolute_value_epigraph max factors");
    _kernel_structural_restore_max_slack<<<1, GPU_PRESOLVE_THREADS>>>(
        x_org, slack.slack_col, t_d, factor_d, static_cast<std::int32_t>(slack.t_cols.size()), lower_bound);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_structural_restore_max_slack");
    cudaFree(t_d);
    cudaFree(factor_d);
  }
}

void restore_absolute_value_epigraph_step(
    double* x_org,
    const AbsoluteValueEpigraphPrimalRecoveryStep& recovery,
    const double* original_l) {
  restore_splits(x_org, recovery.splits);
  restore_outer_pairs(x_org, recovery.outer_pairs);
  restore_linked_slacks(x_org, recovery.linked_slacks);
  restore_max_slacks(x_org, recovery.max_slacks, original_l);
}

void restore_variable_pair_aggregation(
    double* x_org,
    const VariablePairAggregationRecoveryStep& recovery) {
  const std::size_t size = recovery.plus_cols.size();
  if (size == 0) {
    return;
  }
  const std::int32_t count = static_cast<std::int32_t>(size);
  std::int32_t* plus_cols =
      copy_vector_to_device(recovery.plus_cols,
                            "cudaMalloc/copy variable_pair_aggregation plus cols");
  std::int32_t* minus_cols =
      copy_vector_to_device(recovery.minus_cols,
                            "cudaMalloc/copy variable_pair_aggregation minus cols");
  std::int32_t* root_plus_cols =
      copy_vector_to_device(recovery.root_plus_cols,
                            "cudaMalloc/copy variable_pair_aggregation root plus cols");
  std::int32_t* root_minus_cols =
      copy_vector_to_device(recovery.root_minus_cols,
                            "cudaMalloc/copy variable_pair_aggregation root minus cols");
  double* plus_shift =
      copy_vector_to_device(recovery.plus_shift,
                            "cudaMalloc/copy variable_pair_aggregation plus lower");
  double* minus_shift =
      copy_vector_to_device(recovery.minus_shift,
                            "cudaMalloc/copy variable_pair_aggregation minus lower");
  double* plus_cache = nullptr;
  double* minus_cache = nullptr;
  throw_if_cuda_error(cudaMalloc(&plus_cache, sizeof(double) * size),
                      "cudaMalloc variable_pair_aggregation plus cache");
  throw_if_cuda_error(cudaMalloc(&minus_cache, sizeof(double) * size),
                      "cudaMalloc variable_pair_aggregation minus cache");
  const int blocks =
      (count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_cache_variable_pair_root_values<<<blocks, GPU_PRESOLVE_THREADS>>>(
      plus_cache, minus_cache, x_org, root_plus_cols, root_minus_cols, count);
  throw_if_cuda_error(cudaGetLastError(),
                      "_kernel_cache_variable_pair_root_values");
  _kernel_restore_variable_pair_aggregation_members<<<blocks,
                                                GPU_PRESOLVE_THREADS>>>(
      x_org, plus_cols, minus_cols, plus_shift, minus_shift, plus_cache,
      minus_cache, count);
  throw_if_cuda_error(cudaGetLastError(),
                      "_kernel_restore_variable_pair_aggregation_members");
  cudaFree(plus_cols);
  cudaFree(minus_cols);
  cudaFree(root_plus_cols);
  cudaFree(root_minus_cols);
  cudaFree(plus_shift);
  cudaFree(minus_shift);
  cudaFree(plus_cache);
  cudaFree(minus_cache);
}

void restore_device_structural(double *x, const DeviceAbsoluteValueEpigraphStep &s, const double *lower) {
  const auto &a = s.splits;
  int n = a.ids[0].count;
  if (n)
    _kernel_structural_restore_splits<<<(n + 255) / 256, 256>>>(x, a.ids[0].data, a.ids[1].data,
                                                                a.values[0].data, n);
  const auto &b = s.pairs;
  n = b.ids[0].count;
  if (n)
    _kernel_structural_restore_outer_pairs<<<(n + 255) / 256, 256>>>(x, b.ids[0].data,
                                                                     b.ids[1].data, n);
  const auto &c = s.linked;
  n = c.ids[0].count;
  if (n)
    _kernel_structural_restore_linked_slacks<<<(n + 255) / 256, 256>>>(
        x, c.ids[0].data, c.ids[1].data, c.values[0].data, n);
  for (const auto &g : s.max_slacks)
    if (g.ids[1].count)
      _kernel_device_restore_max_slack<<<1, 256>>>(x, g.ids[0].data, g.ids[1].data,
                                                   g.values[0].data, g.ids[1].count, lower);
  throw_if_cuda_error(cudaGetLastError(), "restore device absolute_value_epigraph payload");
}
void restore_device_variable_pair_aggregation(double *x, const DeviceRecoveryGroup &g) {
  int n = g.ids[0].count;
  if (!n)
    return;
  double *plus = nullptr, *minus = nullptr;
  throw_if_cuda_error(cudaMalloc(&plus, sizeof(double) * std::size_t(n)), "allocate plus cache");
  throw_if_cuda_error(cudaMalloc(&minus, sizeof(double) * std::size_t(n)), "allocate minus cache");
  _kernel_cache_variable_pair_root_values<<<(n + 255) / 256, 256>>>(plus, minus, x, g.ids[2].data,
                                                                g.ids[3].data, n);
  _kernel_restore_variable_pair_aggregation_members<<<(n + 255) / 256, 256>>>(
      x, g.ids[0].data, g.ids[1].data, g.values[0].data, g.values[1].data, plus, minus, n);
  throw_if_cuda_error(cudaGetLastError(), "restore device variable_pair_aggregation payload");
  recovery_device_free(plus);
  recovery_device_free(minus);
}

void scatter_reduced_to_original(double* dst,
                                 const double* src,
                                 const std::vector<std::int32_t>& red2org,
                                 std::int32_t original_count,
                                 const char* context) {
  if (original_count > 0) {
    throw_if_cuda_error(cudaMemset(dst, 0, sizeof(double) * static_cast<std::size_t>(original_count)),
                        context);
  }
  if (red2org.empty() || src == nullptr) {
    return;
  }
  std::int32_t* red2org_d = copy_vector_to_device(red2org, context);
  const std::int32_t n_red = static_cast<std::int32_t>(red2org.size());
  const int blocks = (n_red + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_scatter_reduced_to_original<<<blocks, GPU_PRESOLVE_THREADS>>>(dst, src, red2org_d, n_red);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_scatter_reduced_to_original");
  cudaFree(red2org_d);
}

void restore_fixed_values(double* x_org,
                          const std::vector<std::int32_t>& fixed_idx,
                          const std::vector<double>& fixed_val) {
  if (fixed_idx.empty()) {
    return;
  }
  std::int32_t* fixed_idx_d = copy_vector_to_device(fixed_idx, "cudaMalloc/copy fixed_idx");
  double* fixed_val_d = copy_vector_to_device(fixed_val, "cudaMalloc/copy fixed_val");
  const std::int32_t fixed_count = static_cast<std::int32_t>(fixed_idx.size());
  const int blocks = (fixed_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_restore_fixed_values<<<blocks, GPU_PRESOLVE_THREADS>>>(x_org, fixed_idx_d, fixed_val_d, fixed_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_restore_fixed_values");
  cudaFree(fixed_idx_d);
  cudaFree(fixed_val_d);
}

struct DevicePostsolveTape {
  std::int32_t* types = nullptr;
  NnzOffset* index_starts = nullptr;
  NnzOffset* value_starts = nullptr;
  std::int32_t* indices = nullptr;
  double* vals = nullptr;
  NnzOffset count = 0;
  bool owns_storage = false;

  ~DevicePostsolveTape() {
    if (owns_storage) {
      cudaFree(types);
      cudaFree(index_starts);
      cudaFree(value_starts);
      cudaFree(indices);
      cudaFree(vals);
    }
  }
};

void upload_postsolve_tape(const PostsolveTape& tape, DevicePostsolveTape& device_tape) {
  device_tape.count = static_cast<NnzOffset>(tape.types.size());
  if (device_tape.count == 0) {
    return;
  }
  device_tape.owns_storage = true;
  device_tape.types = copy_vector_to_device(tape.types, "cudaMalloc/copy postsolve tape types");
  device_tape.index_starts =
      copy_vector_to_device(tape.index_starts, "cudaMalloc/copy postsolve tape index_starts");
  device_tape.value_starts =
      copy_vector_to_device(tape.value_starts, "cudaMalloc/copy postsolve tape value_starts");
  device_tape.indices = copy_vector_to_device(tape.indices, "cudaMalloc/copy postsolve tape indices");
  device_tape.vals = copy_vector_to_device(tape.vals, "cudaMalloc/copy postsolve tape vals");
}

void borrow_postsolve_tape(const PostsolveTapeGpu& tape,
                           DevicePostsolveTape& device_tape) {
  device_tape.types = tape.types;
  device_tape.index_starts = tape.index_starts;
  device_tape.value_starts = tape.value_starts;
  device_tape.indices = tape.indices;
  device_tape.vals = tape.vals;
  device_tape.count = tape.record_count;
  device_tape.owns_storage = false;
}

void replay_postsolve_tape_range(double* x_org,
                                 double* y_org,
                                 double* z_org,
                                 std::uint8_t* z_retrieved,
                                 const DevicePostsolveTape& tape,
                                 const LPInfoGpu* original_model,
                                 NnzOffset record_begin,
                                 NnzOffset record_end,
                                 bool replay_primal,
                                 bool replay_dual,
                                 double tol) {
  if (record_begin < 0 || record_end < record_begin ||
      record_end > tape.count) {
    throw std::runtime_error("invalid postsolve tape replay range");
  }
  if (record_begin == record_end) {
    return;
  }
  _kernel_replay_postsolve_tape<<<1, 1>>>(x_org,
                                          y_org,
                                          z_org,
                                          z_retrieved,
                                          tape.types,
                                          tape.index_starts,
                                          tape.value_starts,
                                          tape.indices,
                                          tape.vals,
                                          original_model != nullptr ? original_model->A.rowPtr : nullptr,
                                          original_model != nullptr ? original_model->A.colVal : nullptr,
                                          original_model != nullptr ? original_model->A.nzVal : nullptr,
                                          original_model != nullptr ? original_model->AT.rowPtr : nullptr,
                                          original_model != nullptr ? original_model->AT.colVal : nullptr,
                                          original_model != nullptr ? original_model->AT.nzVal : nullptr,
                                          original_model != nullptr ? original_model->c : nullptr,
                                          original_model != nullptr ? original_model->AL : nullptr,
                                          original_model != nullptr ? original_model->AU : nullptr,
                                          original_model != nullptr ? original_model->l : nullptr,
                                          original_model != nullptr ? original_model->u : nullptr,
                                          record_begin,
                                          record_end,
                                          original_model != nullptr ? 1 : 0,
                                          replay_primal ? 1 : 0,
                                          replay_dual ? 1 : 0,
                                          1,
                                          tol);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_replay_postsolve_tape");
}

void mark_retrieved_values(std::uint8_t* z_retrieved, const std::vector<std::int32_t>& red2org) {
  if (z_retrieved == nullptr || red2org.empty()) {
    return;
  }
  std::int32_t* red2org_d = copy_vector_to_device(red2org, "cudaMalloc/copy z_retrieved red2org");
  const std::int32_t n_red = static_cast<std::int32_t>(red2org.size());
  const int blocks = (n_red + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_mark_retrieved<<<blocks, GPU_PRESOLVE_THREADS>>>(z_retrieved, red2org_d, n_red);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_mark_retrieved");
  cudaFree(red2org_d);
}

std::vector<std::uint8_t> build_exact_replay_column_mask(const PostsolveTape& tape, std::int32_t n) {
  std::vector<std::uint8_t> protected_cols(static_cast<std::size_t>(n), std::uint8_t{0});
  for (std::size_t k = 0; k < tape.types.size(); ++k) {
    const NnzOffset idx0 = tape.index_starts[k];
    const auto type = static_cast<PostsolveReductionType>(tape.types[k]);
    if (type == PostsolveReductionType::SubCol) {
      const std::int32_t col = tape.indices[idx0];
      if (0 <= col && col < n) {
        protected_cols[static_cast<std::size_t>(col)] = std::uint8_t{1};
      }
    } else if (type == PostsolveReductionType::DoubletonEquation) {
      const std::int32_t elim_col = tape.indices[idx0];
      const std::int32_t keep_col = tape.indices[idx0 + 1];
      if (0 <= elim_col && elim_col < n) {
        protected_cols[static_cast<std::size_t>(elim_col)] = std::uint8_t{1};
      }
      if (0 <= keep_col && keep_col < n) {
        protected_cols[static_cast<std::size_t>(keep_col)] = std::uint8_t{1};
      }
    } else if (type == PostsolveReductionType::FixedColInf) {
      const std::int32_t col = tape.indices[idx0 + 1];
      if (0 <= col && col < n) {
        protected_cols[static_cast<std::size_t>(col)] = std::uint8_t{1};
      }
    } else if (type == PostsolveReductionType::ParallelColumn) {
      const std::int32_t col = tape.indices[idx0];
      if (0 <= col && col < n) {
        protected_cols[static_cast<std::size_t>(col)] = std::uint8_t{1};
      }
    }
  }
  return protected_cols;
}

__global__ void mark_protected_device(int *mask, const int *types, const NnzOffset *starts,
                                      const int *indices, int count, int n) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= count)
    return;
  int t = types[k];
  NnzOffset p = starts[k];
  int a = -1, b = -1;
  if (t == int(PostsolveReductionType::SubCol) || t == int(PostsolveReductionType::ParallelColumn))
    a = indices[p];
  if (t == int(PostsolveReductionType::DoubletonEquation)) {
    a = indices[p];
    b = indices[p + 1];
  }
  if (t == int(PostsolveReductionType::FixedColInf))
    a = indices[p + 1];
  if (a >= 0 && a < n)
    atomicOr(mask + a, 1);
  if (b >= 0 && b < n)
    atomicOr(mask + b, 1);
}
__global__ void pack_protected_device(unsigned char *out, const int *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = in[i] != 0;
}

void project_column_duals_from_original(double *x_org, double *y_org, double *z_org,
                                        const PresolveRecordGpu &record,
                                        const LPInfoGpu *original_model, double tol) {
  if (x_org == nullptr || y_org == nullptr || z_org == nullptr || original_model == nullptr ||
      original_model->AT.rows <= 0 || original_model->AT.rowPtr == nullptr ||
      original_model->c == nullptr || original_model->l == nullptr || original_model->u == nullptr) {
    return;
  }
  const std::int32_t n = original_model->AT.rows;
  std::uint8_t *protected_cols_d = nullptr;
  if (record.device) {
    int *mask = nullptr;
    throw_if_cuda_error(cudaMalloc(&mask, sizeof(int) * std::size_t(n)), "allocate protected mask");
    throw_if_cuda_error(cudaMemset(mask, 0, sizeof(int) * std::size_t(n)), "clear protected mask");
    throw_if_cuda_error(cudaMalloc(&protected_cols_d, std::size_t(n)), "allocate protected bytes");
    for (const auto &segment : record.device->segments) {
      mark_protected_device<<<(segment.record_count + 255) / 256, 256>>>(
          mask, segment.types, segment.index_starts, segment.indices, segment.record_count, n);
      throw_if_cuda_error(cudaGetLastError(), "mark protected columns");
    }
    pack_protected_device<<<(n + 255) / 256, 256>>>(protected_cols_d, mask, n);
    throw_if_cuda_error(cudaGetLastError(), "pack protected columns");
    recovery_device_free(mask);
  } else {
    auto protected_cols = build_exact_replay_column_mask(record.tape, n);
    protected_cols_d = copy_vector_to_device(protected_cols, "copy protected columns");
  }
  const int blocks = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_project_column_duals_from_original<<<blocks, GPU_PRESOLVE_THREADS>>>(
      x_org,
      z_org,
      protected_cols_d,
      original_model->AT.rowPtr,
      original_model->AT.colVal,
      original_model->AT.nzVal,
      y_org,
      original_model->c,
      original_model->l,
      original_model->u,
      n,
      tol);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_project_column_duals_from_original");
  cudaFree(protected_cols_d);
}

void validate_postsolve_tape(const PostsolveTape& tape) {
  const std::size_t count = tape.types.size();
  if (count > static_cast<std::size_t>(
                  std::numeric_limits<NnzOffset>::max())) {
    throw std::runtime_error("postsolve tape exceeds 64-bit replay range");
  }
  if (tape.index_starts.size() != count + 1 ||
      tape.value_starts.size() != count + 1 ||
      tape.dual_modes.size() != count) {
    throw std::runtime_error("malformed postsolve tape offsets");
  }
  if (tape.index_starts.front() != 0 || tape.value_starts.front() != 0) {
    throw std::runtime_error("postsolve tape offsets must start at zero");
  }
  for (std::size_t k = 0; k < count; ++k) {
    if (tape.index_starts[k] < 0 ||
        tape.index_starts[k] > tape.index_starts[k + 1] ||
        tape.value_starts[k] < 0 ||
        tape.value_starts[k] > tape.value_starts[k + 1]) {
      throw std::runtime_error("postsolve tape offsets are not monotone");
    }
    if (tape.types[k] <
            static_cast<std::int32_t>(PostsolveReductionType::FixedCol) ||
        tape.types[k] >
            static_cast<std::int32_t>(PostsolveReductionType::FmeCol)) {
      throw std::runtime_error("postsolve tape contains an unknown reduction");
    }
    if (tape.dual_modes[k] >
        static_cast<std::uint8_t>(PostsolveDualMode::Minimal)) {
      throw std::runtime_error("postsolve tape contains an unknown dual mode");
    }
  }
  if (static_cast<std::size_t>(tape.index_starts.back()) !=
          tape.indices.size() ||
      static_cast<std::size_t>(tape.value_starts.back()) != tape.vals.size()) {
    throw std::runtime_error("postsolve tape terminal offsets do not match data");
  }
}

void validate_recovery_col(std::int32_t col,
                           std::int32_t n0,
                           const char* context) {
  if (col < 0 || col >= n0) {
    throw std::runtime_error(std::string(context) +
                             " contains an invalid original column");
  }
}

void validate_absolute_value_epigraph_recovery(
    const AbsoluteValueEpigraphPrimalRecoveryStep& recovery,
    std::int32_t n0) {
  for (const AbsoluteValueEpigraphSplitRecovery& split : recovery.splits) {
    validate_recovery_col(split.t_col, n0, "absolute_value_epigraph split recovery");
    validate_recovery_col(split.e_col, n0, "absolute_value_epigraph split recovery");
  }
  for (const AbsoluteValueOuterPairRecovery& pair : recovery.outer_pairs) {
    validate_recovery_col(pair.bound_col, n0,
                          "absolute_value_epigraph outer-pair recovery");
    validate_recovery_col(pair.free_col, n0,
                          "absolute_value_epigraph outer-pair recovery");
  }
  for (const AbsoluteValueLinkedSlackRecovery& slack :
       recovery.linked_slacks) {
    validate_recovery_col(slack.slack_col, n0,
                          "absolute_value_epigraph linked-slack recovery");
    validate_recovery_col(slack.t_col, n0,
                          "absolute_value_epigraph linked-slack recovery");
  }
  for (const AbsoluteValueMaxSlackRecovery& slack : recovery.max_slacks) {
    if (slack.t_cols.size() != slack.factors.size()) {
      throw std::runtime_error(
          "absolute_value_epigraph max-slack recovery has mismatched payloads");
    }
    validate_recovery_col(slack.slack_col, n0,
                          "absolute_value_epigraph max-slack recovery");
    for (const std::int32_t col : slack.t_cols) {
      validate_recovery_col(col, n0, "absolute_value_epigraph max-slack recovery");
    }
  }
}

void validate_variable_pair_aggregation_recovery(
    const VariablePairAggregationRecoveryStep& recovery,
    std::int32_t n0) {
  const std::size_t count = recovery.plus_cols.size();
  if (recovery.minus_cols.size() != count ||
      recovery.root_plus_cols.size() != count ||
      recovery.root_minus_cols.size() != count ||
      recovery.plus_shift.size() != count ||
      recovery.minus_shift.size() != count ||
      count > static_cast<std::size_t>(
                  std::numeric_limits<std::int32_t>::max())) {
    throw std::runtime_error("variable_pair_aggregation recovery has mismatched payloads");
  }
  for (std::size_t i = 0; i < count; ++i) {
    validate_recovery_col(recovery.plus_cols[i], n0,
                          "variable_pair_aggregation recovery");
    validate_recovery_col(recovery.minus_cols[i], n0,
                          "variable_pair_aggregation recovery");
    validate_recovery_col(recovery.root_plus_cols[i], n0,
                          "variable_pair_aggregation recovery");
    validate_recovery_col(recovery.root_minus_cols[i], n0,
                          "variable_pair_aggregation recovery");
  }
}

void validate_primal_recovery_timeline(const PresolveRecordGpu& record) {
  if (!record.device)
    validate_postsolve_tape(record.tape);
  if (record.absolute_value_epigraph_recoveries.size() >
          static_cast<std::size_t>(
              std::numeric_limits<std::int32_t>::max()) ||
      record.variable_pair_aggregation_recoveries.size() >
          static_cast<std::size_t>(
              std::numeric_limits<std::int32_t>::max())) {
    throw std::runtime_error(
        "primal recovery payload exceeds 32-bit checkpoint range");
  }
  std::vector<std::uint8_t> seen_structural(
      record.absolute_value_epigraph_recoveries.size(), std::uint8_t{0});
  std::vector<std::uint8_t> seen_antipodal(
      record.variable_pair_aggregation_recoveries.size(), std::uint8_t{0});
  NnzOffset previous_tape_position = 0;
  const NnzOffset tape_count =
      record.device ? record.device->records : static_cast<NnzOffset>(record.tape.types.size());
  for (const PrimalRecoveryCheckpoint& checkpoint :
       record.primal_recovery_timeline) {
    if (checkpoint.tape_position < previous_tape_position ||
        checkpoint.tape_position > tape_count) {
      throw std::runtime_error(
          "primal recovery checkpoints are not chronological");
    }
    previous_tape_position = checkpoint.tape_position;
    if (checkpoint.kind == PrimalRecoveryKind::AbsoluteValueEpigraph) {
      if (checkpoint.payload_index < 0 ||
          checkpoint.payload_index >=
              static_cast<std::int32_t>(seen_structural.size()) ||
          seen_structural[static_cast<std::size_t>(
              checkpoint.payload_index)] != std::uint8_t{0}) {
        throw std::runtime_error(
            "invalid or duplicate absolute_value_epigraph recovery checkpoint");
      }
      seen_structural[static_cast<std::size_t>(checkpoint.payload_index)] =
          std::uint8_t{1};
      validate_absolute_value_epigraph_recovery(
          record.absolute_value_epigraph_recoveries[static_cast<std::size_t>(
              checkpoint.payload_index)],
          record.n0);
    } else if (checkpoint.kind ==
               PrimalRecoveryKind::VariablePairAggregation) {
      if (checkpoint.payload_index < 0 ||
          checkpoint.payload_index >=
              static_cast<std::int32_t>(seen_antipodal.size()) ||
          seen_antipodal[static_cast<std::size_t>(
              checkpoint.payload_index)] != std::uint8_t{0}) {
        throw std::runtime_error(
            "invalid or duplicate variable_pair_aggregation recovery checkpoint");
      }
      seen_antipodal[static_cast<std::size_t>(checkpoint.payload_index)] =
          std::uint8_t{1};
      validate_variable_pair_aggregation_recovery(
          record.variable_pair_aggregation_recoveries[static_cast<std::size_t>(
              checkpoint.payload_index)],
          record.n0);
    } else {
      throw std::runtime_error("unknown primal recovery checkpoint kind");
    }
  }
  for (const std::uint8_t seen : seen_structural) {
    if (seen == std::uint8_t{0}) {
      throw std::runtime_error(
          "absolute_value_epigraph recovery payload is missing a checkpoint");
    }
  }
  for (const std::uint8_t seen : seen_antipodal) {
    if (seen == std::uint8_t{0}) {
      throw std::runtime_error(
          "variable_pair_aggregation recovery payload is missing a checkpoint");
    }
  }
}

}  // namespace

void postsolve_restore_absolute_value_epigraph_gpu(
    double* x_org,
    const std::vector<AbsoluteValueEpigraphPrimalRecoveryStep>& recoveries,
    const double* original_l) {
  for (auto it = recoveries.rbegin(); it != recoveries.rend(); ++it) {
    restore_absolute_value_epigraph_step(x_org, *it, original_l);
  }
  throw_if_cuda_error(cudaDeviceSynchronize(), "postsolve_restore_absolute_value_epigraph_gpu synchronize");
}

GpuPostsolveResult postsolve_gpu(double* x_red,
                                 double* y_red,
                                 double* z_red,
                                 const PresolveRecordGpu& record,
                                 const double* original_l) {
  return postsolve_gpu(x_red, y_red, z_red, record, nullptr, original_l);
}

GpuPostsolveResult postsolve_gpu(double* x_red,
                                 double* y_red,
                                 double* z_red,
                                 const PresolveRecordGpu& record,
                                 const LPInfoGpu* original_model,
                                 const double* original_l) {
  validate_device_recovery(record);
  validate_primal_recovery_timeline(record);
  GpuPostsolveResult result;
  result.n0 = record.n0;
  result.m0 = record.m0;
  if (original_l == nullptr && original_model != nullptr) {
    original_l = original_model->l;
  }
  if (record.n0 > 0) {
    throw_if_cuda_error(cudaMalloc(&result.x_org, sizeof(double) * static_cast<std::size_t>(record.n0)),
                        "cudaMalloc postsolve x_org");
    throw_if_cuda_error(cudaMalloc(&result.z_org, sizeof(double) * static_cast<std::size_t>(record.n0)),
                        "cudaMalloc postsolve z_org");
  }
  if (record.m0 > 0) {
    throw_if_cuda_error(cudaMalloc(&result.y_org, sizeof(double) * static_cast<std::size_t>(record.m0)),
                        "cudaMalloc postsolve y_org");
  }

  std::uint8_t* z_retrieved = nullptr;
  if (record.n0 > 0) {
    throw_if_cuda_error(cudaMalloc(&z_retrieved, static_cast<std::size_t>(record.n0)),
                        "cudaMalloc postsolve z_retrieved");
    throw_if_cuda_error(cudaMemset(z_retrieved, 0, static_cast<std::size_t>(record.n0)),
                        "cudaMemset postsolve z_retrieved");
  }

  if (record.device) {
    const auto &d = *record.device;
    auto scatter = [&](double *dst, const double *src, const RecoveryArray<int> &map, int count) {
      if (count)
        throw_if_cuda_error(cudaMemset(dst, 0, sizeof(double) * std::size_t(count)),
                            "clear original vector");
      if (src && map.count)
        _kernel_scatter_reduced_to_original<<<(map.count + 255) / 256, 256>>>(dst, src, map.data,
                                                                              map.count);
    };
    scatter(result.x_org, x_red, d.cols, record.n0);
    scatter(result.y_org, y_red, d.rows, record.m0);
    scatter(result.z_org, z_red, d.cols, record.n0);
    if (d.cols.count)
      _kernel_mark_retrieved<<<(d.cols.count + 255) / 256, 256>>>(z_retrieved, d.cols.data,
                                                                  d.cols.count);
    for (const auto &f : d.fixed)
      if (f.indices.count)
        _kernel_restore_fixed_values<<<(f.indices.count + 255) / 256, 256>>>(
            result.x_org, f.indices.data, f.values.data, f.indices.count);
    throw_if_cuda_error(cudaGetLastError(), "device recovery scatter/fixed");
  } else {
    scatter_reduced_to_original(result.x_org, x_red, record.col_red2org, record.n0,
                                "cudaMemset/scatter x_org");
    scatter_reduced_to_original(result.y_org, y_red, record.row_red2org, record.m0,
                                "cudaMemset/scatter y_org");
    scatter_reduced_to_original(result.z_org, z_red, record.col_red2org, record.n0,
                                "cudaMemset/scatter z_org");
    mark_retrieved_values(z_retrieved, record.col_red2org);
    restore_fixed_values(result.x_org, record.fixed_idx, record.fixed_val);
  }
  DevicePostsolveTape device_tape;
  if (record.device) {
    device_tape.count = record.device->records;
  } else if (record.tape_gpu.record_count > 0) {
    if (record.tape_gpu.record_count !=
            static_cast<NnzOffset>(record.tape.types.size()) ||
        record.tape_gpu.index_count !=
            static_cast<NnzOffset>(record.tape.indices.size()) ||
        record.tape_gpu.value_count !=
            static_cast<NnzOffset>(record.tape.vals.size())) {
      throw std::runtime_error(
          "device postsolve tape does not mirror the canonical CPU tape");
    }
    borrow_postsolve_tape(record.tape_gpu, device_tape);
  } else {
    upload_postsolve_tape(record.tape, device_tape);
  }
  auto replay_range = [&](NnzOffset begin, NnzOffset end, bool primal, bool dual) {
    if (!record.device) {
      replay_postsolve_tape_range(result.x_org, result.y_org, result.z_org, z_retrieved,
                                  device_tape, original_model, begin, end, primal, dual, 1e-7);
      return;
    }
    NnzOffset stop = record.device->records;
    for (auto it = record.device->segments.rbegin(); it != record.device->segments.rend(); ++it) {
      NnzOffset start = stop - it->record_count;
      NnzOffset lo = std::max(begin, start), hi = std::min(end, stop);
      if (lo < hi) {
        DevicePostsolveTape view;
        borrow_postsolve_tape(*it, view);
        replay_postsolve_tape_range(result.x_org, result.y_org, result.z_org, z_retrieved, view,
                                    original_model, lo - start, hi - start, primal, dual, 1e-7);
      }
      stop = start;
    }
  };
  NnzOffset range_end = device_tape.count;
  for (auto checkpoint_it = record.primal_recovery_timeline.rbegin();
       checkpoint_it != record.primal_recovery_timeline.rend();
       ++checkpoint_it) {
    replay_range(checkpoint_it->tape_position, range_end, true, false);
    if (checkpoint_it->kind == PrimalRecoveryKind::AbsoluteValueEpigraph) {
      if (record.device)
        restore_device_structural(
            result.x_org, record.device->absolute_value_epigraph[checkpoint_it->payload_index], original_l);
      else
        restore_absolute_value_epigraph_step(result.x_org,
                                       record.absolute_value_epigraph_recoveries[static_cast<std::size_t>(
                                           checkpoint_it->payload_index)],
                                       original_l);
    } else {
      if (record.device)
        restore_device_variable_pair_aggregation(result.x_org,
                                 record.device->variable_pair_aggregation[checkpoint_it->payload_index]);
      else
        restore_variable_pair_aggregation(result.x_org,
                                    record.variable_pair_aggregation_recoveries[static_cast<std::size_t>(
                                        checkpoint_it->payload_index)]);
    }
    range_end = checkpoint_it->tape_position;
  }
  replay_range(0, range_end, true, false);
  replay_range(0, device_tape.count, false, true);
  project_column_duals_from_original(result.x_org, result.y_org, result.z_org, record,
                                     original_model, 1.0e-7);
  throw_if_cuda_error(cudaDeviceSynchronize(), "postsolve_gpu synchronize");
  cudaFree(z_retrieved);
  return result;
}

}  // namespace gpu_presolver::presolve
