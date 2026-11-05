// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/rules/rule_row_sparsification.hpp"
#include "gpu_preprocessor/lp/offset.hpp"
#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include "gpu_preprocessor/lp/presolve/structural_dual.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_postsolve.hpp"
#include "gpu_preprocessor/lp/detail/runtime_options.hpp"
#include "postsolve/parallel_fixed_postsolve.cuh"
#include "postsolve/parallel_bound_postsolve.cuh"
#include "gpu_preprocessor/lp/presolve/compact_bound_record.hpp"
#include "gpu_preprocessor/lp/presolve/packed_bound_record.hpp"
#include "postsolve/parallel_subcol_postsolve.cuh"
#include "postsolve/parallel_components_postsolve.cuh"

#include <cuda_runtime.h>
#include "postsolve/postsolve_common.cuh"
#include "postsolve/postsolve_parallel_columns.cuh"
#include "postsolve/postsolve_parallel_rows.cuh"
#include "postsolve/postsolve_doubleton_equations.cuh"
#include "postsolve/postsolve_singleton_columns.cuh"
#include "postsolve/postsolve_fixed_variables.cuh"
#include "postsolve/postsolve_projected_variables.cuh"
#include "postsolve/postsolve_singleton_rows.cuh"
#include "postsolve/postsolve_bound_strengthening.cuh"
#include "postsolve/postsolve_piecewise_linear_reformulation_absolute_value.hpp"
#include "postsolve/postsolve_piecewise_linear_reformulation_variable_pair.hpp"

#include <cmath>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_preprocessor::lp::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;

using namespace gpu_preprocessor::lp::presolve::postsolve;

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

__global__ void _kernel_restore_fixed_values(double* x_org,
                                             const std::int32_t* fixed_idx,
                                             const double* fixed_val,
                                             std::int32_t fixed_count) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < fixed_count) {
    x_org[fixed_idx[i]] = fixed_val[i];
  }
}

__device__ __forceinline__ void replay_postsolve_records(double* x_org,
                                              double* y_org,
                                              double* z_org,
                                              std::uint8_t* z_retrieved,
                                              double* bound_delta,
                                              std::uint8_t* bound_side,
                                              double* bound_row_delta, double* bound_row_error,
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
                                              double tol,
                                              const int* serial_only) {
  if (serial_only != nullptr && *serial_only == 0) return;
  for (NnzOffset k = record_end - 1; k >= record_begin; --k) {
    const std::int32_t reduction_type = reduction_types[k];
    const NnzOffset idx0 = idx_starts[k];
    const NnzOffset idx1 = idx_starts[k + 1] - 1;
    const NnzOffset val0 = val_starts[k];
    const NnzOffset val1 = val_starts[k + 1] - 1;
    if (replay_bound_strengthening_batch(
          y_org, z_org, bound_delta, bound_side,
          bound_row_delta, bound_row_error, idx_starts, val_starts,
          tape_idx, tape_val, record_begin, replay_dual,
          idx0, idx1, val0, val1,
          k, reduction_type)) continue;
    if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::ParallelColumn)) {
      replay_parallel_column(
          x_org, z_org, z_retrieved, tape_idx, tape_val,
          replay_primal, replay_dual, tol, idx0, val0);
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::DoubletonEquation)) {
      replay_doubleton_equation(
          x_org, y_org, z_org, z_retrieved, tape_idx,
          tape_val, AT_rowPtr, AT_colVal, AT_nzVal, c,
          has_original_model, replay_primal, replay_dual, tol, idx0,
          idx1, val0, val1);
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::SingletonColumnSubstitution)) {
      replay_singleton_column_substitution(
          x_org, y_org, z_org, z_retrieved, tape_idx,
          tape_val, replay_primal, replay_dual, idx0, val0);
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::FixedVariable)) {
      replay_fixed_variable(
          x_org, y_org, z_org, z_retrieved, tape_idx,
          tape_val, replay_primal, replay_dual, idx0, idx1,
          val0);
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::InfiniteBoundDualFix)) {
      replay_infinite_bound_dual_fix(
          x_org, z_org, z_retrieved, tape_idx, tape_val,
          replay_primal, replay_dual, idx0, idx1, val0, val1);
    } else if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::FmeCol)) {
      replay_projected_column(
          x_org, y_org, z_org, z_retrieved, tape_idx,
          tape_val, replay_primal, replay_dual, idx0, val0, idx1 - idx0 + 1, val1 - val0 + 1);
    } else if (replay_dual != 0 &&
               reduction_type == static_cast<std::int32_t>(PostsolveReductionType::RowDualShift)) {
      replay_row_dual_shift(
          y_org, z_org, tape_idx, tape_val, idx0,
          idx1, val0);
    } else if (replay_dual != 0 &&
               reduction_type == static_cast<std::int32_t>(PostsolveReductionType::ParallelRow) &&
               idx1 >= idx0 + 1 && val1 >= val0 + 4) {
      replay_parallel_row(
          y_org, tape_idx, tape_val, idx0, val0);
    } else if (replay_dual != 0 &&
               reduction_type == static_cast<std::int32_t>(PostsolveReductionType::DeletedRow) &&
               idx1 >= idx0 + 1 && val1 >= val0 + 2) {
      replay_deleted_singleton_row(
          x_org, y_org, z_org, z_retrieved, tape_idx,
          tape_val, AT_rowPtr, AT_colVal, AT_nzVal, c,
          l, u, has_original_model, tol, idx0,
          val0, val1);
    } else if (replay_dual != 0 &&
               reduction_type == static_cast<std::int32_t>(PostsolveReductionType::RowImpliedBoundChange) &&
               has_original_model != 0 && idx1 >= idx0 + 1) {
      replay_row_implied_bound_change(
          x_org, y_org, z_org, z_retrieved, tape_idx,
          tape_val, A_rowPtr, A_colVal, A_nzVal, tol,
          idx0, val0);
    }
  }
}

__global__ void _kernel_replay_postsolve_tape(double* x_org,
                                              double* y_org,
                                              double* z_org,
                                              std::uint8_t* z_retrieved,
                                              double* bound_delta,
                                              std::uint8_t* bound_side,
                                              double* bound_row_delta, double* bound_row_error,
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
                                              double tol,
                                              const int* serial_only) {
  if (blockIdx.x || threadIdx.x) return;
  replay_postsolve_records(
      x_org, y_org, z_org, z_retrieved,
      bound_delta, bound_side, bound_row_delta, bound_row_error,
      reduction_types, idx_starts, val_starts, tape_idx, tape_val,
      A_rowPtr, A_colVal, A_nzVal,
      AT_rowPtr, AT_colVal, AT_nzVal,
      c, AL, AU, l, u, record_begin, record_end,
      has_original_model, replay_primal, replay_dual, tol, serial_only);
}

__global__ void _kernel_replay_postsolve_components(double* x_org,
                                              double* y_org,
                                              double* z_org,
                                              std::uint8_t* z_retrieved,
                                              double* bound_delta,
                                              std::uint8_t* bound_side,
                                              double* bound_row_delta, double* bound_row_error,
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
                                              double tol,
                                              const int* serial_only,
                                              const parallel_components_detail::Key* ordered) {
  if (*serial_only) return;
  using Key = parallel_components_detail::Key;
  const auto count = std::uint64_t(record_end - record_begin);
  const auto stride = std::uint64_t(blockDim.x) * gridDim.x;
  for (auto p = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; p < count; p += stride) {
    const Key component = ordered[p] >> 32;
    if (p + 1 < count && ordered[p + 1] >> 32 == component) continue;
    // Each component is one serial reverse subsequence of the original tape.
    // Distinct components share only read-only resources.
    for (std::int64_t q = std::int64_t(p); q >= 0 && ordered[q] >> 32 == component; --q) {
      const NnzOffset record = record_begin + NnzOffset(std::uint32_t(ordered[q]));
      replay_postsolve_records(
          x_org, y_org, z_org, z_retrieved,
          bound_delta, bound_side, bound_row_delta, bound_row_error,
          reduction_types, idx_starts, val_starts, tape_idx, tape_val,
          A_rowPtr, A_colVal, A_nzVal,
          AT_rowPtr, AT_colVal, AT_nzVal,
          c, AL, AU, l, u, record, record + 1,
          has_original_model, replay_primal, replay_dual, tol, nullptr);
    }
  }
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
  RecoveryArray<std::int32_t> red2org_d;
  red2org_d.data = copy_vector_to_device(red2org, context);
  const std::int32_t n_red = static_cast<std::int32_t>(red2org.size());
  const int blocks = (n_red + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_scatter_reduced_to_original<<<blocks, GPU_PRESOLVE_THREADS>>>(dst, src, red2org_d.data, n_red);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_scatter_reduced_to_original");
}

void restore_fixed_values(double* x_org,
                          const std::vector<std::int32_t>& fixed_idx,
                          const std::vector<double>& fixed_val) {
  if (fixed_idx.empty()) {
    return;
  }
  RecoveryArray<std::int32_t> fixed_idx_d;
  RecoveryArray<double> fixed_val_d;
  fixed_idx_d.data = copy_vector_to_device(fixed_idx, "cudaMalloc/copy fixed_idx");
  fixed_val_d.data = copy_vector_to_device(fixed_val, "cudaMalloc/copy fixed_val");
  const std::int32_t fixed_count = static_cast<std::int32_t>(fixed_idx.size());
  const int blocks = (fixed_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_restore_fixed_values<<<blocks, GPU_PRESOLVE_THREADS>>>(x_org, fixed_idx_d.data, fixed_val_d.data, fixed_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_restore_fixed_values");
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
                                 double* bound_delta, std::uint8_t* bound_side,
                                 double* bound_row_delta, double* bound_row_error,
                                 int* parallel_fixed_guard,
                                 int* parallel_bound_guard, int* parallel_subcol_guard,
                                 int* parallel_components_guard,
                                 int original_rows, int original_cols,
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
  int* serial_only = nullptr;
  parallel_components_detail::Plan component_plan;
  bool components = false;
  if (replay_primal && replay_dual && parallel_subcol_detail::try_replay(parallel_subcol_guard,
      x_org, y_org, z_org, z_retrieved, tape.types, tape.index_starts, tape.value_starts,
      tape.indices, tape.vals, record_begin, record_end, original_rows, original_cols)) {
    serial_only = parallel_subcol_guard;
  } else if (replay_dual && parallel_bound_detail::try_replay(parallel_bound_guard, y_org, z_org,
      bound_row_delta, bound_row_error, tape.types, tape.index_starts, tape.value_starts,
      tape.indices, tape.vals, record_begin, record_end, original_rows, original_cols)) {
    serial_only = parallel_bound_guard;
  } else if (replay_primal && replay_dual &&
      parallel_components_detail::prepare(component_plan, parallel_components_guard,
          tape.types, tape.index_starts, tape.value_starts, tape.indices, tape.vals,
          record_begin, record_end, original_rows, original_cols, original_model)) {
    serial_only = parallel_components_guard;
    components = true;
  } else if (parallel_fixed_guard != nullptr &&
      record_end - record_begin >= parallel_fixed_detail::min_records) {
    serial_only = parallel_fixed_guard;
    throw_if_cuda_error(cudaMemsetAsync(serial_only, 0, sizeof(int)), "clear fixed-column replay guard");
    const int blocks = checked_launch_blocks(
        std::min(record_end - record_begin, NnzOffset(1024) * 256), 256);
    parallel_fixed_detail::certify<<<blocks, 256>>>(serial_only, tape.types, tape.index_starts,
                                                  tape.indices, record_begin, record_end);
    parallel_fixed_detail::replay<<<blocks, 256>>>(serial_only, x_org, y_org, z_org, z_retrieved,
        tape.index_starts, tape.value_starts, tape.indices, tape.vals,
        record_begin, record_end, replay_primal, replay_dual);
    throw_if_cuda_error(cudaGetLastError(), "parallel fixed-column replay");
  }
  if (components) {
    _kernel_replay_postsolve_components<<<
        parallel_components_detail::blocks(int(record_end - record_begin)), 256>>>(x_org,
                                          y_org,
                                          z_org,
                                          z_retrieved,
                                          bound_delta, bound_side, bound_row_delta, bound_row_error,
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
                                          tol,
                                          serial_only, component_plan.ordered.data);
    throw_if_cuda_error(cudaGetLastError(), "parallel component replay");
  }
  _kernel_replay_postsolve_tape<<<1, 1>>>(x_org,
                                          y_org,
                                          z_org,
                                          z_retrieved,
                                          bound_delta, bound_side, bound_row_delta, bound_row_error,
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
                                          tol,
                                          serial_only);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_replay_postsolve_tape");
}

void mark_retrieved_values(std::uint8_t* z_retrieved, const std::vector<std::int32_t>& red2org) {
  if (z_retrieved == nullptr || red2org.empty()) {
    return;
  }
  RecoveryArray<std::int32_t> red2org_d;
  red2org_d.data = copy_vector_to_device(red2org, "cudaMalloc/copy z_retrieved red2org");
  const std::int32_t n_red = static_cast<std::int32_t>(red2org.size());
  const int blocks = (n_red + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_mark_retrieved<<<blocks, GPU_PRESOLVE_THREADS>>>(z_retrieved, red2org_d.data, n_red);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_mark_retrieved");
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
            static_cast<std::int32_t>(PostsolveReductionType::FixedVariable) ||
        tape.types[k] >
            static_cast<std::int32_t>(PostsolveReductionType::BoundPackedBatch)) {
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
  for (std::size_t k = 0; k < count; ++k) {
    if (tape.types[k] != int(PostsolveReductionType::InfiniteBoundDualFix)) continue;
    const auto ip = tape.index_starts[k], vp = tape.value_starts[k];
    if (!detail::valid_infinite_bound_record(tape.indices.data() + ip,
            tape.vals.data() + vp, tape.index_starts[k + 1] - ip,
            tape.value_starts[k + 1] - vp))
      throw std::runtime_error("malformed infinite-bound recovery record");
  }
  for (std::size_t k = 0; k < count; ++k) {
    if (tape.types[k] != int(PostsolveReductionType::FmeCol)) continue;
    const auto ip = tape.index_starts[k], vp = tape.value_starts[k];
    if (!detail::valid_projected_column_record(tape.indices.data() + ip,
            tape.vals.data() + vp, tape.index_starts[k + 1] - ip,
            tape.value_starts[k + 1] - vp))
      throw std::runtime_error("malformed projected-column recovery record");
  }
  for(std::size_t k=0;k<count;++k)if(tape.types[k]==int(PostsolveReductionType::BoundPackedBatch)) {
    const auto ip=tape.index_starts[k],vp=tape.value_starts[k];
    if(!valid_packed_bound_header(tape.indices.data()+ip,tape.index_starts[k+1]-ip,tape.value_starts[k+1]-vp))
      throw std::runtime_error("invalid packed bound header");
    const PackedBoundView b{tape.indices.data()+ip};
    for(std::int64_t r=0;r<=b.rows();++r) {
      const auto p=b.start(r);
      if(p<0 || p>b.terms() || (!r && p!=0) || (r==b.rows() && p!=b.terms()) || (r && p<b.start(r-1)))
        throw std::runtime_error("invalid packed bound source offsets");
    }
    for(int t=0;t<b.transfers();++t) {
      const auto lo=b.reference(t,0),up=b.reference(t,1);
      if(lo < -1 || lo>=b.terms() || up < -1 || up>=b.terms() || (lo>=0 && up>=0 && b.column(lo)!=b.column(up)))
        throw std::runtime_error("invalid packed bound coefficient reference");
    }
  }
  for(std::size_t stop=count;stop;) {
    const auto k=stop-1;
    if(tape.types[k]==int(PostsolveReductionType::BoundChangeCompact))
      throw std::runtime_error("compact bound batch has no footer");
    if(tape.types[k]!=int(PostsolveReductionType::BoundBatch)){--stop;continue;}
    const auto ip=tape.index_starts[k];
    if(tape.index_starts[k+1]-ip!=2 || tape.value_starts[k+1]!=tape.value_starts[k])
      throw std::runtime_error("malformed compact bound footer");
    const int rows=tape.indices[ip],columns=tape.indices[ip+1];
    if(rows<0 || columns<1 || std::uint64_t(rows)+columns>k)
      throw std::runtime_error("invalid compact bound batch extent");
    const auto first=k-columns,begin=first-rows;const auto base=tape.value_starts[begin];
    for(auto t=begin;t<first;++t)
      if(tape.types[t]!=int(PostsolveReductionType::BoundSourceRow) ||
         tape.index_starts[t+1]-tape.index_starts[t]!=tape.value_starts[t+1]-tape.value_starts[t]+1)
        throw std::runtime_error("invalid compact bound source");
    for(auto t=first;t<k;++t) {
      const int* ids=tape.indices.data()+tape.index_starts[t];
      if(tape.types[t]!=int(PostsolveReductionType::BoundChangeCompact) ||
         tape.index_starts[t+1]-tape.index_starts[t]!=compact_bound_indices ||
         tape.value_starts[t+1]!=tape.value_starts[t])
        throw std::runtime_error("invalid compact bound column");
      for(int side=0;side<2;++side) {
        const int slot=ids[1+side];const auto ref=compact_bound_reference(ids,side);
        if(slot<0) {if(slot!=-1 || ref!=-1)throw std::runtime_error("invalid missing compact source");continue;}
        if(slot>=rows || ref < -1 ||
           (ref>=0 && (ref<tape.value_starts[begin+slot]-base || ref>=tape.value_starts[begin+slot+1]-base)))
          throw std::runtime_error("compact bound coefficient reference out of range");
      }
    }
    stop=begin;
  }
}

void validate_primal_recovery_timeline(const PresolveRecordGpu& record) {
  if (!record.device)
    validate_postsolve_tape(record.tape);
  if (record.piecewise_linear_reformulation_absolute_value_recoveries.size() >
          static_cast<std::size_t>(
              std::numeric_limits<std::int32_t>::max()) ||
      record.piecewise_linear_reformulation_variable_pair_recoveries.size() >
          static_cast<std::size_t>(
              std::numeric_limits<std::int32_t>::max())) {
    throw std::runtime_error(
        "primal recovery payload exceeds 32-bit checkpoint range");
  }
  std::vector<std::uint8_t> seen_absolute_value(
      record.piecewise_linear_reformulation_absolute_value_recoveries.size(), std::uint8_t{0});
  std::vector<std::uint8_t> seen_variable_pair(
      record.piecewise_linear_reformulation_variable_pair_recoveries.size(), std::uint8_t{0});
  std::vector<std::uint8_t> seen_dual(record.structural_dual_recoveries.size(), 0);
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
    if (checkpoint.dual_index >= 0 &&
        (checkpoint.dual_index >= static_cast<NnzOffset>(record.structural_dual_recoveries.size()) ||
         !record.structural_dual_recoveries[checkpoint.dual_index]))
      throw std::runtime_error("missing structural dual checkpoint");
    if (checkpoint.dual_index >= 0) {
      if (seen_dual[checkpoint.dual_index]++)
        throw std::runtime_error("duplicate structural dual checkpoint");
      const auto& stage = *record.structural_dual_recoveries[checkpoint.dual_index];
      const int m=stage.model.A.rows,n=stage.model.A.cols;
      if (m<=0 || n<=0 || m>record.m0 || n>record.n0 ||
          stage.rows.count!=m || stage.cols.count!=n ||
          stage.keep_rows.count!=m || stage.keep_cols.count!=n ||
          !stage.rows.data || !stage.cols.data || !stage.keep_rows.data || !stage.keep_cols.data ||
          !stage.model.A.rowPtr || !stage.model.AT.rowPtr ||
          stage.model.AT.rows!=n || stage.model.AT.cols!=m ||
          !stage.model.c || !stage.model.l || !stage.model.u || !stage.model.AL || !stage.model.AU)
        throw std::runtime_error("invalid structural stage dimensions or storage");
      if (checkpoint.kind == PrimalRecoveryKind::PiecewiseLinearReformulationVariablePair) {
        if (stage.pairs.ids.size()!=4 || stage.pairs.values.size()!=2 || stage.row_scale.count!=m || !stage.row_scale.data)
          throw std::runtime_error("invalid pair recovery stage");
        const int count=stage.pairs.ids[0].count;
        for (const auto& v:stage.pairs.ids) if(v.count!=count || !v.data)
          throw std::runtime_error("invalid pair recovery indices");
        for (const auto& v:stage.pairs.values) if(v.count!=count || !v.data)
          throw std::runtime_error("invalid pair recovery shifts");
        for (const auto* v:{&stage.tree_parent,&stage.tree_row,&stage.tree_next,&stage.tree_head})
          if(v->count!=count || !v->data) throw std::runtime_error("invalid pair recovery tree");
        if(count<=0 || stage.tree_scale.count!=count || !stage.tree_scale.data)
          throw std::runtime_error("invalid pair recovery edge scales");
      }
    }
    if (checkpoint.kind == PrimalRecoveryKind::ColumnDomination) {
      if (checkpoint.dual_index < 0) throw std::runtime_error("missing domination stage data");
      continue;
    }
    if (checkpoint.kind == PrimalRecoveryKind::PiecewiseLinearReformulationAbsoluteValue) {
      if (checkpoint.payload_index < 0 ||
          checkpoint.payload_index >=
              static_cast<std::int32_t>(seen_absolute_value.size()) ||
          seen_absolute_value[static_cast<std::size_t>(
              checkpoint.payload_index)] != std::uint8_t{0}) {
        throw std::runtime_error(
            "invalid or duplicate piecewise_linear_reformulation_absolute_value recovery checkpoint");
      }
      seen_absolute_value[static_cast<std::size_t>(checkpoint.payload_index)] =
          std::uint8_t{1};
      validate_piecewise_linear_reformulation_absolute_value_recovery(
          record.piecewise_linear_reformulation_absolute_value_recoveries[static_cast<std::size_t>(
              checkpoint.payload_index)],
          record.n0);
    } else if (checkpoint.kind ==
               PrimalRecoveryKind::PiecewiseLinearReformulationVariablePair) {
      if (checkpoint.payload_index < 0 ||
          checkpoint.payload_index >=
              static_cast<std::int32_t>(seen_variable_pair.size()) ||
          seen_variable_pair[static_cast<std::size_t>(
              checkpoint.payload_index)] != std::uint8_t{0}) {
        throw std::runtime_error(
            "invalid or duplicate piecewise_linear_reformulation_variable_pair recovery checkpoint");
      }
      seen_variable_pair[static_cast<std::size_t>(checkpoint.payload_index)] =
          std::uint8_t{1};
      validate_piecewise_linear_reformulation_variable_pair_recovery(
          record.piecewise_linear_reformulation_variable_pair_recoveries[static_cast<std::size_t>(
              checkpoint.payload_index)],
          record.n0);
    } else {
      throw std::runtime_error("unknown primal recovery checkpoint kind");
    }
  }
  for (auto seen : seen_dual)
    if (!seen) throw std::runtime_error("structural dual payload is missing a checkpoint");
  for (const std::uint8_t seen : seen_absolute_value) {
    if (seen == std::uint8_t{0}) {
      throw std::runtime_error(
          "piecewise_linear_reformulation_absolute_value recovery payload is missing a checkpoint");
    }
  }
  for (const std::uint8_t seen : seen_variable_pair) {
    if (seen == std::uint8_t{0}) {
      throw std::runtime_error(
          "piecewise_linear_reformulation_variable_pair recovery payload is missing a checkpoint");
    }
  }
}

}  // namespace

void postsolve_restore_piecewise_linear_reformulation_absolute_value_gpu(
    double* x_org,
    const std::vector<PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep>& recoveries,
    const double* original_l) {
  for (auto it = recoveries.rbegin(); it != recoveries.rend(); ++it) {
    restore_piecewise_linear_reformulation_absolute_value_step(x_org, *it, original_l);
  }
  throw_if_cuda_error(cudaDeviceSynchronize(), "postsolve_restore_piecewise_linear_reformulation_absolute_value_gpu synchronize");
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
  // Ordinary reduction records were made in the post-startup row basis.
  // The caller's original model precedes S and must not be used for this replay.
  if (record.startup_rows) {
    original_model = &record.startup_rows->transformed;
    original_l = original_model->l;
  }
  const LPInfoGpu* poisson_input = original_model;
  if (record.startup_poisson) {
    if (!poisson_input) throw std::invalid_argument("Poisson recovery needs its input model");
    original_model = &record.startup_poisson->transformed;
    original_l = original_model->l;
  }
  validate_device_recovery(record);
  validate_primal_recovery_timeline(record);
  GpuPostsolveResult result;
  struct ResultGuard {
    GpuPostsolveResult& result;
    bool active = true;
    ~ResultGuard() {
      if (active) {
        cudaFree(result.x_org); cudaFree(result.y_org); cudaFree(result.z_org);
      }
    }
  } result_guard{result};
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

  RecoveryArray<std::uint8_t> retrieved_storage;
  auto& z_retrieved = retrieved_storage.data;
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
  RecoveryArray<double> bound_delta, bound_row_delta, bound_row_error;
  RecoveryArray<std::uint8_t> bound_side;
  RecoveryArray<int> parallel_guard;
  const bool parallel_fixed = runtime::env_enabled("GPUPREPROCESSOR_PARALLEL_FIXED_COLS", true);
  const bool parallel_bound = runtime::env_enabled("GPUPREPROCESSOR_PARALLEL_BOUND", true);
  const bool parallel_subcol = runtime::env_enabled("GPUPREPROCESSOR_PARALLEL_SUBCOL", true);
  const bool parallel_components = runtime::env_enabled("GPUPREPROCESSOR_PARALLEL_COMPONENTS", true);
  if (device_tape.count>0 && (parallel_bound ||
      (device_tape.count>=parallel_fixed_detail::min_records && (parallel_fixed || parallel_subcol || parallel_components)))) {
    throw_if_cuda_error(cudaMalloc(&parallel_guard.data, sizeof(int)),
                        "allocate parallel replay guard");
  }
  if(record.n0>0) {
    throw_if_cuda_error(cudaMalloc(&bound_delta.data,sizeof(double)*std::size_t(record.n0)),
                        "allocate joint bound lift deltas");
    throw_if_cuda_error(cudaMalloc(&bound_side.data,std::size_t(record.n0)),
                        "allocate joint bound lift sides");
  }
  if(record.m0>0) {
    throw_if_cuda_error(cudaMalloc(&bound_row_delta.data,sizeof(double)*std::size_t(record.m0)),
                        "allocate shared bound row sums");
    throw_if_cuda_error(cudaMalloc(&bound_row_error.data,sizeof(double)*std::size_t(record.m0)),
                        "allocate shared bound row sum errors");
  }
  auto replay_range = [&](NnzOffset begin, NnzOffset end, bool primal, bool dual) {
    if (!record.device) {
      replay_postsolve_tape_range(result.x_org, result.y_org, result.z_org, z_retrieved,
                                  bound_delta.data, bound_side.data, bound_row_delta.data, bound_row_error.data,
                                  parallel_fixed ? parallel_guard.data : nullptr,
                                  parallel_bound ? parallel_guard.data : nullptr,
                                  parallel_subcol ? parallel_guard.data : nullptr,
                                  parallel_components ? parallel_guard.data : nullptr, record.m0, record.n0,
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
        replay_postsolve_tape_range(result.x_org, result.y_org, result.z_org, z_retrieved,
                                    bound_delta.data, bound_side.data, bound_row_delta.data, bound_row_error.data,
                                    parallel_fixed ? parallel_guard.data : nullptr,
                                    parallel_bound ? parallel_guard.data : nullptr,
                                    parallel_subcol ? parallel_guard.data : nullptr,
                                  parallel_components ? parallel_guard.data : nullptr, record.m0, record.n0, view,
                                    original_model, lo - start, hi - start, primal, dual, 1e-7);
      }
      stop = start;
    }
  };
  NnzOffset range_end = device_tape.count;
  for (auto checkpoint_it = record.primal_recovery_timeline.rbegin();
       checkpoint_it != record.primal_recovery_timeline.rend();
       ++checkpoint_it) {
    replay_range(checkpoint_it->tape_position, range_end, true, true);
    if (checkpoint_it->dual_index >= 0) {
      restore_structural_joint(result.x_org, result.y_org, result.z_org, z_retrieved,
          checkpoint_it->kind, *record.structural_dual_recoveries[checkpoint_it->dual_index]);
      range_end = checkpoint_it->tape_position;
      continue;
    }
    if (checkpoint_it->kind == PrimalRecoveryKind::PiecewiseLinearReformulationAbsoluteValue) {
      if (record.device)
        restore_device_structural(
            result.x_org, record.device->piecewise_linear_reformulation_absolute_value[checkpoint_it->payload_index], original_l);
      else
        restore_piecewise_linear_reformulation_absolute_value_step(result.x_org,
                                       record.piecewise_linear_reformulation_absolute_value_recoveries[static_cast<std::size_t>(
                                           checkpoint_it->payload_index)],
                                       original_l);
    } else {
      if (record.device)
        restore_device_piecewise_linear_reformulation_variable_pair(result.x_org,
                                 record.device->piecewise_linear_reformulation_variable_pair[checkpoint_it->payload_index]);
      else
        restore_piecewise_linear_reformulation_variable_pair(result.x_org,
                                    record.piecewise_linear_reformulation_variable_pair_recoveries[static_cast<std::size_t>(
                                        checkpoint_it->payload_index)]);
    }
    range_end = checkpoint_it->tape_position;
  }
  replay_range(0, range_end, true, true);
  if (record.startup_poisson) {
    auto restored = record.startup_poisson->restore(result, *poisson_input);
    cudaFree(result.x_org); cudaFree(result.y_org); cudaFree(result.z_org);
    result = restored;
  }
  if (record.startup_rows) restore_startup_row_duals(*record.startup_rows, result.y_org);
  throw_if_cuda_error(cudaDeviceSynchronize(), "postsolve_gpu synchronize");
  result_guard.active = false;
  return result;
}

}  // namespace gpu_preprocessor::lp::presolve

#include "postsolve/structural_dual.inc.cu"
#include "postsolve/pair_primal_polish.inc.cu"
