#pragma once

#include "gpu_presolver/offset.hpp"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <climits>
#include <cstdint>
#include <cstddef>
#include <cmath>

namespace gpu_presolver::presolve::detail {

constexpr int DOUBLETON_MERGE_THREADS = 256;
constexpr int DOUBLETON_LONG_ROW_THRESHOLD = 4096;

// Inputs have nondecreasing column indices. Partition only between complete
// column groups: every coefficient retains source-first, delta-second serial
// addition, including duplicate entries and the initial +0.0.
struct DoubletonMergeRow {
  const std::int32_t* cols;
  const double* vals;
  const long long* keys;
  const double* deltas;
  NnzOffset p_begin, p_end, q_begin, q_end;
  std::int32_t n;

  __device__ int dc(NnzOffset q) const { return static_cast<int>(keys[q] % n); }

  __device__ void boundary(int lane, NnzOffset& p, NnzOffset& q) const {
    if (lane == 0) { p = p_begin; q = q_begin; return; }
    if (lane == DOUBLETON_MERGE_THREADS) { p = p_end; q = q_end; return; }
    const NnzOffset pn = p_end - p_begin, qn = q_end - q_begin;
    const std::int64_t rank = (static_cast<std::int64_t>(pn) + qn) * lane /
                              DOUBLETON_MERGE_THREADS;
    NnzOffset low = static_cast<NnzOffset>(rank > qn ? rank - qn : 0);
    NnzOffset high = static_cast<NnzOffset>(rank < pn ? rank : pn);
    NnzOffset i = low, j = static_cast<NnzOffset>(rank - i);
    while (low <= high) {
      i = low + (high - low) / 2;
      j = static_cast<NnzOffset>(rank - i);
      if (i > 0 && j < qn && cols[p_begin + i - 1] > dc(q_begin + j)) {
        high = i - 1;
      } else if (j > 0 && i < pn && dc(q_begin + j - 1) >= cols[p_begin + i]) {
        low = i + 1;
      } else {
        break;
      }
    }
    const int ac = i < pn ? cols[p_begin + i] : INT_MAX;
    const int qc = j < qn ? dc(q_begin + j) : INT_MAX;
    const int key = ac < qc ? ac : qc;
    // Move both inputs to the first entry for this key. Repeated boundaries
    // produce empty segments, never split a floating-point accumulation.
    NnzOffset lo = p_begin, hi = p_end;
    while (lo < hi) {
      const NnzOffset mid = lo + (hi - lo) / 2;
      if (cols[mid] < key) lo = mid + 1; else hi = mid;
    }
    p = lo;
    lo = q_begin; hi = q_end;
    while (lo < hi) {
      const NnzOffset mid = lo + (hi - lo) / 2;
      if (dc(mid) < key) lo = mid + 1; else hi = mid;
    }
    q = lo;
  }

  template<bool Write>
  __device__ NnzOffset merge(NnzOffset p, NnzOffset p_stop,
                            NnzOffset q, NnzOffset q_stop, double zero_tol,
                            NnzOffset write = 0, int* out_cols = nullptr,
                            double* out_vals = nullptr) const {
    NnzOffset count = 0;
    while (p < p_stop || q < q_stop) {
      const int ac = p < p_stop ? cols[p] : INT_MAX;
      const int qc = q < q_stop ? dc(q) : INT_MAX;
      const int next = ac < qc ? ac : qc;
      double merged = 0.0;
      while (p < p_stop && cols[p] == next) merged += vals[p++];
      while (q < q_stop && dc(q) == next) merged += deltas[q++];
      if (fabs(merged) > zero_tol) {
        if constexpr (Write) {
          out_cols[write + count] = next;
          out_vals[write + count] = merged;
        }
        ++count;
      }
    }
    return count;
  }
};

template<bool Write>
__global__ void doubleton_merge_long_rows(
    const NnzOffset* row_ptr, const int* cols, const double* vals,
    const NnzOffset* delta_row_ptr, const long long* keys, const double* deltas,
    const int* rows, const int* row_count, int n, double zero_tol,
    NnzOffset* row_counts, NnzOffset* segment_counts,
    const NnzOffset* out_row_ptr = nullptr, int* out_cols = nullptr,
    double* out_vals = nullptr) {
  using Scan = cub::BlockScan<NnzOffset, DOUBLETON_MERGE_THREADS>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ NnzOffset ps[DOUBLETON_MERGE_THREADS + 1];
  __shared__ NnzOffset qs[DOUBLETON_MERGE_THREADS + 1];
  const int lane = threadIdx.x;
  // Device-resident queue length avoids another host synchronization per batch.
  for (int work = blockIdx.x; work < *row_count; work += gridDim.x) {
    const int row = rows[work];
    const DoubletonMergeRow input{cols, vals, keys, deltas,
        row_ptr[row], row_ptr[row + 1], delta_row_ptr[row], delta_row_ptr[row + 1], n};
    input.boundary(lane, ps[lane], qs[lane]);
    if (lane == 0) input.boundary(DOUBLETON_MERGE_THREADS,
                                ps[DOUBLETON_MERGE_THREADS], qs[DOUBLETON_MERGE_THREADS]);
    __syncthreads();
    const auto offset = static_cast<std::size_t>(work) * DOUBLETON_MERGE_THREADS + lane;
    NnzOffset count;
    if constexpr (Write) count = segment_counts[offset];
    else {
      count = input.merge<false>(ps[lane], ps[lane + 1], qs[lane], qs[lane + 1], zero_tol);
      segment_counts[offset] = count;
    }
    NnzOffset prefix = 0, total = 0;
    Scan(scan).ExclusiveSum(count, prefix, total);
    if constexpr (Write) {
      input.merge<true>(ps[lane], ps[lane + 1], qs[lane], qs[lane + 1], zero_tol,
                        out_row_ptr[row] + prefix, out_cols, out_vals);
    } else if (lane == 0) row_counts[row] = total;
    __syncthreads();
  }
}

}  // namespace gpu_presolver::presolve::detail
