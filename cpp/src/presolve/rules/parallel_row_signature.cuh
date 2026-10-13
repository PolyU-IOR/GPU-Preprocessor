#pragma once

// Compute two magnitude projections of pivot-normalized coefficients.
// This stage selects candidates; the existing hash and proportionality checks
// still decide reductions. Row/column traversal matches their legacy hashes.
namespace row_signature {

using Row = parallel_signature::Projection;
using parallel_signature::Workspace;
using parallel_signature::finish;
using parallel_signature::weight;
using parallel_signature::build_buckets;
using parallel_signature::mark_candidates;
using parallel_signature::IsCandidate;
using parallel_signature::count_nnz;

inline bool enabled(const DeviceCsrMatrix& a) {
  return parallel_signature::enabled(a, runtime::options().row_signatures);
}

inline std::size_t scratch_bytes(std::size_t dimensions) {
  return parallel_signature::scratch_bytes(dimensions, runtime::options().row_signatures);
}

__global__ void short_spmv(Row* output, const std::uint8_t* keep_row,
    const NnzOffset* ptr, const int* columns, const double* values,
    const std::uint8_t* keep_col, int m, double zero_tol) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || ptr[row+1] - ptr[row] > ADAPTIVE_SHORT_ROW) return;
  output[row] = {0, LLONG_MIN, LLONG_MIN, 0};
  if (!keep_row[row]) return;
  LiveRowCursor cursor(ptr[row], ptr[row+1], columns, keep_col);
  if (cursor.pos == cursor.end) return;
  if (fabs(values[cursor.pos]) <= zero_tol) {
    NnzOffset length = 0;
    for (; cursor.pos < cursor.end; cursor.advance()) ++length;
    output[row].length = length; return;
  }
  const double pivot = values[cursor.pos];
  const int first_col = columns[cursor.pos];
  double x = 0, y = 0, abs_sum = 0;
  NnzOffset length = 0;
  for (; cursor.pos < cursor.end; cursor.advance()) {
    const auto p = cursor.pos;
    const double a = fabs(values[p] / pivot);
    x += a * weight(columns[p], 0x243f6a8885a308d3ULL);
    y += a * weight(columns[p], 0x13198a2e03707344ULL);
    abs_sum += fabs(a); ++length;
  }
  finish(output, row, length, first_col, x, y, abs_sum, length, 1);
}

template<int Width>
__global__ void grouped_spmv(Row* output, const std::uint8_t* keep_row,
    const NnzOffset* ptr, const int* columns, const double* values,
    const std::uint8_t* keep_col, double zero_tol, const int* rows, const int* count) {
  const int lane = Width == 32 ? threadIdx.x & 31 : threadIdx.x;
  const int group = Width == 32 ? blockIdx.x * 8 + threadIdx.x / 32 : blockIdx.x;
  const int stride = Width == 32 ? gridDim.x * 8 : gridDim.x;
  using IndexReduce = cub::BlockReduce<NnzOffset, 256>;
  using SumReduce = cub::BlockReduce<double, 256>;
  __shared__ typename IndexReduce::TempStorage index_storage;
  __shared__ typename SumReduce::TempStorage sum_storage;
  __shared__ NnzOffset shared_first, shared_length;
  for (NnzOffset q = group; q < *count; q += stride) {
    const int row = rows[q];
    if (!keep_row[row]) {
      if (lane == 0) output[row] = {0, LLONG_MIN, LLONG_MIN, 0};
      continue;
    }
    const NnzOffset start = ptr[row], end = ptr[row+1];
    NnzOffset first = end, length = 0;
    for (auto p = start + lane; p < end; p += Width) {
      if (keep_col && !keep_col[columns[p]]) continue;
      first = first < p ? first : p; ++length;
    }
    if constexpr (Width == 32) {
      for (int off = 16; off; off /= 2) {
        length += __shfl_down_sync(0xffffffffu, length, off);
        const auto other = __shfl_down_sync(0xffffffffu, first, off);
        first = first < other ? first : other;
      }
      first = __shfl_sync(0xffffffffu, first, 0);
      length = __shfl_sync(0xffffffffu, length, 0);
    } else {
      length = IndexReduce(index_storage).Sum(length); __syncthreads();
      if (!lane) shared_length = length;
      first = IndexReduce(index_storage).Reduce(first, thrust::minimum<NnzOffset>{}); __syncthreads();
      if (!lane) shared_first = first;
      __syncthreads(); first = shared_first; length = shared_length;
    }
    if (!length || fabs(values[first]) <= zero_tol) {
      if (!lane) output[row] = {0, LLONG_MIN, LLONG_MIN, length};
      continue;
    }
    const double pivot = values[first];
    double x = 0, y = 0, abs_sum = 0;
    for (auto p = first + lane; p < end; p += Width) {
      if (keep_col && !keep_col[columns[p]]) continue;
      const double a = fabs(values[p] / pivot);
      x += a * weight(columns[p], 0x243f6a8885a308d3ULL);
      y += a * weight(columns[p], 0x13198a2e03707344ULL);
      abs_sum += fabs(a);
    }
    if constexpr (Width == 32) {
      for (int off = 16; off; off /= 2) {
        x += __shfl_down_sync(0xffffffffu, x, off);
        y += __shfl_down_sync(0xffffffffu, y, off);
        abs_sum += __shfl_down_sync(0xffffffffu, abs_sum, off);
      }
    } else {
      x = SumReduce(sum_storage).Sum(x); __syncthreads();
      y = SumReduce(sum_storage).Sum(y); __syncthreads();
      abs_sum = SumReduce(sum_storage).Sum(abs_sum); __syncthreads();
    }
    if (!lane) finish(output, row, length, columns[first], x, y, abs_sum,
                       (end - first + Width - 1) / Width, Width);
    if constexpr (Width != 32) __syncthreads();
  }
}

inline void filter(Workspace w, const DeviceCsrMatrix& a,
                   const std::uint8_t* keep_row, const std::uint8_t* keep_col, double zero_tol) {
  throw_if_cuda_error(cudaMemset(w.keys, 0, w.size * sizeof(unsigned long long)), "signature keys");
  throw_if_cuda_error(cudaMemset(w.owners, 0xff, w.size * sizeof(int)), "signature owners");
  throw_if_cuda_error(cudaMemset(w.uncertain_support, 0, w.size * sizeof(int)), "signature fallback");
  throw_if_cuda_error(cudaMemset(w.overflow, 0, sizeof(int)), "signature overflow");
  const int blocks = (a.rows + 255) / 256;
  const auto& groups = prepare_adaptive_row_groups(a);
  short_spmv<<<blocks, 256>>>(w.projections, keep_row, a.rowPtr, a.colVal, a.nzVal, keep_col, a.rows, zero_tol);
  grouped_spmv<32><<<std::min(1024, (a.rows+7)/8), 256>>>(w.projections, keep_row,
      a.rowPtr, a.colVal, a.nzVal, keep_col, zero_tol, groups.medium, groups.counts);
  grouped_spmv<256><<<std::min(256, a.rows), 256>>>(w.projections, keep_row,
      a.rowPtr, a.colVal, a.nzVal, keep_col, zero_tol, groups.long_rows, groups.counts+1);
  build_buckets<<<blocks, 256>>>(w, keep_row, a.rows);
  mark_candidates<false><<<blocks, 256>>>(w, keep_row, a.rows);
  throw_if_cuda_error(cudaGetLastError(), "parallel row signature filter");
}

// Audit against original complete hash groups, including accidental hash
// collisions. Zero omissions certifies unchanged live membership/order of
// every original non-singleton group for this invocation.
__global__ void audit_groups(const unsigned long long* hashes, const int* rows,
    const std::uint8_t* keep, const std::uint8_t* candidates, int m, int* missing,
    const Row* projected, DeviceCsrMatrix a, const std::uint8_t* keep_col) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= m || (s && hashes[s-1] == hashes[s])) return;
  int end = s, live = 0, omitted = 0;
  while (end < m && hashes[end] == hashes[s]) {
    const int row = rows[end++];
    if (keep[row]) { ++live; omitted += !candidates[row]; }
  }
  if (live > 1 && omitted) {
    const int ticket = atomicAdd(missing, omitted);
    if (ticket < 4) for (int p = s; p < end && p < s+4; ++p) {
      const int row = rows[p];
      printf("signature audit group=%llx row=%d candidate=%d support=%llx bins=%lld,%lld length=%d\n",
             hashes[s], row, candidates[row], projected[row].support,
             projected[row].x, projected[row].y, int(projected[row].length));
      LiveRowCursor cursor(a.rowPtr[row], a.rowPtr[row+1], a.colVal, keep_col);
      if (cursor.pos != cursor.end) {
        const double pivot = a.nzVal[cursor.pos];
        for (int k = 0; k < 6 && cursor.pos < cursor.end; ++k, cursor.advance())
          printf("signature audit coefficient row=%d col=%d norm=%.17g\n", row,
                 a.colVal[cursor.pos], a.nzVal[cursor.pos] / pivot);
      }
    }
  }
}

}  // namespace row_signature
