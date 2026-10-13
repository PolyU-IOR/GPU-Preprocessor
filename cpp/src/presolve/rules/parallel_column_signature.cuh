#pragma once

// Compute two magnitude projections of pivot-normalized coefficients.
// This stage selects candidates; the existing hash and proportionality checks
// still decide reductions. Row/column traversal matches their legacy hashes.
namespace column_signature {

using Column = parallel_signature::Projection;
using parallel_signature::Workspace;
using parallel_signature::finish;
using parallel_signature::weight;
using parallel_signature::build_buckets;
using parallel_signature::mark_candidates;
using parallel_signature::IsCandidate;
using parallel_signature::count_nnz;

inline bool enabled(const DeviceCsrMatrix& a) {
  return parallel_signature::enabled(a, runtime::options().column_signatures);
}

inline std::size_t scratch_bytes(std::size_t dimensions) {
  return parallel_signature::scratch_bytes(dimensions, runtime::options().column_signatures);
}

__global__ void short_spmv(Column* output, const std::uint8_t* keep_col,
    const NnzOffset* ptr, const int* rows, const double* values,
    const std::uint8_t* keep_row, int n, double zero_tol) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= n || ptr[col+1] - ptr[col] > ADAPTIVE_SHORT_ROW) return;
  output[col] = {0, LLONG_MIN, LLONG_MIN, 0};
  if (!keep_col[col]) return;
  NnzOffset first = ptr[col+1], length = 0;
  for (auto p = ptr[col]; p < ptr[col+1]; ++p) {
    if (!keep_row[rows[p]] || !(fabs(values[p]) > zero_tol)) continue;
    if (!length) first = p;
    ++length;
  }
  if (!length) return;
  const double pivot = values[first];
  double x = 0, y = 0, abs_sum = 0;
  for (auto p = first; p < ptr[col+1]; ++p) {
    if (!keep_row[rows[p]] || !(fabs(values[p]) > zero_tol)) continue;
    const double a = fabs(values[p] / pivot);
    x += a * weight(rows[p], 0x243f6a8885a308d3ULL);
    y += a * weight(rows[p], 0x13198a2e03707344ULL);
    abs_sum += a;
  }
  finish(output, col, length, rows[first], x, y, abs_sum, length, 1);
}

template<int Width>
__global__ void grouped_spmv(Column* output, const std::uint8_t* keep_col,
    const NnzOffset* ptr, const int* columns, const double* values,
    const std::uint8_t* keep_row, double zero_tol, const int* rows, const int* count) {
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
    if (!keep_col[row]) {
      if (lane == 0) output[row] = {0, LLONG_MIN, LLONG_MIN, 0};
      continue;
    }
    const NnzOffset start = ptr[row], end = ptr[row+1];
    NnzOffset first = end, length = 0;
    for (auto p = start + lane; p < end; p += Width) {
      if (!keep_row[columns[p]] || !(fabs(values[p]) > zero_tol)) continue;
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
    if (!length) {
      if (!lane) output[row] = {0, LLONG_MIN, LLONG_MIN, length};
      continue;
    }
    const double pivot = values[first];
    double x = 0, y = 0, abs_sum = 0;
    for (auto p = first + lane; p < end; p += Width) {
      if (!keep_row[columns[p]] || !(fabs(values[p]) > zero_tol)) continue;
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
                   const std::uint8_t* keep_col, const std::uint8_t* keep_row, double zero_tol) {
  throw_if_cuda_error(cudaMemset(w.keys, 0, w.size * sizeof(unsigned long long)), "signature keys");
  throw_if_cuda_error(cudaMemset(w.owners, 0xff, w.size * sizeof(int)), "signature owners");
  throw_if_cuda_error(cudaMemset(w.uncertain_support, 0, w.size * sizeof(int)), "signature fallback");
  throw_if_cuda_error(cudaMemset(w.overflow, 0, sizeof(int)), "signature overflow");
  const int blocks = (a.rows + 255) / 256;
  const auto& groups = prepare_adaptive_row_groups(a);
  short_spmv<<<blocks, 256>>>(w.projections, keep_col, a.rowPtr, a.colVal, a.nzVal, keep_row, a.rows, zero_tol);
  grouped_spmv<32><<<std::min(1024, (a.rows+7)/8), 256>>>(w.projections, keep_col,
      a.rowPtr, a.colVal, a.nzVal, keep_row, zero_tol, groups.medium, groups.counts);
  grouped_spmv<256><<<std::min(256, a.rows), 256>>>(w.projections, keep_col,
      a.rowPtr, a.colVal, a.nzVal, keep_row, zero_tol, groups.long_rows, groups.counts+1);
  build_buckets<<<blocks, 256>>>(w, keep_col, a.rows);
  mark_candidates<true><<<blocks, 256>>>(w, keep_col, a.rows);
  throw_if_cuda_error(cudaGetLastError(), "parallel column signature filter");
}

// Audit against original complete hash groups, including accidental hash
// collisions. Zero omissions certifies unchanged live membership/order of
// every original non-singleton group for this invocation.
__global__ void audit_groups(const unsigned long long* hashes, const int* cols,
    const std::uint8_t* keep, const std::uint8_t* candidates, int n, int* missing) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n || (s && hashes[s-1] == hashes[s])) return;
  int end = s, live = 0, omitted = 0;
  while (end < n && hashes[end] == hashes[s]) {
    const int col = cols[end++];
    live += bool(keep[col]);
    omitted += !candidates[col];
  }
  if (live > 1 && omitted) {
    const int ticket = atomicAdd(missing, omitted);
    if (ticket < 4)
      printf("column signature audit group=%llx size=%d live=%d omitted=%d\n",
             hashes[s], end-s, live, omitted);
  }
}

}  // namespace column_signature
