#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/cleanup_workspace.hpp"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
namespace gpu_presolver::presolve {
namespace {
void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("device recovery: ") + cudaGetErrorString(e));
}
template <class T> void alloc(T *&p, std::size_t n) {
  if (n)
    check(cudaMallocAsync(&p, n * sizeof(T), nullptr));
}
template <class T> RecoveryArray<T> array(std::int32_t n) {
  RecoveryArray<T> a;
  a.count = n;
  alloc(a.data, n);
  return a;
}
template <class T> std::vector<T> download(const T *p, std::size_t n) {
  std::vector<T> v(n);
  if (n)
    check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}
template <class T> void upload(T *p, const std::vector<T> &v) {
  if (!v.empty())
    check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
}
__global__ void identity(int *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    a[i] = i;
}
__global__ void compact_map(int *out, const int *old, const int *index, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && index[i] >= 0)
    out[index[i]] = old[i];
}
__device__ int map_col(int x, const int *cols, int n) { return x >= 0 && x < n ? cols[x] : x; }
__device__ int map_row(int x, const int *rows, int m) { return x >= 0 && x < m ? rows[x] : x; }
struct IndexSpan {
  int *p;
  NnzOffset n;
  __device__ bool empty() const { return n == 0; }
  __device__ NnzOffset size() const { return n; }
  __device__ int &operator[](std::size_t i) { return p[i]; }
};
constexpr int LONG_RECOVERY_INDICES = 256;
__global__ void globalize(int *data, const int *types, const NnzOffset *starts, int count,
                          const int *rows, int m, const int *cols, int n, int* long_records) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= count)
    return;
  IndexSpan indices{data + starts[k], starts[k + 1] - starts[k]};
  const auto type = static_cast<PostsolveReductionType>(types[k]);
  if (long_records && indices.size() > LONG_RECOVERY_INDICES &&
      (type == PostsolveReductionType::FixedCol || type == PostsolveReductionType::SubCol ||
       type == PostsolveReductionType::DoubletonEquation)) {
    long_records[1 + atomicAdd(long_records, 1)] = k;
    return;
  }
  switch (static_cast<PostsolveReductionType>(types[k])) {
  case PostsolveReductionType::FixedCol:
    if (!indices.empty()) {
      indices[0] = map_col(indices[0], cols, n);
      for (std::size_t p = 1; p < indices.size(); ++p) {
        indices[p] = map_row(indices[p], rows, m);
      }
    }
    break;
  case PostsolveReductionType::FixedColInf:
    if (indices.size() >= 2) {
      indices[1] = map_col(indices[1], cols, n);
      std::size_t p = 2;
      while (p < indices.size()) {
        const NnzOffset row_len = indices[p++];
        for (std::int32_t t = 0; t < row_len && p < indices.size(); ++t, ++p) {
          indices[p] = map_col(indices[p], cols, n);
        }
      }
    }
    break;
  case PostsolveReductionType::SubCol:
    if (indices.size() >= 3) {
      indices[0] = map_col(indices[0], cols, n);
      indices[1] = map_row(indices[1], rows, m);
      const std::int32_t support_count = indices[2];
      for (std::int32_t t = 0;
           t < support_count && 3 + t < static_cast<std::int32_t>(indices.size()); ++t) {
        indices[static_cast<std::size_t>(3 + t)] =
            map_col(indices[static_cast<std::size_t>(3 + t)], cols, n);
      }
    }
    break;
  case PostsolveReductionType::ParallelColumn:
    if (indices.size() >= 2) {
      indices[0] = map_col(indices[0], cols, n);
      indices[1] = map_col(indices[1], cols, n);
    }
    break;
  case PostsolveReductionType::ParallelRow:
    if (indices.size() >= 2) {
      indices[0] = map_row(indices[0], rows, m);
      indices[1] = map_row(indices[1], rows, m);
    }
    break;
  case PostsolveReductionType::DeletedRow:
    if (!indices.empty()) {
      indices[0] = map_row(indices[0], rows, m);
      if (indices.size() >= 2) {
        indices[1] = map_col(indices[1], cols, n);
      }
    }
    break;
  case PostsolveReductionType::LhsChange:
  case PostsolveReductionType::RhsChange:
  case PostsolveReductionType::EqToIneq:
    if (!indices.empty()) {
      indices[0] = map_row(indices[0], rows, m);
    }
    break;
  case PostsolveReductionType::BoundChangeNoRow:
    if (!indices.empty()) {
      indices[0] = map_col(indices[0], cols, n);
    }
    break;
  case PostsolveReductionType::BoundChangeTheRow:
    if (indices.size() >= 2) {
      indices[0] = map_col(indices[0], cols, n);
      indices[1] = map_row(indices[1], rows, m);
    }
    break;
  case PostsolveReductionType::DoubletonEquation:
    if (indices.size() >= 4) {
      indices[0] = map_col(indices[0], cols, n);
      indices[1] = map_col(indices[1], cols, n);
      indices[2] = map_row(indices[2], rows, m);
      const std::int32_t support_count = indices[3];
      for (std::int32_t t = 0;
           t < support_count && 4 + t < static_cast<std::int32_t>(indices.size()); ++t) {
        indices[static_cast<std::size_t>(4 + t)] =
            map_row(indices[static_cast<std::size_t>(4 + t)], rows, m);
      }
    }
    break;
  case PostsolveReductionType::FmeCol:
    if (indices.size() >= 3) {
      const std::int32_t involved_count = indices[0];
      indices[1] = map_col(indices[1], cols, n);
      for (std::int32_t t = 0;
           t < involved_count && 2 + t < static_cast<std::int32_t>(indices.size()); ++t) {
        indices[static_cast<std::size_t>(2 + t)] =
            map_row(indices[static_cast<std::size_t>(2 + t)], rows, m);
      }
      const std::int32_t side_count_pos = 2 + involved_count;
      if (side_count_pos < static_cast<std::int32_t>(indices.size())) {
        const std::int32_t side_col_count = indices[static_cast<std::size_t>(side_count_pos)];
        for (std::int32_t t = 0; t < side_col_count &&
                                 side_count_pos + 1 + t < static_cast<std::int32_t>(indices.size());
             ++t) {
          indices[static_cast<std::size_t>(side_count_pos + 1 + t)] =
              map_col(indices[static_cast<std::size_t>(side_count_pos + 1 + t)], cols, n);
        }
      }
    }
    break;
  case PostsolveReductionType::AddedRow:
  case PostsolveReductionType::AddedRows:
    break;
  }
}
// Each block maps one long support. Headers/counts and output positions retain
// their original layout; no floating-point values or record order are changed.
__global__ void globalize_long(int* data, const int* types, const NnzOffset* starts,
                               const int* records, const int* rows, int m,
                               const int* cols, int n) {
  for (int q = blockIdx.x; q < records[0]; q += gridDim.x) {
    const int k = records[q + 1];
    int* indices = data + starts[k];
    const NnzOffset size = starts[k + 1] - starts[k];
    const auto type = static_cast<PostsolveReductionType>(types[k]);
    NnzOffset first = 1, count = size - 1;
    if (type == PostsolveReductionType::SubCol) {
      first = 3;
      count = min(NnzOffset(indices[2]), size - first);
    } else if (type == PostsolveReductionType::DoubletonEquation) {
      first = 4;
      count = min(NnzOffset(indices[3]), size - first);
    }
    if (threadIdx.x == 0) {
      indices[0] = map_col(indices[0], cols, n);
      if (type == PostsolveReductionType::SubCol) indices[1] = map_row(indices[1], rows, m);
      if (type == PostsolveReductionType::DoubletonEquation) {
        indices[1] = map_col(indices[1], cols, n);
        indices[2] = map_row(indices[2], rows, m);
      }
    }
    for (NnzOffset i = threadIdx.x; i < count; i += blockDim.x) {
      int& index = indices[first + i];
      index = type == PostsolveReductionType::SubCol ? map_col(index, cols, n)
                                                    : map_row(index, rows, m);
    }
  }
}
void globalize_segment(PostsolveTapeGpu& s, const DeviceRecoveryState& d) {
  CleanupScratchFrame scratch;
  // Every queued record consumes at least threshold+1 indices.
  const auto capacity = std::min<NnzOffset>(s.record_count, s.index_count / (LONG_RECOVERY_INDICES + 1));
  int* records = nullptr;
  if (capacity) {
    records = scratch.get<int>(5, std::size_t(capacity) + 1);
    check(cudaMemsetAsync(records, 0, sizeof(int), nullptr));
  }
  globalize<<<(s.record_count + 255) / 256, 256>>>(s.indices, s.types, s.index_starts,
      s.record_count, d.rows.data, d.rows.count, d.cols.data, d.cols.count, records);
  if (capacity)
    globalize_long<<<std::min<NnzOffset>(capacity, 256), 256>>>(s.indices, s.types,
        s.index_starts, records, d.rows.data, d.rows.count, d.cols.data, d.cols.count);
  check(cudaGetLastError());
}
__global__ void fixed_counts(int *scan, const unsigned char *keep, const double *l, const double *u,
                             int n, const unsigned char* prior) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    scan[i] = (!prior || prior[i]) && !keep[i] && isfinite(l[i]) && isfinite(u[i]) && fabs(l[i] - u[i]) <= 1e-9;
}
__global__ void fixed_pack(int *idx, double *val, const int *scan, const unsigned char *keep,
                           const double *l, const double *u, const int *map, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && scan[i] > (i ? scan[i - 1] : 0)) {
    int k = scan[i] - 1;
    idx[k] = map[i];
    val[k] = __dmul_rn(0.5, __dadd_rn(l[i], u[i]));
  }
}
void append_fixed(DeviceRecoveryState &d, const PresolvePlanGpu &plan, const LPInfoGpu &lp,
                  int next_n, const unsigned char* keep = nullptr,
                  const unsigned char* prior = nullptr) {
  const int n = lp.A.cols;
  if (n == next_n || !n)
    return;
  auto scan = array<int>(n);
  const double *l = plan.new_l ? plan.new_l : lp.l;
  const double *u = plan.new_u ? plan.new_u : lp.u;
  fixed_counts<<<(n + 255) / 256, 256>>>(scan.data, keep ? keep : plan.keep_col_mask, l, u, n, prior);
  check(cudaGetLastError());
  std::size_t bytes = 0;
  check(cub::DeviceScan::InclusiveSum(nullptr, bytes, scan.data, scan.data, n));
  void *scratch = nullptr;
  check(cudaMallocAsync(&scratch, bytes, nullptr));
  check(cub::DeviceScan::InclusiveSum(scratch, bytes, scan.data, scan.data, n));
  int count = 0;
  check(cudaMemcpy(&count, scan.data + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
  recovery_device_free(scratch);
  if (!count)
    return;
  DeviceFixedSegment fixed;
  fixed.indices = array<int>(count);
  fixed.values = array<double>(count);
  fixed_pack<<<(n + 255) / 256, 256>>>(fixed.indices.data, fixed.values.data, scan.data,
                                       plan.keep_col_mask, l, u, d.cols.data, n);
  check(cudaGetLastError());
  d.fixed.push_back(std::move(fixed));
}
__global__ void simple_counts(int *scan, const unsigned char *a, const unsigned char *b, int n,
                              int kind) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int x = a[i] != 0, y = b && b[i] != 0;
    scan[i] = kind == 3 ? x + y : kind == 4 ? (x || y) : x;
  }
}
__global__ void simple_pack(int *types, unsigned char *modes, NnzOffset *starts, NnzOffset *vstarts,
                            int *indices, double *vals, const int *scan, const unsigned char *a,
                            const unsigned char *b, int n, int kind, int ni, int nv,
                            const int *second, const double *lo, const double *hi,
                            const double *extra) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int begin = i ? scan[i - 1] : 0, end = scan[i];
  for (int k = begin; k < end; ++k) {
    auto ip = NnzOffset(ni) * k, vp = NnzOffset(nv) * k;
    modes[k] = unsigned(PostsolveDualMode::Minimal);
    starts[k + 1] = ip + ni;
    vstarts[k + 1] = vp + nv;
    indices[ip] = i;
    if (kind == 0) {
      types[k] = int(PostsolveReductionType::FixedCol);
      vals[vp] = lo[i];
      vals[vp + 1] = hi[i];
    } else if (kind == 1 || kind == 2) {
      types[k] = int(PostsolveReductionType::DeletedRow);
      vals[vp] = lo[i];
      vals[vp + 1] = hi[i];
      if (kind == 1) {
        indices[ip + 1] = second[i];
        vals[vp + 2] = extra[i];
      }
    } else if (kind == 3) {
      bool lower = a[i] && k == begin;
      types[k] = int(lower ? PostsolveReductionType::LhsChange : PostsolveReductionType::RhsChange);
      vals[vp] = lo[i];
      vals[vp + 1] = hi[i];
      vals[vp + 2] = lower ? -INFINITY : lo[i];
      vals[vp + 3] = lower ? hi[i] : INFINITY;
    } else {
      types[k] = int(PostsolveReductionType::BoundChangeNoRow);
      vals[vp] = lo[i];
      vals[vp + 1] = hi[i];
      vals[vp + 2] = a[i] ? -INFINITY : lo[i];
      vals[vp + 3] = b[i] ? INFINITY : hi[i];
    }
  }
}
struct SegmentView {
  int *types;
  unsigned char *modes;
  NnzOffset *is;
  NnzOffset *vs;
  int *indices;
  double *vals;
};
SegmentView view(PostsolveTapeGpu &s) {
  return {s.types, s.dual_modes, s.index_starts, s.value_starts, s.indices, s.vals};
}
// Wide allocation counters prevent overflow before checking the int32 payload
// lengths. Sparse indices and the stored tape offsets remain NnzOffset/int32.
using Wide = unsigned long long;
void scan_wide(Wide *p, int n) {
  std::size_t bytes = 0;
  check(cub::DeviceScan::InclusiveSum(nullptr, bytes, p, p, n));
  void *tmp = nullptr;
  check(cudaMallocAsync(&tmp, bytes, nullptr));
  check(cub::DeviceScan::InclusiveSum(tmp, bytes, p, p, n));
  recovery_device_free(tmp);
}
__global__ void infinite_counts(Wide *records, Wide *sizes, const unsigned char *fixed,
                                const unsigned char *inf, const unsigned char *keep,
                                DeviceCsrMatrix a, DeviceCsrMatrix at, int n, const unsigned char* entry_cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= n)
    return;
  bool selected = fixed[col] && inf[col];
  records[col] = selected;
  Wide size = selected ? 2 : 0;
  if (selected)
    for (auto p = at.rowPtr[col]; p < at.rowPtr[col + 1]; ++p) {
      int row = at.colVal[p];
      if (keep[row]) {
        ++size;
        if (!entry_cols) size += Wide(a.rowPtr[row + 1] - a.rowPtr[row]);
        else for (auto q = a.rowPtr[row]; q < a.rowPtr[row + 1]; ++q)
          size += entry_cols[a.colVal[q]] != 0;
      }
    }
  sizes[col] = size;
}
__global__ void infinite_pack(SegmentView out, const Wide *scan, const Wide *sizes,
                              const unsigned char *fixed, const unsigned char *inf,
                              const double *value, const unsigned char *keep, const double *l,
                              const double *u, const double *al, const double *au,
                              DeviceCsrMatrix a, DeviceCsrMatrix at, int n, const unsigned char* entry_cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= n || !fixed[col] || !inf[col])
    return;
  auto k = NnzOffset(scan[col] - 1), begin = NnzOffset(col ? sizes[col - 1] : 0);
  auto p = begin;
  bool positive = value[col] > 0;
  out.types[k] = int(PostsolveReductionType::FixedColInf);
  out.modes[k] = unsigned(PostsolveDualMode::Exact);
  out.is[k + 1] = out.vs[k + 1] = NnzOffset(sizes[col]);
  out.indices[p] = positive ? 1 : -1;
  out.vals[p++] = 0;
  out.indices[p] = col;
  out.vals[p++] = positive ? l[col] : u[col];
  int active = 0;
  for (auto t = at.rowPtr[col]; t < at.rowPtr[col + 1]; ++t) {
    int row = at.colVal[t];
    if (!keep[row])
      continue;
    ++active;
    const auto row_header = p;
    out.indices[p] = 0;
    out.vals[p++] = isfinite(al[row]) ? al[row] : au[row];
    for (auto q = a.rowPtr[row]; q < a.rowPtr[row + 1]; ++q) {
      if (entry_cols && !entry_cols[a.colVal[q]]) continue;
      ++out.indices[row_header];
      out.indices[p] = a.colVal[q];
      out.vals[p++] = a.nzVal[q];
    }
  }
  out.vals[begin] = double(active);
}
__device__ int lower_row(DeviceCsrMatrix a, const std::int32_t* rows, const double *al, const double *au,
                         int col, int count, double *pivot, double *bound) {
  for (int j = 0; j < count; ++j) {
    int row = rows[std::size_t(col) * count + j];
    double v = 0;
    int seen = 0;
    for (auto p = a.rowPtr[row]; p < a.rowPtr[row + 1]; ++p)
      if (a.colVal[p] == col) {
        v = a.nzVal[p];
        ++seen;
      }
    if (seen != 1 || !isfinite(v) || v == 0)
      return -1;
    bool lower = isfinite(al[row]) && isinf(au[row]) && au[row] > 0 && v > 0;
    bool upper = isinf(al[row]) && al[row] < 0 && isfinite(au[row]) && v < 0;
    if (lower || upper) {
      *pivot = v;
      *bound = lower ? al[row] : au[row];
      return row;
    }
  }
  return -1;
}
__global__ void projection_counts(Wide *records, Wide *support, int *error,
                                  const unsigned char *selected, DeviceCsrMatrix a,
                                  const std::int32_t* rows, const std::uint8_t* keep_col, const double *al, const double *au, int nr) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= a.cols)
    return;
  records[col] = selected[col] != 0;
  support[col] = 0;
  if (selected[col]) {
    double pivot = 0, bound = 0;
    int row = lower_row(a, rows, al, au, col, nr, &pivot, &bound);
    if (row < 0) {
      atomicExch(error, 1);
      return;
    }
    for (auto p = a.rowPtr[row]; p < a.rowPtr[row + 1]; ++p)
      support[col] += a.colVal[p] != col && (!keep_col || keep_col[a.colVal[p]]);
    if (nr == 3 && support[col] != 2)
      atomicExch(error, 1);
  }
}
__global__ void projection_pack(SegmentView out, const Wide *records, const Wide *support,
                                const unsigned char *selected, DeviceCsrMatrix a,
                                const std::int32_t* rows, const std::uint8_t* keep_col, const double *al, const double *au,
                                const double *lower, int nr) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= a.cols || !selected[col])
    return;
  auto k = NnzOffset(records[col] - 1), s = NnzOffset(col ? support[col - 1] : 0),
       length = NnzOffset(support[col]) - s;
  auto ip = NnzOffset(nr + 3) * k + s, vp = NnzOffset(7) * k + s;
  out.types[k] = int(PostsolveReductionType::FmeCol);
  out.modes[k] = unsigned(PostsolveDualMode::Minimal);
  out.is[k + 1] = ip + nr + 3 + length;
  out.vs[k + 1] = vp + 7 + length;
  out.indices[ip] = nr;
  out.indices[ip + 1] = col;
  for (int j = 0; j < nr; ++j)
    out.indices[ip + 2 + j] = rows[std::size_t(col) * nr + j];
  out.indices[ip + 2 + nr] = length;
  double pivot = 0, bound = 0;
  int row = lower_row(a, rows, al, au, col, nr, &pivot, &bound);
  out.vals[vp] = 2;
  out.vals[vp + 1] = 1;
  out.vals[vp + 2] = 0;
  out.vals[vp + 3] = nr == 2 ? lower[col] : 0;
  out.vals[vp + 4] = 1;
  out.vals[vp + 5] = double(length);
  out.vals[vp + 6] = __ddiv_rn(bound, pivot);
  int j = 0;
  for (auto p = a.rowPtr[row]; p < a.rowPtr[row + 1]; ++p)
    if (a.colVal[p] != col && (!keep_col || keep_col[a.colVal[p]])) {
      out.indices[ip + nr + 3 + j] = a.colVal[p];
      out.vals[vp + 7 + j] = __ddiv_rn(-a.nzVal[p], pivot);
      ++j;
    }
}
__global__ void removed_counts(int *scan, const int *index, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    scan[i] = index[i] < 0;
}
__global__ void removed_pack(int *out, const int *scan, const int *index, const int *map, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && index[i] < 0)
    out[scan[i] - 1] = map[i];
}
void append_removed(std::vector<RecoveryArray<int>> &out, const RecoveryArray<int> &map,
                    const int *index, int next) {
  if (next == map.count)
    return;
  int n = map.count;
  auto scan = array<int>(n);
  removed_counts<<<(n + 255) / 256, 256>>>(scan.data, index, n);
  std::size_t bytes = 0;
  check(cub::DeviceScan::InclusiveSum(nullptr, bytes, scan.data, scan.data, n));
  void *tmp = nullptr;
  check(cudaMallocAsync(&tmp, bytes, nullptr));
  check(cub::DeviceScan::InclusiveSum(tmp, bytes, scan.data, scan.data, n));
  recovery_device_free(tmp);
  auto result = array<int>(n - next);
  removed_pack<<<(n + 255) / 256, 256>>>(result.data, scan.data, index, map.data, n);
  check(cudaGetLastError());
  out.push_back(std::move(result));
}
__global__ void validate_segment(int *error, SegmentView s, int count, NnzOffset indices,
                                 NnzOffset values) {
  unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= unsigned(count))
    return;
  if (s.is[k] < 0 || s.is[k] > s.is[k + 1] || s.is[k + 1] > indices || s.vs[k] < 0 ||
      s.vs[k] > s.vs[k + 1] || s.vs[k + 1] > values || s.types[k] < 0 ||
      s.types[k] > int(PostsolveReductionType::FmeCol) ||
      s.modes[k] > unsigned(PostsolveDualMode::Minimal))
    atomicExch(error, 1);
  if (k == 0 && (s.is[0] != 0 || s.vs[0] != 0 || s.is[count] != indices || s.vs[count] != values))
    atomicExch(error, 1);
}
void clear_host(PostsolveTape &t) {
  t.types.clear();
  t.indices.clear();
  t.vals.clear();
  t.dual_modes.clear();
  t.index_starts.assign(1, 0);
  t.value_starts.assign(1, 0);
}
void append_export(PostsolveTape &host, const PostsolveTapeGpu &s) {
  auto types = download(s.types, s.record_count);
  auto modes = download(s.dual_modes, s.record_count);
  auto indices = download(s.indices, s.index_count);
  auto vals = download(s.vals, s.value_count);
  auto is = download(s.index_starts, s.record_count + 1);
  auto vs = download(s.value_starts, s.record_count + 1);
  const auto ib = host.indices.size(), vb = host.vals.size();
  checked_nnz_offset(ib + indices.size());
  checked_nnz_offset(vb + vals.size());
  host.types.insert(host.types.end(), types.begin(), types.end());
  host.dual_modes.insert(host.dual_modes.end(), modes.begin(), modes.end());
  host.indices.insert(host.indices.end(), indices.begin(), indices.end());
  host.vals.insert(host.vals.end(), vals.begin(), vals.end());
  for (int i = 1; i <= s.record_count; ++i) {
    host.index_starts.push_back(checked_nnz_offset(ib + is[i]));
    host.value_starts.push_back(checked_nnz_offset(vb + vs[i]));
  }
}
__global__ void gather_ids(int *ids, const int *map, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    ids[i] = map[ids[i]];
}
// Payload values stay on device; only group sizes are needed by the allocator.
template <class T> RecoveryArray<T> copy_array(const T* source, int n) {
  auto a = array<T>(n);
  if (n) check(cudaMemcpyAsync(a.data, source, sizeof(T) * std::size_t(n), cudaMemcpyDeviceToDevice));
  return a;
}
__global__ void fill_value(double* values, int n, double value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) values[i] = value;
}
__global__ void pair_recovery_ids(int* plus, int* minus, int* root_plus, int* root_minus,
                                  const int* parent, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    plus[i] = i; minus[i] = i + n;
    root_plus[i] = parent[i]; root_minus[i] = parent[i] + n;
  }
}
void globalize_group(DeviceRecoveryGroup& group, const RecoveryArray<int>& map) {
  for (auto& ids : group.ids)
    if (ids.count) gather_ids<<<(ids.count + 255) / 256, 256>>>(ids.data, map.data, ids.count);
  check(cudaGetLastError());
}
__global__ void fill_int_max(int* values, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) values[i] = INT_MAX;
}
__global__ void first_slack_row(int* first, const int* block, const int* slack,
                               const std::uint8_t* removable, int rows) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < rows && removable[r] && block[r] >= 0 && slack[r] >= 0)
    atomicMin(first + slack[r], r);
}
__global__ void slack_sort_keys(int* keys, int* order, const int* first, const int* block,
                               const int* slack, const std::uint8_t* removable, int rows) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < rows) {
    order[r] = r;
    keys[r] = removable[r] && block[r] >= 0 && slack[r] >= 0 ? first[slack[r]] : INT_MAX;
  }
}
__global__ void slack_group_flags(int* flags, const int* keys, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) flags[i] = keys[i] != INT_MAX && (!i || keys[i] != keys[i-1]);
}
__global__ void slack_group_offsets(int* offsets, const int* scan, const int* keys, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && keys[i] != INT_MAX) {
    if (!i || keys[i] != keys[i-1]) offsets[scan[i]-1] = i;
    if (i == n-1 || keys[i+1] == INT_MAX) offsets[scan[i]] = i+1;
  }
}
__global__ void pack_max_slack(int* slack_id, int* t_ids, double* factors, const int* order,
                             const int* t, const int* row_block, const int* row_slack,
                             const double* row_factor, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    int r = order[i];
    if (!i) slack_id[0] = row_slack[r];
    t_ids[i] = t[row_block[r]];
    factors[i] = row_factor[r];
  }
}
AbsoluteValueEpigraphPrimalRecoveryStep download_absolute_value_epigraph(const DeviceAbsoluteValueEpigraphStep &s) {
  AbsoluteValueEpigraphPrimalRecoveryStep p;
  p.pattern = s.pattern;
  auto ids = [](const auto &g, int i) { return download(g.ids[i].data, g.ids[i].count); };
  auto vals = [](const auto &g, int i) { return download(g.values[i].data, g.values[i].count); };
  auto a = ids(s.splits, 0), b = ids(s.splits, 1);
  auto v = vals(s.splits, 0);
  for (std::size_t k = 0; k < a.size(); ++k)
    p.splits.push_back({a[k], b[k], v[k]});
  a = ids(s.pairs, 0);
  b = ids(s.pairs, 1);
  for (std::size_t k = 0; k < a.size(); ++k)
    p.outer_pairs.push_back({a[k], b[k]});
  a = ids(s.linked, 0);
  b = ids(s.linked, 1);
  v = vals(s.linked, 0);
  for (std::size_t k = 0; k < a.size(); ++k)
    p.linked_slacks.push_back({a[k], b[k], v[k]});
  for (const auto &g : s.max_slacks)
    p.max_slacks.push_back({ids(g, 0)[0], ids(g, 1), vals(g, 0)});
  return p;
}

} // namespace
void recovery_device_free(void *p) noexcept {
  if (p)
    cudaFreeAsync(p, nullptr);
}
bool device_recovery_enabled() {
  return runtime::options().device_recovery;
}
void capture_variable_pair_recovery_device(PresolvePlanGpu& plan, const int* parent, int pairs,
                                          const double* plus_shift, const double* minus_shift) {
  auto g = std::make_shared<DeviceRecoveryGroup>();
  for (int i = 0; i < 4; ++i) g->ids.push_back(array<int>(pairs));
  if (pairs) pair_recovery_ids<<<(pairs + 255) / 256, 256>>>(
      g->ids[0].data, g->ids[1].data, g->ids[2].data, g->ids[3].data, parent, pairs);
  for (const double* shift : {plus_shift, minus_shift}) {
    auto values = shift ? copy_array(shift, pairs) : array<double>(pairs);
    if (!shift && pairs) check(cudaMemsetAsync(values.data, 0, sizeof(double) * std::size_t(pairs)));
    g->values.push_back(std::move(values));
  }
  check(cudaGetLastError());
  plan.variable_pair_aggregation_device = std::move(g);
  plan.has_variable_pair_aggregation_recovery = true;
}
void capture_absolute_value_splits_device(PresolvePlanGpu& plan, const int* t, const int* e,
                                         const double* rho, int count, const char* pattern) {
  auto s = std::make_shared<DeviceAbsoluteValueEpigraphStep>();
  s->pattern = pattern;
  s->splits.ids.push_back(copy_array(t, count));
  s->splits.ids.push_back(copy_array(e, count));
  auto values = rho ? copy_array(rho, count) : array<double>(count);
  if (!rho && count) fill_value<<<(count + 255) / 256, 256>>>(values.data, count, 1.0);
  s->splits.values.push_back(std::move(values));
  s->pairs.ids.resize(2);
  s->linked.ids.resize(2); s->linked.values.resize(1);
  check(cudaGetLastError());
  plan.absolute_value_epigraph_recovery = {};
  plan.absolute_value_epigraph_recovery.pattern = pattern;
  plan.absolute_value_epigraph_device = std::move(s);
  plan.has_absolute_value_epigraph_recovery = true;
}
void capture_absolute_value_outer_device(PresolvePlanGpu& plan, const int* bound, const int* free,
                                        int pairs, const int* slack, const int* t,
                                        const double* factor, int blocks) {
  auto& s = *plan.absolute_value_epigraph_device;
  s.pairs.ids[0] = copy_array(bound, pairs);
  s.pairs.ids[1] = copy_array(free, pairs);
  s.linked.ids[0] = copy_array(slack, blocks);
  s.linked.ids[1] = copy_array(t, blocks);
  s.linked.values[0] = copy_array(factor, blocks);
}
void capture_absolute_value_max_slacks_device(PresolvePlanGpu& plan, const int* t,
    const int* row_block, const int* row_slack, const double* row_factor,
    const std::uint8_t* row_removable, int rows, int cols) {
  if (!rows) return;
  auto first = array<int>(cols);
  // Every valid row is smaller than INT_MAX, including for int32 dimensions.
  auto keys = array<int>(rows), sorted_keys = array<int>(rows);
  auto order = array<int>(rows), sorted_order = array<int>(rows), scan = array<int>(rows);
  if (cols) fill_int_max<<<(cols + 255) / 256, 256>>>(first.data, cols);
  first_slack_row<<<(rows + 255) / 256, 256>>>(first.data, row_block, row_slack, row_removable, rows);
  slack_sort_keys<<<(rows + 255) / 256, 256>>>(keys.data, order.data, first.data,
                                           row_block, row_slack, row_removable, rows);
  check(cudaGetLastError());
  std::size_t sort_bytes = 0, scan_bytes = 0;
  check(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, keys.data, sorted_keys.data,
                                      order.data, sorted_order.data, rows));
  check(cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, scan.data, scan.data, rows));
  void* scratch = nullptr;
  check(cudaMallocAsync(&scratch, std::max(sort_bytes, scan_bytes), nullptr));
  // Stable sort by first incident row preserves both the original group order
  // and the original row order within each group.
  check(cub::DeviceRadixSort::SortPairs(scratch, sort_bytes, keys.data, sorted_keys.data,
                                      order.data, sorted_order.data, rows));
  slack_group_flags<<<(rows + 255) / 256, 256>>>(scan.data, sorted_keys.data, rows);
  check(cudaGetLastError());
  check(cub::DeviceScan::InclusiveSum(scratch, scan_bytes, scan.data, scan.data, rows));
  recovery_device_free(scratch);
  const int groups = download(scan.data + rows - 1, 1)[0];
  if (!groups) return;
  auto offsets = array<int>(checked_nnz_offset(std::int64_t(groups) + 1));
  slack_group_offsets<<<(rows + 255) / 256, 256>>>(offsets.data, scan.data, sorted_keys.data, rows);
  check(cudaGetLastError());
  // Allocation metadata only: no column IDs, coefficients or shifts leave GPU.
  const auto starts = download(offsets.data, groups + 1);
  auto& result = plan.absolute_value_epigraph_device->max_slacks;
  result.reserve(groups);
  for (int i = 0; i < groups; ++i) {
    const int count = starts[i+1] - starts[i];
    DeviceRecoveryGroup g;
    g.ids.push_back(array<int>(1)); g.ids.push_back(array<int>(count));
    g.values.push_back(array<double>(count));
    pack_max_slack<<<(count + 255) / 256, 256>>>(g.ids[0].data, g.ids[1].data,
        g.values[0].data, sorted_order.data + starts[i], t, row_block, row_slack, row_factor, count);
    check(cudaGetLastError());
    result.push_back(std::move(g));
  }
}
PostsolveTapeGpu allocate_recovery_segment(NnzOffset r, NnzOffset i, NnzOffset v) {
  checked_nnz_offset(r);
  checked_nnz_offset(i);
  checked_nnz_offset(v);
  checked_nnz_offset(static_cast<std::int64_t>(r) + 1);
  PostsolveTapeGpu s;
  s.record_count = s.record_capacity = r;
  s.index_count = s.index_capacity = i;
  s.value_count = s.value_capacity = v;
  alloc(s.types, r);
  alloc(s.dual_modes, r);
  alloc(s.index_starts, std::size_t(r) + 1);
  alloc(s.value_starts, std::size_t(r) + 1);
  alloc(s.indices, i);
  alloc(s.vals, v);
  check(cudaMemsetAsync(s.index_starts, 0, sizeof(NnzOffset), nullptr));
  check(cudaMemsetAsync(s.value_starts, 0, sizeof(NnzOffset), nullptr));
  return s;
}
void append_simple_device_recovery(PostsolveTape &t, SimpleRecovery kind, const unsigned char *a,
                                   const unsigned char *b, int n, const int *second,
                                   const double *lo, const double *hi, const double *extra) {
  if (!n)
    return;
  int tag = int(kind);
  checked_nnz_offset(std::int64_t(n) * (tag == 3 ? 2 : 1));
  auto scan = array<int>(n);
  simple_counts<<<(n + 255) / 256, 256>>>(scan.data, a, b, n, tag);
  check(cudaGetLastError());
  std::size_t bytes = 0;
  check(cub::DeviceScan::InclusiveSum(nullptr, bytes, scan.data, scan.data, n));
  void *scratch = nullptr;
  check(cudaMallocAsync(&scratch, bytes, nullptr));
  check(cub::DeviceScan::InclusiveSum(scratch, bytes, scan.data, scan.data, n));
  int count = 0;
  check(cudaMemcpy(&count, scan.data + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
  recovery_device_free(scratch);
  if (!count)
    return;
  int ni = tag == 1 ? 2 : 1, nv = tag == 0 || tag == 2 ? 2 : tag == 1 ? 3 : 4;
  auto segment = allocate_recovery_segment(count, checked_nnz_offset(std::int64_t(ni) * count),
                                           checked_nnz_offset(std::int64_t(nv) * count));
  simple_pack<<<(n + 255) / 256, 256>>>(segment.types, segment.dual_modes, segment.index_starts,
                                        segment.value_starts, segment.indices, segment.vals,
                                        scan.data, a, b, n, tag, ni, nv, second, lo, hi, extra);
  check(cudaGetLastError());
  append_device_recovery_segment(t, std::move(segment));
}
void append_infinite_device_recovery(PostsolveTape &t, const unsigned char *fixed,
                                     const unsigned char *inf, const double *value,
                                     const unsigned char *keep, const double *l, const double *u,
                                     const double *al, const double *au, const DeviceCsrMatrix &a,
                                     const DeviceCsrMatrix &at, int n, const unsigned char* entry_cols) {
  if (!n)
    return;
  auto records = array<Wide>(n), sizes = array<Wide>(n);
  infinite_counts<<<(n + 255) / 256, 256>>>(records.data, sizes.data, fixed, inf, keep, a, at, n, entry_cols);
  check(cudaGetLastError());
  scan_wide(records.data, n);
  scan_wide(sizes.data, n);
  auto rc = download(records.data + n - 1, 1)[0], size = download(sizes.data + n - 1, 1)[0];
  if (!rc)
    return;
  if (rc > std::numeric_limits<NnzOffset>::max() || size > std::numeric_limits<NnzOffset>::max())
    throw std::length_error("infinite fixed recovery exceeds offset range");
  auto s = allocate_recovery_segment(rc, size, size);
  infinite_pack<<<(n + 255) / 256, 256>>>(view(s), records.data, sizes.data, fixed, inf, value,
                                          keep, l, u, al, au, a, at, n, entry_cols);
  check(cudaGetLastError());
  append_device_recovery_segment(t, std::move(s));
}
void append_projection_device_recovery(PostsolveTape &t, const LPInfoGpu &lp,
                                       const unsigned char *selected, int nr,
                                       const std::int32_t* rows, const std::uint8_t* keep_col) {
  int n = lp.A.cols;
  if (!n)
    return;
  auto records = array<Wide>(n), support = array<Wide>(n);
  auto error = array<int>(1);
  check(cudaMemset(error.data, 0, sizeof(int)));
  projection_counts<<<(n + 255) / 256, 256>>>(records.data, support.data, error.data, selected,
                                              lp.A, rows, keep_col, lp.AL, lp.AU, nr);
  check(cudaGetLastError());
  if (download(error.data, 1)[0])
    throw std::runtime_error("projection tape certificate changed before commit");
  scan_wide(records.data, n);
  scan_wide(support.data, n);
  auto rc = download(records.data + n - 1, 1)[0], nz = download(support.data + n - 1, 1)[0];
  if (!rc)
    return;
  auto max = std::numeric_limits<NnzOffset>::max();
  if (rc > max || rc * 7 + nz > max || rc * (nr + 3) + nz > max)
    throw std::length_error("projection recovery exceeds offset range");
  auto s = allocate_recovery_segment(rc, rc * (nr + 3) + nz, rc * 7 + nz);
  projection_pack<<<(n + 255) / 256, 256>>>(view(s), records.data, support.data, selected, lp.A,
                                            rows, keep_col, lp.AL, lp.AU, lp.l, nr);
  check(cudaGetLastError());
  append_device_recovery_segment(t, std::move(s));
}
void flush_host_recovery_segment(PostsolveTape &t) {
  if (t.types.empty())
    return;
  if (t.resident)
    throw std::logic_error("host-built ordinary tape in GPU resident recovery");
  auto s = allocate_recovery_segment(checked_nnz_offset(t.types.size()),
                                     checked_nnz_offset(t.indices.size()),
                                     checked_nnz_offset(t.vals.size()));
  upload(s.types, t.types);
  upload(s.dual_modes, t.dual_modes);
  upload(s.index_starts, t.index_starts);
  upload(s.value_starts, t.value_starts);
  upload(s.indices, t.indices);
  upload(s.vals, t.vals);
  t.segments.push_back(std::move(s));
  clear_host(t);
}
void append_device_recovery_segment(PostsolveTape &t, PostsolveTapeGpu &&s) {
  if (!s.record_count)
    return;
  if (!t.resident) {
    append_export(t, s);
    return;
  }
  flush_host_recovery_segment(t);
  t.segments.push_back(std::move(s));
}
void export_local_recovery_tape(PostsolveTape &t) {
  if (t.segments.empty())
    return;
  flush_host_recovery_segment(t);
  clear_host(t);
  for (const auto &s : t.segments)
    append_export(t, s);
  t.segments.clear();
}
void initialize_device_recovery(PresolveRecordGpu &r) {
  r.device = std::make_shared<DeviceRecoveryState>();
  auto &d = *r.device;
  d.rows = array<int>(r.m0);
  d.cols = array<int>(r.n0);
  if (r.m0)
    identity<<<(r.m0 + 255) / 256, 256>>>(d.rows.data, r.m0);
  if (r.n0)
    identity<<<(r.n0 + 255) / 256, 256>>>(d.cols.data, r.n0);
  check(cudaGetLastError());
}
void update_device_recovery(PresolveRecordGpu &r, PresolvePlanGpu &plan, const LPInfoGpu &lp, int m,
                            int n, double obj, const int *row_index, const int *col_index) {
  auto &d = *r.device;
  d.host_staged_bytes +=
      13 * plan.tape.types.size() + 4 * plan.tape.indices.size() + 8 * plan.tape.vals.size();
  flush_host_recovery_segment(plan.tape);
  if ((plan.has_absolute_value_epigraph_recovery || plan.has_variable_pair_aggregation_recovery) &&
      !plan.tape.segments.empty())
    throw std::runtime_error("mixed absolute_value_epigraph/tape recovery plan");
  for (auto &s : plan.tape.segments) {
    auto records = checked_nnz_offset(std::int64_t(d.records) + s.record_count);
    auto indices = checked_nnz_offset(std::int64_t(d.indices) + s.index_count);
    auto values = checked_nnz_offset(std::int64_t(d.values) + s.value_count);
    globalize_segment(s, d);
    d.segments.push_back(std::move(s));
    d.records = records;
    d.indices = indices;
    d.values = values;
  }
  plan.tape.segments.clear();
  if (plan.has_absolute_value_epigraph_recovery) {
    if (!plan.absolute_value_epigraph_device)
      throw std::logic_error("host-built absolute-value payload in GPU resident recovery");
    auto payload = std::move(*plan.absolute_value_epigraph_device);
    plan.absolute_value_epigraph_device.reset();
    globalize_group(payload.splits, d.cols);
    globalize_group(payload.pairs, d.cols);
    globalize_group(payload.linked, d.cols);
    for (auto& group : payload.max_slacks) globalize_group(group, d.cols);
    d.absolute_value_epigraph.push_back(std::move(payload));
    r.primal_recovery_timeline.push_back({PrimalRecoveryKind::AbsoluteValueEpigraph,
                                          checked_nnz_offset(r.absolute_value_epigraph_recoveries.size()),
                                          d.records});
    AbsoluteValueEpigraphPrimalRecoveryStep descriptor;
    descriptor.pattern = plan.absolute_value_epigraph_recovery.pattern;
    r.absolute_value_epigraph_recoveries.push_back(std::move(descriptor));
  }
  if (plan.has_variable_pair_aggregation_recovery) {
    if (!plan.variable_pair_aggregation_device)
      throw std::logic_error("host-built variable-pair payload in GPU resident recovery");
    auto payload = std::move(*plan.variable_pair_aggregation_device);
    plan.variable_pair_aggregation_device.reset();
    globalize_group(payload, d.cols);
    d.variable_pair_aggregation.push_back(std::move(payload));
    r.primal_recovery_timeline.push_back(
        {PrimalRecoveryKind::VariablePairAggregation,
         checked_nnz_offset(r.variable_pair_aggregation_recoveries.size()), d.records});
    r.variable_pair_aggregation_recoveries.emplace_back();
    r.has_primal_only_variable_pair_aggregation_reduction = true;
  }
  r.has_primal_only_column_domination_reduction |= plan.has_column_domination_reduction;
  r.has_primal_only_projected_auxiliary_reduction |= plan.has_projected_auxiliary_reduction;
  if (!plan.cleanup_recovery_captured) {
  const auto* previous = plan.cleanup_prefix_keep_col;
  if (previous) append_fixed(d, plan, lp, n, previous);
  if (plan.cleanup_dual_keep_col) {
    append_fixed(d, plan, lp, n, plan.cleanup_dual_keep_col, previous);
    previous = plan.cleanup_dual_keep_col;
  }
  append_fixed(d, plan, lp, n, plan.keep_col_mask, previous);
  }
  if (runtime::options().record_removed_indices) {
    append_removed(d.removed_rows, d.rows, row_index, m);
    append_removed(d.removed_cols, d.cols, col_index, n);
  }
  if (m != d.rows.count) {
    auto next = array<int>(m);
    if (d.rows.count)
      compact_map<<<(d.rows.count + 255) / 256, 256>>>(next.data, d.rows.data, row_index,
                                                       d.rows.count);
    d.rows = std::move(next);
  }
  if (n != d.cols.count) {
    auto next = array<int>(n);
    if (d.cols.count)
      compact_map<<<(d.cols.count + 255) / 256, 256>>>(next.data, d.cols.data, col_index,
                                                       d.cols.count);
    d.cols = std::move(next);
  }
  check(cudaGetLastError());
  r.m1 = m;
  r.n1 = n;
  r.obj_constant_new = obj;
}
namespace {
__device__ unsigned long long epoch_mix(unsigned long long x) {
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27; x *= 0x94d049bb133111ebULL; return x ^ (x >> 31);
}
__global__ void epoch_hash(unsigned long long* out,const unsigned char* keep,
    const int* map,const double* a,const double* b,const double* c,int n,int offset,bool normalize_zero=false) {
  int j=blockIdx.x*blockDim.x+threadIdx.x;
  if(j>=n || !keep[j]) return;
  auto x=epoch_mix(static_cast<unsigned long long>(map[j])+1);
  x=epoch_mix(x ^ static_cast<unsigned long long>(__double_as_longlong(normalize_zero && a[j]==0.0 ? 0.0 : a[j])));
  x=epoch_mix(x ^ static_cast<unsigned long long>(__double_as_longlong(normalize_zero && b[j]==0.0 ? 0.0 : b[j])));
  if(c) x=epoch_mix(x ^ static_cast<unsigned long long>(__double_as_longlong(normalize_zero && c[j]==0.0 ? 0.0 : c[j])));
  atomicXor(out+offset,x);atomicAdd(out+offset+1,x);atomicAdd(out+offset+2,1ULL);
}
}
void trace_cleanup_epoch(const PresolveRecordGpu& r,const PresolvePlanGpu& p,
                          const LPInfoGpu& lp,double obj) {
  if (!runtime::options().trace_cleanup_epochs || !r.device) return;
  auto h=array<unsigned long long>(12);check(cudaMemset(h.data,0,12*sizeof(unsigned long long)));
  if(lp.A.rows) epoch_hash<<<(lp.A.rows+255)/256,256>>>(h.data,p.keep_row_mask,r.device->rows.data,p.new_AL,p.new_AU,nullptr,lp.A.rows,0);
  if(lp.A.cols) epoch_hash<<<(lp.A.cols+255)/256,256>>>(h.data,p.keep_col_mask,r.device->cols.data,p.new_l,p.new_u,p.new_c,lp.A.cols,3);
  if(lp.A.rows) epoch_hash<<<(lp.A.rows+255)/256,256>>>(h.data,p.keep_row_mask,r.device->rows.data,p.new_AL,p.new_AU,nullptr,lp.A.rows,6,true);
  if(lp.A.cols) epoch_hash<<<(lp.A.cols+255)/256,256>>>(h.data,p.keep_col_mask,r.device->cols.data,p.new_l,p.new_u,p.new_c,lp.A.cols,9,true);
  check(cudaGetLastError());auto values=download(h.data,12);
  unsigned long long bits;static_assert(sizeof(bits)==sizeof(obj));std::memcpy(&bits,&obj,sizeof(obj));
  std::cerr << ">>> [cleanup-epoch-state]";
  for(int i=0;i<6;++i) std::cerr << ' ' << values[i];
  std::cerr << ' ' << bits << '\n';
  std::cerr << ">>> [cleanup-epoch-numeric-state]";
  for(int i=6;i<12;++i) std::cerr << ' ' << values[i];
  std::cerr << ' ' << (obj==0.0 ? 0ULL : bits) << '\n';

}

void capture_cleanup_recovery(PresolveRecordGpu& r, PresolvePlanGpu& plan,
                              const LPInfoGpu& lp, double obj) {
  // Identity map update globalizes tape indices in the still-physical domain.
  // Fixed records are captured separately with the epoch-entry mask, preserving
  // both prefix/tail order and each column's bounds at the original boundary.
  plan.cleanup_recovery_captured = true;
  update_device_recovery(r, plan, lp, lp.A.rows, lp.A.cols, obj, nullptr, nullptr);
  if (!plan.has_col_action) return;
  auto& d = *r.device;
  // -1 requests a scan while the physical column count is still unchanged.
  const auto* previous = plan.cleanup_entry_keep_col;
  if (plan.cleanup_prefix_keep_col) {
    append_fixed(d, plan, lp, -1, plan.cleanup_prefix_keep_col, previous);
    previous = plan.cleanup_prefix_keep_col;
  }
  append_fixed(d, plan, lp, -1, plan.keep_col_mask, previous);
}

void export_recovery_record(PresolveRecordGpu &r) {
  if (!r.device)
    return;
  const auto &d = *r.device;
  r.row_red2org = download(d.rows.data, d.rows.count);
  r.col_red2org = download(d.cols.data, d.cols.count);
  auto inverse = [](auto &out, const auto &in, int n) {
    out.assign(n, -1);
    for (int k = 0; k < int(in.size()); ++k)
      out[in[k]] = k;
  };
  inverse(r.row_org2red, r.row_red2org, r.m0);
  inverse(r.col_org2red, r.col_red2org, r.n0);
  r.removed_row_idx.clear();
  r.removed_col_idx.clear();
  for (const auto &a : d.removed_rows) {
    auto v = download(a.data, a.count);
    r.removed_row_idx.insert(r.removed_row_idx.end(), v.begin(), v.end());
  }
  for (const auto &a : d.removed_cols) {
    auto v = download(a.data, a.count);
    r.removed_col_idx.insert(r.removed_col_idx.end(), v.begin(), v.end());
  }
  r.fixed_idx.clear();
  r.fixed_val.clear();
  for (const auto &s : d.fixed) {
    auto i = download(s.indices.data, s.indices.count);
    auto v = download(s.values.data, s.values.count);
    r.fixed_idx.insert(r.fixed_idx.end(), i.begin(), i.end());
    r.fixed_val.insert(r.fixed_val.end(), v.begin(), v.end());
  }
  r.absolute_value_epigraph_recoveries.clear();
  for (const auto &s : d.absolute_value_epigraph)
    r.absolute_value_epigraph_recoveries.push_back(download_absolute_value_epigraph(s));
  r.variable_pair_aggregation_recoveries.clear();
  for (const auto &g : d.variable_pair_aggregation) {
    VariablePairAggregationRecoveryStep p;
    p.plus_cols = download(g.ids[0].data, g.ids[0].count);
    p.minus_cols = download(g.ids[1].data, g.ids[1].count);
    p.root_plus_cols = download(g.ids[2].data, g.ids[2].count);
    p.root_minus_cols = download(g.ids[3].data, g.ids[3].count);
    p.plus_shift = download(g.values[0].data, g.values[0].count);
    p.minus_shift = download(g.values[1].data, g.values[1].count);
    r.variable_pair_aggregation_recoveries.push_back(std::move(p));
  }
  clear_host(r.tape);
  for (const auto &s : d.segments)
    append_export(r.tape, s);
}
void validate_device_recovery(const PresolveRecordGpu &r) {
  if (!r.device)
    return;
  const auto &d = *r.device;
  if (d.rows.count != r.m1 || d.cols.count != r.n1 ||
      d.absolute_value_epigraph.size() != r.absolute_value_epigraph_recoveries.size() ||
      d.variable_pair_aggregation.size() != r.variable_pair_aggregation_recoveries.size())
    throw std::runtime_error("device recovery dimensions/checkpoints mismatch");
  std::int64_t nr = 0, ni = 0, nv = 0;
  auto error = array<int>(1);
  check(cudaMemset(error.data, 0, sizeof(int)));
  for (const auto &s : d.segments) {
    if (s.record_count <= 0 || s.index_count < 0 || s.value_count < 0 || !s.types ||
        !s.dual_modes || !s.index_starts || !s.value_starts || (s.index_count && !s.indices) ||
        (s.value_count && !s.vals))
      throw std::runtime_error("malformed device recovery segment");
    nr += s.record_count;
    ni += s.index_count;
    nv += s.value_count;
    SegmentView v{s.types, s.dual_modes, s.index_starts, s.value_starts, s.indices, s.vals};
    validate_segment<<<(std::size_t(s.record_count) + 255) / 256, 256>>>(
        error.data, v, s.record_count, s.index_count, s.value_count);
    check(cudaGetLastError());
  }
  if (nr != d.records || ni != d.indices || nv != d.values || download(error.data, 1)[0])
    throw std::runtime_error("malformed device recovery tape");
}
void trace_recovery_plan(const PresolvePlanGpu &p, const LPInfoGpu &lp) {
  if (!runtime::options().trace_recovery)
    return;
  std::uint64_t h = 1469598103934665603ULL;
  auto bytes = [&](const void *ptr, std::size_t n) {
    auto b = static_cast<const unsigned char *>(ptr);
    for (std::size_t i = 0; i < n; ++i) {
      h ^= b[i];
      h *= 1099511628211ULL;
    }
  };
  auto vector = [&](const auto &v) {
    std::uint64_t n = v.size();
    bytes(&n, sizeof(n));
    bytes(v.data(), v.size() * sizeof(v[0]));
  };
  vector(download(p.keep_row_mask, lp.A.rows));
  vector(download(p.keep_col_mask, lp.A.cols));
  for (auto v : {p.new_c, p.new_l, p.new_u}) {
    if (v)
      vector(download(v, lp.A.cols));
    else
      vector(std::vector<double>{});
  }
  for (auto v : {p.new_AL, p.new_AU}) {
    if (v)
      vector(download(v, lp.A.rows));
    else
      vector(std::vector<double>{});
  }
  bytes(&p.obj_constant_delta, sizeof(double));
  const auto &a = p.has_new_A ? p.new_A : lp.A;
  vector(download(a.rowPtr, a.rows + 1));
  vector(download(a.colVal, a.nnz));
  vector(download(a.nzVal, a.nnz));
  PostsolveTape t;
  for (const auto &segment : p.tape.segments)
    append_export(t, segment);
  const auto ib = t.indices.size(), vb = t.vals.size();
  t.types.insert(t.types.end(), p.tape.types.begin(), p.tape.types.end());
  t.dual_modes.insert(t.dual_modes.end(), p.tape.dual_modes.begin(), p.tape.dual_modes.end());
  t.indices.insert(t.indices.end(), p.tape.indices.begin(), p.tape.indices.end());
  t.vals.insert(t.vals.end(), p.tape.vals.begin(), p.tape.vals.end());
  for (std::size_t i = 1; i < p.tape.index_starts.size(); ++i) {
    t.index_starts.push_back(checked_nnz_offset(ib + p.tape.index_starts[i]));
    t.value_starts.push_back(checked_nnz_offset(vb + p.tape.value_starts[i]));
  }
  auto plan_hash = h;
  h = 1469598103934665603ULL;
  vector(t.types);
  vector(t.dual_modes);
  vector(t.index_starts);
  vector(t.value_starts);
  vector(t.indices);
  vector(t.vals);
  std::cerr << ">>> [recovery-trace] dims=" << lp.A.rows << "," << lp.A.cols << " plan=" << std::hex
            << plan_hash << " tape=" << h << std::dec << " records=" << t.types.size() << "\n";
}

} // namespace gpu_presolver::presolve
