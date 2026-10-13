#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/offset.hpp"
#include "epigraph.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpu_presolver::presolve::detail {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;
constexpr int MAX_QUOTIENT_TERMS = 6;
constexpr int MAX_PROBE_COLUMN_DEGREE = 128;
constexpr int MAX_PROBE_ROWS = 1024;
constexpr int MIN_PROBE_MERGE_PERCENT = 5;
constexpr unsigned long long NO_EDGE = ~0ULL;

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " +
                           cudaGetErrorString(status));
}

bool profile_enabled(const PresolveParams& params) {
  return params.verbose || runtime::options().profile_variable_pair_epigraph;
}

struct VariablePairEpigraphLayout {
  std::int32_t pair_count = 0;
  std::int32_t tail_count = 0;
};

// This host-side suffix read is the common rejection path.  It intentionally
// precedes every allocation and kernel launch.  A candidate must end in a
// short, contiguous, strictly-positive objective suffix and have zero
// objective immediately before it.  Full validation later checks every pair.
bool infer_variable_pair_epigraph_layout(const LPInfoGpu& lp,
                       const PresolveParams& params,
                       VariablePairEpigraphLayout* layout) {
  if (layout == nullptr || !params.enable_variable_pair_aggregation || lp.A.cols <= 2 ||
      lp.c == nullptr) {
    return false;
  }
  const std::int32_t min_pairs = std::max(1, params.variable_pair_epigraph_min_pairs);
  const long long min_cols = 2LL * min_pairs + 1;
  const long long min_rows = 2LL * min_pairs;
  const long long min_nnz = 7LL * min_pairs;
  if (lp.A.cols < min_cols || lp.A.rows < min_rows || lp.A.nnz < min_nnz) {
    return false;
  }
  const std::int32_t max_tail =
      std::max(1, std::min(params.variable_pair_epigraph_max_tail_cols, 8));
  double last = 0.0;
  throw_if_cuda_error(
      cudaMemcpy(&last, lp.c + (lp.A.cols - 1), sizeof(double),
                 cudaMemcpyDeviceToHost),
      "cudaMemcpy variable-pair epigraph final objective");
  if (!std::isfinite(last) || last <= 0.0) {
    return false;
  }
  const std::int32_t copy_count = std::min(lp.A.cols, max_tail + 1);
  double suffix[9] = {};
  throw_if_cuda_error(
      cudaMemcpy(suffix, lp.c + (lp.A.cols - copy_count),
                 sizeof(double) * static_cast<std::size_t>(copy_count),
                 cudaMemcpyDeviceToHost),
      "cudaMemcpy variable-pair epigraph objective suffix");
  std::int32_t tail = 0;
  for (std::int32_t k = copy_count - 1; k >= 0; --k) {
    const double value = suffix[k];
    if (std::isfinite(value) && value > 0.0 && tail < max_tail) {
      ++tail;
      continue;
    }
    break;
  }
  if (tail <= 0 || tail > max_tail || tail >= lp.A.cols) {
    return false;
  }
  // If all copied entries are positive, the suffix may be longer than the
  // configured cap.  Reject rather than guessing the pair offset.
  if (tail == copy_count) {
    return false;
  }
  const double preceding =
      suffix[copy_count - tail - 1];
  const std::int32_t prefix = lp.A.cols - tail;
  if (preceding != 0.0 || (prefix & 1) != 0) {
    return false;
  }
  const std::int32_t pairs = prefix / 2;
  if (pairs < min_pairs) {
    return false;
  }
  layout->pair_count = pairs;
  layout->tail_count = tail;
  return true;
}

__device__ bool valid_pair_bounds(double lower, double upper) {
  return isfinite(lower) && !isnan(upper) && upper >= lower &&
         (!isfinite(upper) || isfinite(upper - lower));
}

__device__ bool valid_row_bounds(double lower, double upper) {
  return !isnan(lower) && !isnan(upper) && lower <= upper &&
         lower != INFINITY && upper != -INFINITY;
}

__global__ void kernel_quick_variable_pair_epigraph_pair_probe(
    std::int32_t* failed,
    const NnzOffset* at_row_ptr,
    const std::int32_t* at_col_val,
    const double* at_nz_val,
    const NnzOffset* row_ptr,
    const std::int32_t* row_col,
    const double* row_val,
    const double* AL,
    const double* AU,
    const double* c,
    const double* l,
    const double* u,
    std::int32_t pair_count,
    std::int32_t cols,
    std::int32_t probe_count) {
  const std::int32_t sample =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (sample >= probe_count) {
    return;
  }
  const std::int32_t pair = static_cast<std::int32_t>(
      (static_cast<long long>(sample) * pair_count) / probe_count);
  const std::int32_t mate = pair + pair_count;
  const NnzOffset begin = at_row_ptr[pair];
  const NnzOffset end = at_row_ptr[pair + 1];
  const NnzOffset mate_begin = at_row_ptr[mate];
  const NnzOffset mate_end = at_row_ptr[mate + 1];
  const std::int32_t degree = end - begin;
  bool ok = degree >= 2 && degree <= MAX_PROBE_COLUMN_DEGREE &&
            degree == mate_end - mate_begin && c[pair] == 0.0 &&
            c[mate] == 0.0 && valid_pair_bounds(l[pair], u[pair]) &&
            valid_pair_bounds(l[mate], u[mate]);
  std::int32_t same_count = 0;
  if (ok) {
    for (NnzOffset p = 0; p < degree; ++p) {
      const std::int32_t row = at_col_val[begin + p];
      const double value = at_nz_val[begin + p];
      const double mate_value = at_nz_val[mate_begin + p];
      if (row != at_col_val[mate_begin + p] || !isfinite(value) ||
          !isfinite(mate_value) || value == 0.0) {
        ok = false;
        break;
      }
      if (value == mate_value) {
        ++same_count;
        const NnzOffset rb = row_ptr[row];
        const NnzOffset len = row_ptr[row + 1] - rb;
        ok = len == 3 && row_col[rb] == pair &&
             row_col[rb + 1] == mate && row_col[rb + 2] >= 2 * pair_count &&
             row_col[rb + 2] < cols && row_val[rb] == row_val[rb + 1] &&
             row_val[rb] < 0.0 && isfinite(row_val[rb + 2]) &&
             row_val[rb + 2] > 0.0 && isfinite(AL[row]) &&
             AU[row] == INFINITY;
        if (!ok) {
          break;
        }
      } else if (value != -mate_value) {
        ok = false;
        break;
      }
    }
  }
  if (!ok || same_count != 1) {
    atomicExch(failed, 1);
  }
}

__global__ void kernel_quick_variable_pair_epigraph_edge_probe(
    std::int32_t* merge_count,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    const double* AL,
    const double* AU,
    std::int32_t rows,
    std::int32_t pair_count,
    std::int32_t probe_count) {
  using BlockReduce = cub::BlockReduce<int, GPU_PRESOLVE_THREADS>;
  __shared__ typename BlockReduce::TempStorage storage;
  const std::int32_t sample =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  bool edge = false;
  if (sample < probe_count) {
    const std::int32_t row = static_cast<std::int32_t>(
        (static_cast<long long>(sample) * rows) / probe_count);
    const NnzOffset begin = row_ptr[row];
    const NnzOffset len = row_ptr[row + 1] - begin;
    if (len == 4 && AL[row] == 0.0 && AU[row] == 0.0) {
      const std::int32_t a = col_val[begin];
      const std::int32_t b = col_val[begin + 1];
      const double va = nz_val[begin];
      const double vb = nz_val[begin + 1];
      edge = a >= 0 && a < pair_count && b >= 0 && b < pair_count &&
             a != b && col_val[begin + 2] == a + pair_count &&
             col_val[begin + 3] == b + pair_count && isfinite(va) &&
             isfinite(vb) && va != 0.0 && va == -vb &&
             va == -nz_val[begin + 2] && vb == -nz_val[begin + 3];
    }
  }
  const int block_edges = BlockReduce(storage).Sum(edge ? 1 : 0);
  if (threadIdx.x == 0 && block_edges != 0) {
    atomicAdd(merge_count, block_edges);
  }
}

__global__ void kernel_validate_variable_pair_epigraph_columns(
    std::int32_t* failed,
    double* pair_shift,
    double* pair_lower_sum,
    const double* c,
    const double* l,
    const double* u,
    std::int32_t pair_count,
    std::int32_t cols) {
  const std::int32_t index =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (index < pair_count) {
    const std::int32_t mate = index + pair_count;
    const double lp = l[index];
    const double ln = l[mate];
    bool ok = c[index] == 0.0 && c[mate] == 0.0 &&
              valid_pair_bounds(lp, u[index]) &&
              valid_pair_bounds(ln, u[mate]);
    const double shift = lp - ln;
    const double sum = lp + ln;
    ok = ok && isfinite(shift) && isfinite(sum);
    if (!ok) {
      atomicOr(failed, 1);
      pair_shift[index] = 0.0;
      pair_lower_sum[index] = 0.0;
    } else {
      pair_shift[index] = shift;
      pair_lower_sum[index] = sum;
    }
  }
  const std::int32_t tail_col = 2 * pair_count + index;
  if (tail_col < cols) {
    const bool ok = isfinite(c[tail_col]) && c[tail_col] > 0.0 &&
                    !isnan(l[tail_col]) && !isnan(u[tail_col]) &&
                    l[tail_col] <= u[tail_col] &&
                    l[tail_col] != INFINITY && u[tail_col] != -INFINITY;
    if (!ok) {
      atomicOr(failed, 1);
    }
  }
}

__global__ void kernel_classify_variable_pair_epigraph_rows(
    std::int32_t* failed,
    std::int32_t* edge_count,
    std::int32_t* g_count,
    std::int32_t* first_bad_row,
    unsigned long long* edge_by_row,
    std::int32_t* g_pair_by_row,
    std::int32_t* pair_g_row,
    std::int32_t* pair_tail_col,
    double* pair_g_coefficient,
    const double* pair_shift,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    const double* AL,
    const double* AU,
    std::int32_t rows,
    std::int32_t cols,
    std::int32_t pair_count,
    std::int32_t max_terms) {
  using BlockReduce = cub::BlockReduce<int, GPU_PRESOLVE_THREADS>;
  __shared__ typename BlockReduce::TempStorage edge_storage;
  __shared__ typename BlockReduce::TempStorage g_storage;
  __shared__ typename BlockReduce::TempStorage bad_storage;
  const std::int32_t row =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  bool is_edge = false;
  bool is_g = false;
  bool bad = false;
  unsigned long long packed = NO_EDGE;
  std::int32_t g_pair = -1;
  if (row < rows) {
    const NnzOffset begin = row_ptr[row];
    const NnzOffset len = row_ptr[row + 1] - begin;
    bool ok = valid_row_bounds(AL[row], AU[row]);
    if (ok && len == 3) {
      const NnzOffset p = col_val[begin];
      const std::int32_t n = col_val[begin + 1];
      const std::int32_t t = col_val[begin + 2];
      const double vp = nz_val[begin];
      const double vn = nz_val[begin + 1];
      const double vt = nz_val[begin + 2];
      ok = p >= 0 && p < pair_count && n == p + pair_count &&
           t >= 2 * pair_count && t < cols && isfinite(vp) && vp < 0.0 &&
           vp == vn && isfinite(vt) && vt > 0.0 && isfinite(AL[row]) &&
           AU[row] == INFINITY;
      if (ok) {
        if (atomicCAS(pair_g_row + p, -1, row) != -1) {
          ok = false;
        } else {
          pair_tail_col[p] = t;
          pair_g_coefficient[p] = vp;
          g_pair = p;
          is_g = true;
        }
      }
    } else if (ok) {
      ok = len >= 2 && (len & 1) == 0;
      const std::int32_t terms = len / 2;
      ok = ok && terms >= 1 && terms <= max_terms &&
           terms <= MAX_QUOTIENT_TERMS;
      if (ok) {
        for (std::int32_t k = 0; k < terms; ++k) {
          const NnzOffset p = col_val[begin + k];
          const std::int32_t n = col_val[begin + terms + k];
          const double vp = nz_val[begin + k];
          const double vn = nz_val[begin + terms + k];
          if (p < 0 || p >= pair_count || n != p + pair_count ||
              !isfinite(vp) || vp == 0.0 || vp != -vn) {
            ok = false;
            break;
          }
        }
      }
      if (ok && terms == 2 && AL[row] == 0.0 && AU[row] == 0.0 &&
          nz_val[begin] == -nz_val[begin + 1]) {
        const std::uint32_t a =
            static_cast<std::uint32_t>(col_val[begin]);
        const std::uint32_t b =
            static_cast<std::uint32_t>(col_val[begin + 1]);
        if (a == b || pair_shift[a] != pair_shift[b]) {
          if (a != b && pair_shift[a] != pair_shift[b]) {
            atomicCAS(first_bad_row, -1, row);
          }
          ok = false;
        } else {
          const std::uint32_t lo = min(a, b);
          const std::uint32_t hi = max(a, b);
          packed = (static_cast<unsigned long long>(lo) << 32) |
                   static_cast<unsigned long long>(hi);
          is_edge = true;
        }
      }
    }
    edge_by_row[row] = packed;
    g_pair_by_row[row] = g_pair;
    bad = !ok;
    if (bad) {
      atomicCAS(first_bad_row, -1, row);
    }
  }
  const int block_edges = BlockReduce(edge_storage).Sum(is_edge ? 1 : 0);
  const int block_g = BlockReduce(g_storage).Sum(is_g ? 1 : 0);
  const int block_bad = BlockReduce(bad_storage).Sum(bad ? 1 : 0);
  if (threadIdx.x == 0 && block_edges != 0) {
    atomicAdd(edge_count, block_edges);
  }
  if (threadIdx.x == 0 && block_g != 0) {
    atomicAdd(g_count, block_g);
  }
  if (threadIdx.x == 0 && block_bad != 0) {
    atomicOr(failed, 2);
  }
}

__global__ void kernel_init_variable_pair_epigraph(std::int32_t* parent,
                                            std::int32_t* component_size,
                                            double* plus_upper,
                                            double* minus_upper,
                                            std::int32_t pair_count) {
  const std::int32_t pair =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (pair < pair_count) {
    parent[pair] = pair;
    component_size[pair] = 0;
    plus_upper[pair] = INFINITY;
    minus_upper[pair] = INFINITY;
  }
}

__device__ std::int32_t find_root_compress(std::int32_t* parent,
                                           std::int32_t vertex) {
  std::int32_t current = vertex;
  while (true) {
    const std::int32_t next = parent[current];
    const std::int32_t grand = parent[next];
    if (next == grand) {
      if (current != next) {
        atomicCAS(parent + current, next, grand);
      }
      return next;
    }
    atomicCAS(parent + current, next, grand);
    current = grand;
  }
}

__global__ void kernel_union_variable_pair_epigraph_edges(
    std::int32_t* parent,
    const unsigned long long* edge_by_row,
    std::int32_t rows) {
  const std::int32_t row =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= rows || edge_by_row[row] == NO_EDGE) {
    return;
  }
  const unsigned long long packed = edge_by_row[row];
  const std::int32_t u = static_cast<std::int32_t>(packed >> 32);
  const std::int32_t v =
      static_cast<std::int32_t>(packed & 0xffffffffULL);
  while (true) {
    const std::int32_t ru = find_root_compress(parent, u);
    const std::int32_t rv = find_root_compress(parent, v);
    if (ru == rv) {
      return;
    }
    const std::int32_t high = max(ru, rv);
    const std::int32_t low = min(ru, rv);
    if (atomicCAS(parent + high, high, low) == high) {
      return;
    }
  }
}

__global__ void kernel_compress_variable_pair_epigraph(std::int32_t* parent,
                                                std::int32_t pair_count) {
  const std::int32_t pair =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (pair < pair_count) {
    parent[pair] = find_root_compress(parent, pair);
  }
}

__global__ void kernel_count_variable_pair_epigraph_roots(std::int32_t* root_count,
                                        const std::int32_t* parent,
                                        std::int32_t pair_count) {
  using BlockReduce = cub::BlockReduce<int, GPU_PRESOLVE_THREADS>;
  __shared__ typename BlockReduce::TempStorage storage;
  const std::int32_t pair =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  const int roots = BlockReduce(storage).Sum(
      pair < pair_count && parent[pair] == pair ? 1 : 0);
  if (threadIdx.x == 0 && roots != 0) {
    atomicAdd(root_count, roots);
  }
}

__device__ void atomic_min_double(double* address, double value) {
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits;
  while (value < __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(
        bits, assumed,
        static_cast<unsigned long long>(__double_as_longlong(value)));
    if (old == assumed) {
      break;
    }
  }
}

__global__ void kernel_aggregate_variable_pair_epigraph_uppers(
    double* plus_upper,
    double* minus_upper,
    std::int32_t* component_size,
    const std::int32_t* parent,
    const double* l,
    const double* u,
    std::int32_t pair_count) {
  const std::int32_t pair =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (pair >= pair_count) {
    return;
  }
  const std::int32_t root = parent[pair];
  atomicAdd(component_size + root, 1);
  atomic_min_double(plus_upper + root, u[pair]);
  atomic_min_double(minus_upper + root, u[pair + pair_count]);
}

__global__ void kernel_validate_variable_pair_epigraph(
    std::int32_t* failed,
    const std::int32_t* parent,
    const std::int32_t* pair_g_row,
    const std::int32_t* pair_tail_col,
    const double* pair_g_coefficient,
    const double* pair_shift,
    const double* pair_lower_sum,
    const double* plus_upper,
    const double* minus_upper,
    const double* l,
    const NnzOffset* row_ptr,
    const double* nz_val,
    const double* AL,
    const double* AU,
    std::int32_t pair_count) {
  const std::int32_t pair =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (pair >= pair_count) {
    return;
  }
  const std::int32_t root = parent[pair];
  const std::int32_t row = pair_g_row[pair];
  const std::int32_t root_row = pair_g_row[root];
  bool ok = root >= 0 && root < pair_count && row >= 0 && root_row >= 0 &&
            pair_shift[pair] == pair_shift[root] &&
            pair_lower_sum[pair] == pair_lower_sum[root] &&
            l[pair] == l[root] &&
            l[pair + pair_count] == l[root + pair_count] &&
            pair_tail_col[pair] == pair_tail_col[root] &&
            pair_g_coefficient[pair] == pair_g_coefficient[root] &&
            AL[row] == AL[root_row] && AU[row] == AU[root_row] &&
            nz_val[row_ptr[row] + 2] == nz_val[row_ptr[root_row] + 2];
  if (pair == root) {
    ok = ok && !isnan(plus_upper[root]) && !isnan(minus_upper[root]) &&
         plus_upper[root] >= l[root] &&
         minus_upper[root] >= l[root + pair_count];
  }
  if (!ok) {
    atomicOr(failed, 4);
  }
}

__device__ std::int32_t collect_variable_pair_epigraph_quotient_row(
    std::int32_t* roots,
    double* coefficients,
    double* shift,
    const std::int32_t* parent,
    const double* pair_shift,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    std::int32_t row) {
  const NnzOffset begin = row_ptr[row];
  const std::int32_t terms = (row_ptr[row + 1] - begin) / 2;
  *shift = 0.0;
  for (std::int32_t k = 0; k < terms; ++k) {
    const std::int32_t pair = col_val[begin + k];
    roots[k] = parent[pair];
    coefficients[k] = nz_val[begin + k];
    const double term = coefficients[k] * pair_shift[pair];
    if (!isfinite(coefficients[k]) || !isfinite(term)) {
      return -1;
    }
    *shift += term;
    if (!isfinite(*shift)) {
      return -1;
    }
  }
  for (std::int32_t i = 1; i < terms; ++i) {
    const std::int32_t root = roots[i];
    const double coefficient = coefficients[i];
    std::int32_t j = i;
    while (j > 0 && roots[j - 1] > root) {
      roots[j] = roots[j - 1];
      coefficients[j] = coefficients[j - 1];
      --j;
    }
    roots[j] = root;
    coefficients[j] = coefficient;
  }
  std::int32_t merged = 0;
  for (std::int32_t i = 0; i < terms; ++i) {
    if (merged > 0 && roots[merged - 1] == roots[i]) {
      coefficients[merged - 1] += coefficients[i];
      if (!isfinite(coefficients[merged - 1])) {
        return -1;
      }
    } else {
      roots[merged] = roots[i];
      coefficients[merged] = coefficients[i];
      ++merged;
    }
  }
  std::int32_t active = 0;
  for (std::int32_t i = 0; i < merged; ++i) {
    if (coefficients[i] != 0.0) {
      roots[active] = roots[i];
      coefficients[active] = coefficients[i];
      ++active;
    }
  }
  return active;
}

__global__ void kernel_prepare_variable_pair_epigraph_columns(
    std::uint8_t* keep_col,
    double* new_c,
    double* new_l,
    double* new_u,
    const std::int32_t* parent,
    const double* plus_upper,
    const double* minus_upper,
    const double* l,
    std::int32_t pair_count) {
  const std::int32_t pair =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (pair >= pair_count) {
    return;
  }
  const bool root = parent[pair] == pair;
  const bool keep_plus = root && plus_upper[pair] > l[pair];
  const bool keep_minus =
      root && minus_upper[pair] > l[pair + pair_count];
  keep_col[pair] = keep_plus ? std::uint8_t{1} : std::uint8_t{0};
  keep_col[pair + pair_count] =
      keep_minus ? std::uint8_t{1} : std::uint8_t{0};
  if (root) {
    new_c[pair] = 0.0;
    new_c[pair + pair_count] = 0.0;
    new_l[pair] = l[pair];
    new_l[pair + pair_count] = l[pair + pair_count];
    new_u[pair] = plus_upper[pair];
    new_u[pair + pair_count] = minus_upper[pair];
  }
}

__global__ void kernel_count_variable_pair_epigraph_rewritten_rows(
    std::int32_t* status,
    std::int32_t* removed_count,
    NnzOffset* row_counts,
    std::uint8_t* keep_row,
    double* new_AL,
    double* new_AU,
    const std::int32_t* parent,
    const std::int32_t* component_size,
    const double* pair_shift,
    const double* pair_lower_sum,
    const double* pair_g_coefficient,
    const unsigned long long* edge_by_row,
    const std::int32_t* g_pair_by_row,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    const double* AL,
    const double* AU,
    double feasibility_tol,
    std::int32_t rows) {
  using BlockReduce = cub::BlockReduce<int, GPU_PRESOLVE_THREADS>;
  __shared__ typename BlockReduce::TempStorage storage;
  const std::int32_t row =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  int removed = 0;
  if (row < rows) {
    const std::int32_t pair = g_pair_by_row[row];
    if (pair >= 0) {
      if (parent[pair] != pair) {
        row_counts[row] = 0;
        keep_row[row] = std::uint8_t{0};
        removed = 1;
      } else {
        // The retained row represents every identical component epigraph row.
        // sqrt(component_size) preserves their contribution to the primal
        // residual 2-norm used by HPR's stopping test.
        const double row_weight =
            sqrt(static_cast<double>(component_size[pair]));
        const double lower_unweighted = AL[row];
        const double upper_unweighted = AU[row];
        const double lower = lower_unweighted * row_weight;
        const double upper = isfinite(upper_unweighted)
                                 ? upper_unweighted * row_weight
                                 : upper_unweighted;
        if (component_size[pair] <= 0 || !isfinite(row_weight) ||
            !valid_row_bounds(lower, upper) ||
            (!isfinite(AL[row]) || isfinite(lower)) == false ||
            (!isfinite(AU[row]) || isfinite(upper)) == false) {
          atomicOr(status, 32);
          row_counts[row] = 0;
          keep_row[row] = std::uint8_t{1};
        } else {
          new_AL[row] = lower;
          new_AU[row] = upper;
          row_counts[row] = 3;
          keep_row[row] = std::uint8_t{1};
        }
      }
    } else if (edge_by_row[row] != NO_EDGE) {
      new_AL[row] = 0.0;
      new_AU[row] = 0.0;
      row_counts[row] = 0;
      keep_row[row] = std::uint8_t{0};
      removed = 1;
    } else {
      std::int32_t roots[MAX_QUOTIENT_TERMS];
      double coefficients[MAX_QUOTIENT_TERMS];
      double shift = 0.0;
      const std::int32_t active = collect_variable_pair_epigraph_quotient_row(
          roots, coefficients, &shift, parent, pair_shift, row_ptr, col_val,
          nz_val, row);
      if (active < 0) {
        atomicOr(status, 32);
        row_counts[row] = 0;
        keep_row[row] = std::uint8_t{1};
      } else {
        const double lower = AL[row];
        const double upper = AU[row];
        bool normalized_ok = true;
        if (active > 0 && isfinite(lower) && lower == upper) {
          const double scale = coefficients[0];
          const double normalized_rhs = lower / scale;
          normalized_ok = isfinite(scale) && scale != 0.0 &&
                          isfinite(normalized_rhs);
          for (std::int32_t k = 0; k < active && normalized_ok; ++k) {
            normalized_ok = isfinite(coefficients[k] / scale);
          }
          if (!normalized_ok) {
            atomicOr(status, 32);
            row_counts[row] = 0;
            keep_row[row] = std::uint8_t{1};
          } else {
            new_AL[row] = normalized_rhs;
            new_AU[row] = normalized_rhs;
          }
        } else {
          new_AL[row] = lower;
          new_AU[row] = upper;
        }
        const bool bounds_ok =
            valid_row_bounds(lower, upper) &&
            (!isfinite(AL[row]) || isfinite(lower)) &&
            (!isfinite(AU[row]) || isfinite(upper));
        if (!normalized_ok || !bounds_ok) {
          atomicOr(status, 32);
          row_counts[row] = 0;
          keep_row[row] = std::uint8_t{1};
        } else {
          row_counts[row] = 2 * active;
          if (active == 0) {
            keep_row[row] = std::uint8_t{0};
            removed = 1;
            if (lower > feasibility_tol || upper < -feasibility_tol) {
              atomicOr(status, 1);
            }
          } else {
            keep_row[row] = std::uint8_t{1};
          }
        }
      }
    }
  }
  const int block_removed = BlockReduce(storage).Sum(removed);
  if (threadIdx.x == 0 && block_removed != 0) {
    atomicAdd(removed_count, block_removed);
  }
}

__global__ void kernel_fill_variable_pair_epigraph_rewritten_rows(
    std::int32_t* out_col,
    double* out_val,
    const NnzOffset* out_row_ptr,
    const std::int32_t* parent,
    const std::int32_t* component_size,
    const double* pair_shift,
    const std::int32_t* pair_tail_col,
    const unsigned long long* edge_by_row,
    const std::int32_t* g_pair_by_row,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    const double* AL,
    const double* AU,
    std::int32_t rows,
    std::int32_t pair_count) {
  const std::int32_t row =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= rows || out_row_ptr[row] == out_row_ptr[row + 1]) {
    return;
  }
  const NnzOffset begin = out_row_ptr[row];
  const std::int32_t pair = g_pair_by_row[row];
  if (pair >= 0) {
    const NnzOffset source = row_ptr[row];
    const double row_weight =
        sqrt(static_cast<double>(component_size[pair]));
    out_col[begin] = pair;
    out_val[begin] = row_weight * nz_val[source];
    out_col[begin + 1] = pair + pair_count;
    out_val[begin + 1] = row_weight * nz_val[source + 1];
    out_col[begin + 2] = pair_tail_col[pair];
    out_val[begin + 2] = row_weight * nz_val[source + 2];
    return;
  }
  if (edge_by_row[row] != NO_EDGE) {
    return;
  }
  std::int32_t roots[MAX_QUOTIENT_TERMS];
  double coefficients[MAX_QUOTIENT_TERMS];
  double ignored_shift = 0.0;
  const std::int32_t active = collect_variable_pair_epigraph_quotient_row(
      roots, coefficients, &ignored_shift, parent, pair_shift, row_ptr,
      col_val, nz_val, row);
  if (active <= 0) {
    return;
  }
  double scale = 1.0;
  if (isfinite(AL[row]) && AL[row] == AU[row]) {
    scale = coefficients[0];
  }
  for (std::int32_t k = 0; k < active; ++k) {
    roots[k] = parent[roots[k]];
    out_col[begin + k] = roots[k];
    out_val[begin + k] = coefficients[k] / scale;
    out_col[begin + active + k] = roots[k] + pair_count;
    out_val[begin + active + k] = -coefficients[k] / scale;
  }
}

template <typename T>
void inclusive_scan_i32(const T* input, T* output,
                        std::int32_t count) {
  if (count <= 0) {
    return;
  }
  void* temp = nullptr;
  std::size_t temp_bytes = 0;
  try {
    throw_if_cuda_error(
        cub::DeviceScan::InclusiveSum(temp, temp_bytes, input, output, count),
        "query variable-pair epigraph row scan");
    throw_if_cuda_error(cudaMalloc(&temp, temp_bytes),
                        "cudaMalloc variable-pair epigraph row scan temp");
    throw_if_cuda_error(
        cub::DeviceScan::InclusiveSum(temp, temp_bytes, input, output, count),
        "run variable-pair epigraph row scan");
    cudaFree(temp);
  } catch (...) {
    cudaFree(temp);
    throw;
  }
}

}  // namespace

static bool quick_probe_variable_pair_epigraph_impl(const LPInfoGpu& lp,
                                             const PresolveParams& params,
                                             VariablePairEpigraphLayout* accepted_layout) {
  const auto start = std::chrono::steady_clock::now();
  VariablePairEpigraphLayout layout;
  bool accepted = false;
  const bool structurally_valid =
      params.enable_variable_pair_aggregation && lp.A.rows > 0 && lp.A.cols > 0 &&
      lp.AT.rows == lp.A.cols && lp.AT.cols == lp.A.rows &&
      lp.AT.nnz == lp.A.nnz && lp.A.rowPtr != nullptr &&
      lp.A.colVal != nullptr && lp.A.nzVal != nullptr &&
      lp.AT.rowPtr != nullptr && lp.AT.colVal != nullptr &&
      lp.AT.nzVal != nullptr && lp.c != nullptr && lp.AL != nullptr &&
      lp.AU != nullptr && lp.l != nullptr && lp.u != nullptr;
  if (structurally_valid && infer_variable_pair_epigraph_layout(lp, params, &layout)) {
    const std::int32_t probes = std::max(
        1, std::min(layout.pair_count, params.variable_pair_epigraph_probe_pairs));
    const std::int32_t row_probes = std::min(lp.A.rows, MAX_PROBE_ROWS);
    std::int32_t* state = nullptr;
    try {
      throw_if_cuda_error(cudaMalloc(&state, 2 * sizeof(std::int32_t)),
                          "cudaMalloc variable-pair epigraph quick state");
      throw_if_cuda_error(cudaMemset(state, 0, 2 * sizeof(std::int32_t)),
                          "cudaMemset variable-pair epigraph quick state");
      const int pair_blocks =
          (probes + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
      kernel_quick_variable_pair_epigraph_pair_probe<<<pair_blocks, GPU_PRESOLVE_THREADS>>>(
          state, lp.AT.rowPtr, lp.AT.colVal, lp.AT.nzVal, lp.A.rowPtr,
          lp.A.colVal, lp.A.nzVal, lp.AL, lp.AU, lp.c, lp.l, lp.u,
          layout.pair_count, lp.A.cols, probes);
      throw_if_cuda_error(cudaGetLastError(),
                          "kernel_quick_variable_pair_epigraph_pair_probe");
      const int row_blocks =
          (row_probes + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
      kernel_quick_variable_pair_epigraph_edge_probe<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
          state + 1, lp.A.rowPtr, lp.A.colVal, lp.A.nzVal, lp.AL, lp.AU,
          lp.A.rows, layout.pair_count, row_probes);
      throw_if_cuda_error(cudaGetLastError(),
                          "kernel_quick_variable_pair_epigraph_edge_probe");
      std::int32_t host[2] = {0, 0};
      throw_if_cuda_error(
          cudaMemcpy(host, state, 2 * sizeof(std::int32_t),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy variable-pair epigraph quick state");
      accepted = host[0] == 0 &&
                 host[1] * 100 >= row_probes * MIN_PROBE_MERGE_PERCENT;
      cudaFree(state);
    } catch (...) {
      cudaFree(state);
      throw;
    }
  }
  if (profile_enabled(params)) {
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
    std::cerr << ">>> [variable_pair_aggregation] case=epigraph stage=probe accepted="
              << (accepted ? "true" : "false")
              << " pairs=" << layout.pair_count
              << " tails=" << layout.tail_count
              << " seconds=" << elapsed.count() << "\n";
  }
  if (accepted && accepted_layout != nullptr) {
    *accepted_layout = layout;
  }
  return accepted;
}

bool quick_probe_variable_pair_epigraph(const LPInfoGpu& lp,
                                 const PresolveParams& params) {
  return quick_probe_variable_pair_epigraph_impl(lp, params, nullptr);
}

void free_variable_pair_epigraph_analysis(VariablePairEpigraphAnalysisGpu& analysis) {
  cudaFree(analysis.edge_by_row);
  cudaFree(analysis.g_pair_by_row);
  cudaFree(analysis.pair_g_row);
  cudaFree(analysis.pair_tail_col);
  cudaFree(analysis.pair_g_coefficient);
  cudaFree(analysis.pair_g_lower);
  cudaFree(analysis.pair_shift);
  cudaFree(analysis.parent);
  cudaFree(analysis.component_size);
  cudaFree(analysis.component_plus_upper);
  cudaFree(analysis.component_minus_upper);
  analysis = VariablePairEpigraphAnalysisGpu{};
}

VariablePairEpigraphAnalysisGpu analyze_variable_pair_epigraph(
    const LPInfoGpu& lp,
    const PresolveParams& params) {
  VariablePairEpigraphAnalysisGpu analysis;
  const auto start = std::chrono::steady_clock::now();
  VariablePairEpigraphLayout layout;
  if (!quick_probe_variable_pair_epigraph_impl(lp, params, &layout)) {
    return analysis;
  }
  const std::int32_t rows = lp.A.rows;
  const std::int32_t pairs = layout.pair_count;
  const int row_blocks =
      (rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const int pair_blocks =
      (pairs + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  std::int32_t* state = nullptr;
  auto reject = [&](const char* reason) {
    if (profile_enabled(params)) {
      const std::chrono::duration<double> elapsed =
          std::chrono::steady_clock::now() - start;
      std::cerr << ">>> [variable_pair_aggregation] case=epigraph stage=full accepted=false reason="
                << reason << " seconds=" << elapsed.count() << "\n";
    }
    cudaFree(state);
    state = nullptr;
    free_variable_pair_epigraph_analysis(analysis);
  };
  try {
    analysis.pair_count = pairs;
    analysis.tail_count = layout.tail_count;
    throw_if_cuda_error(
        cudaMalloc(&analysis.edge_by_row,
                   sizeof(unsigned long long) * static_cast<std::size_t>(rows)),
        "cudaMalloc variable-pair epigraph row edges");
    throw_if_cuda_error(
        cudaMalloc(&analysis.g_pair_by_row,
                   sizeof(std::int32_t) * static_cast<std::size_t>(rows)),
        "cudaMalloc variable-pair epigraph row pairs");
    throw_if_cuda_error(
        cudaMalloc(&analysis.pair_g_row,
                   sizeof(std::int32_t) * static_cast<std::size_t>(pairs)),
        "cudaMalloc variable-pair epigraph pair G rows");
    throw_if_cuda_error(
        cudaMalloc(&analysis.pair_tail_col,
                   sizeof(std::int32_t) * static_cast<std::size_t>(pairs)),
        "cudaMalloc variable-pair epigraph pair tail cols");
    throw_if_cuda_error(
        cudaMalloc(&analysis.pair_g_coefficient,
                   sizeof(double) * static_cast<std::size_t>(pairs)),
        "cudaMalloc variable-pair epigraph pair G coefficients");
    throw_if_cuda_error(
        cudaMalloc(&analysis.pair_g_lower,
                   sizeof(double) * static_cast<std::size_t>(pairs)),
        "cudaMalloc variable-pair epigraph pair lower sums");
    throw_if_cuda_error(
        cudaMalloc(&analysis.pair_shift,
                   sizeof(double) * static_cast<std::size_t>(pairs)),
        "cudaMalloc variable-pair epigraph pair shifts");
    throw_if_cuda_error(cudaMalloc(&state, 4 * sizeof(std::int32_t)),
                        "cudaMalloc variable-pair epigraph full state");
    throw_if_cuda_error(cudaMemset(state, 0, 4 * sizeof(std::int32_t)),
                        "cudaMemset variable-pair epigraph full state");
    throw_if_cuda_error(cudaMemset(state + 3, 0xff, sizeof(std::int32_t)),
                        "cudaMemset variable-pair epigraph first bad row");
    throw_if_cuda_error(
        cudaMemset(analysis.edge_by_row, 0xff,
                   sizeof(unsigned long long) * static_cast<std::size_t>(rows)),
        "cudaMemset variable-pair epigraph row edges");
    throw_if_cuda_error(
        cudaMemset(analysis.g_pair_by_row, 0xff,
                   sizeof(std::int32_t) * static_cast<std::size_t>(rows)),
        "cudaMemset variable-pair epigraph row pairs");
    throw_if_cuda_error(
        cudaMemset(analysis.pair_g_row, 0xff,
                   sizeof(std::int32_t) * static_cast<std::size_t>(pairs)),
        "cudaMemset variable-pair epigraph pair G rows");

    kernel_validate_variable_pair_epigraph_columns<<<pair_blocks, GPU_PRESOLVE_THREADS>>>(
        state, analysis.pair_shift, analysis.pair_g_lower, lp.c, lp.l, lp.u,
        pairs, lp.A.cols);
    throw_if_cuda_error(cudaGetLastError(),
                        "kernel_validate_variable_pair_epigraph_columns");
    const std::int32_t max_terms = std::max(
        1, std::min(MAX_QUOTIENT_TERMS,
                    params.variable_pair_epigraph_max_quotient_terms_per_row));
    kernel_classify_variable_pair_epigraph_rows<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
        state, state + 1, state + 2, state + 3, analysis.edge_by_row,
        analysis.g_pair_by_row, analysis.pair_g_row,
        analysis.pair_tail_col, analysis.pair_g_coefficient,
        analysis.pair_shift, lp.A.rowPtr, lp.A.colVal, lp.A.nzVal, lp.AL,
        lp.AU, rows, lp.A.cols, pairs, max_terms);
    throw_if_cuda_error(cudaGetLastError(), "kernel_classify_variable_pair_epigraph_rows");
    std::int32_t host[4] = {0, 0, 0, 0};
    throw_if_cuda_error(
        cudaMemcpy(host, state, 4 * sizeof(std::int32_t),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy variable-pair epigraph census state");
    analysis.edge_count = host[1];
    if (host[0] != 0 || host[2] != pairs) {
      if (profile_enabled(params)) {
        std::cerr << ">>> [variable_pair_aggregation] case=epigraph census_status=" << host[0]
                  << " edge_count=" << host[1]
                  << " g_count=" << host[2]
                  << " expected_g=" << pairs
                  << " first_bad_row=" << host[3] << "\n";
      }
      reject("row_or_pair_validation");
      return VariablePairEpigraphAnalysisGpu{};
    }
    const double potential_gain =
        static_cast<double>(std::min(analysis.edge_count, pairs - 1)) /
        static_cast<double>(pairs);
    if (analysis.edge_count < std::max(1, params.variable_pair_epigraph_min_pairs) ||
        potential_gain < params.variable_pair_epigraph_min_potential_gain) {
      reject("potential_gain");
      return VariablePairEpigraphAnalysisGpu{};
    }

    throw_if_cuda_error(
        cudaMalloc(&analysis.parent,
                   sizeof(std::int32_t) * static_cast<std::size_t>(pairs)),
        "cudaMalloc variable-pair epigraph parents");
    throw_if_cuda_error(
        cudaMalloc(&analysis.component_size,
                   sizeof(std::int32_t) * static_cast<std::size_t>(pairs)),
        "cudaMalloc variable-pair epigraph component sizes");
    throw_if_cuda_error(
        cudaMalloc(&analysis.component_plus_upper,
                   sizeof(double) * static_cast<std::size_t>(pairs)),
        "cudaMalloc variable-pair epigraph plus uppers");
    throw_if_cuda_error(
        cudaMalloc(&analysis.component_minus_upper,
                   sizeof(double) * static_cast<std::size_t>(pairs)),
        "cudaMalloc variable-pair epigraph minus uppers");
    kernel_init_variable_pair_epigraph<<<pair_blocks, GPU_PRESOLVE_THREADS>>>(
        analysis.parent, analysis.component_size,
        analysis.component_plus_upper, analysis.component_minus_upper,
        pairs);
    throw_if_cuda_error(cudaGetLastError(),
                        "kernel_init_variable_pair_epigraph");
    kernel_union_variable_pair_epigraph_edges<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
        analysis.parent, analysis.edge_by_row, rows);
    throw_if_cuda_error(cudaGetLastError(), "kernel_union_variable_pair_epigraph_edges");
    kernel_compress_variable_pair_epigraph<<<pair_blocks, GPU_PRESOLVE_THREADS>>>(
        analysis.parent, pairs);
    throw_if_cuda_error(cudaGetLastError(),
                        "kernel_compress_variable_pair_epigraph");
    throw_if_cuda_error(cudaMemset(state + 3, 0, sizeof(std::int32_t)),
                        "cudaMemset variable-pair epigraph root count");
    kernel_count_variable_pair_epigraph_roots<<<pair_blocks, GPU_PRESOLVE_THREADS>>>(
        state + 3, analysis.parent, pairs);
    throw_if_cuda_error(cudaGetLastError(), "kernel_count_variable_pair_epigraph_roots");
    throw_if_cuda_error(
        cudaMemcpy(&analysis.component_count, state + 3,
                   sizeof(std::int32_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy variable-pair epigraph root count");
    if (analysis.component_count <= 0) {
      reject("component_count");
      return VariablePairEpigraphAnalysisGpu{};
    }
    const double actual_gain =
        1.0 - static_cast<double>(analysis.component_count) /
                  static_cast<double>(pairs);
    if (actual_gain < params.variable_pair_epigraph_min_actual_gain) {
      reject("actual_gain");
      return VariablePairEpigraphAnalysisGpu{};
    }

    kernel_aggregate_variable_pair_epigraph_uppers<<<pair_blocks, GPU_PRESOLVE_THREADS>>>(
        analysis.component_plus_upper, analysis.component_minus_upper,
        analysis.component_size, analysis.parent, lp.l, lp.u, pairs);
    throw_if_cuda_error(cudaGetLastError(),
                        "kernel_aggregate_variable_pair_epigraph_uppers");
    kernel_validate_variable_pair_epigraph<<<pair_blocks, GPU_PRESOLVE_THREADS>>>(
        state, analysis.parent, analysis.pair_g_row,
        analysis.pair_tail_col, analysis.pair_g_coefficient,
        analysis.pair_shift, analysis.pair_g_lower,
        analysis.component_plus_upper, analysis.component_minus_upper,
        lp.l, lp.A.rowPtr, lp.A.nzVal, lp.AL, lp.AU, pairs);
    throw_if_cuda_error(cudaGetLastError(),
                        "kernel_validate_variable_pair_epigraph");
    throw_if_cuda_error(
        cudaMemcpy(host, state, sizeof(std::int32_t),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy variable-pair epigraph component status");
    if (host[0] != 0) {
      reject("component_metadata");
      return VariablePairEpigraphAnalysisGpu{};
    }
    analysis.applicable = true;
    cudaFree(state);
    state = nullptr;
    if (profile_enabled(params)) {
      const std::chrono::duration<double> elapsed =
          std::chrono::steady_clock::now() - start;
      std::cerr << ">>> [variable_pair_aggregation] case=epigraph stage=full accepted=true pairs="
                << pairs << " tails=" << analysis.tail_count
                << " edges=" << analysis.edge_count
                << " components=" << analysis.component_count
                << " gain=" << actual_gain
                << " seconds=" << elapsed.count() << "\n";
    }
    return analysis;
  } catch (...) {
    reject("exception");
    throw;
  }
}

void build_variable_pair_epigraph_plan(PresolvePlanGpu& plan,
                               const LPInfoGpu& lp,
                               VariablePairEpigraphAnalysisGpu& analysis,
                               const PresolveParams& params) {
  if (!analysis.applicable || analysis.pair_count <= 0) {
    return;
  }
  const auto start = std::chrono::steady_clock::now();
  const std::int32_t rows = lp.A.rows;
  const std::int32_t pairs = analysis.pair_count;
  const int row_blocks =
      (rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const int pair_blocks =
      (pairs + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  NnzOffset* row_counts = nullptr;
  std::int32_t* state = nullptr;
  try {
    throw_if_cuda_error(
        cudaMalloc(&row_counts,
                   sizeof(NnzOffset) * static_cast<std::size_t>(rows)),
        "cudaMalloc variable-pair epigraph row counts");
    throw_if_cuda_error(cudaMalloc(&state, 2 * sizeof(std::int32_t)),
                        "cudaMalloc variable-pair epigraph build state");
    throw_if_cuda_error(cudaMemset(state, 0, 2 * sizeof(std::int32_t)),
                        "cudaMemset variable-pair epigraph build state");
    kernel_prepare_variable_pair_epigraph_columns<<<pair_blocks, GPU_PRESOLVE_THREADS>>>(
        plan.keep_col_mask, plan.new_c, plan.new_l, plan.new_u,
        analysis.parent, analysis.component_plus_upper,
        analysis.component_minus_upper, lp.l, pairs);
    throw_if_cuda_error(cudaGetLastError(), "kernel_prepare_variable_pair_epigraph_columns");
    kernel_count_variable_pair_epigraph_rewritten_rows<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
        state, state + 1, row_counts, plan.keep_row_mask, plan.new_AL,
        plan.new_AU, analysis.parent, analysis.component_size,
        analysis.pair_shift,
        analysis.pair_g_lower, analysis.pair_g_coefficient,
        analysis.edge_by_row, analysis.g_pair_by_row, lp.A.rowPtr,
        lp.A.colVal, lp.A.nzVal, lp.AL, lp.AU, params.feasibility_tol, rows);
    throw_if_cuda_error(cudaGetLastError(),
                        "kernel_count_variable_pair_epigraph_rewritten_rows");

    plan.has_new_A = true;
    plan.new_A.rows = rows;
    plan.new_A.cols = lp.A.cols;
    throw_if_cuda_error(
        cudaMalloc(&plan.new_A.rowPtr,
                   sizeof(NnzOffset) * static_cast<std::size_t>(rows + 1)),
        "cudaMalloc variable-pair epigraph rewritten rowPtr");
    throw_if_cuda_error(cudaMemset(plan.new_A.rowPtr, 0,
                                   sizeof(NnzOffset)),
                        "cudaMemset variable-pair epigraph rewritten rowPtr[0]");
    inclusive_scan_i32(row_counts, plan.new_A.rowPtr + 1, rows);
    std::int32_t host_state[2] = {0, 0};
    throw_if_cuda_error(
        cudaMemcpy(&plan.new_A.nnz, plan.new_A.rowPtr + rows,
                   sizeof(NnzOffset), cudaMemcpyDeviceToHost),
        "cudaMemcpy variable-pair epigraph rewritten nnz");
    throw_if_cuda_error(
        cudaMemcpy(host_state, state, 2 * sizeof(std::int32_t),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy variable-pair epigraph build state");
    analysis.removed_rows = host_state[1];
    if ((host_state[0] & 32) != 0) {
      plan.has_change = false;
      cudaFree(row_counts);
      cudaFree(state);
      return;
    }
    if ((host_state[0] & 1) != 0) {
      plan.has_infeasible = true;
      cudaFree(row_counts);
      cudaFree(state);
      return;
    }
    if (plan.new_A.nnz > 0) {
      throw_if_cuda_error(
          cudaMalloc(&plan.new_A.colVal,
                     sizeof(std::int32_t) *
                         static_cast<std::size_t>(plan.new_A.nnz)),
          "cudaMalloc variable-pair epigraph rewritten columns");
      throw_if_cuda_error(
          cudaMalloc(&plan.new_A.nzVal,
                     sizeof(double) *
                         static_cast<std::size_t>(plan.new_A.nnz)),
          "cudaMalloc variable-pair epigraph rewritten values");
      kernel_fill_variable_pair_epigraph_rewritten_rows<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
          plan.new_A.colVal, plan.new_A.nzVal, plan.new_A.rowPtr,
          analysis.parent, analysis.component_size, analysis.pair_shift,
          analysis.pair_tail_col, analysis.edge_by_row,
          analysis.g_pair_by_row, lp.A.rowPtr, lp.A.colVal, lp.A.nzVal,
          lp.AL, lp.AU, rows, pairs);
      throw_if_cuda_error(cudaGetLastError(),
                          "kernel_fill_variable_pair_epigraph_rewritten_rows");
    }

    if (plan.tape.resident) {
      capture_variable_pair_recovery_device(plan, analysis.parent, pairs);
    } else {
      std::vector<std::int32_t> host_parent(static_cast<std::size_t>(pairs));
      std::vector<double> plus_shift(static_cast<std::size_t>(pairs), 0.0);
      std::vector<double> minus_shift(static_cast<std::size_t>(pairs), 0.0);
      throw_if_cuda_error(
          cudaMemcpy(host_parent.data(), analysis.parent,
                     sizeof(std::int32_t) * static_cast<std::size_t>(pairs),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy variable-pair epigraph recovery parents");
      VariablePairAggregationRecoveryStep recovery;
      recovery.plus_cols.resize(static_cast<std::size_t>(pairs));
      recovery.minus_cols.resize(static_cast<std::size_t>(pairs));
      recovery.root_plus_cols.resize(static_cast<std::size_t>(pairs));
      recovery.root_minus_cols.resize(static_cast<std::size_t>(pairs));
      recovery.plus_shift = std::move(plus_shift);
      recovery.minus_shift = std::move(minus_shift);
      for (std::int32_t pair = 0; pair < pairs; ++pair) {
        recovery.plus_cols[static_cast<std::size_t>(pair)] = pair;
        recovery.minus_cols[static_cast<std::size_t>(pair)] = pair + pairs;
        recovery.root_plus_cols[static_cast<std::size_t>(pair)] =
            host_parent[static_cast<std::size_t>(pair)];
        recovery.root_minus_cols[static_cast<std::size_t>(pair)] =
            host_parent[static_cast<std::size_t>(pair)] + pairs;
      }
      plan.variable_pair_aggregation_recovery = std::move(recovery);
    }
    plan.has_variable_pair_aggregation_recovery = true;
    plan.has_change = true;
    plan.has_row_action = analysis.removed_rows > 0;
    plan.has_col_action = analysis.component_count < pairs;

    cudaFree(row_counts);
    cudaFree(state);
    row_counts = nullptr;
    state = nullptr;
    throw_if_cuda_error(cudaDeviceSynchronize(),
                        "variable-pair epigraph component plan synchronize");
    if (profile_enabled(params)) {
      const std::chrono::duration<double> elapsed =
          std::chrono::steady_clock::now() - start;
      std::cerr << ">>> [variable_pair_aggregation] case=epigraph stage=build applied=true pairs="
                << pairs << " components=" << analysis.component_count
                << " removed_rows=" << analysis.removed_rows
                << " explicit_nnz=" << plan.new_A.nnz
                << " seconds=" << elapsed.count()
                << " postsolve=primal-exact/reduced-dual\n";
    }
  } catch (...) {
    cudaFree(row_counts);
    cudaFree(state);
    throw;
  }
}

}  // namespace gpu_presolver::presolve::detail
