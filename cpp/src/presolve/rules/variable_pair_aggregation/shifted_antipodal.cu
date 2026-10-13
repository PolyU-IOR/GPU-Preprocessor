#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/offset.hpp"
#include "shifted_antipodal.hpp"

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
#include <vector>

namespace gpu_presolver::presolve::detail {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;
constexpr int MAX_QUOTIENT_TERMS = 6;
constexpr int MIN_PROBE_COLUMN_DEGREE = 1;
constexpr int MAX_PROBE_COLUMN_DEGREE = 64;
constexpr int MAX_PROBE_ROWS = 1024;
constexpr int MIN_PROBE_MERGE_PERCENT = 5;
constexpr unsigned long long NO_EDGE = ~0ULL;

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

__device__ bool _valid_shifted_pair_bounds(double lower, double upper) {
  return isfinite(lower) && !isnan(upper) && upper >= lower;
}

__device__ bool _valid_row_bounds(double lower, double upper) {
  return !isnan(lower) && !isnan(upper) && lower <= upper &&
         lower != INFINITY && upper != -INFINITY;
}

__global__ void _kernel_quick_variable_pair_shifted_antipodal_probe(
    std::int32_t* failed,
    const NnzOffset* at_row_ptr,
    const std::int32_t* at_col_val,
    const double* at_nz_val,
    const double* c,
    const double* l,
    const double* u,
    std::int32_t pair_count,
    std::int32_t probe_count) {
  const std::int32_t sample =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (sample >= probe_count) {
    return;
  }
  const std::int32_t col = static_cast<std::int32_t>(
      (static_cast<long long>(sample) * pair_count) / probe_count);
  const std::int32_t mate = col + pair_count;
  const NnzOffset begin = at_row_ptr[col];
  const NnzOffset end = at_row_ptr[col + 1];
  const NnzOffset mate_begin = at_row_ptr[mate];
  const NnzOffset mate_end = at_row_ptr[mate + 1];
  const std::int32_t degree = end - begin;
  bool ok = degree >= MIN_PROBE_COLUMN_DEGREE &&
            degree <= MAX_PROBE_COLUMN_DEGREE &&
            degree == mate_end - mate_begin;
  ok = ok && isfinite(c[col]) && c[col] > 0.0 && c[col] == c[mate];
  ok = ok && _valid_shifted_pair_bounds(l[col], u[col]);
  ok = ok && _valid_shifted_pair_bounds(l[mate], u[mate]);
  ok = ok && (!isfinite(u[col]) || isfinite(u[col] - l[col]));
  ok = ok && (!isfinite(u[mate]) || isfinite(u[mate] - l[mate]));
  if (ok) {
    for (NnzOffset p = 0; p < end - begin; ++p) {
      if (at_col_val[begin + p] != at_col_val[mate_begin + p] ||
          at_nz_val[begin + p] != -at_nz_val[mate_begin + p]) {
        ok = false;
        break;
      }
    }
  }
  if (!ok) {
    atomicExch(failed, 1);
  }
}

__global__ void _kernel_quick_merge_row_probe(
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
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  const std::int32_t sample =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  bool is_merge = false;
  if (sample < probe_count) {
    const std::int32_t row = static_cast<std::int32_t>(
        (static_cast<long long>(sample) * rows) / probe_count);
    const NnzOffset begin = row_ptr[row];
    const NnzOffset len = row_ptr[row + 1] - begin;
    if (len == 4 && AL[row] == 0.0 && AU[row] == 0.0) {
      const std::int32_t plus_a = col_val[begin];
      const std::int32_t plus_b = col_val[begin + 1];
      const double value_a = nz_val[begin];
      const double value_b = nz_val[begin + 1];
      is_merge = plus_a >= 0 && plus_a < pair_count && plus_b >= 0 &&
                 plus_b < pair_count && plus_a != plus_b &&
                 col_val[begin + 2] == plus_a + pair_count &&
                 col_val[begin + 3] == plus_b + pair_count &&
                 isfinite(value_a) && isfinite(value_b) && value_a != 0.0 &&
                 value_a == -value_b &&
                 value_a == -nz_val[begin + 2] &&
                 value_b == -nz_val[begin + 3];
    }
  }
  const int block_merges = BlockReduce(reduce_storage).Sum(is_merge ? 1 : 0);
  if (threadIdx.x == 0 && block_merges != 0) {
    atomicAdd(merge_count, block_merges);
  }
}

__global__ void _kernel_validate_pair_metadata(
    std::int32_t* failed,
    double* pair_shift,
    double* pair_constant,
    const double* c,
    const double* l,
    const double* u,
    std::int32_t pair_count) {
  const std::int32_t pair =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (pair >= pair_count) {
    return;
  }
  const std::int32_t mate = pair + pair_count;
  const double rho = c[pair];
  const double lp = l[pair];
  const double ln = l[mate];
  const double up = u[pair];
  const double un = u[mate];
  const bool ok = isfinite(rho) && rho > 0.0 && rho == c[mate] &&
                  _valid_shifted_pair_bounds(lp, up) &&
                  _valid_shifted_pair_bounds(ln, un) &&
                  (!isfinite(up) || isfinite(up - lp)) &&
                  (!isfinite(un) || isfinite(un - ln));
  if (!ok) {
    atomicExch(failed, 1);
    pair_shift[pair] = 0.0;
    pair_constant[pair] = 0.0;
    return;
  }
  const double shift = lp - ln;
  const double constant = rho * (lp + ln);
  if (!isfinite(shift) || !isfinite(constant)) {
    atomicExch(failed, 1);
    pair_shift[pair] = 0.0;
    pair_constant[pair] = 0.0;
    return;
  }
  pair_shift[pair] = shift;
  pair_constant[pair] = constant;
}

__global__ void _kernel_classify_variable_pair_shifted_antipodal_rows(
    std::int32_t* failed,
    std::int32_t* edge_count,
    unsigned long long* edge_by_row,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    const double* AL,
    const double* AU,
    const double* pair_shift,
    std::int32_t rows,
    std::int32_t pair_count,
    std::int32_t max_terms) {
  using BlockReduce = cub::BlockReduce<int, GPU_PRESOLVE_THREADS>;
  __shared__ typename BlockReduce::TempStorage edge_reduce_storage;
  __shared__ typename BlockReduce::TempStorage invalid_reduce_storage;

  const std::int32_t row =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  bool is_edge = false;
  bool invalid = false;
  unsigned long long packed = NO_EDGE;
  if (row < rows) {
    bool ok = _valid_row_bounds(AL[row], AU[row]);
    const NnzOffset begin = row_ptr[row];
    const NnzOffset len = row_ptr[row + 1] - begin;
    ok = ok && len >= 2 && (len & 1) == 0;
    const std::int32_t terms = len / 2;
    ok = ok && terms >= 1 && terms <= max_terms && terms <= MAX_QUOTIENT_TERMS;
    if (ok) {
      for (std::int32_t t = 0; t < terms; ++t) {
        const std::int32_t plus_col = col_val[begin + t];
        const std::int32_t minus_col = col_val[begin + terms + t];
        const double plus_val = nz_val[begin + t];
        const double minus_val = nz_val[begin + terms + t];
        if (plus_col < 0 || plus_col >= pair_count ||
            minus_col != plus_col + pair_count || !isfinite(plus_val) ||
            !isfinite(minus_val) || plus_val == 0.0 ||
            plus_val != -minus_val) {
          ok = false;
          break;
        }
      }
    }
    if (ok && terms == 2 && AL[row] == 0.0 && AU[row] == 0.0 &&
        nz_val[begin] == -nz_val[begin + 1]) {
      const std::uint32_t a = static_cast<std::uint32_t>(col_val[begin]);
      const std::uint32_t b = static_cast<std::uint32_t>(col_val[begin + 1]);
      if (pair_shift[a] != pair_shift[b]) {
        ok = false;
      } else {
        const std::uint32_t lo = min(a, b);
        const std::uint32_t hi = max(a, b);
        packed = (static_cast<unsigned long long>(lo) << 32) |
                 static_cast<unsigned long long>(hi);
        is_edge = true;
      }
    }
    edge_by_row[row] = packed;
    if (!ok) {
      invalid = true;
    }
  }
  const int block_edges =
      BlockReduce(edge_reduce_storage).Sum(is_edge ? 1 : 0);
  const int block_invalid =
      BlockReduce(invalid_reduce_storage).Sum(invalid ? 1 : 0);
  if (threadIdx.x == 0 && block_edges != 0) {
    atomicAdd(edge_count, block_edges);
  }
  if (threadIdx.x == 0 && block_invalid != 0) {
    atomicExch(failed, 1);
  }
}

__global__ void _kernel_init_components(std::int32_t* parent,
                                        double* plus_cap,
                                        double* minus_cap,
                                        double* rho,
                                        std::int32_t pair_count) {
  const std::int32_t pair =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (pair < pair_count) {
    parent[pair] = pair;
    plus_cap[pair] = INFINITY;
    minus_cap[pair] = INFINITY;
    rho[pair] = 0.0;
  }
}

__device__ std::int32_t _find_component_root_compress(std::int32_t* parent,
                                                      std::int32_t vertex) {
  std::int32_t current = vertex;
  // Every successful hook points a larger root at a strictly smaller root.
  // The forest is therefore acyclic and this loop must terminate; keeping it
  // unbounded avoids rejecting a valid component solely because its current
  // parent chain happens to be long.
  while (true) {
    const std::int32_t next = parent[current];
    const std::int32_t grandparent = parent[next];
    if (next == grandparent) {
      if (current != next) {
        atomicCAS(parent + current, next, grandparent);
      }
      return next;
    }
    atomicCAS(parent + current, next, grandparent);
    current = grandparent;
  }
}

__global__ void _kernel_union_variable_pair_shifted_antipodal_edges(
    std::int32_t* failed,
    std::int32_t* parent,
    const unsigned long long* edge_by_row,
    const double* pair_shift,
    std::int32_t rows) {
  const std::int32_t row =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= rows) {
    return;
  }
  const unsigned long long packed = edge_by_row[row];
  if (packed == NO_EDGE) {
    return;
  }
  const std::int32_t u = static_cast<std::int32_t>(packed >> 32);
  const std::int32_t v = static_cast<std::int32_t>(packed & 0xffffffffULL);
  if (pair_shift[u] != pair_shift[v]) {
    atomicOr(failed, 2);
    return;
  }

  while (true) {
    const std::int32_t root_u = _find_component_root_compress(parent, u);
    const std::int32_t root_v = _find_component_root_compress(parent, v);
    if (root_u == root_v) {
      return;
    }
    const std::int32_t high = max(root_u, root_v);
    const std::int32_t low = min(root_u, root_v);
    if (atomicCAS(parent + high, high, low) == high) {
      return;
    }
  }
}

__global__ void _kernel_compress_components(std::int32_t* parent,
                                            std::int32_t pair_count) {
  const std::int32_t pair =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (pair >= pair_count) {
    return;
  }
  const std::int32_t root = _find_component_root_compress(parent, pair);
  parent[pair] = root;
}

__global__ void _kernel_verify_variable_pair_shifted_antipodal(
    std::int32_t* failed,
    const std::int32_t* parent,
    const unsigned long long* edge_by_row,
    const double* pair_shift,
    std::int32_t rows) {
  const std::int32_t row =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= rows) {
    return;
  }
  const unsigned long long packed = edge_by_row[row];
  if (packed == NO_EDGE) {
    return;
  }
  const std::int32_t u = static_cast<std::int32_t>(packed >> 32);
  const std::int32_t v = static_cast<std::int32_t>(packed & 0xffffffffULL);
  if (parent[u] != parent[v] || pair_shift[u] != pair_shift[v]) {
    atomicOr(failed, 8);
  }
}

__global__ void _kernel_count_component_roots(std::int32_t* root_count,
                                              const std::int32_t* parent,
                                              std::int32_t pair_count) {
  using BlockReduce = cub::BlockReduce<int, GPU_PRESOLVE_THREADS>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  const std::int32_t pair =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  const int block_roots = BlockReduce(reduce_storage).Sum(
      pair < pair_count && parent[pair] == pair ? 1 : 0);
  if (threadIdx.x == 0 && block_roots != 0) {
    atomicAdd(root_count, block_roots);
  }
}

__device__ void _atomic_min_double(double* address, double value) {
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits;
  while (value < __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(bits, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(value)));
    if (old == assumed) {
      break;
    }
  }
}

__global__ void _kernel_aggregate_component_metadata(
    double* plus_cap,
    double* minus_cap,
    double* component_rho,
    const std::int32_t* parent,
    const double* c,
    const double* l,
    const double* u,
    std::int32_t pair_count) {
  const std::int32_t pair =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (pair >= pair_count) {
    return;
  }
  const std::int32_t root = parent[pair];
  _atomic_min_double(plus_cap + root, u[pair] - l[pair]);
  _atomic_min_double(minus_cap + root,
                     u[pair + pair_count] - l[pair + pair_count]);
  atomicAdd(component_rho + root, c[pair]);
}

__global__ void _kernel_validate_component_metadata(
    std::int32_t* failed,
    const std::int32_t* parent,
    const double* plus_cap,
    const double* minus_cap,
    const double* component_rho,
    std::int32_t pair_count) {
  const std::int32_t pair =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (pair >= pair_count || parent[pair] != pair) {
    return;
  }
  if (isnan(plus_cap[pair]) || isnan(minus_cap[pair]) ||
      plus_cap[pair] < 0.0 || minus_cap[pair] < 0.0 ||
      !isfinite(component_rho[pair]) || component_rho[pair] <= 0.0) {
    atomicOr(failed, 16);
  }
}

double reduce_sum_double(const double* values, std::int32_t count) {
  if (count <= 0) {
    return 0.0;
  }
  void* temp = nullptr;
  std::size_t temp_bytes = 0;
  double* output = nullptr;
  throw_if_cuda_error(cudaMalloc(&output, sizeof(double)),
                      "cudaMalloc antipodal reduce output");
  try {
    throw_if_cuda_error(
        cub::DeviceReduce::Sum(temp, temp_bytes, values, output, count),
        "query antipodal objective reduction");
    throw_if_cuda_error(cudaMalloc(&temp, temp_bytes),
                        "cudaMalloc antipodal reduction temp");
    throw_if_cuda_error(
        cub::DeviceReduce::Sum(temp, temp_bytes, values, output, count),
        "run antipodal objective reduction");
    double result = 0.0;
    throw_if_cuda_error(cudaMemcpy(&result, output, sizeof(double),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy antipodal objective reduction");
    cudaFree(temp);
    cudaFree(output);
    return result;
  } catch (...) {
    cudaFree(temp);
    cudaFree(output);
    throw;
  }
}

__device__ std::int32_t _collect_quotient_row(
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
  for (std::int32_t t = 0; t < terms; ++t) {
    const std::int32_t pair = col_val[begin + t];
    roots[t] = parent[pair];
    coefficients[t] = nz_val[begin + t];
    const double shift_term = coefficients[t] * pair_shift[pair];
    if (!isfinite(coefficients[t]) || !isfinite(shift_term)) {
      return -1;
    }
    *shift += shift_term;
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

__global__ void _kernel_prepare_variable_pair_shifted_antipodal_columns(
    std::uint8_t* keep_col,
    double* new_c,
    double* new_l,
    double* new_u,
    const std::int32_t* parent,
    const double* plus_cap,
    const double* minus_cap,
    const double* component_rho,
    std::int32_t pair_count) {
  const std::int32_t pair =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (pair >= pair_count) {
    return;
  }
  const bool root = parent[pair] == pair;
  keep_col[pair] = root ? std::uint8_t{1} : std::uint8_t{0};
  keep_col[pair + pair_count] = root ? std::uint8_t{1} : std::uint8_t{0};
  if (root) {
    new_c[pair] = component_rho[pair];
    new_c[pair + pair_count] = component_rho[pair];
    new_l[pair] = 0.0;
    new_l[pair + pair_count] = 0.0;
    new_u[pair] = plus_cap[pair];
    new_u[pair + pair_count] = minus_cap[pair];
  }
}

__global__ void _kernel_count_rewritten_rows(
    std::int32_t* status,
    std::int32_t* zero_row_count,
    NnzOffset* row_counts,
    std::uint8_t* keep_row,
    double* new_AL,
    double* new_AU,
    const std::int32_t* parent,
    const double* pair_shift,
    const unsigned long long* edge_by_row,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    const double* AL,
    const double* AU,
    double feasibility_tol,
    std::int32_t rows) {
  using BlockReduce = cub::BlockReduce<int, GPU_PRESOLVE_THREADS>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  const std::int32_t row =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  int zero_row_vote = 0;
  if (row < rows) {
    if (edge_by_row[row] != NO_EDGE) {
      new_AL[row] = 0.0;
      new_AU[row] = 0.0;
      row_counts[row] = 0;
      keep_row[row] = std::uint8_t{0};
      zero_row_vote = 1;
    } else {
      std::int32_t roots[MAX_QUOTIENT_TERMS];
      double coefficients[MAX_QUOTIENT_TERMS];
      double shift = 0.0;
      const std::int32_t active = _collect_quotient_row(
          roots, coefficients, &shift, parent, pair_shift, row_ptr, col_val,
          nz_val, row);
      if (active < 0) {
        // Safe placeholders: the host observes status before either array is
        // committed to the working LP.
        new_AL[row] = AL[row];
        new_AU[row] = AU[row];
        row_counts[row] = 0;
        keep_row[row] = std::uint8_t{1};
        atomicOr(status, 32);
      } else {
        const double lower = AL[row] - shift;
        const double upper = AU[row] - shift;
        const bool valid_bounds = _valid_row_bounds(lower, upper) &&
            (!isfinite(AL[row]) || isfinite(lower)) &&
            (!isfinite(AU[row]) || isfinite(upper));
        if (!valid_bounds) {
          new_AL[row] = AL[row];
          new_AU[row] = AU[row];
          row_counts[row] = 0;
          keep_row[row] = std::uint8_t{1};
          atomicOr(status, 32);
        } else {
          new_AL[row] = lower;
          new_AU[row] = upper;
          row_counts[row] = 2 * active;
          if (active == 0) {
            keep_row[row] = std::uint8_t{0};
            zero_row_vote = 1;
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
  const int block_zero_rows =
      BlockReduce(reduce_storage).Sum(zero_row_vote);
  if (threadIdx.x == 0 && block_zero_rows != 0) {
    atomicAdd(zero_row_count, block_zero_rows);
  }
}

__global__ void _kernel_fill_rewritten_rows(
    std::int32_t* out_col,
    double* out_val,
    const NnzOffset* out_row_ptr,
    const std::int32_t* parent,
    const double* pair_shift,
    const NnzOffset* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    std::int32_t rows,
    std::int32_t pair_count) {
  const std::int32_t row =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row >= rows) {
    return;
  }
  if (out_row_ptr[row] == out_row_ptr[row + 1]) {
    return;
  }
  std::int32_t roots[MAX_QUOTIENT_TERMS];
  double coefficients[MAX_QUOTIENT_TERMS];
  double ignored_shift = 0.0;
  const std::int32_t active = _collect_quotient_row(
      roots, coefficients, &ignored_shift, parent, pair_shift, row_ptr,
      col_val, nz_val, row);
  if (active <= 0) {
    return;
  }
  const NnzOffset begin = out_row_ptr[row];
  for (std::int32_t i = 0; i < active; ++i) {
    out_col[begin + i] = roots[i];
    out_val[begin + i] = coefficients[i];
    out_col[begin + active + i] = roots[i] + pair_count;
    out_val[begin + active + i] = -coefficients[i];
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
        "query antipodal row scan");
    throw_if_cuda_error(cudaMalloc(&temp, temp_bytes),
                        "cudaMalloc antipodal row scan temp");
    throw_if_cuda_error(
        cub::DeviceScan::InclusiveSum(temp, temp_bytes, input, output, count),
        "run antipodal row scan");
    cudaFree(temp);
  } catch (...) {
    cudaFree(temp);
    throw;
  }
}

}  // namespace

bool quick_probe_variable_pair_shifted_antipodal(const LPInfoGpu& lp,
                                      const PresolveParams& params) {
  if (!params.enable_variable_pair_aggregation || lp.A.cols <= 0 ||
      (lp.A.cols & 1) != 0 || lp.A.rows <= 0 || lp.AT.rows != lp.A.cols ||
      lp.AT.cols != lp.A.rows || lp.AT.nnz != lp.A.nnz ||
      lp.A.rowPtr == nullptr || lp.A.colVal == nullptr ||
      lp.A.nzVal == nullptr || lp.AT.rowPtr == nullptr ||
      lp.AT.colVal == nullptr || lp.AT.nzVal == nullptr || lp.c == nullptr ||
      lp.AL == nullptr || lp.AU == nullptr || lp.l == nullptr ||
      lp.u == nullptr) {
    return false;
  }
  const std::int32_t pair_count = lp.A.cols / 2;
  if (pair_count < std::max(1, params.variable_pair_shifted_antipodal_min_pairs)) {
    return false;
  }
  const std::int32_t probes =
      std::max(1, std::min(pair_count, params.variable_pair_shifted_antipodal_probe_pairs));
  const std::int32_t row_probes = std::min(lp.A.rows, MAX_PROBE_ROWS);
  std::int32_t* probe_state = nullptr;
  throw_if_cuda_error(cudaMalloc(&probe_state, 2 * sizeof(std::int32_t)),
                      "cudaMalloc antipodal quick state");
  try {
    throw_if_cuda_error(cudaMemset(probe_state, 0, 2 * sizeof(std::int32_t)),
                        "cudaMemset antipodal quick state");
    const int blocks = (probes + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    _kernel_quick_variable_pair_shifted_antipodal_probe<<<blocks, GPU_PRESOLVE_THREADS>>>(
        probe_state, lp.AT.rowPtr, lp.AT.colVal, lp.AT.nzVal, lp.c, lp.l, lp.u,
        pair_count, probes);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_quick_variable_pair_shifted_antipodal_probe");
    const int row_blocks =
        (row_probes + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    _kernel_quick_merge_row_probe<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
        probe_state + 1, lp.A.rowPtr, lp.A.colVal, lp.A.nzVal, lp.AL, lp.AU,
        lp.A.rows, pair_count, row_probes);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_quick_merge_row_probe");
    std::int32_t host_state[2] = {0, 0};
    throw_if_cuda_error(cudaMemcpy(host_state, probe_state,
                                   2 * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy antipodal quick state");
    cudaFree(probe_state);
    return host_state[0] == 0 &&
           host_state[1] * 100 >= row_probes * MIN_PROBE_MERGE_PERCENT;
  } catch (...) {
    cudaFree(probe_state);
    throw;
  }
}

void free_variable_pair_shifted_antipodal_analysis(
    VariablePairShiftedAntipodalAnalysisGpu& analysis) {
  cudaFree(analysis.edge_by_row);
  cudaFree(analysis.parent);
  cudaFree(analysis.pair_shift);
  cudaFree(analysis.component_plus_cap);
  cudaFree(analysis.component_minus_cap);
  cudaFree(analysis.component_rho);
  analysis = VariablePairShiftedAntipodalAnalysisGpu{};
}

VariablePairShiftedAntipodalAnalysisGpu analyze_variable_pair_shifted_antipodal(
    const LPInfoGpu& lp,
    const PresolveParams& params) {
  VariablePairShiftedAntipodalAnalysisGpu analysis;
  if (!quick_probe_variable_pair_shifted_antipodal(lp, params)) {
    return analysis;
  }
  const auto analysis_start = std::chrono::steady_clock::now();
  const std::int32_t rows = lp.A.rows;
  const std::int32_t pair_count = lp.A.cols / 2;
  const int row_blocks =
      (rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const int pair_blocks =
      (pair_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  std::int32_t* status = nullptr;
  std::int32_t* edge_count_d = nullptr;
  std::int32_t* root_count_d = nullptr;
  double* pair_constant = nullptr;
  auto reject = [&]() {
    cudaFree(status);
    cudaFree(edge_count_d);
    cudaFree(root_count_d);
    cudaFree(pair_constant);
    status = nullptr;
    edge_count_d = nullptr;
    root_count_d = nullptr;
    pair_constant = nullptr;
    free_variable_pair_shifted_antipodal_analysis(analysis);
  };

  try {
    analysis.pair_count = pair_count;
    throw_if_cuda_error(
        cudaMalloc(&analysis.edge_by_row,
                   sizeof(unsigned long long) * static_cast<std::size_t>(rows)),
        "cudaMalloc antipodal row edges");
    throw_if_cuda_error(
        cudaMalloc(&analysis.pair_shift,
                   sizeof(double) * static_cast<std::size_t>(pair_count)),
        "cudaMalloc antipodal pair shifts");
    throw_if_cuda_error(cudaMalloc(&pair_constant,
                                   sizeof(double) * static_cast<std::size_t>(pair_count)),
                        "cudaMalloc antipodal pair constants");
    throw_if_cuda_error(cudaMalloc(&status, sizeof(std::int32_t)),
                        "cudaMalloc antipodal status");
    throw_if_cuda_error(cudaMalloc(&edge_count_d, sizeof(std::int32_t)),
                        "cudaMalloc antipodal edge count");
    throw_if_cuda_error(cudaMemset(status, 0, sizeof(std::int32_t)),
                        "cudaMemset antipodal status");
    throw_if_cuda_error(cudaMemset(edge_count_d, 0, sizeof(std::int32_t)),
                        "cudaMemset antipodal edge count");

    _kernel_validate_pair_metadata<<<pair_blocks, GPU_PRESOLVE_THREADS>>>(
        status, analysis.pair_shift, pair_constant, lp.c, lp.l, lp.u,
        pair_count);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_validate_pair_metadata");
    const std::int32_t max_terms = std::max(
        1, std::min(MAX_QUOTIENT_TERMS,
                    params.variable_pair_shifted_antipodal_max_quotient_terms_per_row));
    _kernel_classify_variable_pair_shifted_antipodal_rows<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
        status, edge_count_d, analysis.edge_by_row, lp.A.rowPtr, lp.A.colVal,
        lp.A.nzVal, lp.AL, lp.AU, analysis.pair_shift, rows, pair_count,
        max_terms);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_classify_variable_pair_shifted_antipodal_rows");

    std::int32_t host_status = 0;
    throw_if_cuda_error(cudaMemcpy(&host_status, status, sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy antipodal full status");
    throw_if_cuda_error(cudaMemcpy(&analysis.edge_count, edge_count_d,
                                   sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy antipodal edge count");
    if (host_status != 0 || analysis.edge_count < 1) {
      reject();
      return VariablePairShiftedAntipodalAnalysisGpu{};
    }
    const double potential_gain =
        static_cast<double>(std::min(analysis.edge_count, pair_count - 1)) /
        static_cast<double>(pair_count);
    if (analysis.edge_count < std::max(1, params.variable_pair_shifted_antipodal_min_pairs) ||
        potential_gain < params.variable_pair_shifted_antipodal_min_potential_gain) {
      reject();
      return VariablePairShiftedAntipodalAnalysisGpu{};
    }

    // The rejection path above covers common [B,-B] models that have no
    // useful merge graph.  Allocate the four pair-sized DSU/aggregate arrays
    // only after the cheap edge and potential-gain gates have passed.
    throw_if_cuda_error(
        cudaMalloc(&analysis.parent,
                   sizeof(std::int32_t) * static_cast<std::size_t>(pair_count)),
        "cudaMalloc antipodal parents");
    throw_if_cuda_error(
        cudaMalloc(&analysis.component_plus_cap,
                   sizeof(double) * static_cast<std::size_t>(pair_count)),
        "cudaMalloc antipodal plus caps");
    throw_if_cuda_error(
        cudaMalloc(&analysis.component_minus_cap,
                   sizeof(double) * static_cast<std::size_t>(pair_count)),
        "cudaMalloc antipodal minus caps");
    throw_if_cuda_error(
        cudaMalloc(&analysis.component_rho,
                   sizeof(double) * static_cast<std::size_t>(pair_count)),
        "cudaMalloc antipodal component costs");
    throw_if_cuda_error(cudaMalloc(&root_count_d, sizeof(std::int32_t)),
                        "cudaMalloc antipodal root count");
    _kernel_init_components<<<pair_blocks, GPU_PRESOLVE_THREADS>>>(
        analysis.parent, analysis.component_plus_cap,
        analysis.component_minus_cap, analysis.component_rho, pair_count);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_init_components");
    _kernel_union_variable_pair_shifted_antipodal_edges<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
        status, analysis.parent, analysis.edge_by_row, analysis.pair_shift,
        rows);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_union_variable_pair_shifted_antipodal_edges");
    _kernel_compress_components<<<pair_blocks, GPU_PRESOLVE_THREADS>>>(
        analysis.parent, pair_count);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_compress_components");
    _kernel_verify_variable_pair_shifted_antipodal<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
        status, analysis.parent, analysis.edge_by_row, analysis.pair_shift,
        rows);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_verify_variable_pair_shifted_antipodal");
    throw_if_cuda_error(cudaMemset(root_count_d, 0, sizeof(std::int32_t)),
                        "cudaMemset antipodal root count");
    _kernel_count_component_roots<<<pair_blocks, GPU_PRESOLVE_THREADS>>>(
        root_count_d, analysis.parent, pair_count);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_count_component_roots");
    throw_if_cuda_error(cudaMemcpy(&host_status, status, sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy antipodal union status");
    throw_if_cuda_error(cudaMemcpy(&analysis.component_count, root_count_d,
                                   sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy antipodal component count");
    if (host_status != 0 || analysis.component_count <= 0) {
      reject();
      return VariablePairShiftedAntipodalAnalysisGpu{};
    }
    const double actual_gain =
        1.0 - static_cast<double>(analysis.component_count) /
                  static_cast<double>(pair_count);
    if (actual_gain < params.variable_pair_shifted_antipodal_min_actual_gain) {
      reject();
      return VariablePairShiftedAntipodalAnalysisGpu{};
    }

    _kernel_aggregate_component_metadata<<<pair_blocks,
                                           GPU_PRESOLVE_THREADS>>>(
        analysis.component_plus_cap, analysis.component_minus_cap,
        analysis.component_rho, analysis.parent, lp.c, lp.l, lp.u,
        pair_count);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_aggregate_component_metadata");
    _kernel_validate_component_metadata<<<pair_blocks,
                                          GPU_PRESOLVE_THREADS>>>(
        status, analysis.parent, analysis.component_plus_cap,
        analysis.component_minus_cap, analysis.component_rho, pair_count);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_validate_component_metadata");
    throw_if_cuda_error(cudaMemcpy(&host_status, status, sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy antipodal metadata status");
    if (host_status != 0) {
      reject();
      return VariablePairShiftedAntipodalAnalysisGpu{};
    }
    analysis.objective_constant_delta =
        reduce_sum_double(pair_constant, pair_count);
    if (!std::isfinite(analysis.objective_constant_delta) ||
        !std::isfinite(lp.obj_constant +
                       analysis.objective_constant_delta)) {
      reject();
      return VariablePairShiftedAntipodalAnalysisGpu{};
    }
    analysis.applicable = true;

    cudaFree(status);
    cudaFree(edge_count_d);
    cudaFree(root_count_d);
    cudaFree(pair_constant);
    status = nullptr;
    edge_count_d = nullptr;
    root_count_d = nullptr;
    pair_constant = nullptr;
    if (params.verbose || runtime::options().profile_shifted_antipodal) {
      const std::chrono::duration<double> elapsed =
          std::chrono::steady_clock::now() - analysis_start;
      std::cerr << ">>> [variable_pair_aggregation] case=shifted_antipodal full-check=true pairs="
                << pair_count << " merge_edges=" << analysis.edge_count
                << " components=" << analysis.component_count
                << " gain=" << actual_gain << " analysis="
                << elapsed.count() << "s\n";
    }
    return analysis;
  } catch (...) {
    reject();
    throw;
  }
}

void build_variable_pair_shifted_antipodal_plan(PresolvePlanGpu& plan,
                                    const LPInfoGpu& lp,
                                    VariablePairShiftedAntipodalAnalysisGpu& analysis,
                                    const PresolveParams& params) {
  if (!analysis.applicable || analysis.pair_count <= 0) {
    return;
  }
  const std::int32_t rows = lp.A.rows;
  const std::int32_t pair_count = analysis.pair_count;
  const int row_blocks =
      (rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const int pair_blocks =
      (pair_count + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  NnzOffset* row_counts = nullptr;
  std::int32_t* status = nullptr;
  std::int32_t* zero_rows_d = nullptr;
  try {
    throw_if_cuda_error(
        cudaMalloc(&row_counts,
                   sizeof(NnzOffset) * static_cast<std::size_t>(rows)),
        "cudaMalloc antipodal row counts");
    throw_if_cuda_error(cudaMalloc(&status, sizeof(std::int32_t)),
                        "cudaMalloc antipodal rewrite status");
    throw_if_cuda_error(cudaMalloc(&zero_rows_d, sizeof(std::int32_t)),
                        "cudaMalloc antipodal zero rows");
    throw_if_cuda_error(cudaMemset(status, 0, sizeof(std::int32_t)),
                        "cudaMemset antipodal rewrite status");
    throw_if_cuda_error(cudaMemset(zero_rows_d, 0, sizeof(std::int32_t)),
                        "cudaMemset antipodal zero rows");

    _kernel_prepare_variable_pair_shifted_antipodal_columns<<<pair_blocks, GPU_PRESOLVE_THREADS>>>(
        plan.keep_col_mask, plan.new_c, plan.new_l, plan.new_u,
        analysis.parent, analysis.component_plus_cap,
        analysis.component_minus_cap, analysis.component_rho, pair_count);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_prepare_variable_pair_shifted_antipodal_columns");
    _kernel_count_rewritten_rows<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
        status, zero_rows_d, row_counts, plan.keep_row_mask, plan.new_AL,
        plan.new_AU, analysis.parent, analysis.pair_shift,
        analysis.edge_by_row, lp.A.rowPtr, lp.A.colVal, lp.A.nzVal, lp.AL,
        lp.AU, params.feasibility_tol, rows);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_count_rewritten_rows");

    plan.has_new_A = true;
    plan.new_A.rows = rows;
    plan.new_A.cols = lp.A.cols;
    throw_if_cuda_error(
        cudaMalloc(&plan.new_A.rowPtr,
                   sizeof(NnzOffset) *
                       static_cast<std::size_t>(rows + 1)),
        "cudaMalloc antipodal rewritten rowPtr");
    throw_if_cuda_error(cudaMemset(plan.new_A.rowPtr, 0, sizeof(NnzOffset)),
                        "cudaMemset antipodal rewritten rowPtr[0]");
    inclusive_scan_i32(row_counts, plan.new_A.rowPtr + 1, rows);
    throw_if_cuda_error(cudaMemcpy(&plan.new_A.nnz, plan.new_A.rowPtr + rows,
                                   sizeof(NnzOffset), cudaMemcpyDeviceToHost),
                        "cudaMemcpy antipodal rewritten nnz");
    throw_if_cuda_error(cudaMemcpy(&analysis.zero_rows, zero_rows_d,
                                   sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy antipodal zero rows");
    std::int32_t host_status = 0;
    throw_if_cuda_error(cudaMemcpy(&host_status, status, sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy antipodal rewrite status");
    if (host_status != 0) {
      plan.has_infeasible = true;
      cudaFree(row_counts);
      cudaFree(status);
      cudaFree(zero_rows_d);
      return;
    }
    if (plan.new_A.nnz > 0) {
      throw_if_cuda_error(
          cudaMalloc(&plan.new_A.colVal,
                     sizeof(std::int32_t) *
                         static_cast<std::size_t>(plan.new_A.nnz)),
          "cudaMalloc antipodal rewritten columns");
      throw_if_cuda_error(
          cudaMalloc(&plan.new_A.nzVal,
                     sizeof(double) *
                         static_cast<std::size_t>(plan.new_A.nnz)),
          "cudaMalloc antipodal rewritten values");
      _kernel_fill_rewritten_rows<<<row_blocks, GPU_PRESOLVE_THREADS>>>(
          plan.new_A.colVal, plan.new_A.nzVal, plan.new_A.rowPtr,
          analysis.parent, analysis.pair_shift, lp.A.rowPtr, lp.A.colVal,
          lp.A.nzVal, rows, pair_count);
      throw_if_cuda_error(cudaGetLastError(),
                          "_kernel_fill_rewritten_rows");
    }

    if (plan.tape.resident) {
      capture_variable_pair_recovery_device(plan, analysis.parent, pair_count, lp.l, lp.l + pair_count);
    } else {
      std::vector<std::int32_t> host_parent(
          static_cast<std::size_t>(pair_count));
      std::vector<double> plus_shift(static_cast<std::size_t>(pair_count));
      std::vector<double> minus_shift(static_cast<std::size_t>(pair_count));
      throw_if_cuda_error(
          cudaMemcpy(host_parent.data(), analysis.parent,
                     sizeof(std::int32_t) * static_cast<std::size_t>(pair_count),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy antipodal recovery parents");
      throw_if_cuda_error(
          cudaMemcpy(plus_shift.data(), lp.l,
                     sizeof(double) * static_cast<std::size_t>(pair_count),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy antipodal recovery plus lower");
      throw_if_cuda_error(
          cudaMemcpy(minus_shift.data(), lp.l + pair_count,
                     sizeof(double) * static_cast<std::size_t>(pair_count),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy antipodal recovery minus lower");

      VariablePairAggregationRecoveryStep recovery;
      recovery.plus_cols.resize(static_cast<std::size_t>(pair_count));
      recovery.minus_cols.resize(static_cast<std::size_t>(pair_count));
      recovery.root_plus_cols.resize(static_cast<std::size_t>(pair_count));
      recovery.root_minus_cols.resize(static_cast<std::size_t>(pair_count));
      recovery.plus_shift = std::move(plus_shift);
      recovery.minus_shift = std::move(minus_shift);
      for (std::int32_t pair = 0; pair < pair_count; ++pair) {
        recovery.plus_cols[static_cast<std::size_t>(pair)] = pair;
        recovery.minus_cols[static_cast<std::size_t>(pair)] = pair + pair_count;
        recovery.root_plus_cols[static_cast<std::size_t>(pair)] =
            host_parent[static_cast<std::size_t>(pair)];
        recovery.root_minus_cols[static_cast<std::size_t>(pair)] =
            host_parent[static_cast<std::size_t>(pair)] + pair_count;
      }
      plan.variable_pair_aggregation_recovery = std::move(recovery);
    }
    plan.has_variable_pair_aggregation_recovery = true;
    plan.obj_constant_delta += analysis.objective_constant_delta;
    plan.has_change = true;
    plan.has_row_action = analysis.zero_rows > 0;
    plan.has_col_action = analysis.component_count < pair_count;

    cudaFree(row_counts);
    cudaFree(status);
    cudaFree(zero_rows_d);
    row_counts = nullptr;
    status = nullptr;
    zero_rows_d = nullptr;

    throw_if_cuda_error(cudaDeviceSynchronize(),
                        "antipodal component plan synchronize");
  } catch (...) {
    cudaFree(row_counts);
    cudaFree(status);
    cudaFree(zero_rows_d);
    throw;
  }
}

}  // namespace gpu_presolver::presolve::detail
