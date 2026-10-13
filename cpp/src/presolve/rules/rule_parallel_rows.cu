#include "parallel_signature_common.cuh"
#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/live_matrix.cuh"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/duplicate_rule_workspace.hpp"
#include "gpu_presolver/presolve/rules/rule_parallel_rows.hpp"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "gpu_presolver/presolve/gpu_presolve_kernels.hpp"
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cfloat>
#include <climits>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

constexpr int GPU_PRESOLVE_THREADS = 256;
constexpr unsigned long long FNV_OFFSET = 0xcbf29ce484222325ULL;
constexpr unsigned long long FNV_PRIME = 0x100000001b3ULL;

template <typename T>
std::size_t reserve_scratch(std::size_t& bytes, std::size_t count) {
  const std::size_t alignment = alignof(T);
  bytes = (bytes + alignment - 1) & ~(alignment - 1);
  const std::size_t offset = bytes;
  bytes += sizeof(T) * count;
  return offset;
}

template <typename T>
T* scratch_at(void* storage, std::size_t offset) {
  return reinterpret_cast<T*>(static_cast<unsigned char*>(storage) + offset);
}

struct HashRowLess {
  __host__ __device__ bool operator()(const thrust::tuple<unsigned long long, std::int32_t>& lhs,
                                      const thrust::tuple<unsigned long long, std::int32_t>& rhs) const {
    const unsigned long long lhs_hash = thrust::get<0>(lhs);
    const unsigned long long rhs_hash = thrust::get<0>(rhs);
    if (lhs_hash != rhs_hash) {
      return lhs_hash < rhs_hash;
    }
    return thrust::get<1>(lhs) < thrust::get<1>(rhs);
  }
};

struct ParallelRowTapeRecord {
  std::int32_t step;
  std::int32_t kept_row;
  std::int32_t deleted_row;
  double ratio;
  double kept_old_AL;
  double kept_old_AU;
  double deleted_old_AL;
  double deleted_old_AU;
};

struct RecoveryOrder {
  __host__ __device__ bool operator()(const ParallelRowTapeRecord &a,
                                      const ParallelRowTapeRecord &b) const {
    return a.step != b.step ? a.step < b.step : a.deleted_row < b.deleted_row;
  }
};
__global__ void pack_recovery_segment(int *types, unsigned char *modes, NnzOffset *is,
                                      NnzOffset *vs, int *indices, double *vals,
                                      const ParallelRowTapeRecord *records, int count) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k > count)
    return;
  is[k] = NnzOffset(2) * k;
  vs[k] = NnzOffset(5) * k;
  if (k == count)
    return;
  const auto r = records[k];
  types[k] = int(PostsolveReductionType::ParallelRow);
  modes[k] = unsigned(PostsolveDualMode::Exact);
  indices[NnzOffset(2) * k + 0] = r.kept_row;
  indices[NnzOffset(2) * k + 1] = r.deleted_row;
  vals[NnzOffset(5) * k + 0] = r.ratio;
  vals[NnzOffset(5) * k + 1] = r.kept_old_AL;
  vals[NnzOffset(5) * k + 2] = r.kept_old_AU;
  vals[NnzOffset(5) * k + 3] = r.deleted_old_AL;
  vals[NnzOffset(5) * k + 4] = r.deleted_old_AU;
}

void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}

void append_postsolve_record(PostsolveTape& tape,
                             PostsolveReductionType type,
                             const std::int32_t* indices,
                             std::size_t index_count,
                             const double* vals,
                             std::size_t value_count,
                             PostsolveDualMode dual_mode) {
  tape.types.push_back(static_cast<std::int32_t>(type));
  tape.indices.insert(tape.indices.end(), indices, indices + index_count);
  tape.vals.insert(tape.vals.end(), vals, vals + value_count);
  tape.index_starts.push_back(static_cast<NnzOffset>(tape.indices.size()));
  tape.value_starts.push_back(static_cast<NnzOffset>(tape.vals.size()));
  tape.dual_modes.push_back(static_cast<std::uint8_t>(dual_mode));
}

__device__ __host__ unsigned long long _parallel_row_hash_mix(unsigned long long h,
                                                              unsigned long long x) {
  return (h ^ x) * FNV_PRIME;
}

__device__ unsigned long long _double_bits(double value) {
  return static_cast<unsigned long long>(__double_as_longlong(value));
}

__global__ void _kernel_parallel_row_hashes(unsigned long long* row_hash,
                                            const std::uint8_t* keep_row,
                                            const NnzOffset* row_ptr,
                                            const std::int32_t* col_val,
                                            const double* nz_val,
                                            double coeff_tol,
                                            std::int32_t m, int max_length, bool use_xor,
                                            const std::uint8_t* keep_col = nullptr,
                                            const std::uint8_t* candidates = nullptr) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < m && (max_length == 0 || row_ptr[i+1]-row_ptr[i] <= max_length)) {
    if (candidates && !candidates[i]) return;
    if (keep_row[i] == std::uint8_t{0}) {
      row_hash[i] = static_cast<unsigned long long>(i);
      return;
    }

    LiveRowCursor first(row_ptr[i], row_ptr[i+1], col_val, keep_col);
    const NnzOffset start_i = first.pos;
    const NnzOffset stop_i = row_ptr[i + 1];
    NnzOffset len_i = stop_i - start_i;
    if (keep_col) {
      len_i = 0;
      for (auto cursor = first; cursor.pos < cursor.end; cursor.advance()) ++len_i;
    }
    if (len_i <= 0) {
      row_hash[i] = _parallel_row_hash_mix(FNV_OFFSET, static_cast<unsigned long long>(i));
      return;
    }

    const double pivot = nz_val[start_i];
    if (fabs(pivot) <= coeff_tol) {
      row_hash[i] = _parallel_row_hash_mix(FNV_OFFSET, static_cast<unsigned long long>(i));
      return;
    }

    unsigned long long h = _parallel_row_hash_mix(FNV_OFFSET, static_cast<unsigned long long>(len_i));
    unsigned long long aggregate=0;
    for (auto cursor = first; cursor.pos < cursor.end; cursor.advance()) {
      const NnzOffset p = cursor.pos;
      const double norm = nz_val[p] / pivot;
      if (use_xor) {
        auto entry=_parallel_row_hash_mix(FNV_OFFSET, static_cast<unsigned long long>(col_val[p]));
        aggregate^=_parallel_row_hash_mix(entry,_double_bits(norm));
      } else {
        h = _parallel_row_hash_mix(h, static_cast<unsigned long long>(col_val[p]));
        h = _parallel_row_hash_mix(h, _double_bits(norm));
      }
    }
    row_hash[i] = use_xor ? _parallel_row_hash_mix(h,aggregate) : h;
  }
}

__global__ void _kernel_parallel_row_hashes_warp(unsigned long long* row_hash,
                                                 const std::uint8_t* keep_row,
                                                 const NnzOffset* row_ptr,
                                                 const std::int32_t* col_val,
                                                 const double* nz_val,
                                                 double coeff_tol,
                                                 std::int32_t m,
                                                 const std::uint8_t* candidates = nullptr) {
  constexpr std::int32_t warp_size = 32;
  const std::int32_t lane =
      static_cast<std::int32_t>(threadIdx.x) & (warp_size - 1);
  const std::int32_t warps_per_block =
      static_cast<std::int32_t>(blockDim.x) / warp_size;
  const std::int32_t warp =
      static_cast<std::int32_t>(threadIdx.x) / warp_size;
  const std::int32_t i =
      static_cast<std::int32_t>(blockIdx.x) * warps_per_block + warp;
  if (i >= m) {
    return;
  }
  if (candidates && !candidates[i]) return;

  if (keep_row[i] == std::uint8_t{0}) {
    if (lane == 0) {
      row_hash[i] = static_cast<unsigned long long>(i);
    }
    return;
  }

  const NnzOffset start_i = row_ptr[i];
  const NnzOffset stop_i = row_ptr[i + 1];
  const NnzOffset len_i = stop_i - start_i;
  if (len_i <= 0) {
    if (lane == 0) {
      row_hash[i] = _parallel_row_hash_mix(
          FNV_OFFSET, static_cast<unsigned long long>(i));
    }
    return;
  }

  const double pivot = nz_val[start_i];
  if (fabs(pivot) <= coeff_tol) {
    if (lane == 0) {
      row_hash[i] = _parallel_row_hash_mix(
          FNV_OFFSET, static_cast<unsigned long long>(i));
    }
    return;
  }

  unsigned long long aggregate = 0ULL;
  for (NnzOffset p = start_i + lane; p < stop_i; p += warp_size) {
    unsigned long long entry = _parallel_row_hash_mix(
        FNV_OFFSET, static_cast<unsigned long long>(col_val[p]));
    const double norm = nz_val[p] / pivot;
    entry = _parallel_row_hash_mix(entry, _double_bits(norm));
    aggregate ^= entry;
  }
  for (int offset = warp_size / 2; offset > 0; offset /= 2) {
    aggregate ^= __shfl_down_sync(0xffffffffU, aggregate, offset);
  }
  if (lane == 0) {
    const unsigned long long seed = _parallel_row_hash_mix(
        FNV_OFFSET, static_cast<unsigned long long>(len_i));
    row_hash[i] = _parallel_row_hash_mix(seed, aggregate);
  }
}

struct HashXor {
  __device__ unsigned long long operator()(unsigned long long a,unsigned long long b) const { return a^b; }
};
template<int Width>
__global__ void grouped_parallel_row_hash(
    unsigned long long* output,const std::uint8_t* keep,const NnzOffset* ptr,const int* indices,
    const double* values,double tol,const int* rows,const int* count,bool use_xor,const std::uint8_t* keep_col,
    const std::uint8_t* candidates = nullptr) {
  const int lane=Width==32 ? threadIdx.x&31 : threadIdx.x;
  const int group=Width==32 ? blockIdx.x*8+threadIdx.x/32 : blockIdx.x;
  const NnzOffset stride=Width==32 ? gridDim.x*8 : gridDim.x;
  const NnzOffset base=Width==32 ? (threadIdx.x/32)*32 : 0;
  __shared__ unsigned long long norms[256];
  __shared__ int columns[256];
  using Reduce=cub::BlockReduce<unsigned long long,256>;
  __shared__ typename Reduce::TempStorage storage;
  using IndexReduce = cub::BlockReduce<NnzOffset,256>;
  __shared__ typename IndexReduce::TempStorage index_storage;
  __shared__ NnzOffset live_first, live_length;
  for(NnzOffset q=group;q<*count;q+=stride) {
    const int row=rows[q];
    if (candidates && !candidates[row]) continue;
    const NnzOffset start=ptr[row], last=ptr[row+1];
    NnzOffset first=start, length=last-start;
    if (keep_col && keep[row]) {
      first=last; length=0;
      for (NnzOffset p=start+lane;p<last;p+=Width) if(keep_col[indices[p]]) {
        first=first<p?first:p; ++length;
      }
      if constexpr(Width==32) {
        for(int off=16;off;off/=2) {
          length+=__shfl_down_sync(0xffffffffu,length,off);
          const auto other=__shfl_down_sync(0xffffffffu,first,off);
          first=first<other?first:other;
        }
        first=__shfl_sync(0xffffffffu,first,0);
        length=__shfl_sync(0xffffffffu,length,0);
      } else {
        length=IndexReduce(index_storage).Sum(length); __syncthreads();
        if(lane==0) live_length=length;
        first=IndexReduce(index_storage).Reduce(first,thrust::minimum<NnzOffset>{}); __syncthreads();
        if(lane==0) live_first=first;
        __syncthreads(); first=live_first; length=live_length;
      }
    }
    if(!keep[row] || length==0 || fabs(values[first])<=tol) {
      if(lane==0) output[row]=!keep[row] ? static_cast<unsigned long long>(row)
          : _parallel_row_hash_mix(FNV_OFFSET,static_cast<unsigned long long>(row));
      continue;
    }
    const double pivot=values[first];
    unsigned long long h=_parallel_row_hash_mix(FNV_OFFSET,static_cast<unsigned long long>(length));
    if(use_xor) {
      unsigned long long aggregate=0;
      for(NnzOffset p=first+lane;p<last;p+=Width) {
        if(keep_col && !keep_col[indices[p]]) continue;
        const auto entry=_parallel_row_hash_mix(FNV_OFFSET,static_cast<unsigned long long>(indices[p]));
        aggregate^=_parallel_row_hash_mix(entry,_double_bits(values[p]/pivot));
      }
      if constexpr(Width==32) {
        for(int off=16;off;off/=2) aggregate^=__shfl_down_sync(0xffffffffu,aggregate,off);
      } else { aggregate=Reduce(storage).Reduce(aggregate,HashXor{});__syncthreads(); }
      if(lane==0) output[row]=_parallel_row_hash_mix(h,aggregate);
    } else {
      // Parallelize loads/divisions, but keep the legacy FNV fold in CSR order.
      for(NnzOffset chunk=first;chunk<last;chunk+=Width) {
        const NnzOffset p=chunk+lane;
        const bool live=p<last && (!keep_col || keep_col[indices[p]]);
        columns[threadIdx.x]=live?indices[p]:-1;
        if(live) norms[threadIdx.x]=_double_bits(values[p]/pivot);
        if constexpr(Width==32) __syncwarp(); else __syncthreads();
        if(lane==0) for(int j=0;j<Width && chunk+j<last;++j) {
          if(columns[base+j]<0) continue;
          h=_parallel_row_hash_mix(h,static_cast<unsigned long long>(columns[base+j]));
          h=_parallel_row_hash_mix(h,norms[base+j]);
        }
        if constexpr(Width==32) __syncwarp(); else __syncthreads();
      }
      if(lane==0) output[row]=h;
    }
    if constexpr(Width!=32) __syncthreads();
  }
}

__device__ bool _parallel_row_ratio(std::int32_t i,
                                    std::int32_t k,
                                    const NnzOffset* row_ptr,
                                    const std::int32_t* col_val,
                                    const double* nz_val,
                                    double coeff_tol,
                                    double* ratio_out, const std::uint8_t* keep_col) {
  LiveRowCursor left(row_ptr[i], row_ptr[i+1], col_val, keep_col);
  LiveRowCursor right(row_ptr[k], row_ptr[k+1], col_val, keep_col);
  if (left.pos == left.end && right.pos == right.end) {
    *ratio_out = 1.0;
    return true;
  }

  double ratio = 0.0;
  bool ratio_set = false;
  for (; left.pos < left.end && right.pos < right.end; left.advance(), right.advance()) {
    const NnzOffset p_i = left.pos, p_k = right.pos;
    if (col_val[p_i] != col_val[p_k]) {
      return false;
    }
    const double a_i = nz_val[p_i];
    const double a_k = nz_val[p_k];
    if (!ratio_set) {
      if (fabs(a_k) <= coeff_tol) {
        return false;
      }
      ratio = a_i / a_k;
      ratio_set = true;
    }
    if (fabs(a_i - ratio * a_k) > coeff_tol) {
      return false;
    }
  }
  *ratio_out = ratio;
  return ratio_set && left.pos == left.end && right.pos == right.end;
}

__device__ void _scaled_interval_for_parallel_row(double lower,
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

__device__ bool _intervals_disjoint(double l1, double u1, double l2, double u2, double tol) {
  return fmax(l1, l2) > fmin(u1, u2) + tol;
}

__device__ bool _interval_contains(double outer_l,
                                   double outer_u,
                                   double inner_l,
                                   double inner_u,
                                   double tol) {
  return outer_l <= inner_l + tol && inner_u <= outer_u + tol;
}

__device__ bool _row_interval_is_equality(double lower, double upper, double tol) {
  return isfinite(lower) && isfinite(upper) && fabs(lower - upper) <= tol;
}

__global__ void _kernel_parallel_row_groups(std::int32_t* status_flag,
                                            std::uint8_t* row_delete,
                                            std::int32_t* merge_to,
                                            double* merge_ratio,
                                            double* kept_old_AL,
                                            double* kept_old_AU,
                                            double* deleted_old_AL,
                                            double* deleted_old_AU,
                                            std::int32_t* merge_step,
                                            const unsigned long long* sorted_hash,
                                            const std::int32_t* sorted_rows,
                                            const std::uint8_t* keep_row,
                                            const std::uint8_t* keep_col,
                                            double* AL,
                                            double* AU,
                                            const NnzOffset* row_ptr,
                                            const std::int32_t* col_val,
                                            const double* nz_val,
                                            double coeff_tol,
                                            double interval_tol,
                                            std::int32_t m) {
  const std::int32_t s = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (s < m) {
    const unsigned long long hs = sorted_hash[s];
    if (s > 0 && sorted_hash[s - 1] == hs) {
      return;
    }

    std::int32_t e = s;
    while (e + 1 < m && sorted_hash[e + 1] == hs) {
      ++e;
    }
    if (e <= s) {
      return;
    }

    std::int32_t rep_seed = -1;
    bool rep_seed_is_eq = false;
    for (std::int32_t a = s; a <= e; ++a) {
      const std::int32_t row = sorted_rows[a];
      if (keep_row[row] == std::uint8_t{0}) {
        continue;
      }
      const double row_l = AL[row];
      const double row_u = AU[row];
      const bool row_is_eq = _row_interval_is_equality(row_l, row_u, interval_tol);
      if (rep_seed < 0 || (row_is_eq && !rep_seed_is_eq)) {
        rep_seed = row;
        rep_seed_is_eq = row_is_eq;
      }
    }
    if (rep_seed < 0) {
      return;
    }

    std::int32_t rep = rep_seed;
    const double rep_seed_l = AL[rep_seed];
    const double rep_seed_u = AU[rep_seed];
    double rep_scaled_l = rep_seed_l;
    double rep_scaled_u = rep_seed_u;
    for (std::int32_t a = s; a <= e; ++a) {
      const std::int32_t row = sorted_rows[a];
      if (row == rep_seed || keep_row[row] == std::uint8_t{0}) {
        continue;
      }

      double ratio_seed = 0.0;
      if (!_parallel_row_ratio(rep_seed, row, row_ptr, col_val, nz_val, coeff_tol, &ratio_seed, keep_col)) {
        continue;
      }

      double row_scaled_l = 0.0;
      double row_scaled_u = 0.0;
      _scaled_interval_for_parallel_row(AL[row], AU[row], ratio_seed, &row_scaled_l, &row_scaled_u);
      if (_interval_contains(rep_scaled_l, rep_scaled_u, row_scaled_l, row_scaled_u, interval_tol)) {
        rep = row;
        rep_scaled_l = row_scaled_l;
        rep_scaled_u = row_scaled_u;
      }
    }

    double rep_l = AL[rep];
    double rep_u = AU[rep];
    bool found_parallel_peer = false;
    std::int32_t step = 0;
    for (std::int32_t a = s; a <= e; ++a) {
      const std::int32_t row = sorted_rows[a];
      if (row == rep || keep_row[row] == std::uint8_t{0}) {
        continue;
      }

      double ratio = 0.0;
      if (!_parallel_row_ratio(rep, row, row_ptr, col_val, nz_val, coeff_tol, &ratio, keep_col)) {
        continue;
      }
      found_parallel_peer = true;

      double row_l = 0.0;
      double row_u = 0.0;
      _scaled_interval_for_parallel_row(AL[row], AU[row], ratio, &row_l, &row_u);
      if (_intervals_disjoint(rep_l, rep_u, row_l, row_u, interval_tol)) {
        atomicMax(&status_flag[0], 1);
        return;
      }

      ++step;
      merge_to[row] = rep;
      merge_ratio[row] = ratio;
      kept_old_AL[row] = rep_l;
      kept_old_AU[row] = rep_u;
      deleted_old_AL[row] = AL[row];
      deleted_old_AU[row] = AU[row];
      merge_step[row] = step;

      rep_l = fmax(rep_l, row_l);
      rep_u = fmin(rep_u, row_u);
      row_delete[row] = std::uint8_t{1};
    }

    if (found_parallel_peer) {
      AL[rep] = rep_l;
      AU[rep] = rep_u;
    }
  }
}

__global__ void _kernel_apply_parallel_row_deletions(std::uint8_t* keep_row,
                                                     const std::uint8_t* row_delete,
                                                     std::int32_t* changed,
                                                     std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (i < m && row_delete[i] != std::uint8_t{0}) {
    keep_row[i] = std::uint8_t{0};
    atomicAdd(changed, 1);
  }
}

__global__ void _kernel_pack_parallel_row_tape(
    ParallelRowTapeRecord* records,
    std::int32_t* count,
    const std::uint8_t* row_delete,
    const std::int32_t* merge_to,
    const double* merge_ratio,
    const double* kept_old_AL,
    const double* kept_old_AU,
    const double* deleted_old_AL,
    const double* deleted_old_AU,
    const std::int32_t* merge_step,
    std::int32_t m) {
  const std::int32_t row =
      static_cast<std::int32_t>(NnzOffset(blockIdx.x) * blockDim.x + threadIdx.x);
  if (row < m && row_delete[row] != std::uint8_t{0}) {
    const std::int32_t slot = atomicAdd(count, 1);
    records[slot] = ParallelRowTapeRecord{merge_step[row],
                                          merge_to[row],
                                          row,
                                          merge_ratio[row],
                                          kept_old_AL[row],
                                          kept_old_AU[row],
                                          deleted_old_AL[row],
                                          deleted_old_AU[row]};
  }
}

}  // namespace

#include "parallel_row_signature.cuh"

void compute_parallel_row_hashes(unsigned long long* row_hash,
                                   const DeviceCsrMatrix& matrix,
                                   const std::uint8_t* keep_row, double zero_tol, int hash_mode,
                                   const std::uint8_t* keep_col = nullptr,
                                   const std::uint8_t* candidates = nullptr) {
  const int m=matrix.rows;
  if (m<=0) return;
  const int blocks = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const bool use_warp_hash = hash_mode >= 0 ? hash_mode != 0 :
      static_cast<std::int64_t>(matrix.nnz) >
      32LL * static_cast<std::int64_t>(m);
  if (adaptive_hash_scan_enabled()) {
    const auto& groups=prepare_adaptive_row_groups(matrix);
    _kernel_parallel_row_hashes<<<blocks,GPU_PRESOLVE_THREADS>>>(
        row_hash,keep_row,matrix.rowPtr,matrix.colVal,matrix.nzVal,zero_tol,m,ADAPTIVE_SHORT_ROW,use_warp_hash,keep_col,candidates);
    grouped_parallel_row_hash<32><<<std::min(1024,(m+7)/8),256>>>(
        row_hash,keep_row,matrix.rowPtr,matrix.colVal,matrix.nzVal,zero_tol,groups.medium,groups.counts,use_warp_hash,keep_col,candidates);
    grouped_parallel_row_hash<256><<<std::min(256,m),256>>>(
        row_hash,keep_row,matrix.rowPtr,matrix.colVal,matrix.nzVal,zero_tol,groups.long_rows,groups.counts+1,use_warp_hash,keep_col,candidates);
    throw_if_cuda_error(cudaGetLastError(),"grouped parallel row hash");
  } else if (keep_col) {
    _kernel_parallel_row_hashes<<<blocks, GPU_PRESOLVE_THREADS>>>(
        row_hash, keep_row, matrix.rowPtr, matrix.colVal, matrix.nzVal,
        zero_tol, m, 0, use_warp_hash, keep_col, candidates);
    throw_if_cuda_error(cudaGetLastError(), "masked parallel row hash");
  } else if (use_warp_hash) {
    constexpr int warps_per_block = GPU_PRESOLVE_THREADS / 32;
    const int warp_blocks = (m + warps_per_block - 1) / warps_per_block;
    _kernel_parallel_row_hashes_warp<<<warp_blocks, GPU_PRESOLVE_THREADS>>>(
        row_hash, keep_row, matrix.rowPtr, matrix.colVal, matrix.nzVal,
        zero_tol, m, candidates);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_parallel_row_hashes_warp");
  } else {
    _kernel_parallel_row_hashes<<<blocks, GPU_PRESOLVE_THREADS>>>(
        row_hash, keep_row, matrix.rowPtr, matrix.colVal, matrix.nzVal,
        zero_tol, m, 0, false, nullptr, candidates);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_parallel_row_hashes");
  }
}

void compute_parallel_row_hashes(unsigned long long* row_hash,
                                  const DeviceCsrMatrix& matrix,
                                  const std::uint8_t* keep_row, double zero_tol) {
  compute_parallel_row_hashes(row_hash, matrix, keep_row, zero_tol, -1);
}

std::size_t parallel_rows_scratch_bytes(std::size_t rows) {
  std::size_t bytes = 0;
  reserve_scratch<unsigned long long>(bytes, rows);
  reserve_scratch<std::int32_t>(bytes, rows);
  reserve_scratch<std::int32_t>(bytes, 1);
  reserve_scratch<std::uint8_t>(bytes, rows);
  reserve_scratch<std::int32_t>(bytes, rows);
  for (int i = 0; i < 5; ++i) reserve_scratch<double>(bytes, rows);
  reserve_scratch<std::int32_t>(bytes, rows);
  reserve_scratch<std::int32_t>(bytes, 1);
  reserve_scratch<unsigned long long>(bytes, (row_signature::scratch_bytes(rows) + 7) / 8);
  return bytes;
}

void apply_rule_parallel_rows(PresolvePlanGpu& plan,
                              const LPInfoGpu& lp,
                              const PresolveStatsGpu& stats,
                              const PresolveParams& pparams) {
  (void)stats;
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }

  const std::int32_t m = lp.A.rows;
  if (m <= 1) {
    return;
  }
  const bool profile = runtime::options().profile_parallel_rows;
  const auto rule_start = std::chrono::steady_clock::now();
  auto profile_stage = [&](const char* stage, const std::chrono::steady_clock::time_point& stage_start) {
    if (!profile) {
      return;
    }
    throw_if_cuda_error(cudaDeviceSynchronize(), "parallel_rows profile synchronize");
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - stage_start;
    std::cerr << ">>> [parallel_rows C++] " << stage << " = " << elapsed.count() << "s\n";
  };

  const std::size_t m_size = static_cast<std::size_t>(m);
  std::size_t scratch_bytes = 0;
  const auto row_hash_offset = reserve_scratch<unsigned long long>(scratch_bytes, m_size);
  const auto sorted_rows_offset = reserve_scratch<std::int32_t>(scratch_bytes, m_size);
  const auto status_flag_offset = reserve_scratch<std::int32_t>(scratch_bytes, 1);
  const auto row_delete_offset = reserve_scratch<std::uint8_t>(scratch_bytes, m_size);
  const auto merge_to_offset = reserve_scratch<std::int32_t>(scratch_bytes, m_size);
  const auto merge_ratio_offset = reserve_scratch<double>(scratch_bytes, m_size);
  const auto kept_old_AL_offset = reserve_scratch<double>(scratch_bytes, m_size);
  const auto kept_old_AU_offset = reserve_scratch<double>(scratch_bytes, m_size);
  const auto deleted_old_AL_offset = reserve_scratch<double>(scratch_bytes, m_size);
  const auto deleted_old_AU_offset = reserve_scratch<double>(scratch_bytes, m_size);
  const auto merge_step_offset = reserve_scratch<std::int32_t>(scratch_bytes, m_size);
  const auto changed_offset = reserve_scratch<std::int32_t>(scratch_bytes, 1);
  const bool signature_enabled = row_signature::enabled(lp.A);
  const auto signature_offset = reserve_scratch<unsigned long long>(scratch_bytes,
      (row_signature::scratch_bytes(m_size) + 7) / 8);
  DuplicateRuleScratchFrame scratch_frame;
  void* scratch_storage = scratch_frame.get(scratch_bytes);
  auto* row_hash = scratch_at<unsigned long long>(scratch_storage, row_hash_offset);
  auto* sorted_rows = scratch_at<std::int32_t>(scratch_storage, sorted_rows_offset);
  auto* status_flag = scratch_at<std::int32_t>(scratch_storage, status_flag_offset);
  auto* row_delete = scratch_at<std::uint8_t>(scratch_storage, row_delete_offset);
  auto* merge_to = scratch_at<std::int32_t>(scratch_storage, merge_to_offset);
  auto* merge_ratio = scratch_at<double>(scratch_storage, merge_ratio_offset);
  auto* kept_old_AL = scratch_at<double>(scratch_storage, kept_old_AL_offset);
  auto* kept_old_AU = scratch_at<double>(scratch_storage, kept_old_AU_offset);
  auto* deleted_old_AL = scratch_at<double>(scratch_storage, deleted_old_AL_offset);
  auto* deleted_old_AU = scratch_at<double>(scratch_storage, deleted_old_AU_offset);
  auto* merge_step = scratch_at<std::int32_t>(scratch_storage, merge_step_offset);
  auto* changed = scratch_at<std::int32_t>(scratch_storage, changed_offset);
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)), "cudaMemset parallel_rows status_flag");
  throw_if_cuda_error(cudaMemset(row_delete, 0, static_cast<std::size_t>(m)), "cudaMemset parallel_rows row_delete");
  throw_if_cuda_error(cudaMemset(changed, 0, sizeof(std::int32_t)), "cudaMemset parallel_rows changed");

  const int blocks = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  auto* sort_hash = row_hash;
  int sort_count = m;
  const auto* keep_col = stats.live_masks ? plan.keep_col_mask : nullptr;
  const std::uint8_t* candidates = nullptr;
  const bool signature_audit = signature_enabled && runtime::options().audit_row_signatures;
  auto hash_start = rule_start;
  if (signature_enabled) {
    row_signature::Workspace signature(scratch_at<void>(scratch_storage, signature_offset), m_size);
    row_signature::filter(signature, lp.A, plan.keep_row_mask, keep_col, pparams.zero_tol);
    candidates = signature.candidates;
    profile_stage("signature", rule_start);
    if (signature_audit) {
      const auto audit_start = std::chrono::steady_clock::now();
      compute_parallel_row_hashes(row_hash, lp.A, plan.keep_row_mask, pparams.zero_tol,
                                 plan.parallel_row_hash_mode, keep_col);
      auto hashes = thrust::device_pointer_cast(row_hash);
      auto rows = thrust::device_pointer_cast(sorted_rows);
      thrust::sequence(rows, rows + m, std::int32_t{0});
      auto zipped = thrust::make_zip_iterator(thrust::make_tuple(hashes, rows));
      thrust::sort(zipped, zipped + m, HashRowLess{});
      throw_if_cuda_error(cudaMemset(signature.overflow, 0, sizeof(int)), "signature audit reset");
      row_signature::audit_groups<<<blocks, GPU_PRESOLVE_THREADS>>>(row_hash, sorted_rows,
          plan.keep_row_mask, candidates, m, signature.overflow, signature.projections, lp.A, keep_col);
      int missing = 0;
      throw_if_cuda_error(cudaMemcpy(&missing, signature.overflow, sizeof(int), cudaMemcpyDeviceToHost),
                          "signature audit result");
      std::cerr << ">>> [parallel_rows signature] audit_missing = " << missing << "\n";
      if (missing) throw std::runtime_error("parallel row signature omitted original hash-group members");
      profile_stage("signature_audit", audit_start);
    }
    hash_start = std::chrono::steady_clock::now();
  }
  // Audit sorts the full hash array, so recompute in row-id order as usual.
  compute_parallel_row_hashes(row_hash,lp.A,plan.keep_row_mask,pparams.zero_tol,
                               plan.parallel_row_hash_mode, keep_col, candidates);
  throw_if_cuda_error(cudaDeviceSynchronize(), "parallel_rows hash synchronize");
  profile_stage("hash", hash_start);

  const auto sort_start = std::chrono::steady_clock::now();
  auto row_hash_begin = thrust::device_pointer_cast(row_hash);
  auto sorted_rows_begin = thrust::device_pointer_cast(sorted_rows);
  if (signature_enabled) {
    const auto first = thrust::make_counting_iterator(std::int32_t{0});
    const auto end = thrust::copy_if(first, first + m, thrust::device_pointer_cast(candidates),
                                   sorted_rows_begin, row_signature::IsCandidate{});
    sort_count = static_cast<int>(end - sorted_rows_begin);
    if (profile) {
      row_signature::Workspace signature(scratch_at<void>(scratch_storage, signature_offset), m_size);
      throw_if_cuda_error(cudaMemset(signature.counts, 0, 2*sizeof(unsigned long long)), "signature nnz reset");
      row_signature::count_nnz<<<blocks, GPU_PRESOLVE_THREADS>>>(signature, m);
      unsigned long long counts[2];
      throw_if_cuda_error(cudaMemcpy(counts, signature.counts, sizeof(counts), cudaMemcpyDeviceToHost),
                          "signature nnz counts");
      std::cerr << ">>> [parallel_rows signature] candidate_live_nnz = " << counts[1] << "/" << counts[0] << "\n";
    }
    // Projection data is dead after candidate marking. Reuse it for gather.
    sort_hash = scratch_at<unsigned long long>(scratch_storage, signature_offset);
    auto output = thrust::device_pointer_cast(sort_hash);
    thrust::gather(sorted_rows_begin, end, row_hash_begin, output);
    row_hash_begin = output;
    if (profile || signature_audit)
      std::cerr << ">>> [parallel_rows signature] candidates = " << sort_count << "/" << m << "\n";
  } else {
    thrust::sequence(sorted_rows_begin, sorted_rows_begin + m, std::int32_t{0});
  }
  auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(row_hash_begin, sorted_rows_begin));
  thrust::sort(zipped_begin, zipped_begin + sort_count, HashRowLess{});
  throw_if_cuda_error(cudaGetLastError(), "thrust parallel_rows sort hashes");
  profile_stage("device_sort", sort_start);

  const auto groups_start = std::chrono::steady_clock::now();
  _kernel_parallel_row_groups<<<blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag,
      row_delete,
      merge_to,
      merge_ratio,
      kept_old_AL,
      kept_old_AU,
      deleted_old_AL,
      deleted_old_AU,
      merge_step,
      sort_hash,
      sorted_rows,
      plan.keep_row_mask,
      stats.live_masks ? plan.keep_col_mask : nullptr,
      plan.new_AL,
      plan.new_AU,
      lp.A.rowPtr,
      lp.A.colVal,
      lp.A.nzVal,
      pparams.zero_tol,
      pparams.feasibility_tol,
      sort_count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_parallel_row_groups");
  profile_stage("groups", groups_start);

  const auto apply_start = std::chrono::steady_clock::now();
  std::int32_t status = 0;
  throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy parallel_rows status");
  if (status != 0) {
    plan.has_infeasible = true;
  } else {
    _kernel_apply_parallel_row_deletions<<<blocks, GPU_PRESOLVE_THREADS>>>(
        plan.keep_row_mask,
        row_delete,
        changed,
        m);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_apply_parallel_row_deletions");
    throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_parallel_rows synchronize");
    std::int32_t changed_host = 0;
    throw_if_cuda_error(cudaMemcpy(&changed_host, changed, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy parallel_rows changed");
    profile_stage("apply_delete", apply_start);
    if (changed_host != 0) {
      if (pparams.record_postsolve_tape) {
        const auto tape_start = std::chrono::steady_clock::now();
        ParallelRowTapeRecord* packed_records = nullptr;
        throw_if_cuda_error(
            cudaMalloc(&packed_records,
                       sizeof(ParallelRowTapeRecord) * static_cast<std::size_t>(changed_host)),
            "cudaMalloc packed parallel-row tape");
        throw_if_cuda_error(cudaMemset(changed, 0, sizeof(std::int32_t)),
                            "cudaMemset parallel-row packed count");
        _kernel_pack_parallel_row_tape<<<blocks, GPU_PRESOLVE_THREADS>>>(
            packed_records, changed, row_delete, merge_to, merge_ratio, kept_old_AL,
            kept_old_AU, deleted_old_AL, deleted_old_AU, merge_step, m);
        throw_if_cuda_error(cudaGetLastError(), "_kernel_pack_parallel_row_tape");
        std::vector<ParallelRowTapeRecord> tape_records(
            static_cast<std::size_t>(plan.tape.resident ? 0 : changed_host));
        if (plan.tape.resident) {
          auto ptr = thrust::device_pointer_cast(packed_records);
          thrust::sort(ptr, ptr + changed_host, RecoveryOrder{});
          auto segment = allocate_recovery_segment(
              changed_host, checked_nnz_offset(2 * std::int64_t(changed_host)),
              checked_nnz_offset(5 * std::int64_t(changed_host)));
          pack_recovery_segment<<<(changed_host + 256) / 256, 256>>>(
              segment.types, segment.dual_modes, segment.index_starts, segment.value_starts,
              segment.indices, segment.vals, packed_records, changed_host);
          throw_if_cuda_error(cudaGetLastError(), "pack duplicate recovery segment");
          append_device_recovery_segment(plan.tape, std::move(segment));
        } else {
          throw_if_cuda_error(cudaMemcpy(tape_records.data(), packed_records,
                                         sizeof(ParallelRowTapeRecord) * tape_records.size(),
                                         cudaMemcpyDeviceToHost),
                              "cudaMemcpy packed parallel-row tape");
        }
        cudaFree(packed_records);
        std::sort(tape_records.begin(), tape_records.end(),
                  [](const ParallelRowTapeRecord& lhs, const ParallelRowTapeRecord& rhs) {
                    if (lhs.step != rhs.step) {
                      return lhs.step < rhs.step;
                    }
                    return lhs.deleted_row < rhs.deleted_row;
                  });
        plan.tape.types.reserve(plan.tape.types.size() + tape_records.size());
        plan.tape.indices.reserve(plan.tape.indices.size() + 2 * tape_records.size());
        plan.tape.vals.reserve(plan.tape.vals.size() + 5 * tape_records.size());
        plan.tape.index_starts.reserve(plan.tape.index_starts.size() + tape_records.size());
        plan.tape.value_starts.reserve(plan.tape.value_starts.size() + tape_records.size());
        plan.tape.dual_modes.reserve(plan.tape.dual_modes.size() + tape_records.size());
        for (const ParallelRowTapeRecord& record : tape_records) {
          const std::int32_t indices[] = {record.kept_row, record.deleted_row};
          const double vals[] = {record.ratio,
                                 record.kept_old_AL,
                                 record.kept_old_AU,
                                 record.deleted_old_AL,
                                 record.deleted_old_AU};
          append_postsolve_record(
              plan.tape,
              PostsolveReductionType::ParallelRow,
              indices,
              2,
              vals,
              5,
              PostsolveDualMode::Exact);
        }
        profile_stage("tape", tape_start);
      }
      plan.has_row_action = true;
      plan.has_change = true;
    }
  }

  profile_stage("total", rule_start);
}

}  // namespace gpu_presolver::presolve
