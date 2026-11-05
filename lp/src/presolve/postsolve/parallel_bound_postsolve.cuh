// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include "gpu_preprocessor/lp/presolve/compact_bound_record.hpp"
#include "gpu_preprocessor/lp/presolve/packed_bound_record.hpp"
#include <cub/cub.cuh>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gpu_preprocessor::lp::presolve::parallel_bound_detail {

constexpr NnzOffset min_records = 1024;
using Key = std::uint64_t;
struct Shape {
  NnzOffset begin = 0, first = 0, end = 0, value_begin = 0;
  int rows = 0, transfers = 0, terms = 0;
  bool compact = false, packed = false;
};
inline void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("parallel bound replay: ") + cudaGetErrorString(e));
}
template<class T> bool allocate(RecoveryArray<T>& a, std::size_t n) {
  const auto status = cudaMalloc(&a.data, checked_nnz_bytes(checked_nnz_offset(n), sizeof(T)));
  if (status == cudaErrorMemoryAllocation) {
    if (cudaPeekAtLastError() == cudaErrorMemoryAllocation) cudaGetLastError();
    return false;
  }
  check(status);
  return true;
}
template<class T> void read_control(T& value, const T* device) {
  // Control copies stay inside module timing, separate from final vector D2H.
  check(cudaMemcpyAsync(&value, device, sizeof(T), cudaMemcpyDeviceToHost));
  check(cudaStreamSynchronize(nullptr));
}
inline int blocks(int n) { return checked_launch_blocks(std::min(n, 256 * 1024), 256); }

__global__ void inspect(Shape* out, const int* types, const int* ids, const NnzOffset* is, const NnzOffset* vs,
                         const double* values, NnzOffset begin, NnzOffset end) {
  if (blockIdx.x || threadIdx.x) return;
  *out = Shape{};
  if(types[end-1]==int(PostsolveReductionType::BoundPackedBatch)) {
    if(end-begin!=1 || !valid_packed_bound_header(ids+is[begin],is[end]-is[begin],vs[end]-vs[begin]))return;
    const PackedBoundView b{ids+is[begin]};
    if(b.terms()>INT_MAX || b.terms()<1)return;
    out->begin=out->first=begin;out->end=end;out->value_begin=vs[begin];
    out->rows=b.rows();out->transfers=b.transfers();out->terms=int(b.terms());out->packed=true;return;
  }
  const auto vp = vs[end - 1];
  const bool compact=types[end-1]==int(PostsolveReductionType::BoundBatch);
  if(compact ? (is[end]-is[end-1]!=2 || vs[end]!=vp) : vs[end]-vp!=12) return;
  const double r = compact?ids[is[end-1]+1]:values[vp + 8];
  const double u = compact?ids[is[end-1]]:values[vp + 11];
  if (!isfinite(r) || !isfinite(u) || r < 1 || u < 1 ||
      r > INT_MAX || u > INT_MAX || floor(r) != r || floor(u) != u ||
      r + u + int(compact) != double(end - begin)) return;
  const auto first = end - NnzOffset(r)-int(compact);
  const auto terms = vs[first] - vs[begin];
  if (terms <= 0 || terms > INT_MAX ||
      is[first] - is[begin] - NnzOffset(u) != terms) return;
  out->begin = begin; out->first = first; out->end = end-int(compact);out->compact=compact;
  out->value_begin = vs[begin]; out->rows = int(u);
  out->transfers = int(r); out->terms = int(terms);
}

__device__ int source_row(Shape s,const NnzOffset* is,const int* ids,int r) {
  return s.packed?PackedBoundView{ids+is[s.begin]}.row(r):ids[is[s.begin+NnzOffset(r)]];
}
__global__ void source_keys(int* serial_only, Key* keys, int* owners, Shape s,
    const int* types, const NnzOffset* is, const NnzOffset* vs,
    const int* ids, const double* values, const double* z, int m, int n) {
  const auto stride = std::uint64_t(blockDim.x) * gridDim.x;
  for (auto r = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       r < std::uint64_t(s.rows); r += stride) {
    if(s.packed) {
      const PackedBoundView b{ids+is[s.begin]};const auto begin=b.start(int(r)),end=b.start(int(r+1));
      const int row=b.row(int(r));
      if(row<0 || row>=m || (r && b.row(int(r-1))>=row))atomicExch(serial_only,1);
      for(auto q=begin;q<end;++q) {
        const int col=b.column(q);keys[q]=(Key(std::uint32_t(col))<<32)|Key(q);owners[q]=int(r);
        if(col<0 || col>=n || !isfinite(values[s.value_begin+q]) || !isfinite(z[col]))atomicExch(serial_only,1);
      }
      continue;
    }
    const auto k = s.begin + NnzOffset(r), ip = is[k], vp = vs[k];
    const auto count = vs[k + 1] - vp;
    if (types[k] != int(PostsolveReductionType::BoundSourceRow) || count < 0 ||
        is[k + 1] - ip != count + 1 || vp < s.value_begin ||
        vs[k + 1] - s.value_begin > s.terms) {
      atomicExch(serial_only, 1); continue;
    }
    const int row = ids[ip];
    if (row < 0 || row >= m || (r && ids[is[k - 1]] >= row)) {
      atomicExch(serial_only, 1); continue;
    }
    for (NnzOffset q = 0; q < count; ++q) {
      const int col = ids[ip + 1 + q];
      const auto ordinal = vp + q - s.value_begin;
      if (col < 0 || col >= n || !isfinite(values[vp + q]) || !isfinite(z[col])) {
        atomicExch(serial_only, 1); continue;
      }
      // Coefficient ordinals preserve source-row and within-row entry order.
      keys[ordinal] = (Key(col) << 32) | Key(ordinal);
      owners[ordinal] = int(r);
    }
  }
}

__global__ void transfer_keys(int* serial_only, Key* keys, double* delta, Shape s,
    const int* types, const NnzOffset* is, const NnzOffset* vs,
    const int* ids, const double* values, const double* z, const int* source_owners, int n) {
  const auto stride = std::uint64_t(blockDim.x) * gridDim.x;
  for (auto r = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       r < std::uint64_t(s.transfers); r += stride) {
    keys[r] = (Key(s.rows) << 32) | r; delta[r] = 0;
    if(s.packed) {
      const PackedBoundView b{ids+is[s.begin]};const auto lo=b.reference(int(r),0),up=b.reference(int(r),1),any=lo>=0?lo:up;
      if(any<0)continue;
      if(any>=s.terms) {atomicExch(serial_only,1);continue;}
      const int col=b.column(any);if(col<0 || col>=n){atomicExch(serial_only,1);continue;}
      const double incoming=z[col];const auto ref=incoming>0.0?lo:up;
      if(ref<0)continue;if(ref>=s.terms){atomicExch(serial_only,1);continue;}
      const double coefficient=values[s.value_begin+ref];if(coefficient==0.0)continue;
      const int slot=source_owners[ref];
      if(slot<0 || slot>=s.rows || b.column(ref)!=col || !isfinite(incoming) || !isfinite(coefficient)) {atomicExch(serial_only,1);continue;}
      const double d=incoming/coefficient;if(!isfinite(d)){atomicExch(serial_only,1);continue;}
      keys[r]=(Key(slot)<<32)|r;delta[r]=d;continue;
    }
    const auto k = s.first + NnzOffset(r), ip = is[k], vp = vs[k];
    if (types[k] != int(s.compact?PostsolveReductionType::BoundChangeCompact:PostsolveReductionType::BoundChangeSharedRow) ||
        is[k + 1] - ip != (s.compact?compact_bound_indices:5) || vs[k + 1] - vp != (s.compact?0:12) ||
        (!s.compact && (values[vp + 8] != double(r + 1) || values[vp + 11] != double(s.rows)))) {
      atomicExch(serial_only, 1); continue;
    }
    const int col = ids[ip];
    if (col < 0 || col >= n) { atomicExch(serial_only, 1); continue; }
    const double incoming = z[col];
    const int side = incoming > 0.0 ? 0 : 1;
    const int slot = ids[ip + (s.compact?1:3) + side];
    const auto ref=s.compact?compact_bound_reference(ids+ip,side):NnzOffset(-1);
    if(s.compact && (slot<0 || ref<0)) continue;
    if(s.compact && ref>=s.terms) {atomicExch(serial_only,1);continue;}
    const double coefficient = values[s.compact?s.value_begin+ref:vp+6+side];
    if ((!s.compact && values[vp + 4 + side] == 0.0) || coefficient == 0.0) continue;
    if (slot < 0 || slot >= s.rows || (!s.compact && ids[is[s.begin + slot]] != ids[ip + 1 + side]) ||
        !isfinite(incoming) || !isfinite(coefficient)) {
      atomicExch(serial_only, 1); continue;
    }
    const double d = incoming / coefficient;
    if (!isfinite(d)) { atomicExch(serial_only, 1); continue; }
    keys[r] = (Key(slot) << 32) | r; delta[r] = d;
  }
}

__device__ int lower_key(const Key* keys, int n, Key value) {
  int lo = 0, hi = n;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (keys[mid] < value) lo = mid + 1; else hi = mid;
  }
  return lo;
}
__global__ void row_sums(int* serial_only, double* sums, double* errors, Shape s,
    const Key* keys, const double* deltas, const NnzOffset* is, const int* ids) {
  const auto stride = std::uint64_t(blockDim.x) * gridDim.x;
  for (auto r = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       r < std::uint64_t(s.rows); r += stride) {
    const int begin = lower_key(keys, s.transfers, Key(r) << 32);
    const int end = lower_key(keys, s.transfers, Key(r + 1) << 32);
    double sum = 0.0, error = 0.0;
    for (int p = begin; p < end; ++p) {
      const double d = deltas[p], old = sum, next = old + d;
      error += fabs(old) >= fabs(d) ? (old - next) + d : (d - next) + old;
      sum = next;
    }
    const int row = source_row(s,is,ids,int(r));
    sums[row] = sum; errors[row] = error;
    if (!isfinite(sum) || !isfinite(error) || !isfinite(sum + error)) atomicExch(serial_only, 1);
  }
}
__global__ void apply_rows(const int* serial_only, double* y, const double* sums,
    const double* errors, Shape s, const NnzOffset* is, const int* ids) {
  if (*serial_only || !y) return;
  const auto stride = std::uint64_t(blockDim.x) * gridDim.x;
  for (auto r = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       r < std::uint64_t(s.rows); r += stride) {
    const int row = source_row(s,is,ids,int(r));
    const double d = sums[row] + errors[row];
    if (d != 0.0) y[row] += d;
  }
}
__global__ void apply_columns(const int* serial_only, double* z, const double* sums,
    const double* errors, Shape s, const Key* keys, const int* owners,
    const NnzOffset* is, const int* ids, const double* values) {
  if (*serial_only) return;
  const auto stride = std::uint64_t(blockDim.x) * gridDim.x;
  for (auto p = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       p < std::uint64_t(s.terms); p += stride) {
    const int col = int(keys[p] >> 32);
    if (p && int(keys[p - 1] >> 32) == col) continue;
    double value = z[col];
    for (auto q = p; q < std::uint64_t(s.terms) && int(keys[q] >> 32) == col; ++q) {
      const int row = source_row(s,is,ids,owners[q]);
      const double d = sums[row] + errors[row];
      if (d != 0.0) value -= values[s.value_begin + NnzOffset(std::uint32_t(keys[q]))] * d;
    }
    z[col] = value;
  }
}

// Only a complete shared-row batch is eligible. All validation and allocation
// finish before any y/z write; failed eligibility or scratch OOM uses serial.
inline bool try_replay(int* serial_only, double* y, double* z,
    double* sums, double* errors, const int* types, const NnzOffset* is,
    const NnzOffset* vs, const int* ids, const double* values,
    NnzOffset begin, NnzOffset end, int m, int n) {
  if (!serial_only || !z || end<=begin) return false;
  int last = -1;
  read_control(last, types + end - 1);
  if (last != int(PostsolveReductionType::BoundChangeSharedRow) &&
      last != int(PostsolveReductionType::BoundBatch) && last != int(PostsolveReductionType::BoundPackedBatch)) return false;
  if(last!=int(PostsolveReductionType::BoundPackedBatch) && end-begin<min_records)return false;
  RecoveryArray<Shape> device_shape;
  if (!allocate(device_shape, 1)) return false;
  inspect<<<1, 1>>>(device_shape.data, types, ids, is, vs, values, begin, end);
  check(cudaGetLastError());
  Shape s;
  read_control(s, device_shape.data);
  if (!s.rows || s.transfers<min_records) return false;
  RecoveryArray<Key> tk_in, tk_out, ck_in, ck_out;
  RecoveryArray<double> td_in, td_out;
  RecoveryArray<int> co_in, co_out;
  if (!allocate(tk_in, s.transfers) || !allocate(tk_out, s.transfers) ||
      !allocate(td_in, s.transfers) || !allocate(td_out, s.transfers) ||
      !allocate(ck_in, s.terms) || !allocate(ck_out, s.terms) ||
      !allocate(co_in, s.terms) || !allocate(co_out, s.terms)) return false;
  std::size_t transfer_bytes = 0, column_bytes = 0;
  check(cub::DeviceRadixSort::SortPairs(nullptr, transfer_bytes, tk_in.data, tk_out.data,
                                       td_in.data, td_out.data, s.transfers));
  check(cub::DeviceRadixSort::SortPairs(nullptr, column_bytes, ck_in.data, ck_out.data,
                                       co_in.data, co_out.data, s.terms));
  RecoveryArray<std::uint8_t> temp;
  const auto bytes = std::max(transfer_bytes, column_bytes);
  // CUB storage is a byte count, independent of the tape's offset width.
  const auto allocated=cudaMalloc(&temp.data,bytes);
  if(allocated==cudaErrorMemoryAllocation) {
    if(cudaPeekAtLastError()==cudaErrorMemoryAllocation)cudaGetLastError();
    return false;
  }
  check(allocated);
  check(cudaMemsetAsync(serial_only, 0, sizeof(int)));
  source_keys<<<blocks(s.rows), 256>>>(serial_only, ck_in.data, co_in.data, s,
      types, is, vs, ids, values, z, m, n);
  transfer_keys<<<blocks(s.transfers), 256>>>(serial_only, tk_in.data, td_in.data, s,
      types, is, vs, ids, values, z, co_in.data, n);
  check(cudaGetLastError());
  int invalid = 0;
  read_control(invalid, serial_only);
  if (invalid) return false;
  check(cub::DeviceRadixSort::SortPairs(temp.data, transfer_bytes, tk_in.data, tk_out.data,
                                       td_in.data, td_out.data, s.transfers));
  check(cub::DeviceRadixSort::SortPairs(temp.data, column_bytes, ck_in.data, ck_out.data,
                                       co_in.data, co_out.data, s.terms));
  row_sums<<<blocks(s.rows), 256>>>(serial_only, sums, errors, s, tk_out.data, td_out.data, is, ids);
  apply_rows<<<blocks(s.rows), 256>>>(serial_only, y, sums, errors, s, is, ids);
  apply_columns<<<blocks(s.terms), 256>>>(serial_only, z, sums, errors, s,
      ck_out.data, co_out.data, is, ids, values);
  check(cudaGetLastError());
  return true;
}

} // namespace gpu_preprocessor::lp::presolve::parallel_bound_detail
