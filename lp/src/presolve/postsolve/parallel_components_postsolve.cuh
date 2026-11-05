// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once
#include "../infinite_bound_record.hpp"
#include "../projected_column_record.hpp"

#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include <cub/cub.cuh>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gpu_preprocessor::lp::presolve::parallel_components_detail {

constexpr NnzOffset min_records = 1024;
using Key = std::uint64_t;
inline void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("component postsolve: ") + cudaGetErrorString(e));
}
template<class T> bool allocate(RecoveryArray<T>& a, std::size_t n) {
  if (!n) return true;
  const auto e = cudaMalloc(&a.data, n * sizeof(T));
  if (e == cudaErrorMemoryAllocation) {
    if (cudaPeekAtLastError() == cudaErrorMemoryAllocation) cudaGetLastError();
    return false;
  }
  check(e); return true;
}
inline int blocks(int n) { return checked_launch_blocks(std::min(n, 256 * 1024), 256); }

// Parents only decrease. Every concurrent parent access is atomic; a later
// kernel reads the final forest after all unions have completed on stream 0.
__device__ inline int root(int* parent, int a) {
  for (;;) { const int b = atomicAdd(parent + a, 0); if (a == b) return a; a = b; }
}
__device__ inline void unite(int* parent, int a, int b) {
  for (;;) {
    a = root(parent, a); b = root(parent, b);
    if (a == b) return;
    const int hi = max(a, b), lo = min(a, b);
    const int previous = atomicMin(parent + hi, lo);
    if (previous == hi) return;
    a = previous; b = lo;
  }
}
struct Access {
  int *invalid, *columns, *rows, *parent;
  int m, n, record;
  bool connect;
  __device__ bool require(bool ok) const {
    if (!ok) atomicExch(invalid, 1);
    return ok;
  }
  __device__ void use(int id, bool row, bool write) const {
    if (!require(id >= 0 && id < (row ? m : n))) return;
    int* owners = row ? rows : columns;
    if (!connect) { if (write) atomicCAS(owners + id, -1, record); }
    else { const int owner = owners[id]; if (owner >= 0) unite(parent, record, owner); }
  }
  __device__ void col(int id, bool write = false) const { use(id, false, write); }
  __device__ void row(int id, bool write = false) const { use(id, true, write); }
};

// A column resource conservatively combines x, z and the retrieved mask.
// Read/read sharing does not join records. Any possible write/read or
// write/write overlap does, even if a numerical branch would skip the write.
__device__ inline void accesses(Access a, int type, const int* ids, NnzOffset ni,
    const double* v, NnzOffset nv, const LPInfoGpu& model, bool original) {
  using T = PostsolveReductionType;
  switch (static_cast<T>(type)) {
  case T::FixedVariable:
    if (!a.require(ni >= 1 && nv == ni + 1)) return;
    a.col(ids[0], true);
    for (NnzOffset q = 1; q < ni; ++q) a.row(ids[q]);
    break;
  case T::SingletonColumnSubstitution:
    if (!a.require(ni >= 3 && nv >= 6)) return;
    if (!a.require(ids[2] >= 0 && ni - 3 == ids[2] && nv - 6 == ids[2])) return;
    a.col(ids[0], true); a.row(ids[1], true);
    for (int q = 0; q < ids[2]; ++q) a.col(ids[3 + q]);
    break;
  case T::DoubletonEquation: {
    if (!a.require(ni >= 3 && nv >= 10)) return;
    a.col(ids[0], true); a.col(ids[1], true); a.row(ids[2], true);
    if (ni >= 4 && nv >= 12) {
      if (!a.require(ids[3] >= 0 && ni - 4 == ids[3] && nv - 12 == ids[3])) return;
      for (int q = 0; q < ids[3]; ++q) a.row(ids[4 + q]);
    } else if (original) {
      if (!a.require(model.AT.rowPtr && model.AT.colVal && ids[0] >= 0 && ids[0] < a.n)) return;
      for (auto q = model.AT.rowPtr[ids[0]]; q < model.AT.rowPtr[ids[0] + 1]; ++q)
        a.row(model.AT.colVal[q]);
    }
    break;
  }
  case T::ParallelColumn:
    if (!a.require(ni == 2 && nv >= 5)) return;
    a.col(ids[0], true); a.col(ids[1], true); break;
  case T::ParallelRow:
    if (!a.require(ni == 2 && nv >= 5)) return;
    a.row(ids[0], true); a.row(ids[1], true); break;
  case T::DeletedRow:
    // Plain row deletion has no inverse. Endpoint-owner and legacy singleton
    // inverses read the column; legacy stationarity also reads original rows.
    if (ni < 2 || nv < 3) break;
    a.col(ids[1], true); a.row(ids[0], true);
    if (nv < 9 && original) {
      if (!a.require(model.AT.rowPtr && model.AT.colVal && ids[1] >= 0 && ids[1] < a.n)) return;
      for (auto q = model.AT.rowPtr[ids[1]]; q < model.AT.rowPtr[ids[1] + 1]; ++q)
        a.row(model.AT.colVal[q]);
    }
    break;
  case T::RowDualShift:
    if (!a.require(ni >= 1 && nv >= 1)) return;
    a.row(ids[0], true);
    if (ni > 1) {
      if (!a.require(ids[1] >= 0 && ni - 2 == ids[1] && nv - 1 == ids[1])) return;
      for (int q = 0; q < ids[1]; ++q) a.col(ids[2 + q], true);
    }
    break;
  case T::InfiniteBoundDualFix: {
    detail::InfiniteBoundHeader header;
    if (!a.require(detail::read_infinite_bound_header(ids, v, ni, nv, header))) return;
    a.col(ids[1], true);
    NnzOffset ip = header.indices, vp = header.values;
    for (int r = 0; r < header.rows; ++r) {
      if (!a.require(ip < ni && vp < nv)) return;
      const int length = ids[ip];
      if (!a.require(length >= 0 && length <= ni - ip - 1 && length <= nv - vp - 1)) return;
      for (int q = 0; q < length; ++q) a.col(ids[ip + 1 + q]);
      ip += NnzOffset(length) + 1; vp += NnzOffset(length) + 1;
    }
    a.require(ip == ni && vp == nv); break;
  }
  case T::FmeCol: {
    detail::ProjectedColumnHeader header;
    if (!a.require(detail::valid_projected_column_record(ids, v, ni, nv) &&
                   detail::read_projected_column_header(ids, v, ni, nv, header))) return;
    a.col(ids[1], true);
    for (int q = 0; q < ids[0]; ++q) a.row(ids[2 + q], true);
    for (NnzOffset q = header.support_begin; q < ni; ++q) a.col(ids[q]);
    break;
  }
  case T::AddedRow: case T::AddedRows: case T::RowLowerBoundChange: case T::RowUpperBoundChange:
  case T::VariableBoundChange:
    break; // These records have no scalar inverse arithmetic.
  default:
    // Bound batches consume several records together. Preserve their existing
    // specialized or scalar batch interpreter rather than splitting them.
    a.require(false);
  }
}

static __global__ void inspect(int* invalid, int* owners, int* parent, const int* types,
    const NnzOffset* is, const NnzOffset* vs, const int* ids, const double* values,
    NnzOffset begin, int count, int m, int n, LPInfoGpu model, bool original, bool connect) {
  const auto stride = std::uint64_t(blockDim.x) * gridDim.x;
  for (auto p = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; p < std::uint64_t(count); p += stride) {
    if (!connect) parent[p] = int(p);
    const auto k = begin + NnzOffset(p), ip = is[k], vp = vs[k];
    Access a{invalid, owners, owners + n, parent, m, n, int(p), connect};
    accesses(a, types[k], ids + ip, is[k + 1] - ip, values + vp, vs[k + 1] - vp, model, original);
  }
}
static __global__ void keys(Key* out, const int* parent, int count) {
  const auto stride = std::uint64_t(blockDim.x) * gridDim.x;
  for (auto p = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; p < std::uint64_t(count); p += stride) {
    int r = int(p);
    while (parent[r] != r) r = parent[r];
    out[p] = (Key(r) << 32) | Key(p);
  }
}
struct Plan {
  RecoveryArray<int> owners, parent;
  RecoveryArray<Key> unsorted, ordered;
  RecoveryArray<std::uint8_t> temporary;
};
inline bool prepare(Plan& plan, int* serial_only, const int* types,
    const NnzOffset* is, const NnzOffset* vs, const int* ids, const double* values,
    NnzOffset begin, NnzOffset end, int m, int n, const LPInfoGpu* model) {
  const auto length = end - begin;
  if (!serial_only || length < min_records || length > INT_MAX || m < 0 || n <= 0) return false;
  int last = -1;
  check(cudaMemcpyAsync(&last, types + end - 1, sizeof(int), cudaMemcpyDeviceToHost));
  check(cudaStreamSynchronize(nullptr));
  using T = PostsolveReductionType;
  if (last == int(T::FixedVariable) || last == int(T::SingletonColumnSubstitution) || last == int(T::RowImpliedBoundChange) ||
      last == int(T::BoundChangeSharedRow) || last == int(T::BoundSourceRow) ||
      last == int(T::BoundBatch) || last == int(T::BoundChangeCompact) || last == int(T::BoundPackedBatch)) return false;
  const int count = int(length);
  if (!allocate(plan.owners, std::size_t(m) + n) || !allocate(plan.parent, count) ||
      !allocate(plan.unsorted, count) || !allocate(plan.ordered, count)) return false;
  std::size_t bytes = 0;
  check(cub::DeviceRadixSort::SortKeys(nullptr, bytes, plan.unsorted.data, plan.ordered.data, count));
  if (bytes > INT_MAX || !allocate(plan.temporary, bytes)) return false;
  check(cudaMemsetAsync(serial_only, 0, sizeof(int)));
  check(cudaMemsetAsync(plan.owners.data, 0xff, (std::size_t(m) + n) * sizeof(int)));
  const LPInfoGpu original = model ? *model : LPInfoGpu{};
  inspect<<<blocks(count), 256>>>(serial_only, plan.owners.data, plan.parent.data, types,
      is, vs, ids, values, begin, count, m, n, original, model != nullptr, false);
  inspect<<<blocks(count), 256>>>(serial_only, plan.owners.data, plan.parent.data, types,
      is, vs, ids, values, begin, count, m, n, original, model != nullptr, true);
  keys<<<blocks(count), 256>>>(plan.unsorted.data, plan.parent.data, count);
  check(cudaGetLastError());
  check(cub::DeviceRadixSort::SortKeys(plan.temporary.data, bytes, plan.unsorted.data, plan.ordered.data, count));
  return true;
}

} // namespace gpu_preprocessor::lp::presolve::parallel_components_detail
