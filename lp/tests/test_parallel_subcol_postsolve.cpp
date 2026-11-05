// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_postsolve.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::size_t fail_bytes = 0;
int allocation_failures = 0;
}
extern "C" cudaError_t __real_cudaMalloc(void**, std::size_t);
extern "C" cudaError_t __wrap_cudaMalloc(void** p, std::size_t bytes) {
  if (fail_bytes && bytes == fail_bytes) {
    *p = nullptr; ++allocation_failures; return cudaErrorMemoryAllocation;
  }
  return __real_cudaMalloc(p, bytes);
}

namespace {
using namespace gpu_preprocessor::lp::presolve;
constexpr int count = 4097, retained = 9, n = count + retained;
const char* option = "GPUPREPROCESSOR_PARALLEL_SUBCOL";
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> T* upload(const std::vector<T>& v) {
  T* p = nullptr;
  if (!v.empty()) {
    check(cudaMalloc(&p, v.size() * sizeof(T)));
    check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  return p;
}
template<class T> void put(T* p, const std::vector<T>& v) {
  check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
}
std::vector<double> read(const double* p, int length) {
  std::vector<double> v(length);
  if (length) check(cudaMemcpy(v.data(), p, length * sizeof(double), cudaMemcpyDeviceToHost));
  return v;
}
void same(const std::vector<double>& a, const std::vector<double>& b) {
  assert(a.size() == b.size());
  assert(std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}
enum class Case { Independent, PermutedColumns, DuplicateColumns, SharedRows, UnorderedRows,
                  ForwardDependency, ReverseDependency, Mixed, NonfiniteX, NonfiniteY,
                  ZeroPivot, NegativeRow, EmptySupport, NullDuals, Split, Checkpoint, ScratchOOM };
void fixture(PresolveRecordGpu& r, Case which, bool device) {
  r.n0 = n; r.n1 = retained; r.m0 = r.m1 = count;
  for (int j = 0; j < retained; ++j) r.col_red2org.push_back(count + j);
  for (int i = 0; i < count; ++i) r.row_red2org.push_back(i);
  auto& t = r.tape;
  for (int k = 0; k < count; ++k) {
    const int terms = which == Case::EmptySupport || k % 13 == 0 ? 0 : (k % 19 ? 7 : 128);
    t.types.push_back(int(PostsolveReductionType::SingletonColumnSubstitution));
    t.dual_modes.push_back(std::uint8_t(PostsolveDualMode::Minimal));
    const int col = which == Case::PermutedColumns ? count - 1 - k : k;
    t.indices.insert(t.indices.end(), {col, k, terms});
    t.vals.insert(t.vals.end(), {k % 2 ? -2.0 : 0.5, k / 7.0, -INFINITY, INFINITY,
                               k / 13.0, k % 3 == 0 ? 1.0 : 0.0});
    for (int q = 0; q < terms; ++q) {
      t.indices.push_back(count + q % retained);
      t.vals.push_back(q % 4 == 3 ? -0.0 : 0.25 + (k % 7) / 17.0);
    }
    t.index_starts.push_back(NnzOffset(t.indices.size()));
    t.value_starts.push_back(NnzOffset(t.vals.size()));
  }
  const int mid = count / 2;
  const auto ip = t.index_starts[mid], vp = t.value_starts[mid];
  if (which == Case::DuplicateColumns) t.indices[ip] = 0;
  if (which == Case::SharedRows) t.indices[ip + 1] = mid - 1;
  if (which == Case::UnorderedRows)
    std::swap(t.indices[ip + 1], t.indices[t.index_starts[mid + 1] + 1]);
  if (which == Case::ForwardDependency || which == Case::ReverseDependency) {
    for (int k = 1; k + 1 < count; ++k)
      if (t.indices[t.index_starts[k] + 2])
        t.indices[t.index_starts[k] + 3] = which == Case::ForwardDependency ? k + 1 : k - 1;
  }
  if (which == Case::Mixed) {
    t.types[mid] = int(PostsolveReductionType::RowDualShift);
    t.indices[ip] = 0; t.indices[ip + 1] = 1; t.indices[ip + 2] = count;
    t.vals[vp] = -1; t.vals[vp + 1] = 0.3;
  }
  if (which == Case::ZeroPivot) t.vals[vp] = 0.0;
  if (which == Case::NegativeRow) { t.indices[ip + 1] = -1; t.vals[vp + 5] = 1.0; }
  if (which == Case::Checkpoint) {
    PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep step;
    step.splits.push_back(PiecewiseLinearReformulationAbsoluteValueSplitRecovery{count, count + 1, 2.0});
    r.piecewise_linear_reformulation_absolute_value_recoveries.push_back(step);
    r.primal_recovery_timeline.push_back(
        PrimalRecoveryCheckpoint{PrimalRecoveryKind::PiecewiseLinearReformulationAbsoluteValue, 0, mid, -1});
  }
  if (!device) return;
  r.device = std::make_shared<DeviceRecoveryState>();
  auto& d = *r.device;
  d.cols.data = upload(r.col_red2org); d.cols.count = retained;
  d.rows.data = upload(r.row_red2org); d.rows.count = count;
  auto segment = [&](int begin, int end) {
    const auto ib = t.index_starts[begin], ie = t.index_starts[end];
    const auto vb = t.value_starts[begin], ve = t.value_starts[end];
    auto s = allocate_recovery_segment(end - begin, ie - ib, ve - vb);
    put(s.types, std::vector<int>(t.types.begin() + begin, t.types.begin() + end));
    put(s.dual_modes, std::vector<std::uint8_t>(t.dual_modes.begin() + begin, t.dual_modes.begin() + end));
    std::vector<NnzOffset> is, vs;
    for (int k = begin; k <= end; ++k) {
      is.push_back(t.index_starts[k] - ib); vs.push_back(t.value_starts[k] - vb);
    }
    put(s.index_starts, is); put(s.value_starts, vs);
    put(s.indices, std::vector<int>(t.indices.begin() + ib, t.indices.begin() + ie));
    put(s.vals, std::vector<double>(t.vals.begin() + vb, t.vals.begin() + ve));
    d.records += s.record_count; d.indices += s.index_count; d.values += s.value_count;
    d.segments.push_back(std::move(s));
  };
  if (which == Case::Split) { segment(0, mid); segment(mid, count); }
  else segment(0, count);
}
void run(Case which, bool device) {
  PresolveRecordGpu r;
  fixture(r, which, device);
  std::vector<double> xv{1e16, 1.0, -1e16, -0.0, 0.3, -0.7, 0.0, 1.5, -2.5}, yv(count), zv(retained);
  for (int j = 0; j < count; ++j) yv[j] = j % 5 ? j / 11.0 : -0.0;
  for (int j = 0; j < retained; ++j) zv[j] = j % 3 ? j / 17.0 : -0.0;
  if (which == Case::NonfiniteX) xv[0] = INFINITY;
  if (which == Case::NonfiniteY) yv[count / 2] = INFINITY;
  auto x = upload(xv);
  auto y = which == Case::NullDuals ? nullptr : upload(yv);
  auto z = which == Case::NullDuals ? nullptr : upload(zv);
  setenv(option, "0", 1);
  auto serial = postsolve_gpu(x, y, z, r);
  const auto sx = read(serial.x_org, n), sy = read(serial.y_org, count), sz = read(serial.z_org, n);
  setenv(option, "1", 1);
  if (which == Case::ScratchOOM) { fail_bytes = std::size_t(n) * sizeof(int); allocation_failures = 0; }
  auto parallel = postsolve_gpu(x, y, z, r);
  if (which == Case::ScratchOOM) { assert(allocation_failures == 1); fail_bytes = 0; }
  same(sx, read(parallel.x_org, n)); same(sy, read(parallel.y_org, count)); same(sz, read(parallel.z_org, n));
  same(xv, read(x, retained));
  if (y) same(yv, read(y, count));
  if (z) same(zv, read(z, retained));
  for (auto p : {serial.x_org, serial.y_org, serial.z_org,
                 parallel.x_org, parallel.y_org, parallel.z_org, x, y, z}) check(cudaFree(p));
}
} // namespace
int main() {
  const char* previous = std::getenv(option);
  const bool existed = previous != nullptr;
  const std::string saved = previous ? previous : "";
  for (bool device : {false, true})
    for (auto which : {Case::Independent, Case::PermutedColumns, Case::DuplicateColumns, Case::SharedRows,
                       Case::UnorderedRows, Case::ForwardDependency, Case::ReverseDependency, Case::Mixed,
                       Case::NonfiniteX, Case::NonfiniteY, Case::ZeroPivot, Case::NegativeRow,
                       Case::EmptySupport, Case::NullDuals, Case::Split, Case::ScratchOOM}) run(which, device);
  run(Case::Checkpoint, false);
  if (existed) setenv(option, saved.c_str(), 1); else unsetenv(option);
  std::cout << "test_parallel_subcol_postsolve passed\n";
}
