// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_postsolve.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace gpu_preprocessor::lp::presolve;
constexpr int count = 4097;
const char* option = "GPUPREPROCESSOR_PARALLEL_FIXED_COLS";
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
std::vector<double> read(const double* p, int n) {
  std::vector<double> v(n);
  if (n) check(cudaMemcpy(v.data(), p, n * sizeof(double), cudaMemcpyDeviceToHost));
  return v;
}
enum class Shape { Sorted, Duplicate, Unordered, Mixed, Checkpoint };
void fixture(PresolveRecordGpu& r, Shape shape, bool device, bool split, bool zero_rows) {
  r.n0 = count + 1; r.n1 = 1; r.m0 = r.m1 = zero_rows ? 0 : 4;
  r.col_red2org = {count};
  if (!zero_rows) r.row_red2org = {0, 1, 2, 3};
  auto& t = r.tape;
  for (int j = 0; j < count; ++j) {
    t.types.push_back(int(PostsolveReductionType::FixedVariable));
    t.dual_modes.push_back(std::uint8_t(PostsolveDualMode::Exact));
    t.indices.push_back(j);
    t.vals.push_back(j % 5 ? j / 7.0 : -0.0);
    t.vals.push_back(j % 11 ? j / 13.0 : -0.0);
    const int terms = !zero_rows && j % 17 ? (j % 23 ? 4 : 100) : 0;
    for (int q = 0; q < terms; ++q) {
      t.indices.push_back(q % 4);
      t.vals.push_back(q % 4 == 1 ? 1.0 : 0.1 + (j % 7) / 17.0);
    }
    t.index_starts.push_back(NnzOffset(t.indices.size()));
    t.value_starts.push_back(NnzOffset(t.vals.size()));
  }
  const auto mid = count / 2;
  if (shape == Shape::Duplicate) t.indices[t.index_starts[mid]] = 0;
  if (shape == Shape::Unordered)
    std::swap(t.indices[t.index_starts[mid]], t.indices[t.index_starts[mid + 1]]);
  if (shape == Shape::Mixed) {
    // A row-dual update changes the input of all earlier fixed-column lifts.
    t.types[mid] = int(PostsolveReductionType::RowDualShift);
    auto ip = t.index_starts[mid], vp = t.value_starts[mid];
    t.indices[ip] = 0; t.indices[ip + 1] = 1; t.indices[ip + 2] = 0;
    t.vals[vp] = -1; t.vals[vp + 1] = 0.3;
  }
  if (shape == Shape::Checkpoint) {
    PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep step;
    step.splits.push_back(PiecewiseLinearReformulationAbsoluteValueSplitRecovery{0, 1, 2.0});
    r.piecewise_linear_reformulation_absolute_value_recoveries.push_back(step);
    r.primal_recovery_timeline.push_back(
        PrimalRecoveryCheckpoint{PrimalRecoveryKind::PiecewiseLinearReformulationAbsoluteValue, 0, mid, -1});
  }
  if (!device) return;
  r.device = std::make_shared<DeviceRecoveryState>();
  auto& d = *r.device;
  d.cols.data = upload(r.col_red2org); d.cols.count = 1;
  d.rows.data = upload(r.row_red2org); d.rows.count = r.m1;
  auto segment = [&](int begin, int end) {
    const auto ib = t.index_starts[begin], ie = t.index_starts[end];
    const auto vb = t.value_starts[begin], ve = t.value_starts[end];
    auto s = allocate_recovery_segment(end - begin, ie - ib, ve - vb);
    put(s.types, std::vector<int>(t.types.begin() + begin, t.types.begin() + end));
    put(s.dual_modes, std::vector<std::uint8_t>(t.dual_modes.begin() + begin, t.dual_modes.begin() + end));
    std::vector<NnzOffset> is, vs;
    for (int j = begin; j <= end; ++j) {
      is.push_back(t.index_starts[j] - ib); vs.push_back(t.value_starts[j] - vb);
    }
    put(s.index_starts, is); put(s.value_starts, vs);
    put(s.indices, std::vector<int>(t.indices.begin() + ib, t.indices.begin() + ie));
    put(s.vals, std::vector<double>(t.vals.begin() + vb, t.vals.begin() + ve));
    d.records += s.record_count; d.indices += s.index_count; d.values += s.value_count;
    d.segments.push_back(std::move(s));
  };
  if (split) { segment(0, mid); segment(mid, count); }
  else segment(0, count);
}
void run(Shape shape, bool device, bool split = false, bool null_duals = false, bool zero_rows = false) {
  PresolveRecordGpu r;
  fixture(r, shape, device, split, zero_rows);
  auto x = upload<double>({2.0});
  auto y = null_duals || zero_rows ? nullptr : upload<double>({1e16, 1.0, -1e16, -0.0});
  auto z = null_duals ? nullptr : upload<double>({-0.0});
  const auto x_before = read(x, 1);
  setenv(option, "0", 1);
  auto serial = postsolve_gpu(x, y, z, r);
  const auto sx = read(serial.x_org, r.n0), sy = read(serial.y_org, r.m0), sz = read(serial.z_org, r.n0);
  setenv(option, "1", 1);
  auto parallel = postsolve_gpu(x, y, z, r);
  const auto px = read(parallel.x_org, r.n0), py = read(parallel.y_org, r.m0), pz = read(parallel.z_org, r.n0);
  assert(std::memcmp(sx.data(), px.data(), sx.size() * sizeof(double)) == 0);
  assert(std::memcmp(sy.data(), py.data(), sy.size() * sizeof(double)) == 0);
  assert(std::memcmp(sz.data(), pz.data(), sz.size() * sizeof(double)) == 0);
  const auto x_after = read(x, 1);
  assert(std::memcmp(x_before.data(), x_after.data(), sizeof(double)) == 0);
  if (y) assert((read(y, 4) == std::vector<double>{1e16, 1.0, -1e16, -0.0}));
  if (z) assert(std::signbit(read(z, 1)[0]));
  for (auto p : {serial.x_org, serial.y_org, serial.z_org,
                 parallel.x_org, parallel.y_org, parallel.z_org, x, y, z}) check(cudaFree(p));
}
} // namespace
int main() {
  const char* value = std::getenv(option);
  const bool existed = value != nullptr;
  const std::string saved = value ? value : "";
  for (bool device : {false, true}) {
    for (Shape shape : {Shape::Sorted, Shape::Duplicate, Shape::Unordered, Shape::Mixed}) run(shape, device);
    run(Shape::Sorted, device, true);
    run(Shape::Sorted, device, false, true);
    run(Shape::Sorted, device, false, true, true);
  }
  run(Shape::Checkpoint, false);
  if (existed) setenv(option, saved.c_str(), 1); else unsetenv(option);
  std::cout << "test_parallel_fixed_postsolve passed\n";
}
