// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_postsolve.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <climits>
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
extern "C" cudaError_t __wrap_cudaMalloc(void** pointer, std::size_t bytes) {
  if (fail_bytes && bytes == fail_bytes) {
    *pointer = nullptr;
    ++allocation_failures;
    return cudaErrorMemoryAllocation;
  }
  return __real_cudaMalloc(pointer, bytes);
}

namespace {
using namespace gpu_preprocessor::lp::presolve;
constexpr int n = 2048, m = 17;
const char* option = "GPUPREPROCESSOR_PARALLEL_BOUND";
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Memory {
  std::vector<void*> pointers;
  ~Memory() { for (auto p : pointers) cudaFree(p); }
  template<class T> T* put(const std::vector<T>& v) {
    T* p = nullptr;
    if (!v.empty()) {
      check(cudaMalloc(&p, v.size() * sizeof(T))); pointers.push_back(p);
      check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    return p;
  }
};
std::vector<double> read(const double* p, int count) {
  std::vector<double> v(count);
  if (count) check(cudaMemcpy(v.data(), p, v.size() * sizeof(double), cudaMemcpyDeviceToHost));
  return v;
}
void same(const std::vector<double>& a, const std::vector<double>& b) {
  assert(a.size() == b.size());
  assert(std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}
enum class Case { Overlap, ReverseColumns, ReverseRows, Adjacent, Missing, Overflow, Nonfinite, Zero,
                  Checkpoint, Mixed, DuplicateEntries, ScratchOOM };
void run(Case which, bool host) {
  Memory mem;
  // The legacy mutation below intentionally decouples a target from its pivot.
  setenv("GPUPREPROCESSOR_PACKED_BOUND_RECORDS",which==Case::DuplicateEntries?"0":"1",1);
  std::vector<NnzOffset> starts{0};
  std::vector<int> cols;
  std::vector<double> coefficients;
  const bool overflow = which == Case::Overflow;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (overflow ? i == 0 : (j % m == i || (j + 3) % m == i || j % 5 == 0)) {
        cols.push_back(j);
        coefficients.push_back(overflow ? 1.0 :
            (i == 8 || j % 23 == 0 ? 0.0 : (j % 2 ? -1.0 : 1.0) * (0.25 + (j % 7) / 5.0 + i / 17.0)));
      }
    }
    starts.push_back(NnzOffset(cols.size()));
  }
  LPInfoGpu lp;
  lp.A = {m, n, NnzOffset(cols.size()), mem.put(starts), mem.put(cols), mem.put(coefficients)};
  PresolveRecordGpu record;
  record.n0 = record.n1 = n; record.m0 = record.m1 = m;
  initialize_device_recovery(record);
  std::vector<int> cm(n), rm(m);
  for (int j = 0; j < n; ++j) cm[j] = which == Case::ReverseColumns ? n - 1 - j : j;
  for (int i = 0; i < m; ++i) rm[i] = which == Case::ReverseRows ? m - 1 - i : i;
  check(cudaMemcpy(record.device->cols.data, cm.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  check(cudaMemcpy(record.device->rows.data, rm.data(), m * sizeof(int), cudaMemcpyHostToDevice));
  std::vector<double> lower(n, 0), upper(n, 100), next_lower(n, 1), next_upper(n, 99), exact(n, 1);
  std::vector<int> lr(n), ur(n);
  std::vector<std::uint8_t> live(n, 1);
  for (int j = 0; j < n; ++j) {
    lr[j] = overflow ? 0 : (j % 19 ? j % m : INT_MAX);
    ur[j] = overflow ? 0 : (j % 29 ? (j + 3) % m : INT_MAX);
    if (which == Case::Missing) lr[j] = ur[j] = INT_MAX;
    if (!overflow && j % 31 == 0) live[j] = 0;
    if (j % 7 == 0) exact[j] += 1e-5;
  }
  for (int batch = 0; batch < (which == Case::Adjacent ? 2 : 1); ++batch) {
    PresolvePlanGpu plan; plan.tape.resident = true;
    append_bound_batch_recovery(plan.tape, lp.A, n, mem.put(live), mem.put(lower), mem.put(upper),
        mem.put(next_lower), mem.put(next_upper), mem.put(exact), mem.put(next_upper), mem.put(lr), mem.put(ur));
    update_device_recovery(record, plan, lp, m, n, 0, nullptr, nullptr);
  }
  if (host) {
    export_recovery_record(record); record.device.reset();
    auto& t = record.tape;
    if (which == Case::Checkpoint) {
      // A complete batch starts at a nonzero tape position after a transform.
      t.types.insert(t.types.begin(), int(PostsolveReductionType::FixedVariable));
      t.dual_modes.insert(t.dual_modes.begin(), std::uint8_t(PostsolveDualMode::Exact));
      t.indices.insert(t.indices.begin(), 0); t.vals.insert(t.vals.begin(), {-0.0, 0.0});
      for (auto& p : t.index_starts) ++p;
      for (auto& p : t.value_starts) p += 2;
      t.index_starts.insert(t.index_starts.begin(), 0);
      t.value_starts.insert(t.value_starts.begin(), 0);
      PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep step;
      step.splits.push_back(PiecewiseLinearReformulationAbsoluteValueSplitRecovery{0, 1, 2.0});
      record.piecewise_linear_reformulation_absolute_value_recoveries.push_back(step);
      record.primal_recovery_timeline.push_back(
          PrimalRecoveryCheckpoint{PrimalRecoveryKind::PiecewiseLinearReformulationAbsoluteValue, 0, 1, -1});
    } else if (which == Case::Mixed) {
      // The later row-dual lift must precede the complete bound batch.
      t.types.push_back(int(PostsolveReductionType::RowDualShift));
      t.dual_modes.push_back(std::uint8_t(PostsolveDualMode::Exact));
      t.indices.insert(t.indices.end(), {0, 1, 0}); t.vals.insert(t.vals.end(), {-1.0, 0.3});
      t.index_starts.push_back(NnzOffset(t.indices.size()));
      t.value_starts.push_back(NnzOffset(t.vals.size()));
    } else if (which == Case::DuplicateEntries) {
      // Preserve repeated target updates and explicit zero coefficients too.
      const auto ip = t.index_starts[0], vp = t.value_starts[0];
      t.indices[ip + 2] = t.indices[ip + 1];
      t.vals[vp + 1] = 0.0;
    }
  }
  std::vector<double> xv(n), yv(m), zv(n);
  for (int j = 0; j < n; ++j) {
    xv[j] = j % 7 ? j / 13.0 : -0.0;
    if (overflow) zv[j] = 1e308;
    else if (which == Case::Zero) zv[j] = j % 2 ? -0.0 : 0.0;
    else zv[j] = j % 4 == 0 ? 1e16 : j % 4 == 1 ? 1.0 : j % 4 == 2 ? -1e16 : -0.0;
  }
  if (which == Case::Nonfinite) zv[1] = std::numeric_limits<double>::infinity();
  for (int i = 0; i < m; ++i) yv[i] = i / 11.0;
  auto x = mem.put(xv), y = mem.put(yv), z = mem.put(zv);
  setenv(option, "0", 1);
  auto before = postsolve_gpu(x, y, z, record);
  const auto bx = read(before.x_org, n), by = read(before.y_org, m), bz = read(before.z_org, n);
  setenv(option, "1", 1);
  if (which == Case::ScratchOOM) {
    fail_bytes = std::count(live.begin(), live.end(), std::uint8_t{1}) * sizeof(std::uint64_t);
    allocation_failures = 0;
  }
  auto after = postsolve_gpu(x, y, z, record);
  if (which == Case::ScratchOOM) { assert(allocation_failures == 1); fail_bytes = 0; }
  same(bx, read(after.x_org, n)); same(by, read(after.y_org, m)); same(bz, read(after.z_org, n));
  same(xv, read(x, n)); same(yv, read(y, m)); same(zv, read(z, n));
  for (auto p : {before.x_org, before.y_org, before.z_org, after.x_org, after.y_org, after.z_org}) check(cudaFree(p));
}
} // namespace
int main() {
  const char* previous = std::getenv(option);
  const bool existed = previous != nullptr;
  const std::string saved = previous ? previous : "";
  for (bool host : {false, true})
    for (auto which : {Case::Overlap, Case::ReverseColumns, Case::ReverseRows, Case::Adjacent,
                       Case::Missing, Case::Overflow, Case::Nonfinite, Case::Zero, Case::ScratchOOM}) run(which, host);
  for (auto which : {Case::Checkpoint, Case::Mixed, Case::DuplicateEntries}) run(which, true);
  if (existed) setenv(option, saved.c_str(), 1); else unsetenv(option);
  std::cout << "test_parallel_bound_postsolve passed\n";
}
