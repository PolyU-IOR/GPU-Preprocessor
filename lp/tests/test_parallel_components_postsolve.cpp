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
#include <string>
#include <vector>

using namespace gpu_preprocessor::lp::presolve;
int check_component_plan(const PostsolveTapeGpu&, int, int, const LPInfoGpu*);
namespace { std::size_t fail_bytes = 0; int failures = 0; }
extern "C" cudaError_t __real_cudaMalloc(void**, std::size_t);
extern "C" cudaError_t __wrap_cudaMalloc(void** p, std::size_t bytes) {
  if (fail_bytes && bytes == fail_bytes) { *p = nullptr; ++failures; return cudaErrorMemoryAllocation; }
  return __real_cudaMalloc(p, bytes);
}
namespace {
constexpr int count = 2048, n = 4 * count + 3, m = 3 * count + 5;
constexpr const char* option = "GPUPREPROCESSOR_PARALLEL_COMPONENTS";
using T = PostsolveReductionType;
void check(cudaError_t e) { assert(e == cudaSuccess); }
template<class V> V* upload(const std::vector<V>& v) {
  V* p = nullptr;
  if (!v.empty()) { check(cudaMalloc(&p, v.size() * sizeof(V))); check(cudaMemcpy(p, v.data(), v.size() * sizeof(V), cudaMemcpyHostToDevice)); }
  return p;
}
template<class V> void put(V* p, const std::vector<V>& v) {
  if (!v.empty()) check(cudaMemcpy(p, v.data(), v.size() * sizeof(V), cudaMemcpyHostToDevice));
}
std::vector<double> read(const double* p, int size) {
  std::vector<double> v(size);
  if (size) check(cudaMemcpy(v.data(), p, size * sizeof(double), cudaMemcpyDeviceToHost));
  return v;
}
void equal(const std::vector<double>& a, const std::vector<double>& b) {
  assert(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}
void append(PostsolveTape& t, T type, std::vector<int> ids, std::vector<double> values) {
  t.types.push_back(int(type)); t.dual_modes.push_back(std::uint8_t(PostsolveDualMode::Exact));
  t.indices.insert(t.indices.end(), ids.begin(), ids.end());
  t.vals.insert(t.vals.end(), values.begin(), values.end());
  t.index_starts.push_back(NnzOffset(t.indices.size())); t.value_starts.push_back(NnzOffset(t.vals.size()));
}
PostsolveTapeGpu segment(const PostsolveTape& t, int begin, int end) {
  const auto ib = t.index_starts[begin], ie = t.index_starts[end], vb = t.value_starts[begin], ve = t.value_starts[end];
  auto s = allocate_recovery_segment(end - begin, ie - ib, ve - vb);
  put(s.types, std::vector<int>(t.types.begin() + begin, t.types.begin() + end));
  put(s.dual_modes, std::vector<std::uint8_t>(t.dual_modes.begin() + begin, t.dual_modes.begin() + end));
  put(s.indices, std::vector<int>(t.indices.begin() + ib, t.indices.begin() + ie));
  put(s.vals, std::vector<double>(t.vals.begin() + vb, t.vals.begin() + ve));
  std::vector<NnzOffset> is, vs;
  for (int k = begin; k <= end; ++k) { is.push_back(t.index_starts[k] - ib); vs.push_back(t.value_starts[k] - vb); }
  put(s.index_starts, is); put(s.value_starts, vs); return s;
}
enum class Pattern { MixedProjectedFormats, IntegerProjected, MixedInfiniteFormats, Independent, Chains, SingleComponent, Mixed, NullDuals, OOM, Checkpoint, Legacy, Unsupported, XChain, YChain, Interleaved, Nonfinite, ZeroPivot, Split };
void fixture(PresolveRecordGpu& r, T type, Pattern pattern, bool device) {
  r.n0 = r.n1 = n; r.m0 = r.m1 = m;
  for (int j = 0; j < n; ++j) r.col_red2org.push_back(j);
  for (int i = 0; i < m; ++i) r.row_red2org.push_back(i);
  for (int k = 0; k < count; ++k) {
    const int owner = pattern == Pattern::SingleComponent ? 0 : (pattern == Pattern::Chains ? k / 8 * 8 : (pattern == Pattern::Interleaved ? k % (count / 8) : k));
    const int keep = count + owner;
    const double sign = k % 2 ? -1.0 : 1.0;
    auto kind = pattern == Pattern::Mixed ? T(k % 15) : type;
    // Batch records are not independent scalar operations.
    if (kind == T::RowImpliedBoundChange) kind = T::DoubletonEquation;
    if (k == count - 1 && pattern == Pattern::Mixed) kind = T::DoubletonEquation;
    switch (kind) {
    case T::DoubletonEquation:
      if (pattern == Pattern::Legacy) append(r.tape, kind, {k, keep, k}, {sign * 2, 0.75, k / 7.0, -INFINITY, INFINITY, 0, 9, 1, 8, k / 11.0});
      else append(r.tape, kind, {k, keep, k, 4, 2 * count, 2 * count + 1, 2 * count + 2, 2 * count},
          {sign * 2, 0.75, k / 7.0, -INFINITY, INFINITY, 0, 9, 1, 8, k / 11.0, double(k % 3), double((k + 1) % 3), 1e16, 1, -1e16, -0.0});
      break;
    case T::ParallelColumn: append(r.tape, kind, {k, keep}, {sign * 2, -2, 4, -3, 5}); break;
    case T::ParallelRow: append(r.tape, kind, {count + owner, k}, {sign * 2, -2, 4, -1, 2}); break;
    case T::DeletedRow:
      append(r.tape, kind, {k, keep}, pattern == Pattern::Legacy ? std::vector<double>{-2, 4, sign * 2} :
          std::vector<double>{-2, 4, sign * 2, 0, 9, 1, 8, double(k % 2), double((k + 1) % 2)}); break;
    case T::RowDualShift:
      if (pattern == Pattern::Legacy) append(r.tape, kind, {owner}, {sign * 0.25});
      else append(r.tape, kind, {owner, 3, keep, keep, 2 * count + k}, {sign, 1e16, -1e16, 0.25});
      break;
    case T::InfiniteBoundDualFix:
      if (pattern == Pattern::Legacy || (pattern == Pattern::MixedInfiniteFormats && k % 2))
        append(r.tape, kind, {k % 2 ? -1 : 1, k, 3, k, keep, keep},
            {1, -sign * INFINITY, k / 7.0, sign * 2, 1e16, -1e16});
      else append(r.tape, kind, {k % 2 ? -1 : 1, k, 1, 3, k, keep, keep},
          {-sign * INFINITY, k / 7.0, sign * 2, 1e16, -1e16});
      break;
    case T::FmeCol:
      if (pattern == Pattern::IntegerProjected || (pattern == Pattern::MixedProjectedFormats && k % 2))
        append(r.tape, kind, {1, k, owner, 3, 2, 1, 0, 1, 3, keep, keep, 2 * count + k},
            {-3, k / 7.0, 1e16, -1e16, 0.25});
      else append(r.tape, kind, {1, k, owner, 3, keep, keep, 2 * count + k},
          {2, 1, 0, -3, 1, 3, k / 7.0, 1e16, -1e16, 0.25}); break;
    case T::FixedVariable: append(r.tape, kind, {k, 2 * count}, {k / 7.0, k / 11.0, 0.25}); break;
    case T::SingletonColumnSubstitution: append(r.tape, kind, {k, k, 2, keep, keep}, {sign * 2, k / 7.0, -INFINITY, INFINITY, 0.5, 0, 1e16, -1e16}); break;
    default: append(r.tape, kind, {k}, {0}); break;
    }
  }
  if (pattern == Pattern::XChain || pattern == Pattern::YChain) {
    auto& t = r.tape;
    for (int k = 0; k < count; ++k) if (k % 8 != 7) {
      const auto ip = t.index_starts[k];
      if (pattern == Pattern::XChain) t.indices[ip + 1] = k + 1;
      else t.indices[ip + 4] = k + 1;
    }
  }
  if (pattern == Pattern::ZeroPivot) r.tape.vals[r.tape.value_starts[count / 2]] = 0;
  if (pattern == Pattern::Unsupported) {
    // Well-formed joint legacy bound batch in a mixed range must fall back.
    auto& t = r.tape; t = PostsolveTape{};
    append(t, T::RowImpliedBoundChange, {0, 0, 0, 1, 1, 0, 0}, {0, 9, 1, 8, 1, 1, 2, 2, 1, 2, 2});
    for (int k = 1; k < count; ++k) append(t, T::ParallelColumn, {k, count + k}, {1, -2, 4, -3, 5});
  }
  if (pattern == Pattern::Checkpoint) {
    PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep step;
    step.splits.push_back({count, count + 1, 2});
    r.piecewise_linear_reformulation_absolute_value_recoveries.push_back(step);
    r.primal_recovery_timeline.push_back({PrimalRecoveryKind::PiecewiseLinearReformulationAbsoluteValue, 0, count / 2, -1});
  }
  auto s = segment(r.tape, 0, count);
  const int groups = check_component_plan(s, m, n, nullptr);
  if (pattern == Pattern::IntegerProjected || pattern == Pattern::MixedProjectedFormats ||
      pattern == Pattern::MixedInfiniteFormats || pattern == Pattern::Independent || pattern == Pattern::Legacy || pattern == Pattern::NullDuals || pattern == Pattern::OOM)
    assert(groups == count);
  if (pattern == Pattern::Chains && type != T::InfiniteBoundDualFix) assert(groups == count / 8);
  if (pattern == Pattern::SingleComponent && type != T::InfiniteBoundDualFix) assert(groups == 1);
  if (pattern == Pattern::Unsupported) assert(groups == -1);
  if (pattern == Pattern::XChain || pattern == Pattern::YChain || pattern == Pattern::Interleaved) assert(groups == count / 8);
  if (type == T::InfiniteBoundDualFix && (pattern == Pattern::Chains || pattern == Pattern::SingleComponent)) assert(groups == count);
  if (!device) return;
  r.device = std::make_shared<DeviceRecoveryState>(); auto& d = *r.device;
  d.cols.data = upload(r.col_red2org); d.cols.count = n; d.rows.data = upload(r.row_red2org); d.rows.count = m;
  d.records = s.record_count; d.indices = s.index_count; d.values = s.value_count;
  if (pattern == Pattern::Split) {
    d.segments.push_back(segment(r.tape, 0, count / 2));
    d.segments.push_back(segment(r.tape, count / 2, count));
  } else d.segments.push_back(std::move(s));
}
void run(T type, Pattern pattern, bool device, bool with_model = false) {
  PresolveRecordGpu r; fixture(r, type, pattern, device);
  std::vector<double> xv(n), yv(m), zv(n);
  for (int j = 0; j < n; ++j) { xv[j] = j % 7 ? (j % 17 - 8) / 3.0 : -0.0; zv[j] = j % 11 ? (j % 13 - 6) / 7.0 : -0.0; }
  for (int i = 0; i < m; ++i) yv[i] = i % 5 ? (i % 19 - 9) / 11.0 : -0.0;
  if (pattern == Pattern::Nonfinite) { xv[count] = INFINITY; yv[2 * count] = INFINITY; zv[count + 1] = -INFINITY; }
  auto x = upload(xv), y = pattern == Pattern::NullDuals ? nullptr : upload(yv), z = pattern == Pattern::NullDuals ? nullptr : upload(zv);
  LPInfoGpu model{};
  std::vector<void*> model_storage;
  auto copy = [&](const auto& v) { auto p = upload(v); model_storage.push_back(p); return p; };
  if (with_model) {
    // Legacy inverses obtain support from AT. A read of the next deleted row
    // joins eight otherwise disjoint records, including interleaved GPU lanes.
    std::vector<NnzOffset> atptr(n + 1), aptr(m + 1);
    std::vector<int> atids, aids;
    std::vector<double> av;
    std::vector<std::vector<int>> rows(m);
    for (int j = 0; j < n; ++j) {
      const int k = type == T::DoubletonEquation ? j : j - count;
      if (k >= 0 && k < count) {
        const int row = k % 8 == 7 ? 2 * count : k + 1;
        atids.push_back(row); rows[row].push_back(j);
      }
      atptr[j + 1] = NnzOffset(atids.size());
    }
    for (int i = 0; i < m; ++i) {
      aids.insert(aids.end(), rows[i].begin(), rows[i].end()); aptr[i + 1] = NnzOffset(aids.size());
    }
    av.assign(atids.size(), 0.25);
    model.A.rows = m; model.A.cols = n; model.A.nnz = NnzOffset(aids.size());
    model.A.rowPtr = copy(aptr); model.A.colVal = copy(aids); model.A.nzVal = copy(av);
    model.AT.rows = n; model.AT.cols = m; model.AT.nnz = NnzOffset(atids.size());
    model.AT.rowPtr = copy(atptr); model.AT.colVal = copy(atids); model.AT.nzVal = copy(av);
    model.c = copy(std::vector<double>(n, 0.5)); model.l = copy(std::vector<double>(n, -2)); model.u = copy(std::vector<double>(n, 4));
    model.AL = copy(std::vector<double>(m, -2)); model.AU = copy(std::vector<double>(m, 4));
    auto tape = segment(r.tape, 0, count);
    assert(check_component_plan(tape, m, n, &model) == count / 8);
  }
  const LPInfoGpu* original = with_model ? &model : nullptr;
  setenv(option, "0", 1); auto old = postsolve_gpu(x, y, z, r, original);
  auto ox = read(old.x_org, n), oy = read(old.y_org, m), oz = read(old.z_org, n);
  setenv(option, "1", 1);
  if (pattern == Pattern::OOM) { fail_bytes = std::size_t(n + m) * sizeof(int); failures = 0; }
  auto next = postsolve_gpu(x, y, z, r, original);
  if (pattern == Pattern::OOM) { assert(failures == 1); fail_bytes = 0; }
  equal(ox, read(next.x_org, n)); equal(oy, read(next.y_org, m)); equal(oz, read(next.z_org, n));
  equal(xv, read(x, n)); if (y) equal(yv, read(y, m)); if (z) equal(zv, read(z, n));
  for (auto p : model_storage) check(cudaFree(p));
  for (auto p : {old.x_org, old.y_org, old.z_org, next.x_org, next.y_org, next.z_org, x, y, z}) check(cudaFree(p));
}
} // namespace
int main() {
  const char* before = std::getenv(option); const bool existed = before != nullptr; const std::string saved = before ? before : "";
  for (bool device : {false, true}) {
    for (auto type : {T::DoubletonEquation, T::ParallelColumn, T::ParallelRow, T::DeletedRow, T::RowDualShift, T::InfiniteBoundDualFix, T::FmeCol})
      for (auto pattern : {Pattern::Independent, Pattern::Chains, Pattern::SingleComponent, Pattern::NullDuals}) run(type, pattern, device);
    for (auto type : {T::DoubletonEquation, T::DeletedRow, T::RowDualShift, T::InfiniteBoundDualFix}) run(type, Pattern::Legacy, device);
    run(T::InfiniteBoundDualFix, Pattern::MixedInfiniteFormats, device);
    for (auto type : {T::DoubletonEquation, T::DeletedRow}) run(type, Pattern::Legacy, device, true);
    for (auto pattern : {Pattern::Mixed, Pattern::OOM, Pattern::Unsupported, Pattern::XChain, Pattern::YChain, Pattern::Interleaved, Pattern::Nonfinite, Pattern::ZeroPivot, Pattern::Split}) run(T::DoubletonEquation, pattern, device);
  }
  for (bool resident : {false, true}) {
    run(T::FmeCol, Pattern::IntegerProjected, resident);
    run(T::FmeCol, Pattern::MixedProjectedFormats, resident);
  }
  run(T::DoubletonEquation, Pattern::Checkpoint, false);
  if (existed) setenv(option, saved.c_str(), 1); else unsetenv(option);
  std::cout << "test_parallel_components_postsolve passed\n";
}
