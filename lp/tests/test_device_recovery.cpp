// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include "gpu_preprocessor/lp/presolve/cleanup_workspace.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_postsolve.hpp"
#include <cassert>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace gpu_preprocessor::lp::presolve;
static void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
template <class T> T *device(const std::vector<T> &v) {
  T *p = nullptr;
  if (!v.empty()) {
    check(cudaMalloc(&p, sizeof(T) * v.size()));
    check(cudaMemcpy(p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
  }
  return p;
}
static std::vector<double> host(const double *p, int n) {
  std::vector<double> v(n);
  check(cudaMemcpy(v.data(), p, sizeof(double) * n, cudaMemcpyDeviceToHost));
  return v;
}
static PostsolveTapeGpu subcol(int row, double rhs) {
  PostsolveTape t;
  t.types = {int(PostsolveReductionType::SingletonColumnSubstitution)};
  t.dual_modes = {unsigned(PostsolveDualMode::Minimal)};
  t.index_starts = {0, 4};
  t.value_starts = {0, 7};
  t.indices = {0, row, 1, 1};
  t.vals = {1, rhs, -10, 10, 0, 0, 1};
  flush_host_recovery_segment(t);
  return std::move(t.segments.front());
}
static void apply(PresolveRecordGpu &r, bool structural, int row, double rhs) {
  const int n = r.n1, m = r.m1;
  LPInfoGpu lp;
  lp.A.rows = m;
  lp.A.cols = n;
  lp.l = device(std::vector<double>(n, -10));
  lp.u = device(std::vector<double>(n, 10));
  PresolvePlanGpu plan;
  plan.tape.resident = true;
  std::vector<int> cols(n);
  std::vector<unsigned char> mask(n, 1);
  for (int i = 0; i < n; ++i)
    cols[i] = i;
  if (structural) {
    auto t = device(std::vector<int>{1}), e = device(std::vector<int>{2});
    auto rho = device(std::vector<double>{2});
    capture_absolute_value_splits_device(plan, t, e, rho, 1, "checkpoint");
    cudaFree(t); cudaFree(e); cudaFree(rho);
  } else {
    mask[0] = 0;
    for (int i = 0; i < n; ++i)
      cols[i] = i - 1;
    append_device_recovery_segment(plan.tape, subcol(row, rhs));
  }
  plan.keep_col_mask = device(mask);
  plan.keep_row_mask = device(std::vector<unsigned char>(m, 1));
  auto ci = device(cols), ri = device(std::vector<int>{0, 1});
  update_device_recovery(r, plan, lp, m, n - (structural ? 0 : 1), 0, structural ? nullptr : ri, structural ? nullptr : ci);
  cudaFree(ci);
  cudaFree(ri);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.keep_row_mask);
  cudaFree(lp.l);
  cudaFree(lp.u);
}
// Exercise threshold boundaries, multiple queue iterations, and mixed serial /
// parallel records against an independently built host index mapping.
static void long_support_mapping(bool reuse) {
  setenv("GPUPREPROCESSOR_CLEANUP_WORKSPACE", reuse ? "1" : "0", 1);
  CleanupWorkspaceScope workspace;
  constexpr int size = 4096;
  PresolveRecordGpu r;
  r.m0 = r.m1 = r.n0 = r.n1 = size;
  initialize_device_recovery(r);
  std::vector<int> rows(size), cols(size);
  for (int i = 0; i < size; ++i) { rows[i] = size-1-i; cols[i] = (37*i+13)%size; }
  check(cudaMemcpy(r.device->rows.data, rows.data(), size*sizeof(int), cudaMemcpyHostToDevice));
  check(cudaMemcpy(r.device->cols.data, cols.data(), size*sizeof(int), cudaMemcpyHostToDevice));
  auto map = [&](int i, bool row) { return i >= 0 && i < size ? (row ? rows[i] : cols[i]) : i; };
  LPInfoGpu lp; lp.A.rows = lp.A.cols = size;
  for (int pass = 0; pass < 2; ++pass) {
    PostsolveTape tape;
    std::vector<int> expected;
    tape.index_starts = {0}; tape.value_starts = {0};
    auto add = [&](PostsolveReductionType type, int length, bool integer_count = false) {
      std::vector<int> ids, mapped;
      auto index = [&](int i, bool row) { ids.push_back(i); mapped.push_back(map(i,row)); };
      auto literal = [&](int i) { ids.push_back(i); mapped.push_back(i); };
      auto support = [&](bool row) {
        for (int j = 0; j < length; ++j) index(j%31 == 0 ? -1 : (j*17)%size, row);
      };
      if (type == PostsolveReductionType::FixedVariable) { index(11,false); support(true); }
      if (type == PostsolveReductionType::SingletonColumnSubstitution) {
        index(19,false); index(23,true); literal(length); support(false);
      }
      if (type == PostsolveReductionType::DoubletonEquation) {
        index(7,false); index(13,false); index(29,true); literal(length); support(true);
      }
      if (type == PostsolveReductionType::RowDualShift) {
        index(23,true); literal(length); support(false);
      }
      if (type == PostsolveReductionType::BoundSourceRow) { index(23,true); support(false); }
      if (type == PostsolveReductionType::InfiniteBoundDualFix) {
        literal(1); index(17,false); if(integer_count)literal(2); literal(length); support(false);
        literal(2); index(31,false); index(size+1,false);
      }
      if (type == PostsolveReductionType::FmeCol) {
        literal(2); index(17,false); index(23,true); index(29,true); literal(length);
        if (integer_count) { literal(2); literal(1); literal(0); literal(-1); literal(length); }
        support(false);
      }
      tape.types.push_back(int(type)); tape.dual_modes.push_back(unsigned(PostsolveDualMode::Minimal));
      tape.indices.insert(tape.indices.end(), ids.begin(), ids.end());
      expected.insert(expected.end(), mapped.begin(), mapped.end());
      tape.index_starts.push_back(tape.indices.size());
      if (type == PostsolveReductionType::InfiniteBoundDualFix) {
        std::vector<double> values(ids.size() - (integer_count ? 2 : 0), 1.25);
        if (!integer_count) values[0] = 2;
        tape.vals.insert(tape.vals.end(), values.begin(), values.end());
      } else if (type == PostsolveReductionType::FmeCol) {
        std::vector<double> values(length, 0.25);
        if (integer_count) { values.insert(values.begin(), {0.0, 1.25}); }
        else { values.insert(values.begin(), {2.0, 1.0, 0.0, 0.0, -1.0, double(length), 1.25}); }
        tape.vals.insert(tape.vals.end(), values.begin(), values.end());
      } else tape.vals.push_back(1.25);
      tape.value_starts.push_back(tape.vals.size());
    };
    for (auto type : {PostsolveReductionType::FixedVariable, PostsolveReductionType::SingletonColumnSubstitution,
                     PostsolveReductionType::DoubletonEquation, PostsolveReductionType::InfiniteBoundDualFix,
                     PostsolveReductionType::RowDualShift, PostsolveReductionType::BoundSourceRow,
                     PostsolveReductionType::FmeCol})
      for (int len : {0,1,252,253,254,255,256,257,1024,65536}) add(type,len);
    for (int len : {0,1,252,253,254,255,256,257,1024,65536})
      { add(PostsolveReductionType::InfiniteBoundDualFix,len,true);
        add(PostsolveReductionType::FmeCol,len,true); }
    for (int i=0;i<300;++i) add(PostsolveReductionType::SingletonColumnSubstitution,257);
    const auto starts=tape.index_starts, value_starts=tape.value_starts;
    const auto types=tape.types; const auto values=tape.vals; const auto modes=tape.dual_modes;
    flush_host_recovery_segment(tape);
    PresolvePlanGpu plan; plan.tape.resident=true;
    append_device_recovery_segment(plan.tape,std::move(tape.segments.front()));
    update_device_recovery(r,plan,lp,size,size,0,nullptr,nullptr);
    auto& segment=r.device->segments.back();
    std::vector<int> actual(expected.size());
    check(cudaMemcpy(actual.data(),segment.indices,actual.size()*sizeof(int),cudaMemcpyDeviceToHost));
    assert(actual==expected);
    auto unchanged = [&](const auto& original, const auto* ptr) {
      auto actual = original;
      check(cudaMemcpy(actual.data(),ptr,actual.size()*sizeof(*ptr),cudaMemcpyDeviceToHost));
      assert(actual==original);
    };
    unchanged(starts,segment.index_starts); unchanged(value_starts,segment.value_starts);
    unchanged(types,segment.types); unchanged(modes,segment.dual_modes); unchanged(values,segment.vals);
  }
}
// Reject malformed headers and truncated row payloads on both validation paths.
static void invalid_infinite_records() {
  const std::vector<std::pair<std::vector<int>, std::vector<double>>> cases{
      {{1,0,-1}, {0}}, {{1,0,1}, {0}}, {{1,0,0,0}, {0,0}},
      {{1,0,1,2,0}, {0,0,1}}, {{1,0}, {0.5,0}},
      {{1,0}, {INFINITY,0}}, {{1,0,-1}, {1,0,0}}};
  for (const auto& item : cases) for (bool resident : {false, true}) {
    PresolveRecordGpu r;
    r.n0 = r.n1 = 1;
    r.col_red2org = {0}; r.col_org2red = {0};
    auto& t = r.tape;
    t.types = {int(PostsolveReductionType::InfiniteBoundDualFix)};
    t.dual_modes = {unsigned(PostsolveDualMode::Exact)};
    t.indices = item.first; t.vals = item.second;
    t.index_starts = {0,NnzOffset(t.indices.size())};
    t.value_starts = {0,NnzOffset(t.vals.size())};
    if (resident) {
      initialize_device_recovery(r);
      r.device->records = 1; r.device->indices = t.indices.size();
      r.device->values = t.vals.size();
      flush_host_recovery_segment(t);
      r.device->segments.push_back(std::move(t.segments.front()));
    }
    bool rejected = false;
    try {
      if (resident) validate_device_recovery(r);
      else {
        auto result = postsolve_gpu(nullptr, nullptr, nullptr, r);
        cudaFree(result.x_org); cudaFree(result.y_org); cudaFree(result.z_org);
      }
    } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
  }
}
static void invalid_projected_records() {
  const std::vector<std::pair<std::vector<int>,std::vector<double>>> cases{
    {{0,0,1,1,1,2,1},{0.,1.}}, // Length exceeds the payload.
    {{0,0,1,1,1,-1,1},{0.,1.}},
    {{0,0,1,1,1,1,1},{0.}},
    {{0,0,1,1},{1.,1.,1.5,0.,1.}}, // Fractional legacy length.
    {{0,0,1,1},{INFINITY,1.,1.,0.,1.}},
    {{-1,0,0},{0.}}
  };
  for (const auto& entry : cases) for (bool resident : {false,true}) {
    PresolveRecordGpu r; r.m0=r.m1=0; r.n0=r.n1=2; r.col_red2org={0,1};
    auto& t=r.tape; t.types={int(PostsolveReductionType::FmeCol)};
    t.dual_modes={unsigned(PostsolveDualMode::Minimal)};
    t.indices=entry.first; t.vals=entry.second;
    t.index_starts={0,NnzOffset(t.indices.size())};
    t.value_starts={0,NnzOffset(t.vals.size())};
    bool rejected=false;
    try {
      if (resident) {
        initialize_device_recovery(r); flush_host_recovery_segment(t);
        auto segment=std::move(t.segments.front());
        r.device->records=segment.record_count; r.device->indices=segment.index_count;
        r.device->values=segment.value_count; r.device->segments.push_back(std::move(segment));
        t=PostsolveTape{}; validate_device_recovery(r);
      } else {
        auto out=postsolve_gpu(nullptr,nullptr,nullptr,r);
        cudaFree(out.x_org);cudaFree(out.y_org);cudaFree(out.z_org);
      }
    } catch(const std::runtime_error&) { rejected=true; }
    assert(rejected);
  }
}

// The same affine arithmetic must recover identical bytes from legacy,
// integer-metadata and mixed tapes, including dependent expressions.
static void projected_record_formats() {
  std::vector<double> reference;
  for (bool resident : {false, true}) for (int format : {0, 1, 2}) {
    PresolveRecordGpu r;
    r.m0 = 2; r.m1 = 1; r.n0 = 5; r.n1 = 3;
    r.row_red2org = {1}; r.col_red2org = {2, 3, 4};
    auto add = [&](bool integer, bool first) {
      std::vector<int> ids;
      std::vector<double> vals;
      if (first) {
        ids = integer ? std::vector<int>{0,1,1,1,-1,1,0} : std::vector<int>{0,1,1,0};
        vals = integer ? std::vector<double>{2.,.5} : std::vector<double>{1.,-1.,1.,2.,.5};
      } else {
        ids = integer ? std::vector<int>{1,0,0,3,2,1,0,1,3,2,3,2} : std::vector<int>{1,0,0,3,2,3,2};
        vals = integer ? std::vector<double>{-3.,.5,1e16,-1e16,.25} :
                         std::vector<double>{2.,1.,0.,-3.,1.,3.,.5,1e16,-1e16,.25};
      }
      auto& t = r.tape; t.types.push_back(int(PostsolveReductionType::FmeCol));
      t.dual_modes.push_back(unsigned(PostsolveDualMode::Minimal));
      t.indices.insert(t.indices.end(),ids.begin(),ids.end());
      t.vals.insert(t.vals.end(),vals.begin(),vals.end());
      t.index_starts.push_back(NnzOffset(t.indices.size()));
      t.value_starts.push_back(NnzOffset(t.vals.size()));
    };
    add(format != 0, true); add(format == 1, false);
    if (resident) {
      initialize_device_recovery(r);
      cudaFree(r.device->rows.data); cudaFree(r.device->cols.data);
      r.device->rows.data = device(r.row_red2org); r.device->rows.count = r.m1;
      r.device->cols.data = device(r.col_red2org); r.device->cols.count = r.n1;
      flush_host_recovery_segment(r.tape);
      auto segment = std::move(r.tape.segments.front());
      r.device->records = segment.record_count; r.device->indices = segment.index_count;
      r.device->values = segment.value_count;
      r.device->segments.push_back(std::move(segment));
      r.tape = PostsolveTape{};
      validate_device_recovery(r);
    }
    auto* x = device(std::vector<double>{1.,1.,2.});
    auto* y = device(std::vector<double>{.5});
    auto* z = device(std::vector<double>{.25,-.5,0.});
    auto out = postsolve_gpu(x,y,z,r);
    auto values = host(out.x_org,5), yy = host(out.y_org,2), zz = host(out.z_org,5);
    assert(values[0] == .25 && values[1] == 2.125);
    values.insert(values.end(),yy.begin(),yy.end()); values.insert(values.end(),zz.begin(),zz.end());
    if (reference.empty()) reference = values;
    else assert(values.size()==reference.size() &&
                std::memcmp(values.data(),reference.data(),values.size()*sizeof(double))==0);
    assert(host(x,3)==std::vector<double>({1.,1.,2.}));
    cudaFree(x); cudaFree(y); cudaFree(z);
    cudaFree(out.x_org); cudaFree(out.y_org); cudaFree(out.z_org);
  }
}

int main() {
  invalid_projected_records();
  projected_record_formats();
  invalid_infinite_records();
  long_support_mapping(false);
  long_support_mapping(true);
  PresolveRecordGpu r;
  r.m0 = r.m1 = 2;
  r.n0 = r.n1 = 4;
  initialize_device_recovery(r);
  apply(r, false, 0, 3);
  apply(r, true, 0, 0);
  apply(r, false, 1, 4);
  assert(r.tape.types.empty() && r.row_red2org.empty() && r.col_red2org.empty());
  assert(r.piecewise_linear_reformulation_absolute_value_recoveries[0].splits.empty());
  assert(r.device->segments.size() == 2 && r.device->records == 2 &&
         r.device->host_staged_bytes == 0);
  auto x = device(std::vector<double>{1, 2}), y = device(std::vector<double>{0, 0}),
       z = device(std::vector<double>{0, 0});
  auto gpu = postsolve_gpu(x, y, z, r); // This must work without any host snapshot.
  auto gx = host(gpu.x_org, 4), gy = host(gpu.y_org, 2), gz = host(gpu.z_org, 4);
  export_recovery_record(r);
  assert((r.col_red2org == std::vector<int>{2, 3}));
  assert((r.col_org2red == std::vector<int>{-1, -1, 0, 1}));
  assert((r.tape.indices == std::vector<int>{0, 0, 1, 1, 1, 1, 1, 2}));
  assert((r.tape.index_starts == std::vector<NnzOffset>{0, 4, 8}));
  assert(r.primal_recovery_timeline.size() == 1 &&
         r.primal_recovery_timeline[0].tape_position == 1);
  assert(r.piecewise_linear_reformulation_absolute_value_recoveries[0].splits[0].t_col == 2 &&
         r.piecewise_linear_reformulation_absolute_value_recoveries[0].splits[0].e_col == 3);
  auto retained = r.device;
  r.device.reset();
  auto legacy = postsolve_gpu(x, y, z, r);
  assert(gx == host(legacy.x_org, 4));
  assert(gy == host(legacy.y_org, 2));
  assert(gz == host(legacy.z_org, 4));
  for (auto p :
       {gpu.x_org, gpu.y_org, gpu.z_org, legacy.x_org, legacy.y_org, legacy.z_org, x, y, z})
    cudaFree(p);
  r.device = retained;
  int invalid_type = 999;
  check(
      cudaMemcpy(r.device->segments[0].types, &invalid_type, sizeof(int), cudaMemcpyHostToDevice));
  bool malformed = false;
  try {
    validate_device_recovery(r);
  } catch (const std::runtime_error &) {
    malformed = true;
  }
  assert(malformed);
  int valid_type = int(PostsolveReductionType::SingletonColumnSubstitution);
  check(cudaMemcpy(r.device->segments[0].types, &valid_type, sizeof(int), cudaMemcpyHostToDevice));
  validate_device_recovery(r);
  bool rejected = false;
  try {
    auto bad = allocate_recovery_segment(-1, 0, 0);
  } catch (const std::exception &) {
    rejected = true;
  }
  assert(rejected);
  PostsolveTape invalid;
  invalid.resident = true;
  invalid.types = {0};
  rejected = false;
  try {
    flush_host_recovery_segment(invalid);
  } catch (const std::logic_error &) {
    rejected = true;
  }
  assert(rejected);
  check(cudaDeviceSynchronize());
  std::cout << "device segments, old-map globalization, interleaved checkpoint, direct replay and "
               "explicit export passed\n";
}
