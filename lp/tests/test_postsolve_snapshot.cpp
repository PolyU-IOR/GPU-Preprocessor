// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "../tools/postsolve_replay/snapshot.hpp"
#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_row_sparsification.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

using namespace gpu_preprocessor::lp;
template<class T> T* device(std::vector<T> values) {
  T* result = nullptr;
  if (cudaMalloc(&result, values.size()*sizeof(T)) != cudaSuccess ||
      cudaMemcpy(result, values.data(), values.size()*sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
    throw std::runtime_error("Device allocation failed");
  return result;
}
template<class T> void array(presolve::RecoveryArray<T>& out, std::vector<T> values) {
  out.count = values.size(); out.data = device(std::move(values));
}
template<class F> void rejects(F operation) {
  bool failed = false;
  try { operation(); } catch (const std::exception&) { failed = true; }
  assert(failed);
}
int main() {
  diagnostics::PostsolveSnapshot snapshot;
  auto& p = snapshot.pipeline;
  p.original_rows = 2; p.original_cols = 4;
  p.folded_rows = 1; p.folded_cols = 2; p.folding_applied = true;
  auto& f = p.folding;
  f.applied = f.owns_folded_lp = true;
  f.original_rows = 2; f.original_cols = 4; f.folded_rows = 1; f.folded_cols = 2;
  f.folded_lp.A.rows = 1; f.folded_lp.A.cols = 2;
  auto& map = f.map;
  map.owns_buffers = true; map.original_rows = 2; map.original_cols = 4;
  map.reduced_rows = 1; map.reduced_cols = 2;
  map.row_color_id = device<int>({1,1}); map.col_color_id = device<int>({1,1,2,2});
  map.row_scale = device<double>({0.5,0.5}); map.col_scale = device<double>({0.5,0.5,0.5,0.5});
  auto& s = p.presolve;
  s.original_rows = 1; s.original_cols = 2; s.reduced_rows = s.reduced_cols = 1;
  s.owns_reduced_lp = true;
  s.reduced_lp.A = {1,1,1,device<NnzOffset>({0,1}),device<int>({0}),device<double>({1})};
  s.reduced_lp.c = device<double>({9}); s.reduced_lp.l = device<double>({0}); s.reduced_lp.u = device<double>({10});
  s.reduced_lp.AL = device<double>({2}); s.reduced_lp.AU = device<double>({2});
  s.reduced_lp.obj_constant = 7;
  auto& r = s.record;
  r.m0 = r.m1 = 1; r.n0 = 2; r.n1 = 1;
  r.device = std::make_shared<presolve::DeviceRecoveryState>();
  array(r.device->rows, std::vector<int>{0}); array(r.device->cols, std::vector<int>{0});
  presolve::PostsolveTape tape;
  tape.types = {int(presolve::PostsolveReductionType::FixedVariable),
                int(presolve::PostsolveReductionType::InfiniteBoundDualFix)};
  tape.dual_modes = {unsigned(presolve::PostsolveDualMode::Exact),
                     unsigned(presolve::PostsolveDualMode::Exact)};
  tape.index_starts = {0,1,3}; tape.value_starts = {0,2,4};
  tape.indices = {1,-1,1}; tape.vals = {3,0,0,3};
  presolve::flush_host_recovery_segment(tape);
  r.device->segments.push_back(std::move(tape.segments.front()));
  r.device->records = 2; r.device->indices = 3; r.device->values = 4;
  snapshot.original.A.rows = 2; snapshot.original.A.cols = 4;
  snapshot.x = device<double>({2}); snapshot.y = device<double>({4}); snapshot.z = device<double>({5});
  auto run = [](const diagnostics::PostsolveSnapshot& v) {
    return folding::postsolve_and_unfold_to_host(v.x,v.y,v.z,v.pipeline,&v.original,v.primal_tolerance);
  };
  char directory[] = "postsolve-snapshot-test-XXXXXX";
  if (!mkdtemp(directory)) throw std::runtime_error("Cannot create test directory");
  const std::string path = std::string(directory) + "/state.bin";
  auto original = run(snapshot);
  assert((original.x == std::vector<double>{2,2,3,3}));
  assert((original.y == std::vector<double>{2,2}));
  assert((original.z == std::vector<double>{2.5,2.5,0,0}));
  diagnostics::save_snapshot(path,snapshot.x,snapshot.y,snapshot.z,p,snapshot.original,0);
  diagnostics::save_reference(path+".reference",original);
  auto loaded = diagnostics::load_snapshot(path);
  assert(loaded->format_version == 7 && loaded->pipeline.presolve.owns_reduced_lp);
  assert(loaded->pipeline.presolve.reduced_lp.A.nnz == 1);
  auto replayed = run(*loaded);
  assert(original.x == replayed.x && original.y == replayed.y && original.z == replayed.z);
  auto reference = diagnostics::load_reference(path+".reference");
  assert(original.x == reference.x && original.y == reference.y && original.z == reference.z);
  rejects([&] { diagnostics::save_reference(path+".reference",original); });
  std::ifstream file(path,std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(file)),{});
  { std::ofstream out(path+".truncated",std::ios::binary); out.write(bytes.data(),bytes.size()-3); }
  rejects([&] { diagnostics::load_snapshot(path+".truncated"); });
  const std::uint32_t old_version = 6;
  std::memcpy(bytes.data() + 16, &old_version, sizeof(old_version));
  { std::ofstream out(path+".legacy-v6",std::ios::binary); out.write(bytes.data(),bytes.size()); }
  auto historical = diagnostics::load_snapshot(path+".legacy-v6");
  auto historical_point = run(*historical);
  assert(historical->format_version == 6);
  assert(original.x == historical_point.x && original.y == historical_point.y && original.z == historical_point.z);

  presolve::PostsolveTape mixed;
  mixed.types = {int(presolve::PostsolveReductionType::FixedVariable),
                 int(presolve::PostsolveReductionType::InfiniteBoundDualFix),
                 int(presolve::PostsolveReductionType::InfiniteBoundDualFix)};
  mixed.dual_modes = std::vector<std::uint8_t>(3, unsigned(presolve::PostsolveDualMode::Exact));
  mixed.index_starts = {0,1,3,6}; mixed.value_starts = {0,2,4,5};
  mixed.indices = {1,-1,1,-1,1,0}; mixed.vals = {3,0,0,3,3};
  presolve::flush_host_recovery_segment(mixed);
  r.device->segments.clear(); r.device->segments.push_back(std::move(mixed.segments.front()));
  r.device->records = 3; r.device->indices = 6; r.device->values = 5;
  diagnostics::save_snapshot(path+".mixed-v7",snapshot.x,snapshot.y,snapshot.z,p,snapshot.original,0);
  auto mixed_snapshot = diagnostics::load_snapshot(path+".mixed-v7");
  auto mixed_point = run(*mixed_snapshot);
  assert(mixed_snapshot->format_version == 7);
  assert(original.x == mixed_point.x && original.y == mixed_point.y && original.z == mixed_point.z);

  bytes[16] = 99;
  { std::ofstream out(path+".version",std::ios::binary); out.write(bytes.data(),bytes.size()); }
  rejects([&] { diagnostics::load_snapshot(path+".version"); });

  p.startup = std::make_shared<folding::StartupPresolveState>();
  auto& startup = p.startup->summary;
  startup.original_rows = startup.reduced_rows = 2;
  startup.original_cols = startup.reduced_cols = 4;
  startup.record.m0 = startup.record.m1 = 2; startup.record.n0 = startup.record.n1 = 4;
  presolve::initialize_device_recovery(startup.record);
  startup.reduced_lp.A.rows = 2; startup.reduced_lp.A.cols = 4; startup.owns_reduced_lp = true;
  startup.record.startup_rows = std::make_shared<presolve::StartupRowSparsification>();
  auto& rows = *startup.record.startup_rows;
  rows.transformed.A.rows = 2; rows.transformed.A.cols = 4;
  rows.parents = device<int>({-1,0}); rows.alpha = device<int>({0,1});
  original = run(snapshot);
  diagnostics::save_snapshot(path+".startup",snapshot.x,snapshot.y,snapshot.z,p,snapshot.original,0);
  loaded = diagnostics::load_snapshot(path+".startup"); replayed = run(*loaded);
  assert(original.x == replayed.x && original.y == replayed.y && original.z == replayed.z);
  auto& sr = startup.record;
  sr.piecewise_linear_reformulation_absolute_value_recoveries.resize(1);
  sr.piecewise_linear_reformulation_absolute_value_recoveries[0].pattern = "snapshot-test";
  sr.piecewise_linear_reformulation_absolute_value_recoveries[0].splits.push_back({2,3,2});
  sr.device->piecewise_linear_reformulation_absolute_value.resize(1);
  auto& av = sr.device->piecewise_linear_reformulation_absolute_value[0];
  av.pattern = "snapshot-test";
  av.splits.ids.resize(2); av.splits.values.resize(1);
  array(av.splits.ids[0],std::vector<int>{2}); array(av.splits.ids[1],std::vector<int>{3});
  array(av.splits.values[0],std::vector<double>{2});
  av.pairs.ids.resize(2); av.linked.ids.resize(2); av.linked.values.resize(1);
  sr.primal_recovery_timeline.push_back({presolve::PrimalRecoveryKind::PiecewiseLinearReformulationAbsoluteValue,0,0});
  sr.piecewise_linear_reformulation_variable_pair_recoveries.resize(1);
  auto& hp = sr.piecewise_linear_reformulation_variable_pair_recoveries[0];
  hp.plus_cols = {0}; hp.minus_cols = {1}; hp.root_plus_cols = {2}; hp.root_minus_cols = {3};
  hp.plus_shift = {1}; hp.minus_shift = {-1};
  sr.device->piecewise_linear_reformulation_variable_pair.resize(1);
  auto& pair = sr.device->piecewise_linear_reformulation_variable_pair[0];
  pair.ids.resize(4); pair.values.resize(2);
  array(pair.ids[0],hp.plus_cols); array(pair.ids[1],hp.minus_cols);
  array(pair.ids[2],hp.root_plus_cols); array(pair.ids[3],hp.root_minus_cols);
  array(pair.values[0],hp.plus_shift); array(pair.values[1],hp.minus_shift);
  sr.primal_recovery_timeline.push_back({presolve::PrimalRecoveryKind::PiecewiseLinearReformulationVariablePair,0,0});
  original = run(snapshot);
  diagnostics::save_snapshot(path+".structural",snapshot.x,snapshot.y,snapshot.z,p,snapshot.original,0);
  loaded = diagnostics::load_snapshot(path+".structural"); replayed = run(*loaded);
  assert(original.x == replayed.x && original.y == replayed.y && original.z == replayed.z);
  assert(loaded->pipeline.startup->summary.record.piecewise_linear_reformulation_absolute_value_recoveries[0].pattern == "snapshot-test");
  std::printf("Snapshot roundtrip, folding, tape, structural checkpoints, Startup row inverse and format checks passed: %s\n",directory);
}
