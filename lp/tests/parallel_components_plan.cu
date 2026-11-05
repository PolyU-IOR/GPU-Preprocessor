// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "../src/presolve/postsolve/parallel_components_postsolve.cuh"
#include <cassert>
#include <vector>

// Test the actual GPU plan as well as its recovered vectors. In particular,
// a byte-equal result obtained only through the fallback is not a parallel pass.
int check_component_plan(const gpu_preprocessor::lp::presolve::PostsolveTapeGpu& tape,
                         int m, int n, const gpu_preprocessor::lp::presolve::LPInfoGpu* model) {
  using namespace gpu_preprocessor::lp::presolve;
  using namespace parallel_components_detail;
  RecoveryArray<int> guard;
  if (!allocate(guard, 1)) throw std::runtime_error("cannot allocate test guard");
  Plan plan;
  if (!prepare(plan, guard.data, tape.types, tape.index_starts, tape.value_starts,
               tape.indices, tape.vals, 0, tape.record_count, m, n, model)) return -1;
  int invalid = 0;
  check(cudaMemcpy(&invalid, guard.data, sizeof(int), cudaMemcpyDeviceToHost));
  if (invalid) return -1;
  std::vector<Key> keys(tape.record_count);
  check(cudaMemcpy(keys.data(), plan.ordered.data, keys.size() * sizeof(Key), cudaMemcpyDeviceToHost));
  int groups = 0;
  std::vector<bool> seen(keys.size());
  for (std::size_t p = 0; p < keys.size(); ++p) {
    const auto record = std::uint32_t(keys[p]);
    assert(record < keys.size() && !seen[record]); seen[record] = true;
    assert(p == 0 || keys[p - 1] < keys[p]);
    if (!p || keys[p - 1] >> 32 != keys[p] >> 32) ++groups;
  }
  return groups;
}
