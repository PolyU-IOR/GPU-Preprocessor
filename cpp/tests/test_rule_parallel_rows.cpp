#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/rules/rule_parallel_rows.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace runtime = gpu_presolver::runtime;

namespace {

void check(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

template <class T>
T* copy_to_device(const std::vector<T>& values) {
  T* device = nullptr;
  check(cudaMalloc(&device, sizeof(T) * values.size()), "cudaMalloc");
  check(cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy H2D");
  return device;
}

template <class T>
std::vector<T> copy_to_host(T* device, std::size_t size) {
  std::vector<T> values(size);
  check(cudaMemcpy(values.data(), device, sizeof(T) * size, cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H");
  return values;
}

void free_plan(gpu_presolver::presolve::PresolvePlanGpu& plan) {
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
}

void free_matrix(gpu_presolver::presolve::LPInfoGpu& lp) {
  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
}

}  // namespace

void test_deletes_parallel_row_and_tightens_representative() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  LPInfoGpu lp;
  lp.A = {2, 2, 4,
          copy_to_device<NnzOffset>({0, 2, 4}),
          copy_to_device<std::int32_t>({0, 1, 0, 1}),
          copy_to_device<double>({1.0, 1.0, 2.0, 2.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_AL = copy_to_device<double>({0.0, 4.0});
  plan.new_AU = copy_to_device<double>({10.0, 12.0});
  PresolveStatsGpu stats;
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_parallel_rows(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep = copy_to_host(plan.keep_row_mask, 2);
  const std::vector<double> AL = copy_to_host(plan.new_AL, 2);
  const std::vector<double> AU = copy_to_host(plan.new_AU, 2);
  assert((keep[0] == 0 && keep[1] == 1) || (keep[0] == 1 && keep[1] == 0));
  const int rep = keep[0] == 1 ? 0 : 1;
  assert(AL[rep] <= AU[rep]);
  assert(plan.has_row_action);
  assert(plan.has_change);
  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::ParallelRow)}));
  assert((plan.tape.index_starts == std::vector<NnzOffset>{0, 2}));
  assert((plan.tape.value_starts == std::vector<NnzOffset>{0, 5}));
  assert((plan.tape.dual_modes == std::vector<std::uint8_t>{
                                      static_cast<std::uint8_t>(PostsolveDualMode::Exact)}));
  assert(plan.tape.indices.size() == 2);
  assert(plan.tape.indices[0] == rep);
  assert(plan.tape.indices[1] == (rep == 0 ? 1 : 0));
  assert(plan.tape.vals.size() == 5);

  free_plan(plan);
  free_matrix(lp);
}

void test_detects_disjoint_parallel_row_intervals() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  LPInfoGpu lp;
  lp.A = {2, 2, 4,
          copy_to_device<NnzOffset>({0, 2, 4}),
          copy_to_device<std::int32_t>({0, 1, 0, 1}),
          copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_AL = copy_to_device<double>({0.0, 3.0});
  plan.new_AU = copy_to_device<double>({1.0, 4.0});
  PresolveStatsGpu stats;
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_parallel_rows(plan, lp, stats, params);
  assert(plan.has_infeasible);

  free_plan(plan);
  free_matrix(lp);
}

// Compare the exact plan/tape, not just dimensions, under scalar, warp and
// block dispatch. Masked physical row lengths deliberately differ while the
// live rows remain identical; cancellation and tiny pivots exercise fallback.
void test_signature_preserves_plan_and_tape() {
  using namespace gpu_presolver::presolve;
  for (int hash_mode : {0, 1}) for (bool adaptive : {false, true}) {
    for (bool uncertain_case : {false, true}) for (int width : {2, 8, 31, 1024, 1100}) {
      std::vector<NnzOffset> ptr{0};
      std::vector<int> columns;
      std::vector<double> coefficients;
      std::vector<std::uint8_t> col_keep(width + 1, 1);
      col_keep[width] = 0;
      for (int row = 0; row < 10; ++row) {
        for (int col = 0; col < width; ++col) {
          columns.push_back(col);
          const double scale = row == 1 ? -2.0 : row == 2 ? 4.0 : 1.0;
          double value = col == 0 ? 1.0 : ((col & 1) ? 3.0 : -3.0);
          if (row == 3 && col == width-1) value += 0.125;
          if (row >= 4) value = col == 0 ? 1.0 : row + 0.125 * col;
          if (uncertain_case && row == 4) value = col == 0 ? 1e-9 : 1e100;
          if (uncertain_case && row == 5) value = col == 0 ? 2e-9 : 2e100;
          if (uncertain_case && row == 6) value = col == 0 ? 1e-12 : 10.0;
          if (uncertain_case && row == 7) value = col == 0 ? 1.0 : std::numeric_limits<double>::max();
          if (uncertain_case && row == 8) value = col == 0 ? 2.0 : std::numeric_limits<double>::max();
          if (row == 9) value = col == 0 ? 1.0 : 0.0;
          coefficients.push_back(value * scale);
          if ((row & 1) && col % 7 == 3) {
            columns.push_back(width); coefficients.push_back(987.0);
          }
        }
        // These entries must never influence pivot, support or projections.
        if (row & 1) { columns.push_back(width); coefficients.push_back(987.0); }
        ptr.push_back(coefficients.size());
      }
      LPInfoGpu lp;
      lp.A = {10, width+1, static_cast<NnzOffset>(coefficients.size()),
              copy_to_device(ptr), copy_to_device(columns), copy_to_device(coefficients)};
      struct Snapshot {
        std::vector<std::uint8_t> keep;
        std::vector<double> lower, upper;
        PostsolveTape tape;
        bool changed, infeasible;
      } saved;
      for (int signature : {0, 1}) {
        runtime::options().row_signatures = (signature) ? runtime::SignatureMode::Enabled : runtime::SignatureMode::Disabled;
        runtime::options().audit_row_signatures = true;
        runtime::options().hash_scan = adaptive;
        PresolvePlanGpu plan;
        plan.keep_row_mask = copy_to_device<std::uint8_t>({1,1,1,1,1,1,1,1,0,1});
        plan.keep_col_mask = copy_to_device(col_keep);
        plan.new_AL = copy_to_device<double>({0,-12,8,0,0,0,0,0,0,0});
        plan.new_AU = copy_to_device<double>({10,-4,24,10,10,10,10,10,10,10});
        plan.parallel_row_hash_mode = hash_mode;
        PresolveStatsGpu stats; stats.live_masks = true;
        PresolveParams params;
        apply_rule_parallel_rows(plan, lp, stats, params);
        auto keep = copy_to_host(plan.keep_row_mask, 10);
        auto lower = copy_to_host(plan.new_AL, 10);
        auto upper = copy_to_host(plan.new_AU, 10);
        if (!signature) {
          saved = {keep, lower, upper, plan.tape, plan.has_change, plan.has_infeasible};
        } else {
          assert(keep == saved.keep);
          assert(std::memcmp(lower.data(), saved.lower.data(), 10*sizeof(double)) == 0);
          assert(std::memcmp(upper.data(), saved.upper.data(), 10*sizeof(double)) == 0);
          assert(plan.tape.types == saved.tape.types);
          assert(plan.tape.index_starts == saved.tape.index_starts);
          assert(plan.tape.value_starts == saved.tape.value_starts);
          assert(plan.tape.dual_modes == saved.tape.dual_modes);
          assert(plan.tape.indices == saved.tape.indices);
          assert(plan.tape.vals.size() == saved.tape.vals.size());
          assert(plan.tape.vals.empty() || std::memcmp(plan.tape.vals.data(), saved.tape.vals.data(),
                 plan.tape.vals.size()*sizeof(double)) == 0);
          assert(plan.has_change == saved.changed && plan.has_infeasible == saved.infeasible);
        }
        cudaFree(plan.keep_col_mask); free_plan(plan);
      }
      free_matrix(lp);
    }
  }
  runtime::options().row_signatures = runtime::Options{}.row_signatures;
  runtime::options().audit_row_signatures = runtime::Options{}.audit_row_signatures;
  runtime::options().hash_scan = runtime::Options{}.hash_scan;
}

void test_signature_retains_legacy_sign_collision_representative() {
  using namespace gpu_presolver::presolve;
  // Both legacy hashes collide for these three rows. The original seed is
  // row 0, which is not parallel to rows 1/2, so the baseline makes no change.
  // A signed sketch would drop row 0 and newly merge rows 1/2, changing the path.
  for (int mode : {0, 1}) for (int signature : {0, 1}) {
    runtime::options().row_signatures = (signature) ? runtime::SignatureMode::Enabled : runtime::SignatureMode::Disabled;
    runtime::options().audit_row_signatures = true;
    LPInfoGpu lp;
    lp.A = {3, 3, 9, copy_to_device<NnzOffset>({0,3,6,9}),
            copy_to_device<int>({0,1,2,0,1,2,0,1,2}),
            copy_to_device<double>({1,1,1,1,-1,-1,2,-2,-2})};
    PresolvePlanGpu plan;
    plan.keep_row_mask = copy_to_device<std::uint8_t>({1,1,1});
    plan.new_AL = copy_to_device<double>({0,2,4});
    plan.new_AU = copy_to_device<double>({10,6,12});
    plan.parallel_row_hash_mode = mode;
    PresolveStatsGpu stats; PresolveParams params;
    apply_rule_parallel_rows(plan, lp, stats, params);
    assert(!plan.has_change && !plan.has_infeasible);
    assert(copy_to_host(plan.keep_row_mask, 3) == std::vector<std::uint8_t>({1,1,1}));
    assert(copy_to_host(plan.new_AL, 3) == std::vector<double>({0,2,4}));
    assert(copy_to_host(plan.new_AU, 3) == std::vector<double>({10,6,12}));
    assert(plan.tape.types.empty());
    free_plan(plan); free_matrix(lp);
  }
  runtime::options().row_signatures = runtime::Options{}.row_signatures;
  runtime::options().audit_row_signatures = runtime::Options{}.audit_row_signatures;
}

int main() {
  test_deletes_parallel_row_and_tightens_representative();
  test_detects_disjoint_parallel_row_intervals();
  test_signature_preserves_plan_and_tape();
  test_signature_retains_legacy_sign_collision_representative();
  std::cout << "test_rule_parallel_rows passed\n";
  return 0;
}
