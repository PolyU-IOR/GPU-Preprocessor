#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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

void test_postsolve_scatter_and_fixed_column_restore() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::PresolveRecordGpu;

  PresolveRecordGpu record;
  record.m0 = 3;
  record.n0 = 4;
  record.m1 = 2;
  record.n1 = 2;
  record.row_red2org = {0, 2};
  record.row_org2red = {0, -1, 1};
  record.col_red2org = {0, 2};
  record.col_org2red = {0, -1, 1, -1};
  record.fixed_idx = {1, 3};
  record.fixed_val = {5.0, -2.0};

  double* x_red = copy_to_device<double>({10.0, 20.0});
  double* y_red = copy_to_device<double>({1.5, -3.0});
  double* z_red = copy_to_device<double>({0.25, 0.75});

  GpuPostsolveResult result = gpu_presolver::presolve::postsolve_gpu(x_red, y_red, z_red, record);
  const std::vector<double> x = copy_to_host(result.x_org, 4);
  const std::vector<double> y = copy_to_host(result.y_org, 3);
  const std::vector<double> z = copy_to_host(result.z_org, 4);

  assert((x == std::vector<double>{10.0, 5.0, 20.0, -2.0}));
  assert((y == std::vector<double>{1.5, 0.0, -3.0}));
  assert((z == std::vector<double>{0.25, 0.0, 0.75, 0.0}));

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
}

void test_postsolve_allows_null_reduced_dual_inputs() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::PresolveRecordGpu;

  PresolveRecordGpu record;
  record.m0 = 1;
  record.n0 = 1;
  record.m1 = 1;
  record.n1 = 1;
  record.row_red2org = {0};
  record.row_org2red = {0};
  record.col_red2org = {0};
  record.col_org2red = {0};

  double* x_red = copy_to_device<double>({2.0});
  GpuPostsolveResult result = gpu_presolver::presolve::postsolve_gpu(x_red, nullptr, nullptr, record);
  const std::vector<double> x = copy_to_host(result.x_org, 1);
  const std::vector<double> y = copy_to_host(result.y_org, 1);
  const std::vector<double> z = copy_to_host(result.z_org, 1);
  assert((x == std::vector<double>{2.0}));
  assert((y == std::vector<double>{0.0}));
  assert((z == std::vector<double>{0.0}));

  cudaFree(x_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
}

void test_postsolve_keeps_unretrieved_fixed_column_dual_tape_based_with_original_model() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveRecordGpu;

  PresolveRecordGpu record;
  record.m0 = 1;
  record.n0 = 2;
  record.m1 = 1;
  record.n1 = 1;
  record.row_red2org = {0};
  record.row_org2red = {0};
  record.col_red2org = {1};
  record.col_org2red = {-1, 0};
  record.fixed_idx = {0};
  record.fixed_val = {2.0};

  LPInfoGpu original;
  original.AT = {2, 1, 1,
                 copy_to_device<NnzOffset>({0, 1, 1}),
                 copy_to_device<std::int32_t>({0}),
                 copy_to_device<double>({2.0})};
  original.c = copy_to_device<double>({10.0, 5.0});

  double* x_red = copy_to_device<double>({7.0});
  double* y_red = copy_to_device<double>({3.0});
  double* z_red = copy_to_device<double>({1.5});

  GpuPostsolveResult result =
      gpu_presolver::presolve::postsolve_gpu(x_red, y_red, z_red, record, &original);
  const std::vector<double> x = copy_to_host(result.x_org, 2);
  const std::vector<double> z = copy_to_host(result.z_org, 2);
  assert((x == std::vector<double>{2.0, 7.0}));
  assert(std::fabs(z[0]) < 1.0e-12);
  assert(std::fabs(z[1] - 1.5) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
  cudaFree(original.AT.rowPtr);
  cudaFree(original.AT.colVal);
  cudaFree(original.AT.nzVal);
  cudaFree(original.c);
}

void test_presolve_record_drives_fixed_column_postsolve() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::GpuPresolveSummary;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;

  const double inf = std::numeric_limits<double>::infinity();
  LPInfoGpu lp;
  lp.A = {1, 2, 2,
          copy_to_device<NnzOffset>({0, 2}),
          copy_to_device<std::int32_t>({0, 1}),
          copy_to_device<double>({2.0, 3.0})};
  lp.AT = {2, 1, 2,
           copy_to_device<NnzOffset>({0, 1, 2}),
           copy_to_device<std::int32_t>({0, 0}),
           copy_to_device<double>({2.0, 3.0})};
  lp.c = copy_to_device<double>({1.0, 2.0});
  lp.l = copy_to_device<double>({7.0, 0.0});
  lp.u = copy_to_device<double>({7.0, inf});
  lp.AL = copy_to_device<double>({10.0});
  lp.AU = copy_to_device<double>({10.0});

  PresolveParams params;
  params.max_iters = 1;
  params.enable_empty_rows = false;
  params.enable_singleton_rows = false;
  params.enable_infeasible_and_redundant_rows = false;
  params.enable_bound_strengthening = false;
  params.enable_parallel_rows = false;
  params.enable_empty_columns = false;

  params.enable_singleton_columns = false;
  params.enable_doubleton_equations = false;
  params.enable_dual_fix = false;
  params.enable_parallel_columns = false;
  params.enable_absolute_value_epigraph_reduction = false;
  params.enable_startup = false;

  GpuPresolveSummary summary = gpu_presolver::presolve::run_gpu_presolve_with_record(lp, params);
  gpu_presolver::presolve::export_recovery_record(summary.record);
  assert(summary.reduced_rows == 1);
  assert(summary.reduced_cols == 1);
  assert(summary.record.m0 == 1);
  assert(summary.record.n0 == 2);
  assert(summary.record.m1 == 1);
  assert(summary.record.n1 == 1);
  assert((summary.record.col_red2org == std::vector<std::int32_t>{1}));
  assert((summary.record.fixed_idx == std::vector<std::int32_t>{0}));
  assert(std::fabs(summary.record.fixed_val[0] - 7.0) < 1.0e-12);

  double* x_red = copy_to_device<double>({4.0});
  double* y_red = copy_to_device<double>({1.0});
  double* z_red = copy_to_device<double>({0.5});
  GpuPostsolveResult result =
      gpu_presolver::presolve::postsolve_gpu(x_red, y_red, z_red, summary.record, &lp);
  const std::vector<double> x = copy_to_host(result.x_org, 2);
  const std::vector<double> y = copy_to_host(result.y_org, 1);
  const std::vector<double> z = copy_to_host(result.z_org, 2);
  assert((x == std::vector<double>{7.0, 4.0}));
  assert((y == std::vector<double>{1.0}));
  assert((z == std::vector<double>{-1.0, 0.0}));

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(lp.c);
  cudaFree(lp.l);
  cudaFree(lp.u);
  cudaFree(lp.AL);
  cudaFree(lp.AU);
}

void test_infeasible_and_fixed_variables_postsolve(double width, bool expect_fixed) {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::GpuPresolveSummary;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;

  constexpr double lower = 2.0;
  const double upper = lower + width;
  const double midpoint = lower + 0.5 * (upper - lower);
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {1, 2, 2,
          copy_to_device<NnzOffset>({0, 2}),
          copy_to_device<std::int32_t>({0, 1}),
          copy_to_device<double>({1.0, 1.0})};
  lp.AT = {2, 1, 2,
           copy_to_device<NnzOffset>({0, 1, 2}),
           copy_to_device<std::int32_t>({0, 0}),
           copy_to_device<double>({1.0, 1.0})};
  lp.c = copy_to_device<double>({1.0, 0.0});
  lp.l = copy_to_device<double>({lower, 0.0});
  lp.u = copy_to_device<double>({upper, inf});
  lp.AL = copy_to_device<double>({0.0});
  lp.AU = copy_to_device<double>({10.0});

  PresolveParams params;
  params.max_iters = 1;
  params.record_postsolve_tape_cpu = false;
  params.enable_startup = false;
  params.enable_empty_rows = false;
  params.enable_singleton_rows = false;
  params.enable_infeasible_and_redundant_rows = false;
  params.enable_bound_strengthening = false;
  params.enable_parallel_rows = false;
  params.enable_empty_columns = false;

  params.enable_singleton_columns = false;
  params.enable_doubleton_equations = false;
  params.enable_dual_fix = false;
  params.enable_parallel_columns = false;
  params.enable_absolute_value_epigraph_reduction = false;
  params.enable_variable_pair_aggregation = false;
  params.enable_column_domination = false;

  assert(params.bound_tol == 1.0e-6);
  assert(!params.record_postsolve_tape_cpu);
  GpuPresolveSummary summary = gpu_presolver::presolve::run_gpu_presolve_with_record(lp, params);
  gpu_presolver::presolve::export_recovery_record(summary.record);
  assert(summary.reduced_rows == 1);
  if (expect_fixed) {
    assert(summary.reduced_cols == 1);
    assert((summary.record.col_red2org == std::vector<std::int32_t>{1}));
    assert((summary.record.fixed_idx == std::vector<std::int32_t>{0}));
    assert(summary.record.fixed_val == std::vector<double>{midpoint});
  } else {
    assert(summary.reduced_cols == 2);
    assert((summary.record.col_red2org == std::vector<std::int32_t>{0, 1}));
    assert(summary.record.fixed_idx.empty());
    assert(summary.record.fixed_val.empty());
  }

  double* x_red = copy_to_device(expect_fixed ? std::vector<double>{4.0}
                                              : std::vector<double>{lower, 4.0});
  GpuPostsolveResult result =
      gpu_presolver::presolve::postsolve_gpu(x_red, nullptr, nullptr, summary.record);
  const std::vector<double> x = copy_to_host(result.x_org, 2);
  // Recover accepted midpoints exactly; preserve the solver's endpoint when
  // the approximate fixing was rejected by the row-error guard.
  assert(std::fabs(x[0] - (expect_fixed ? midpoint : lower)) < 1.0e-12);
  assert(x[0] >= lower && x[0] <= upper);
  assert(std::fabs(x[1] - 4.0) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(lp.c);
  cudaFree(lp.l);
  cudaFree(lp.u);
  cudaFree(lp.AL);
  cudaFree(lp.AU);
}

void test_empty_column_fixed_value_is_recorded_for_postsolve() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::GpuPresolveSummary;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;

  LPInfoGpu lp;
  lp.A = {0, 1, 0,
          copy_to_device<NnzOffset>({0}),
          nullptr,
          nullptr};
  lp.AT = {1, 0, 0,
           copy_to_device<NnzOffset>({0, 0}),
           nullptr,
           nullptr};
  lp.c = copy_to_device<double>({2.0});
  lp.l = copy_to_device<double>({3.0});
  lp.u = copy_to_device<double>({10.0});
  lp.AL = nullptr;
  lp.AU = nullptr;

  PresolveParams params;
  params.max_iters = 1;
  params.enable_startup = false;
  params.enable_infeasible_and_fixed_variables = false;
  params.enable_empty_rows = false;
  params.enable_singleton_rows = false;
  params.enable_infeasible_and_redundant_rows = false;
  params.enable_bound_strengthening = false;
  params.enable_parallel_rows = false;

  params.enable_singleton_columns = false;
  params.enable_doubleton_equations = false;
  params.enable_dual_fix = false;
  params.enable_parallel_columns = false;
  params.enable_absolute_value_epigraph_reduction = false;

  GpuPresolveSummary summary = gpu_presolver::presolve::run_gpu_presolve_with_record(lp, params);
  gpu_presolver::presolve::export_recovery_record(summary.record);
  assert(summary.reduced_rows == 0);
  assert(summary.reduced_cols == 0);
  assert((summary.record.fixed_idx == std::vector<std::int32_t>{0}));
  assert(std::fabs(summary.record.fixed_val[0] - 3.0) < 1.0e-12);

  GpuPostsolveResult result = gpu_presolver::presolve::postsolve_gpu(nullptr, nullptr, nullptr, summary.record);
  const std::vector<double> x = copy_to_host(result.x_org, 1);
  assert((x == std::vector<double>{3.0}));

  cudaFree(result.x_org);
  cudaFree(result.z_org);
  cudaFree(lp.A.rowPtr);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.c);
  cudaFree(lp.l);
  cudaFree(lp.u);
}

void test_parallel_columnumn_tape_replay_restores_primal_and_dual() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveRecordGpu;

  PresolveRecordGpu record;
  record.m0 = 0;
  record.n0 = 2;
  record.m1 = 0;
  record.n1 = 1;
  record.col_red2org = {0};
  record.col_org2red = {0, -1};
  record.tape.types = {static_cast<std::int32_t>(PostsolveReductionType::ParallelColumn)};
  record.tape.index_starts = {0, 2};
  record.tape.value_starts = {0, 5};
  record.tape.dual_modes = {static_cast<std::uint8_t>(PostsolveDualMode::Minimal)};
  record.tape.indices = {1, 0};
  record.tape.vals = {2.0, 0.0, 4.0, 1.0, 1.0};

  double* x_red = copy_to_device<double>({5.0});
  double* z_red = copy_to_device<double>({0.5});
  GpuPostsolveResult result = gpu_presolver::presolve::postsolve_gpu(x_red, nullptr, z_red, record);
  const std::vector<double> x = copy_to_host(result.x_org, 2);
  const std::vector<double> z = copy_to_host(result.z_org, 2);
  assert(std::fabs(x[0] - 1.0) < 1.0e-12);
  assert(std::fabs(x[1] - 2.0) < 1.0e-12);
  assert(std::fabs(z[0] - 0.5) < 1.0e-12);
  assert(std::fabs(z[1] - 1.0) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(z_red);
  cudaFree(result.x_org);
  cudaFree(result.z_org);
}

void test_doubleton_equations_tape_replay_restores_primal_and_basic_dual() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveRecordGpu;

  PresolveRecordGpu record;
  record.m0 = 1;
  record.n0 = 2;
  record.m1 = 0;
  record.n1 = 1;
  record.col_red2org = {1};
  record.col_org2red = {-1, 0};
  record.tape.types = {static_cast<std::int32_t>(PostsolveReductionType::DoubletonEquation)};
  record.tape.index_starts = {0, 4};
  record.tape.value_starts = {0, 12};
  record.tape.dual_modes = {static_cast<std::uint8_t>(PostsolveDualMode::Minimal)};
  record.tape.indices = {0, 1, 0, 0};
  record.tape.vals = {2.0, 3.0, 8.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 4.0, 0.0, 0.0};

  double* x_red = copy_to_device<double>({2.0});
  double* z_red = copy_to_device<double>({0.5});
  GpuPostsolveResult result = gpu_presolver::presolve::postsolve_gpu(x_red, nullptr, z_red, record);
  const std::vector<double> x = copy_to_host(result.x_org, 2);
  const std::vector<double> y = copy_to_host(result.y_org, 1);
  const std::vector<double> z = copy_to_host(result.z_org, 2);
  assert(std::fabs(x[0] - 1.0) < 1.0e-12);
  assert(std::fabs(x[1] - 2.0) < 1.0e-12);
  assert(std::fabs(y[0] - 2.0) < 1.0e-12);
  assert(std::fabs(z[0]) < 1.0e-12);
  assert(std::fabs(z[1] - 0.5) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(z_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
}

void test_doubleton_equations_original_model_replay_splits_induced_bound_dual() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveRecordGpu;

  const double inf = std::numeric_limits<double>::infinity();
  LPInfoGpu original;
  original.A = {2, 2, 3,
                copy_to_device<NnzOffset>({0, 2, 3}),
                copy_to_device<std::int32_t>({0, 1, 0}),
                copy_to_device<double>({2.0, -2.0, 3.0})};
  original.AT = {2, 2, 3,
                 copy_to_device<NnzOffset>({0, 2, 3}),
                 copy_to_device<std::int32_t>({0, 1, 0}),
                 copy_to_device<double>({2.0, 3.0, -2.0})};
  original.c = copy_to_device<double>({10.0, 0.0});
  original.l = copy_to_device<double>({0.0, 5.0});
  original.u = copy_to_device<double>({inf, 10.0});
  original.AL = copy_to_device<double>({0.0, -inf});
  original.AU = copy_to_device<double>({0.0, inf});

  PresolveRecordGpu record;
  record.m0 = 2;
  record.n0 = 2;
  record.m1 = 1;
  record.n1 = 1;
  record.row_red2org = {1};
  record.col_red2org = {1};
  record.col_org2red = {-1, 0};
  record.tape.types = {static_cast<std::int32_t>(PostsolveReductionType::DoubletonEquation)};
  record.tape.index_starts = {0, 4};
  record.tape.value_starts = {0, 12};
  record.tape.dual_modes = {static_cast<std::uint8_t>(PostsolveDualMode::Minimal)};
  record.tape.indices = {0, 1, 0, 0};
  record.tape.vals = {2.0, -2.0, 0.0, 0.0, inf, 0.0, 10.0, 5.0, 10.0, 10.0, 2.0, 0.0};

  double* x_red = copy_to_device<double>({5.0});
  double* y_red = copy_to_device<double>({1.0});
  double* z_red = copy_to_device<double>({4.0});
  GpuPostsolveResult result = gpu_presolver::presolve::postsolve_gpu(x_red, y_red, z_red, record, &original);
  const std::vector<double> x = copy_to_host(result.x_org, 2);
  const std::vector<double> y = copy_to_host(result.y_org, 2);
  const std::vector<double> z = copy_to_host(result.z_org, 2);
  assert(std::fabs(x[0] - 5.0) < 1.0e-12);
  assert(std::fabs(x[1] - 5.0) < 1.0e-12);
  assert(std::fabs(y[0] - 3.0) < 1.0e-12);
  assert(std::fabs(y[1] - 1.0) < 1.0e-12);
  assert(std::fabs(z[0] - 4.0) < 1.0e-12);
  assert(std::fabs(z[1]) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
  cudaFree(original.A.rowPtr);
  cudaFree(original.A.colVal);
  cudaFree(original.A.nzVal);
  cudaFree(original.AT.rowPtr);
  cudaFree(original.AT.colVal);
  cudaFree(original.AT.nzVal);
  cudaFree(original.c);
  cudaFree(original.l);
  cudaFree(original.u);
  cudaFree(original.AL);
  cudaFree(original.AU);
}

void test_sub_col_tape_replay_restores_primal_and_dual() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveRecordGpu;

  PresolveRecordGpu record;
  record.m0 = 1;
  record.n0 = 3;
  record.m1 = 0;
  record.n1 = 2;
  record.col_red2org = {1, 2};
  record.col_org2red = {-1, 0, 1};
  record.tape.types = {static_cast<std::int32_t>(PostsolveReductionType::SubCol)};
  record.tape.index_starts = {0, 5};
  record.tape.value_starts = {0, 8};
  record.tape.dual_modes = {static_cast<std::uint8_t>(PostsolveDualMode::Minimal)};
  record.tape.indices = {0, 0, 2, 1, 2};
  record.tape.vals = {2.0, 13.0, -10.0, 10.0, 6.0, 1.0, 1.0, 3.0};

  double* x_red = copy_to_device<double>({2.0, 3.0});
  double* z_red = copy_to_device<double>({0.5, 0.7});
  GpuPostsolveResult result = gpu_presolver::presolve::postsolve_gpu(x_red, nullptr, z_red, record);
  const std::vector<double> x = copy_to_host(result.x_org, 3);
  const std::vector<double> y = copy_to_host(result.y_org, 1);
  const std::vector<double> z = copy_to_host(result.z_org, 3);
  assert(std::fabs(x[0] - 1.0) < 1.0e-12);
  assert(std::fabs(y[0] - 3.0) < 1.0e-12);
  assert(std::fabs(z[0]) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(z_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
}

void test_fixed_col_tape_replay_restores_exact_reduced_cost() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveRecordGpu;

  PresolveRecordGpu record;
  record.m0 = 1;
  record.n0 = 2;
  record.m1 = 1;
  record.n1 = 1;
  record.row_red2org = {0};
  record.col_red2org = {1};
  record.tape.types = {static_cast<std::int32_t>(PostsolveReductionType::FixedCol)};
  record.tape.index_starts = {0, 2};
  record.tape.value_starts = {0, 3};
  record.tape.dual_modes = {static_cast<std::uint8_t>(PostsolveDualMode::Minimal)};
  record.tape.indices = {0, 0};
  record.tape.vals = {4.0, 10.0, 2.0};

  double* x_red = copy_to_device<double>({8.0});
  double* y_red = copy_to_device<double>({3.0});
  double* z_red = copy_to_device<double>({0.5});
  GpuPostsolveResult result = gpu_presolver::presolve::postsolve_gpu(x_red, y_red, z_red, record);
  const std::vector<double> x = copy_to_host(result.x_org, 2);
  const std::vector<double> z = copy_to_host(result.z_org, 2);
  assert(std::fabs(x[0] - 4.0) < 1.0e-12);
  assert(std::fabs(z[0] - 4.0) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
}

void test_fme_and_eq_to_ineq_tape_replay() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveRecordGpu;

  PresolveRecordGpu record;
  record.m0 = 2;
  record.n0 = 2;
  record.m1 = 1;
  record.n1 = 1;
  record.row_red2org = {1};
  record.col_red2org = {1};
  record.tape.types = {static_cast<std::int32_t>(PostsolveReductionType::FmeCol),
                       static_cast<std::int32_t>(PostsolveReductionType::EqToIneq)};
  record.tape.index_starts = {0, 5, 6};
  record.tape.value_starts = {0, 5, 6};
  record.tape.dual_modes = {static_cast<std::uint8_t>(PostsolveDualMode::Minimal),
                            static_cast<std::uint8_t>(PostsolveDualMode::Minimal)};
  record.tape.indices = {1, 0, 0, 1, 1, 1};
  record.tape.vals = {1.0, 1.0, 1.0, 3.0, 2.0, 1.25};

  double* x_red = copy_to_device<double>({2.0});
  double* y_red = copy_to_device<double>({0.75});
  double* z_red = copy_to_device<double>({0.5});
  GpuPostsolveResult result = gpu_presolver::presolve::postsolve_gpu(x_red, y_red, z_red, record);
  const std::vector<double> x = copy_to_host(result.x_org, 2);
  const std::vector<double> y = copy_to_host(result.y_org, 2);
  const std::vector<double> z = copy_to_host(result.z_org, 2);
  assert(std::fabs(x[0] - 7.0) < 1.0e-12);
  assert(std::fabs(y[0]) < 1.0e-12);
  assert(std::fabs(y[1] - 2.0) < 1.0e-12);
  assert(std::fabs(z[0]) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
}

void test_parallel_row_original_model_replay_splits_dual_between_rows() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveRecordGpu;

  const double inf = std::numeric_limits<double>::infinity();
  LPInfoGpu original;
  original.A = {2, 1, 2,
                copy_to_device<NnzOffset>({0, 1, 2}),
                copy_to_device<std::int32_t>({0, 0}),
                copy_to_device<double>({1.0, 1.0})};
  original.AT = {1, 2, 2,
                 copy_to_device<NnzOffset>({0, 2}),
                 copy_to_device<std::int32_t>({0, 1}),
                 copy_to_device<double>({1.0, 1.0})};
  original.c = copy_to_device<double>({0.0});
  original.l = copy_to_device<double>({0.0});
  original.u = copy_to_device<double>({inf});
  original.AL = copy_to_device<double>({0.0, 2.0});
  original.AU = copy_to_device<double>({10.0, inf});

  PresolveRecordGpu record;
  record.m0 = 2;
  record.n0 = 1;
  record.m1 = 1;
  record.n1 = 1;
  record.row_red2org = {0};
  record.col_red2org = {0};
  record.tape.types = {static_cast<std::int32_t>(PostsolveReductionType::ParallelRow)};
  record.tape.index_starts = {0, 2};
  record.tape.value_starts = {0, 5};
  record.tape.dual_modes = {static_cast<std::uint8_t>(PostsolveDualMode::Minimal)};
  record.tape.indices = {0, 1};
  record.tape.vals = {1.0, 0.0, 10.0, 2.0, inf};

  double* x_red = copy_to_device<double>({2.0});
  double* y_red = copy_to_device<double>({5.0});
  double* z_red = copy_to_device<double>({0.0});
  GpuPostsolveResult result = gpu_presolver::presolve::postsolve_gpu(x_red, y_red, z_red, record, &original);
  const std::vector<double> y = copy_to_host(result.y_org, 2);
  assert(std::fabs(y[0]) < 1.0e-12);
  assert(std::fabs(y[1] - 5.0) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
  cudaFree(original.A.rowPtr);
  cudaFree(original.A.colVal);
  cudaFree(original.A.nzVal);
  cudaFree(original.AT.rowPtr);
  cudaFree(original.AT.colVal);
  cudaFree(original.AT.nzVal);
  cudaFree(original.c);
  cudaFree(original.l);
  cudaFree(original.u);
  cudaFree(original.AL);
  cudaFree(original.AU);
}

void test_deleted_row_and_bound_change_replay_with_original_model() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveRecordGpu;

  LPInfoGpu original;
  original.A = {1, 1, 1,
                copy_to_device<NnzOffset>({0, 1}),
                copy_to_device<std::int32_t>({0}),
                copy_to_device<double>({2.0})};
  original.AT = {1, 1, 1,
                 copy_to_device<NnzOffset>({0, 1}),
                 copy_to_device<std::int32_t>({0}),
                 copy_to_device<double>({2.0})};
  original.c = copy_to_device<double>({6.0});
  original.l = copy_to_device<double>({0.0});
  original.u = copy_to_device<double>({10.0});
  original.AL = copy_to_device<double>({4.0});
  original.AU = copy_to_device<double>({4.0});

  PresolveRecordGpu record;
  record.m0 = 1;
  record.n0 = 1;
  record.m1 = 0;
  record.n1 = 1;
  record.col_red2org = {0};
  record.tape.types = {static_cast<std::int32_t>(PostsolveReductionType::DeletedRow)};
  record.tape.index_starts = {0, 2};
  record.tape.value_starts = {0, 3};
  record.tape.dual_modes = {static_cast<std::uint8_t>(PostsolveDualMode::Minimal)};
  record.tape.indices = {0, 0};
  record.tape.vals = {4.0, 4.0, 2.0};

  double* x_red = copy_to_device<double>({2.0});
  double* z_red = copy_to_device<double>({0.0});
  GpuPostsolveResult result = gpu_presolver::presolve::postsolve_gpu(x_red, nullptr, z_red, record, &original);
  const std::vector<double> y = copy_to_host(result.y_org, 1);
  const std::vector<double> z = copy_to_host(result.z_org, 1);
  assert(std::fabs(y[0] - 3.0) < 1.0e-12);
  assert(std::fabs(z[0]) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(z_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
  cudaFree(original.A.rowPtr);
  cudaFree(original.A.colVal);
  cudaFree(original.A.nzVal);
  cudaFree(original.AT.rowPtr);
  cudaFree(original.AT.colVal);
  cudaFree(original.AT.nzVal);
  cudaFree(original.c);
  cudaFree(original.l);
  cudaFree(original.u);
  cudaFree(original.AL);
  cudaFree(original.AU);
}

void test_row_bound_change_replay_refines_live_row_duals_from_original_model() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveRecordGpu;

  const double inf = std::numeric_limits<double>::infinity();
  LPInfoGpu original;
  original.A = {1, 1, 1,
                copy_to_device<NnzOffset>({0, 1}),
                copy_to_device<std::int32_t>({0}),
                copy_to_device<double>({2.0})};
  original.AT = {1, 1, 1,
                 copy_to_device<NnzOffset>({0, 1}),
                 copy_to_device<std::int32_t>({0}),
                 copy_to_device<double>({2.0})};
  original.c = copy_to_device<double>({6.0});
  original.l = copy_to_device<double>({0.0});
  original.u = copy_to_device<double>({10.0});
  original.AL = copy_to_device<double>({4.0});
  original.AU = copy_to_device<double>({inf});

  PresolveRecordGpu record;
  record.m0 = 1;
  record.n0 = 1;
  record.m1 = 1;
  record.n1 = 1;
  record.row_red2org = {0};
  record.col_red2org = {0};
  record.tape.types = {static_cast<std::int32_t>(PostsolveReductionType::BoundChangeTheRow)};
  record.tape.index_starts = {0, 2};
  record.tape.value_starts = {0, 4};
  record.tape.dual_modes = {static_cast<std::uint8_t>(PostsolveDualMode::Minimal)};
  record.tape.indices = {0, 0};
  record.tape.vals = {0.0, 10.0, 2.0, 10.0};

  double* x_red = copy_to_device<double>({2.0});
  double* y_red = copy_to_device<double>({0.0});
  double* z_red = copy_to_device<double>({6.0});
  GpuPostsolveResult result = gpu_presolver::presolve::postsolve_gpu(x_red, y_red, z_red, record, &original);
  const std::vector<double> y = copy_to_host(result.y_org, 1);
  const std::vector<double> z = copy_to_host(result.z_org, 1);
  assert(std::fabs(y[0] - 3.0) < 1.0e-12);
  assert(std::fabs(z[0]) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
  cudaFree(original.A.rowPtr);
  cudaFree(original.A.colVal);
  cudaFree(original.A.nzVal);
  cudaFree(original.AT.rowPtr);
  cudaFree(original.AT.colVal);
  cudaFree(original.AT.nzVal);
  cudaFree(original.c);
  cudaFree(original.l);
  cudaFree(original.u);
  cudaFree(original.AL);
  cudaFree(original.AU);
}

void test_bound_change_no_row_replay_projects_unprotected_column_dual() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveRecordGpu;

  const double inf = std::numeric_limits<double>::infinity();
  LPInfoGpu original;
  original.A = {1, 1, 1,
                copy_to_device<NnzOffset>({0, 1}),
                copy_to_device<std::int32_t>({0}),
                copy_to_device<double>({2.0})};
  original.AT = {1, 1, 1,
                 copy_to_device<NnzOffset>({0, 1}),
                 copy_to_device<std::int32_t>({0}),
                 copy_to_device<double>({2.0})};
  original.c = copy_to_device<double>({5.0});
  original.l = copy_to_device<double>({0.0});
  original.u = copy_to_device<double>({inf});
  original.AL = copy_to_device<double>({-inf});
  original.AU = copy_to_device<double>({inf});

  PresolveRecordGpu record;
  record.m0 = 1;
  record.n0 = 1;
  record.m1 = 1;
  record.n1 = 1;
  record.row_red2org = {0};
  record.col_red2org = {0};
  record.tape.types = {static_cast<std::int32_t>(PostsolveReductionType::BoundChangeNoRow)};
  record.tape.index_starts = {0, 1};
  record.tape.value_starts = {0, 0};
  record.tape.dual_modes = {static_cast<std::uint8_t>(PostsolveDualMode::Minimal)};
  record.tape.indices = {0};

  double* x_red = copy_to_device<double>({0.0});
  double* y_red = copy_to_device<double>({1.0});
  double* z_red = copy_to_device<double>({99.0});
  GpuPostsolveResult result = gpu_presolver::presolve::postsolve_gpu(x_red, y_red, z_red, record, &original);
  const std::vector<double> z = copy_to_host(result.z_org, 1);
  assert(std::fabs(z[0] - 3.0) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
  cudaFree(original.A.rowPtr);
  cudaFree(original.A.colVal);
  cudaFree(original.A.nzVal);
  cudaFree(original.AT.rowPtr);
  cudaFree(original.AT.colVal);
  cudaFree(original.AT.nzVal);
  cudaFree(original.c);
  cudaFree(original.l);
  cudaFree(original.u);
  cudaFree(original.AL);
  cudaFree(original.AU);
}

void test_dual_tape_replay_runs_after_absolute_value_epigraph_recovery() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PrimalRecoveryCheckpoint;
  using gpu_presolver::presolve::PrimalRecoveryKind;
  using gpu_presolver::presolve::PresolveRecordGpu;
  using gpu_presolver::presolve::AbsoluteValueEpigraphPrimalRecoveryStep;
  using gpu_presolver::presolve::AbsoluteValueEpigraphSplitRecovery;

  const double inf = std::numeric_limits<double>::infinity();
  LPInfoGpu original;
  original.A = {1, 2, 2,
                copy_to_device<NnzOffset>({0, 2}),
                copy_to_device<std::int32_t>({0, 1}),
                copy_to_device<double>({1.0, 1.0})};
  original.AT = {2, 1, 2,
                 copy_to_device<NnzOffset>({0, 1, 2}),
                 copy_to_device<std::int32_t>({0, 0}),
                 copy_to_device<double>({1.0, 1.0})};
  original.c = copy_to_device<double>({5.0, 0.0});
  original.l = copy_to_device<double>({3.5, -inf});
  original.u = copy_to_device<double>({inf, inf});
  original.AL = copy_to_device<double>({-inf});
  original.AU = copy_to_device<double>({inf});

  PresolveRecordGpu record;
  record.m0 = 1;
  record.n0 = 2;
  record.m1 = 1;
  record.n1 = 2;
  record.row_red2org = {0};
  record.col_red2org = {0, 1};
  record.tape.types = {static_cast<std::int32_t>(PostsolveReductionType::BoundChangeNoRow)};
  record.tape.index_starts = {0, 1};
  record.tape.value_starts = {0, 0};
  record.tape.dual_modes = {static_cast<std::uint8_t>(PostsolveDualMode::Minimal)};
  record.tape.indices = {0};

  AbsoluteValueEpigraphPrimalRecoveryStep step;
  step.pattern = "test_split";
  step.splits.push_back(AbsoluteValueEpigraphSplitRecovery{0, 1, 1.0});
  record.absolute_value_epigraph_recoveries.push_back(step);
  record.primal_recovery_timeline.push_back(PrimalRecoveryCheckpoint{
      PrimalRecoveryKind::AbsoluteValueEpigraph, 0, 0});

  double* x_red = copy_to_device<double>({3.0, 1.0});
  double* y_red = copy_to_device<double>({0.0});
  double* z_red = copy_to_device<double>({99.0, 0.0});
  GpuPostsolveResult result = gpu_presolver::presolve::postsolve_gpu(x_red, y_red, z_red, record, &original);
  const std::vector<double> x = copy_to_host(result.x_org, 2);
  const std::vector<double> z = copy_to_host(result.z_org, 2);
  assert(std::fabs(x[0] - 4.0) < 1.0e-12);
  assert(std::fabs(x[1] - 2.0) < 1.0e-12);
  assert(std::fabs(z[0]) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
  cudaFree(original.A.rowPtr);
  cudaFree(original.A.colVal);
  cudaFree(original.A.nzVal);
  cudaFree(original.AT.rowPtr);
  cudaFree(original.AT.colVal);
  cudaFree(original.AT.nzVal);
  cudaFree(original.c);
  cudaFree(original.l);
  cudaFree(original.u);
  cudaFree(original.AL);
  cudaFree(original.AU);
}

void test_tape_checkpoint_replays_after_structural_primal() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PrimalRecoveryCheckpoint;
  using gpu_presolver::presolve::PrimalRecoveryKind;
  using gpu_presolver::presolve::PresolveRecordGpu;
  using gpu_presolver::presolve::AbsoluteValueEpigraphPrimalRecoveryStep;
  using gpu_presolver::presolve::AbsoluteValueEpigraphSplitRecovery;

  PresolveRecordGpu record;
  record.m0 = 0;
  record.n0 = 3;
  record.m1 = 0;
  record.n1 = 2;
  record.col_red2org = {0, 1};
  record.tape.types = {
      static_cast<std::int32_t>(PostsolveReductionType::FmeCol)};
  record.tape.index_starts = {0, 4};
  record.tape.value_starts = {0, 5};
  record.tape.dual_modes = {
      static_cast<std::uint8_t>(PostsolveDualMode::Minimal)};
  // K=0 deleted rows, eliminate col 2, one support occurrence: col 0.
  record.tape.indices = {0, 2, 1, 0};
  // One lower affine side: z >= x_0.
  record.tape.vals = {1.0, 1.0, 1.0, 0.0, 1.0};

  AbsoluteValueEpigraphPrimalRecoveryStep step;
  step.pattern = "test_tape_checkpoint_order";
  step.splits.push_back(AbsoluteValueEpigraphSplitRecovery{0, 1, 1.0});
  record.absolute_value_epigraph_recoveries.push_back(step);
  record.primal_recovery_timeline.push_back(PrimalRecoveryCheckpoint{
      PrimalRecoveryKind::AbsoluteValueEpigraph, 0, 1});

  double* x_red = copy_to_device<double>({3.0, 1.0});
  double* z_red = copy_to_device<double>({0.0, 0.0});
  GpuPostsolveResult result = gpu_presolver::presolve::postsolve_gpu(
      x_red, nullptr, z_red, record);
  const std::vector<double> x = copy_to_host(result.x_org, 3);
  assert(std::fabs(x[0] - 4.0) < 1.0e-12);
  assert(std::fabs(x[1] - 2.0) < 1.0e-12);
  assert(std::fabs(x[2] - 4.0) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(z_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
}

void test_structural_tape_structural_timeline_replays_exact_reverse_order() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PrimalRecoveryCheckpoint;
  using gpu_presolver::presolve::PrimalRecoveryKind;
  using gpu_presolver::presolve::PresolveRecordGpu;
  using gpu_presolver::presolve::AbsoluteValueEpigraphPrimalRecoveryStep;
  using gpu_presolver::presolve::AbsoluteValueEpigraphSplitRecovery;

  PresolveRecordGpu record;
  record.m0 = 0;
  record.n0 = 4;
  record.m1 = 0;
  record.n1 = 3;
  record.col_red2org = {0, 1, 3};
  record.tape.types = {
      static_cast<std::int32_t>(PostsolveReductionType::FmeCol)};
  record.tape.index_starts = {0, 4};
  record.tape.value_starts = {0, 5};
  record.tape.dual_modes = {
      static_cast<std::uint8_t>(PostsolveDualMode::Minimal)};
  // Forward chronology: S1, eliminate col 2 with x_2 >= x_0, then S2.
  record.tape.indices = {0, 2, 1, 0};
  record.tape.vals = {1.0, 1.0, 1.0, 0.0, 1.0};

  AbsoluteValueEpigraphPrimalRecoveryStep first;
  first.pattern = "timeline_s1";
  first.splits.push_back(AbsoluteValueEpigraphSplitRecovery{0, 1, 1.0});
  record.absolute_value_epigraph_recoveries.push_back(first);
  record.primal_recovery_timeline.push_back(PrimalRecoveryCheckpoint{
      PrimalRecoveryKind::AbsoluteValueEpigraph, 0, 0});

  AbsoluteValueEpigraphPrimalRecoveryStep second;
  second.pattern = "timeline_s2";
  second.splits.push_back(AbsoluteValueEpigraphSplitRecovery{0, 3, 1.0});
  record.absolute_value_epigraph_recoveries.push_back(second);
  record.primal_recovery_timeline.push_back(PrimalRecoveryCheckpoint{
      PrimalRecoveryKind::AbsoluteValueEpigraph, 1, 1});

  double* x_red = copy_to_device<double>({2.0, 1.0, 1.0});
  double* z_red = copy_to_device<double>({0.0, 0.0, 0.0});
  GpuPostsolveResult result = gpu_presolver::presolve::postsolve_gpu(
      x_red, nullptr, z_red, record);
  const std::vector<double> x = copy_to_host(result.x_org, 4);
  assert((x == std::vector<double>{4.0, 2.0, 3.0, 1.0}));

  cudaFree(x_red);
  cudaFree(z_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
}

void test_postsolve_rejects_invalid_primal_recovery_checkpoint() {
  using gpu_presolver::presolve::PrimalRecoveryCheckpoint;
  using gpu_presolver::presolve::PrimalRecoveryKind;
  using gpu_presolver::presolve::PresolveRecordGpu;
  using gpu_presolver::presolve::AbsoluteValueEpigraphPrimalRecoveryStep;
  using gpu_presolver::presolve::AbsoluteValueEpigraphSplitRecovery;

  PresolveRecordGpu record;
  record.n0 = 2;
  AbsoluteValueEpigraphPrimalRecoveryStep step;
  step.splits.push_back(AbsoluteValueEpigraphSplitRecovery{0, 1, 1.0});
  record.absolute_value_epigraph_recoveries.push_back(step);
  record.primal_recovery_timeline.push_back(PrimalRecoveryCheckpoint{
      PrimalRecoveryKind::AbsoluteValueEpigraph, 0, 1});

  bool threw = false;
  try {
    (void)gpu_presolver::presolve::postsolve_gpu(
        nullptr, nullptr, nullptr, record);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
}

void test_parallel_columnumn_rule_records_globalized_postsolve_tape() {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::GpuPresolveSummary;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;

  LPInfoGpu lp;
  lp.A = {2, 2, 4,
          copy_to_device<NnzOffset>({0, 2, 4}),
          copy_to_device<std::int32_t>({0, 1, 0, 1}),
          copy_to_device<double>({1.0, 2.0, 2.0, 4.0})};
  lp.AT = {2, 2, 4,
           copy_to_device<NnzOffset>({0, 2, 4}),
           copy_to_device<std::int32_t>({0, 1, 0, 1}),
           copy_to_device<double>({1.0, 2.0, 2.0, 4.0})};
  lp.c = copy_to_device<double>({1.0, 2.0});
  lp.l = copy_to_device<double>({0.0, 1.0});
  lp.u = copy_to_device<double>({10.0, 3.0});
  lp.AL = copy_to_device<double>({0.0, 0.0});
  lp.AU = copy_to_device<double>({10.0, 10.0});

  PresolveParams params;
  // Propagation, Aggregate, then Parallel Detection.
  params.max_iters = 3;
  params.enable_startup = false;
  params.enable_infeasible_and_fixed_variables = false;
  params.enable_empty_rows = false;
  params.enable_singleton_rows = false;
  params.enable_infeasible_and_redundant_rows = false;
  params.enable_bound_strengthening = false;
  params.enable_parallel_rows = false;
  params.enable_empty_columns = false;

  params.enable_singleton_columns = false;
  params.enable_doubleton_equations = false;
  params.enable_dual_fix = false;
  params.enable_absolute_value_epigraph_reduction = false;

  GpuPresolveSummary summary = gpu_presolver::presolve::run_gpu_presolve_with_record(lp, params);
  gpu_presolver::presolve::export_recovery_record(summary.record);
  assert(summary.reduced_rows == 2);
  assert(summary.reduced_cols == 1);
  assert(summary.record.tape.types.size() == 1);
  assert(summary.record.tape_gpu.record_count == 0);
  assert(summary.record.tape_gpu.index_count == 0);
  assert(summary.record.tape_gpu.value_count == 0);
  assert(summary.record.tape.types[0] == static_cast<std::int32_t>(PostsolveReductionType::ParallelColumn));
  assert((summary.record.tape.indices == std::vector<std::int32_t>{1, 0}));
  assert(std::fabs(summary.record.tape.vals[0] - 2.0) < 1.0e-12);

  double* x_red = copy_to_device<double>({5.0});
  double* y_red = copy_to_device<double>({0.0, 0.0});
  double* z_red = copy_to_device<double>({0.4});
  GpuPostsolveResult result = gpu_presolver::presolve::postsolve_gpu(x_red, y_red, z_red, summary.record);
  const std::vector<double> x = copy_to_host(result.x_org, 2);
  const std::vector<double> z = copy_to_host(result.z_org, 2);
  assert(std::fabs(x[0] - 1.5) < 1.0e-12);
  assert(std::fabs(x[1] - 1.75) < 1.0e-12);
  assert(std::fabs(z[0] - 0.4) < 1.0e-12);
  assert(std::fabs(z[1] - 0.8) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  cudaFree(result.x_org);
  cudaFree(result.y_org);
  cudaFree(result.z_org);
  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(lp.c);
  cudaFree(lp.l);
  cudaFree(lp.u);
  cudaFree(lp.AL);
  cudaFree(lp.AU);
}

}  // namespace

int main() {
  test_postsolve_scatter_and_fixed_column_restore();
  test_postsolve_allows_null_reduced_dual_inputs();
  test_postsolve_keeps_unretrieved_fixed_column_dual_tape_based_with_original_model();
  test_presolve_record_drives_fixed_column_postsolve();
  test_infeasible_and_fixed_variables_postsolve(0.0, true);
  test_infeasible_and_fixed_variables_postsolve(1.0e-8, true);
  test_infeasible_and_fixed_variables_postsolve(5.0e-7, false);
  test_empty_column_fixed_value_is_recorded_for_postsolve();
  test_parallel_columnumn_tape_replay_restores_primal_and_dual();
  test_doubleton_equations_tape_replay_restores_primal_and_basic_dual();
  test_doubleton_equations_original_model_replay_splits_induced_bound_dual();
  test_sub_col_tape_replay_restores_primal_and_dual();
  test_fixed_col_tape_replay_restores_exact_reduced_cost();
  test_fme_and_eq_to_ineq_tape_replay();
  test_parallel_row_original_model_replay_splits_dual_between_rows();
  test_deleted_row_and_bound_change_replay_with_original_model();
  test_row_bound_change_replay_refines_live_row_duals_from_original_model();
  test_bound_change_no_row_replay_projects_unprotected_column_dual();
  test_dual_tape_replay_runs_after_absolute_value_epigraph_recovery();
  test_tape_checkpoint_replays_after_structural_primal();
  test_structural_tape_structural_timeline_replays_exact_reverse_order();
  test_postsolve_rejects_invalid_primal_recovery_checkpoint();
  test_parallel_columnumn_rule_records_globalized_postsolve_tape();
  std::cout << "test_gpu_postsolve passed\n";
  return 0;
}
