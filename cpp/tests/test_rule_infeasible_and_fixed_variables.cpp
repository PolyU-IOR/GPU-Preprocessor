#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/rules/rule_infeasible_and_fixed_variables.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
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
  check(cudaMalloc(&device, sizeof(T) * values.size()), "cudaMalloc copy_to_device");
  check(cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy copy_to_device");
  return device;
}

template <class T>
std::vector<T> copy_to_host(T* device, std::size_t size) {
  std::vector<T> values(size);
  check(cudaMemcpy(values.data(), device, sizeof(T) * size, cudaMemcpyDeviceToHost),
        "cudaMemcpy copy_to_host");
  return values;
}

void free_all(std::initializer_list<void*> pointers) {
  for (void* pointer : pointers) {
    if (pointer != nullptr) {
      cudaFree(pointer);
    }
  }
}

}  // namespace

void test_fixed_variable(double width) {
  using gpu_presolver::presolve::DeviceCsrMatrix;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;

  constexpr double lower = 2.0;
  const double upper = lower + width;
  const double midpoint = lower + 0.5 * (upper - lower);

  // Julia indexing equivalent:
  // AT has 2 columns-as-rows. Column 1 appears in row 1 with coeff 2.0 and row 2 with coeff -1.0.
  // Column 2 appears in row 2 with coeff 3.0.
  NnzOffset* at_row_ptr = copy_to_device<NnzOffset>({0, 2, 3});
  std::int32_t* at_col_val = copy_to_device<std::int32_t>({0, 1, 1});
  double* at_nz_val = copy_to_device<double>({2.0, -1.0, 3.0});

  std::uint8_t* keep_col = copy_to_device<std::uint8_t>({1, 1});
  std::uint8_t* keep_row = copy_to_device<std::uint8_t>({1, 1});
  double* new_c = copy_to_device<double>({4.0, 5.0});
  double* new_l = copy_to_device<double>({lower, 0.0});
  double* new_u = copy_to_device<double>({upper, 10.0});
  double* new_AL = copy_to_device<double>({5.0, -3.0});
  double* new_AU = copy_to_device<double>({7.0, 9.0});

  LPInfoGpu lp;
  lp.A.rows = 2;
  lp.A.cols = 2;
  lp.A.nnz = 3;
  lp.AT = DeviceCsrMatrix{2, 2, 3, at_row_ptr, at_col_val, at_nz_val};

  PresolvePlanGpu plan;
  plan.keep_row_mask = keep_row;
  plan.keep_col_mask = keep_col;
  plan.new_c = new_c;
  plan.new_l = new_l;
  plan.new_u = new_u;
  plan.new_AL = new_AL;
  plan.new_AU = new_AU;

  PresolveParams params;
  params.bound_tol = 1.0e-6;

  gpu_presolver::presolve::apply_rule_infeasible_and_fixed_variables(plan, lp, params);

  const std::vector<std::uint8_t> keep_col_after = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> l_after = copy_to_host(plan.new_l, 2);
  const std::vector<double> u_after = copy_to_host(plan.new_u, 2);
  const std::vector<double> al_after = copy_to_host(plan.new_AL, 2);
  const std::vector<double> au_after = copy_to_host(plan.new_AU, 2);

  assert(keep_col_after[0] == 0);
  assert(keep_col_after[1] == 1);
  assert(std::fabs(l_after[0] - midpoint) < 1.0e-12);
  assert(std::fabs(u_after[0] - midpoint) < 1.0e-12);
  assert(std::fabs(al_after[0] - (5.0 - 2.0 * midpoint)) < 1.0e-12);
  assert(std::fabs(au_after[0] - (7.0 - 2.0 * midpoint)) < 1.0e-12);
  assert(std::fabs(al_after[1] - (-3.0 + midpoint)) < 1.0e-12);
  assert(std::fabs(au_after[1] - (9.0 + midpoint)) < 1.0e-12);
  assert(std::fabs(plan.obj_constant_delta - 4.0 * midpoint) < 1.0e-12);
  assert(plan.has_col_action);
  assert(plan.has_change);
  assert(plan.tape.types.empty());

  free_all({
      at_row_ptr,
      at_col_val,
      at_nz_val,
      keep_col,
      keep_row,
      new_c,
      new_l,
      new_u,
      new_AL,
      new_AU,
  });

}

void test_narrow_intervals_preserve_feasible_equality(std::int32_t narrow_count,
                                                     double coefficient,
                                                     bool alternate_signs = false) {
  using namespace gpu_presolver::presolve;
  // A known feasible point is x[0] = 2, x[1:] = 0. The worst-case activity
  // perturbation exceeds feasibility_tol, both for many small contributions
  // and one large one. Opposite coefficient signs must not cancel this bound.
  const std::int32_t n = narrow_count + 1;
  const double width = 5.0e-8;
  PresolveParams params;
  assert(narrow_count * coefficient * width / 2 > params.feasibility_tol);
  assert(width < params.bound_tol);

  std::vector<std::int32_t> at_ptr(n + 1);
  for (std::int32_t j = 0; j <= n; ++j) at_ptr[j] = j;
  std::vector<double> values(n, coefficient), lower(n, 0.0), upper(n, width);
  if (alternate_signs) {
    for (std::int32_t j = 2; j < n; j += 2) values[j] = -coefficient;
  }
  values[0] = 1.0;
  lower[0] = upper[0] = 2.0;

  LPInfoGpu lp;
  lp.A.rows = 1;
  lp.A.cols = n;
  lp.A.nnz = n;
  lp.AT = DeviceCsrMatrix{n, 1, n, copy_to_device(std::vector<NnzOffset>(at_ptr.begin(),at_ptr.end())),
      copy_to_device(std::vector<std::int32_t>(n, 0)), copy_to_device(values)};
  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device(std::vector<std::uint8_t>(n, 1));
  plan.new_c = copy_to_device(std::vector<double>(n, 1.0));
  plan.new_l = copy_to_device(lower);
  plan.new_u = copy_to_device(upper);
  plan.new_AL = copy_to_device<double>({2.0});
  plan.new_AU = copy_to_device<double>({2.0});

  apply_rule_infeasible_and_fixed_variables(plan, lp, params);
  const auto keep = copy_to_host(plan.keep_col_mask, n);
  const auto l = copy_to_host(plan.new_l, n);
  const auto u = copy_to_host(plan.new_u, n);
  assert(keep[0] == 0);
  for (std::int32_t j = 1; j < n; ++j) {
    assert(keep[j] == 1);
    assert(l[j] == 0.0 && u[j] == width);
  }
  // The original feasible point still satisfies the transformed equality.
  assert(copy_to_host(plan.new_AL, 1)[0] == 0.0);
  assert(copy_to_host(plan.new_AU, 1)[0] == 0.0);
  assert(plan.obj_constant_delta == 2.0);
  assert(!plan.has_infeasible && !plan.has_unbounded);
  // A subsequent rule call must not gradually fix a rejected batch.
  apply_rule_infeasible_and_fixed_variables(plan, lp, params);
  assert(copy_to_host(plan.keep_col_mask, n) == keep);
  assert(copy_to_host(plan.new_AL, 1)[0] == 0.0);
  assert(plan.obj_constant_delta == 2.0);
  free_all({lp.AT.rowPtr, lp.AT.colVal, lp.AT.nzVal, plan.keep_row_mask,
            plan.keep_col_mask, plan.new_c, plan.new_l, plan.new_u,
            plan.new_AL, plan.new_AU});
}

int main() {
  test_fixed_variable(0.0);
  test_fixed_variable(1.0e-8);
  test_narrow_intervals_preserve_feasible_equality(64, 1.0);
  test_narrow_intervals_preserve_feasible_equality(64, 1.0, true);
  test_narrow_intervals_preserve_feasible_equality(1, 100.0);
  std::cout << "test_rule_infeasible_and_fixed_variables passed\n";
  return 0;
}
