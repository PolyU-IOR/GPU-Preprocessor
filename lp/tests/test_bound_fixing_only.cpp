// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/gpu_presolve.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_postsolve.hpp"
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace gp = gpu_preprocessor::lp::presolve;
void require(bool value, const char* text) {
  if (!value) throw std::runtime_error(text);
}
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> T* upload(std::vector<T> values) {
  T* result = nullptr;
  check(cudaMalloc(&result, values.size() * sizeof(T)));
  check(cudaMemcpy(result, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  return result;
}
std::vector<double> download(double* pointer, int size) {
  std::vector<double> values(size);
  check(cudaMemcpy(values.data(), pointer, size * sizeof(double), cudaMemcpyDeviceToHost));
  return values;
}
struct Fixture {
  gp::LPInfoGpu lp{};
  Fixture(double a, double lower, double upper, double row_lower,
          double row_upper, double cost) {
    lp.A = {1,2,2,upload<NnzOffset>({0,2}),upload<int>({0,1}),upload<double>({a,1})};
    lp.AT = {2,1,2,upload<NnzOffset>({0,1,2}),upload<int>({0,0}),upload<double>({a,1})};
    lp.l = upload<double>({lower,3}); lp.u = upload<double>({upper,3});
    lp.AL = upload<double>({row_lower}); lp.AU = upload<double>({row_upper});
    lp.c = upload<double>({cost,0});
  }
  ~Fixture() {
    for (void* p : {(void*)lp.A.rowPtr,(void*)lp.A.colVal,(void*)lp.A.nzVal,
                    (void*)lp.AT.rowPtr,(void*)lp.AT.colVal,(void*)lp.AT.nzVal,
                    (void*)lp.l,(void*)lp.u,(void*)lp.AL,(void*)lp.AU,(void*)lp.c})
      cudaFree(p);
  }
};
gp::PresolveParams params() {
  gp::PresolveParams p;
  p.max_iters = 4;
  p.enable_structure_specific_rules = false;
  p.enable_infeasible_and_fixed_variables = p.enable_empty_rows = false;
  p.enable_singleton_rows = p.enable_infeasible_and_redundant_rows = false;
  p.enable_parallel_rows = p.enable_empty_columns = p.enable_singleton_columns = false;
  p.enable_doubleton_equations = p.enable_dual_fix = p.enable_parallel_columns = false;
  return p;
}
bool cone(double value, double lower, double upper, double dual) {
  if (value < lower - 1e-12 || value > upper + 1e-12) return false;
  return (dual <= 0 || (std::isfinite(lower) && std::abs(value-lower)<1e-12)) &&
         (dual >= 0 || (std::isfinite(upper) && std::abs(value-upper)<1e-12));
}
void fixture(bool fixing_only, double a, double lower, double upper,
             double row_lower, double row_upper, double cost,
             bool expect_fixed, bool expect_tightening) {
  setenv("GPUPREPROCESSOR_BOUND_STRENGTHENING_FIX_ONLY", fixing_only ? "1" : "0", 1);
  Fixture f(a,lower,upper,row_lower,row_upper,cost);
  auto summary = gp::run_gpu_presolve_with_reduced_lp(f.lp, params());
  require(!summary.has_infeasible && !summary.has_unbounded, "unexpected status");
  require(summary.reduced_cols == (expect_fixed ? 1 : 2), "wrong deletion");
  require(summary.reduced_rows == 1, "a required row was deleted");
  if (!expect_fixed) {
    auto l = download(summary.reduced_lp.l,2), u = download(summary.reduced_lp.u,2);
    require(l[0] == lower, "nonfixed lower bound changed");
    require(u[0] == (expect_tightening ? 2 : upper), "nonfixed upper bound changed");
  }
  double* x = upload<double>(expect_fixed ? std::vector<double>{3}
                                         : std::vector<double>{2,3});
  const double row_dual = expect_fixed || expect_tightening ? 0 : cost/a;
  double* y = upload<double>({row_dual});
  double* z = upload<double>(expect_fixed ? std::vector<double>{0}
                : std::vector<double>{cost-a*row_dual,-row_dual});
  auto recovered = gp::postsolve_gpu(x,y,z,summary.record,&f.lp);
  auto ox=download(recovered.x_org,2), oy=download(recovered.y_org,1);
  auto oz=download(recovered.z_org,2);
  require(std::abs(ox[0]-2)<1e-12 && ox[1]==3, "wrong primal recovery");
  require(std::abs(cost-a*oy[0]-oz[0])<1e-12 &&
          std::abs(oy[0]+oz[1])<1e-12, "stationarity recovery");
  require(cone(a*ox[0]+ox[1],row_lower,row_upper,oy[0]), "row cone");
  require(cone(ox[0],lower,upper,oz[0]) && cone(ox[1],3,3,oz[1]), "bound cones");
  const auto rc=download(summary.reduced_lp.c,summary.reduced_cols);
  const auto rx=download(x,summary.reduced_cols);
  double objective=summary.reduced_lp.obj_constant;
  for (int j=0;j<summary.reduced_cols;++j) objective += rc[j]*rx[j];
  require(std::abs(objective-cost*ox[0])<1e-12, "objective offset recovery");
  for (void* p : {(void*)x,(void*)y,(void*)z,(void*)recovered.x_org,
                  (void*)recovered.y_org,(void*)recovered.z_org}) cudaFree(p);
  gp::free_gpu_presolve_reduced_lp(summary);
}
int main() {
  try {
    unsetenv("GPUPREPROCESSOR_DISABLE_BOUND_STRENGTHENING");
    fixture(false,1,0,10,-INFINITY,5,-1,false,true);
    fixture(true,1,0,10,-INFINITY,5,-1,false,false);
    fixture(true,1,2,10,-INFINITY,5,-1,true,false);
    fixture(true,1,2,INFINITY,-INFINITY,5,-1,true,false);
    fixture(true,-1,2,10,1,INFINITY,-1,true,false);
    fixture(true,1,-INFINITY,INFINITY,5,5,1,true,false);
    fixture(true,1,2-1e-8,10,-INFINITY,5,-1,false,false);
    fixture(true,1,2,10,-INFINITY,5,0,false,false);
    Fixture bad(1,3,10,-INFINITY,5,-1);
    auto rejected=gp::run_gpu_presolve_with_reduced_lp(bad.lp,params());
    require(rejected.has_infeasible,"infeasibility check was lost");
    gp::free_gpu_presolve_reduced_lp(rejected);
    std::cout << "PASS: fixing-only selection and exact primal-dual recovery (9 cases)\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
