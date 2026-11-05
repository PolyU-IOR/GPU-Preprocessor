// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "kkt.hpp"
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace gpu_preprocessor::lp::diagnostics {
namespace {
struct Totals {
  EndpointMetrics endpoints[4];
  double row2 = 0, bound2 = 0, rhs2 = 0, cost2 = 0, dual2 = 0;
  double p = 0, d = 0, comp = 0, cone = 0, row_max = 0, bound_max = 0, dual_max = 0, comp_max = 0;
  int row_id = -1, bound_id = -1, dual_id = -1;
  bool finite = true, domain = true;
};
__host__ __device__ void update_max(double value, int id, double& maximum, int& worst) {
  if (value > maximum || (value == maximum && value > 0 && id >= 0 && (worst < 0 || id < worst))) {
    maximum = value; worst = id;
  }
}
struct Sum {
  __host__ __device__ Totals operator()(Totals a, const Totals& b) const {
    a.row2 += b.row2; a.bound2 += b.bound2; a.rhs2 += b.rhs2;
    a.cost2 += b.cost2; a.dual2 += b.dual2; a.p += b.p; a.d += b.d; a.comp += b.comp;
    a.cone = fmax(a.cone,b.cone); a.comp_max = fmax(a.comp_max,b.comp_max);
    if (b.row_max > a.row_max || (b.row_max == a.row_max && b.row_id >= 0 && (a.row_id < 0 || b.row_id < a.row_id))) { a.row_max=b.row_max; a.row_id=b.row_id; }
    if (b.bound_max > a.bound_max || (b.bound_max == a.bound_max && b.bound_id >= 0 && (a.bound_id < 0 || b.bound_id < a.bound_id))) { a.bound_max=b.bound_max; a.bound_id=b.bound_id; }
    if (b.dual_max > a.dual_max || (b.dual_max == a.dual_max && b.dual_id >= 0 && (a.dual_id < 0 || b.dual_id < a.dual_id))) { a.dual_max=b.dual_max; a.dual_id=b.dual_id; }
    for (int k = 0; k < 4; ++k) {
      auto& dst = a.endpoints[k]; const auto& src = b.endpoints[k];
      update_max(src.complementarity_inf, src.worst_complementarity_index,
                 dst.complementarity_inf, dst.worst_complementarity_index);
      update_max(src.cone_inf, src.worst_cone_index, dst.cone_inf, dst.worst_cone_index);
    }
    a.finite = a.finite && b.finite; a.domain = a.domain && b.domain;
    return a;
  }
};
__device__ double violation(double value, double lo, double hi) {
  return fmax(0.0, fmax(isfinite(lo) ? lo-value : 0.0, isfinite(hi) ? value-hi : 0.0));
}
__device__ void endpoint(Totals& s, double value, double multiplier, double lo, double hi, int base, int index) {
  const double cone = fmax(0.0, fmax(!isfinite(lo) ? multiplier : 0.0, !isfinite(hi) ? -multiplier : 0.0));
  s.cone = fmax(s.cone,cone); s.domain = s.domain && cone == 0.0;
  if (multiplier == 0.0) return;
  auto& entry = s.endpoints[base + (multiplier < 0 ? 1 : 0)];
  update_max(cone, index, entry.cone_inf, entry.worst_cone_index);
  const double side = multiplier > 0 ? lo : hi;
  if (isfinite(side)) {
    s.d += multiplier*side;
    const double error = fabs(multiplier*(value-side));
    s.comp += error; s.comp_max=fmax(s.comp_max,error);
    update_max(error, index, entry.complementarity_inf, entry.worst_complementarity_index);
  }
}
struct Row {
  presolve::LPInfoGpu model; const double *x, *y; double* aty;
  __device__ Totals operator()(int i) const {
    Totals s; double ax=0;
    const auto a=model.A;
    for (auto p=a.rowPtr[i]; p<a.rowPtr[i+1]; ++p) {
      ax += a.nzVal[p]*x[a.colVal[p]];
      atomicAdd(aty+a.colVal[p],a.nzVal[p]*y[i]);
    }
    const double lo=model.AL[i], hi=model.AU[i];
    const double v=violation(ax,lo,hi);
    const double scale=fmax(isfinite(lo)?fabs(lo):0.0,isfinite(hi)?fabs(hi):0.0);
    s.row2=v*v; s.rhs2=scale*scale; s.row_max=v; s.row_id=v>0?i:-1;
    s.finite=isfinite(ax)&&isfinite(y[i]);
    endpoint(s,ax,y[i],lo,hi,0,i); return s;
  }
};
struct Col {
  presolve::LPInfoGpu model; const double *x,*z,*aty;
  __device__ Totals operator()(int j) const {
    Totals s;
    const double residual=model.c[j]-aty[j]-z[j];
    const double v=violation(x[j],model.l[j],model.u[j]);
    s.bound2=v*v; s.cost2=model.c[j]*model.c[j]; s.dual2=residual*residual;
    s.p=model.c[j]*x[j]; s.bound_max=v; s.bound_id=v>0?j:-1;
    s.dual_max=fabs(residual); s.dual_id=residual!=0?j:-1;
    s.finite=isfinite(x[j])&&isfinite(z[j])&&isfinite(residual);
    endpoint(s,x[j],z[j],model.l[j],model.u[j],2,j); return s;
  }
};
void checked(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
const char* check_name(KktCheck check) {
  return check == KktCheck::PrimalDualGap ? "pdg" : "full";
}
bool KktMetrics::passed(double t, KktCheck check) const {
  if (check == KktCheck::PrimalDualGap) {
    return available && pdg_finite && std::isfinite(primal) && std::isfinite(dual) &&
        std::isfinite(gap) && primal <= t && dual <= t && gap <= t;
  }
  return available && finite && dual_domain_valid && primal<=t && dual<=t && gap<=t && cone<=t && complementarity<=t;
}
KktMetrics check_kkt(const presolve::LPInfoGpu* lp,const double* x,const double* y,const double* z) {
  KktMetrics out;
  if (!lp || (lp->A.cols && (!lp->c || !lp->l || !lp->u || !x || !z)) ||
      (lp->A.rows && (!lp->AL || !lp->AU || !lp->A.rowPtr || !y)) ||
      (lp->A.nnz && (!lp->A.colVal || !lp->A.nzVal))) return out;
  out.available=true;
  double* aty=nullptr;
  checked(cudaMalloc(&aty,sizeof(double)*std::max(1,lp->A.cols)));
  try {
    checked(cudaMemset(aty,0,sizeof(double)*std::max(1,lp->A.cols)));
    auto begin=thrust::make_counting_iterator<int>(0);
    auto rows=thrust::transform_reduce(thrust::cuda::par,begin,begin+lp->A.rows,Row{*lp,x,y,aty},Totals{},Sum{});
    auto cols=thrust::transform_reduce(thrust::cuda::par,begin,begin+lp->A.cols,Col{*lp,x,z,aty},Totals{},Sum{});
    const auto s=Sum{}(rows,cols); const double scale=1+std::abs(s.p)+std::abs(s.d);
    out.primal=std::max(std::sqrt(s.row2),std::sqrt(s.bound2))/(1+std::sqrt(s.rhs2));
    out.row_violation_l2=std::sqrt(s.row2); out.bound_violation_l2=std::sqrt(s.bound2);
    out.rhs_l2=std::sqrt(s.rhs2);
    out.stationarity_l2=std::sqrt(s.dual2); out.objective_l2=std::sqrt(s.cost2);
    out.dual=out.stationarity_l2/(1+out.objective_l2); out.gap=std::abs(s.p-s.d)/scale;
    out.row_lower=s.endpoints[0]; out.row_upper=s.endpoints[1];
    out.col_lower=s.endpoints[2]; out.col_upper=s.endpoints[3];
    out.cone=s.cone; out.complementarity=s.comp/scale;
    out.complementarity_l1=s.comp; out.objective_scale=scale;
    out.primal_inf=std::max(s.row_max,s.bound_max); out.stationarity_inf=s.dual_max;
    out.complementarity_inf=s.comp_max; out.primal_objective=s.p+lp->obj_constant;
    out.dual_objective=s.d+lp->obj_constant; out.dual_domain_valid=s.domain;
    out.worst_primal_row=s.row_id; out.worst_primal_col=s.bound_id; out.worst_stationarity_col=s.dual_id;
    // Ancillary complementarity overflow must not change a three-metric gate.
    out.pdg_finite=s.finite && std::isfinite(s.row2+s.bound2+s.rhs2+s.cost2+s.dual2) &&
                   std::isfinite(out.primal_objective) && std::isfinite(out.dual_objective);
    out.finite=s.finite && std::isfinite(s.row2+s.bound2+s.rhs2+s.cost2+s.dual2+s.comp) &&
               std::isfinite(out.primal_objective) && std::isfinite(out.dual_objective);
    checked(cudaFree(aty)); return out;
  } catch (...) { cudaFree(aty); throw; }
}
void print_kkt(const char* stage,const KktMetrics& m,double tol,KktCheck check) {
  const bool pdg = check == KktCheck::PrimalDualGap;
  const bool pass = m.passed(tol, check);
  std::printf("KKT_STAGE {\"stage\":\"%s\",\"criterion\":\"%s\",\"available\":%s,\"finite\":%s,\"pass\":%s,\"status\":\"%s\",\"dual_domain_valid\":%s",
      stage,check_name(check),m.available?"true":"false",(pdg?m.pdg_finite:m.finite)?"true":"false",
      m.available?(pass?"true":"false"):"null",m.available?(pass?"pass":"fail"):"unavailable",
      m.dual_domain_valid?"true":"false");
  auto number=[&](const char* name,double v,bool valid=true) {
    std::printf(",\"%s\":",name); if (valid && std::isfinite(v)) std::printf("%.17g",v); else std::printf("null");
  };
  number("tolerance",tol); number("primal",m.primal,m.available); number("dual",m.dual,m.available);
  number("gap",m.gap,m.available&&(pdg||m.dual_domain_valid));
  number("finite_endpoint_gap",m.gap,m.available); number("cone",m.cone,m.available);
  number("complementarity",m.complementarity,m.available);
  number("complementarity_l1",m.complementarity_l1,m.available); number("objective_scale",m.objective_scale,m.available);
  number("stationarity_l2",m.stationarity_l2,m.available); number("objective_l2",m.objective_l2,m.available);
  number("row_violation_l2",m.row_violation_l2,m.available);
  number("bound_violation_l2",m.bound_violation_l2,m.available); number("rhs_l2",m.rhs_l2,m.available);
  number("primal_inf",m.primal_inf,m.available); number("stationarity_inf",m.stationarity_inf,m.available);
  number("complementarity_inf",m.complementarity_inf,m.available);
  const char* names[] = {"row_lower", "row_upper", "col_lower", "col_upper"};
  const EndpointMetrics* sides[] = {&m.row_lower, &m.row_upper, &m.col_lower, &m.col_upper};
  for (int k = 0; k < 4; ++k) {
    char field[80];
    std::snprintf(field,sizeof(field),"%s_complementarity_inf",names[k]);
    number(field,sides[k]->complementarity_inf,m.available);
    std::snprintf(field,sizeof(field),"%s_cone_inf",names[k]);
    number(field,sides[k]->cone_inf,m.available);
    std::printf(",\"%s_worst_complementarity_index\":%d,\"%s_worst_cone_index\":%d",
        names[k],sides[k]->worst_complementarity_index,names[k],sides[k]->worst_cone_index);
  }
  number("primal_objective",m.primal_objective,m.available);
  number("dual_objective",m.dual_objective,m.available&&m.dual_domain_valid);
  std::printf(",\"worst_primal_row\":%d,\"worst_primal_col\":%d,\"worst_stationarity_col\":%d}\n",
      m.worst_primal_row,m.worst_primal_col,m.worst_stationarity_col); std::fflush(stdout);
}
}  // namespace gpu_preprocessor::lp::diagnostics
