// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "../tools/postsolve_replay/kkt.hpp"
#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_postsolve.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_presolve.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_row_sparsification.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_doubleton_equations.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_singleton_columns.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_singleton_rows.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_infeasible_and_fixed_variables.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_empty_rows_and_columns.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_dual_fix.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_bound_strengthening.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
using namespace gpu_preprocessor::lp;
using namespace gpu_preprocessor::lp::presolve;
const double inf=std::numeric_limits<double>::infinity();
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Memory {
  std::vector<void*> pointers;
  ~Memory() { for(void* p:pointers) cudaFree(p); }
  template<class T> T* put(const std::vector<T>& v) {
    if(v.empty()) return nullptr;
    T* p=nullptr; check(cudaMalloc(&p,sizeof(T)*v.size())); pointers.push_back(p);
    check(cudaMemcpy(p,v.data(),sizeof(T)*v.size(),cudaMemcpyHostToDevice)); return p;
  }
  template<class T> std::vector<T> get(const T* p,int n) {
    std::vector<T> v(n); if(n) check(cudaMemcpy(v.data(),p,sizeof(T)*n,cudaMemcpyDeviceToHost)); return v;
  }
  LPInfoGpu model(const std::vector<std::vector<double>>& a,const std::vector<double>& c,
                  const std::vector<double>& l,const std::vector<double>& u,
                  const std::vector<double>& lo,const std::vector<double>& hi) {
    const int m=a.size(),n=c.size(); LPInfoGpu lp;
    auto matrix=[&](bool trans) {
      std::vector<NnzOffset> ptr{0}; std::vector<int> ids; std::vector<double> vals;
      for(int i=0;i<(trans?n:m);++i) {
        for(int j=0;j<(trans?m:n);++j) {
          const double v=trans?a[j][i]:a[i][j]; if(v!=0) { ids.push_back(j); vals.push_back(v); }
        }
        ptr.push_back(ids.size());
      }
      return DeviceCsrMatrix{trans?n:m,trans?m:n,static_cast<NnzOffset>(ids.size()),put(ptr),put(ids),put(vals)};
    };
    lp.A=matrix(false); lp.AT=matrix(true); lp.c=put(c); lp.l=put(l); lp.u=put(u); lp.AL=put(lo); lp.AU=put(hi); return lp;
  }
  PresolvePlanGpu plan(const LPInfoGpu& lp,bool resident) {
    PresolvePlanGpu p; p.tape.resident=resident;
    p.keep_row_mask=put(std::vector<std::uint8_t>(lp.A.rows,1));
    p.keep_col_mask=put(std::vector<std::uint8_t>(lp.A.cols,1));
    p.new_c=put(get(lp.c,lp.A.cols)); p.new_l=put(get(lp.l,lp.A.cols)); p.new_u=put(get(lp.u,lp.A.cols));
    p.new_AL=put(get(lp.AL,lp.A.rows)); p.new_AU=put(get(lp.AU,lp.A.rows)); return p;
  }
  PresolveRecordGpu record(PresolvePlanGpu& p,const LPInfoGpu& lp) {
    PresolveRecordGpu r; r.m0=lp.A.rows; r.n0=lp.A.cols; r.m1=r.m0; r.n1=r.n0;
    const auto rows=get(p.keep_row_mask,r.m0),cols=get(p.keep_col_mask,r.n0);
    std::vector<int> ri(r.m0,-1),ci(r.n0,-1);
    for(int i=0;i<r.m0;++i) if(rows[i]) { ri[i]=r.row_red2org.size(); r.row_red2org.push_back(i); }
    for(int j=0;j<r.n0;++j) if(cols[j]) { ci[j]=r.col_red2org.size(); r.col_red2org.push_back(j); }
    if(p.tape.resident) {
      initialize_device_recovery(r);
      update_device_recovery(r,p,lp,r.row_red2org.size(),r.col_red2org.size(),0,put(ri),put(ci));
    } else r.tape=std::move(p.tape);
    r.m1=r.row_red2org.size(); r.n1=r.col_red2org.size(); return r;
  }
};
void valid(const LPInfoGpu& lp,const double* x,const double* y,const double* z) {
  auto k=diagnostics::check_kkt(&lp,x,y,z);
  if(!k.passed(1e-12)) diagnostics::print_kkt("test_failure",k,1e-12);
  assert(k.passed(1e-12)); assert(k.primal_inf<1e-12 && k.stationarity_inf<1e-12 && k.complementarity_inf<1e-12);
}
void doubleton_support(bool resident,bool batch,double scale) {
  Memory mem;
  auto lp=mem.model({{scale,scale,0},{0,1,1},{1,0,1}}, {0,0,1}, {0,0,0},{inf,inf,inf},
                    {scale,1,-inf},{scale,inf,10});
  auto plan=mem.plan(lp,resident); PresolveStatsGpu stats; PresolveParams params;
  params.doubleton_equations_scan=batch; params.doubleton_equations_min_selected_per_batch=1;
  params.doubleton_equations_min_selected_ratio=0;
  apply_rule_doubleton_equations(plan,lp,stats,params);
  if(plan.has_new_A) { mem.pointers.push_back(plan.new_A.rowPtr); mem.pointers.push_back(plan.new_A.colVal); mem.pointers.push_back(plan.new_A.nzVal); }
  assert((mem.get(plan.keep_col_mask,3)==std::vector<std::uint8_t>{1,0,1}));
  auto r=mem.record(plan,lp);
  auto xr=mem.put<double>({0,0}),yr=mem.put<double>({1,0}),zr=mem.put<double>({1,0});
  auto reduced=mem.model({{-1,1},{1,1}},{0,1},{0,0},{1,inf},{0,-inf},{inf,10});
  valid(reduced,xr,yr,zr);
  auto post=postsolve_gpu(xr,yr,zr,r,&lp);
  mem.pointers.insert(mem.pointers.end(),{post.x_org,post.y_org,post.z_org});
  valid(lp,post.x_org,post.y_org,post.z_org);
  assert(std::abs(mem.get(post.y_org,3)[0]+1/scale)<1e-12);
  export_recovery_record(r);
  assert(r.tape.indices.size()==5 && r.tape.indices[3]==1 && r.tape.indices[4]==1);
  assert(r.tape.vals.size()==13 && r.tape.vals[12]==1);
}
void fixed_intersection(double multiplier,bool induced_lower) {
  Memory mem;
  // e=s-2 or e=2-s, with e>=0 and the opposite inherited endpoint on s.
  const double alpha=induced_lower?1:-1, rhs=induced_lower?-2:2;
  auto lp=mem.model({{1,-alpha}},{0,multiplier},{0,induced_lower?0.0:2.0},{inf,induced_lower?2.0:4.0},{rhs},{rhs});
  PresolveRecordGpu r; r.m0=1;r.n0=2;r.m1=0;r.n1=1;r.col_red2org={1};
  r.tape.types={int(PostsolveReductionType::DoubletonEquation)}; r.tape.dual_modes={unsigned(PostsolveDualMode::Minimal)};
  r.tape.index_starts={0,4};r.tape.value_starts={0,12};r.tape.indices={0,1,0,0};
  r.tape.vals={1,-alpha,rhs,0,inf,induced_lower?0.0:2.0,induced_lower?2.0:4.0,2,2,0,
                induced_lower?2.0:1.0,induced_lower?1.0:2.0};
  auto p=postsolve_gpu(mem.put<double>({2}),nullptr,mem.put<double>({multiplier}),r,&lp);
  mem.pointers.insert(mem.pointers.end(),{p.x_org,p.y_org,p.z_org});valid(lp,p.x_org,p.y_org,p.z_org);
}
void fixed_production(bool resident, bool batch, double cost) {
  Memory mem;
  auto lp=mem.model({{-1,1,0},{0,1,1},{1,0,1}}, {cost,0,1},
                    {0,0,0},{2,inf,inf},{-2,1,-inf},{-2,inf,4});
  auto plan=mem.plan(lp,resident); PresolveStatsGpu stats; PresolveParams params;
  params.doubleton_equations_scan=batch; params.doubleton_equations_min_selected_per_batch=1;
  params.doubleton_equations_min_selected_ratio=0;
  apply_rule_doubleton_equations(plan,lp,stats,params);
  if(plan.has_new_A) mem.pointers.insert(mem.pointers.end(),
      {plan.new_A.rowPtr,plan.new_A.colVal,plan.new_A.nzVal});
  auto r=mem.record(plan,lp);
  double *x, *y, *z;
  if(batch) {
    assert((r.col_red2org==std::vector<int>{1,2}));
    x=mem.put<double>({0,1}); y=mem.put<double>({-5,1,0}); z=mem.put<double>({4,0});
    auto reduced=mem.model({{1,0},{1,1},{0,1}}, {0,1}, {0,0},{inf,inf},
                           {0,1,-inf},{0,inf,2});
    valid(reduced,x,y,z);
  } else {
    assert((r.col_red2org==std::vector<int>{0,2}));
    x=mem.put<double>({2,1}); y=mem.put<double>({1,0}); z=mem.put<double>({cost-1,0});
    auto reduced=mem.model({{1,1},{1,1}}, {cost,1}, {2,0},{2,inf},
                           {3,-inf},{inf,4});
    valid(reduced,x,y,z);
  }
  // The local lift must recover stationarity from its saved stage data.
  auto post=postsolve_gpu(x,y,z,r,static_cast<const LPInfoGpu*>(nullptr));
  mem.pointers.insert(mem.pointers.end(),{post.x_org,post.y_org,post.z_org});
  valid(lp,post.x_org,post.y_org,post.z_org);
  assert((mem.get(post.x_org,3)==std::vector<double>{2,0,1}));
}
void restriction(bool resident,double sign) {
  Memory mem;
  auto lp=mem.model({{sign,sign}},{1,0},{1,0},{inf,0},{sign>0?1:-inf},{sign>0?inf:-1});
  auto plan=mem.plan(lp,resident); PresolveStatsGpu stats;
  stats.singleton_column_mask=mem.put<std::uint8_t>({1,0}); stats.singleton_column_row=mem.put<int>({0,-1});
  stats.singleton_column_val=mem.put<double>({sign,0}); PresolveParams params;
  apply_rule_singleton_columns(plan,lp,stats,params);
  assert(plan.has_row_action && !plan.has_col_action);
  auto r=mem.record(plan,lp);
  auto x=mem.put<double>({1,0}),y=mem.put<double>({-2*sign}),z=mem.put<double>({3,2});
  auto reduced=mem.model({{sign,sign}},{1,0},{1,0},{inf,0},{sign},{sign}); valid(reduced,x,y,z);
  auto post=postsolve_gpu(x,y,z,r,&lp); mem.pointers.insert(mem.pointers.end(),{post.x_org,post.y_org,post.z_org});
  valid(lp,post.x_org,post.y_org,post.z_org);
  assert(mem.get(post.y_org,1)[0]==0);
  assert((mem.get(post.z_org,2)==std::vector<double>{1,0}));
}
void singleton_row_batch(bool resident, double sign, double cost, int inherited) {
  Memory mem;
  // Duplicate endpoint owners and a fixed final intersection exercise both signs.
  const double lower = (inherited & 1) ? 2.0 : 0.0;
  const double upper = (inherited & 2) ? 2.0 : inf;
  const double endpoint = 2*std::abs(sign);
  const std::vector<double> lo = sign > 0 ? std::vector<double>{endpoint,-inf,-inf}
                                            : std::vector<double>{-inf,-endpoint,-endpoint};
  const std::vector<double> hi = sign > 0 ? std::vector<double>{inf,endpoint,endpoint}
                                            : std::vector<double>{-endpoint,inf,inf};
  auto lp=mem.model({{sign},{sign},{sign}}, {cost}, {lower},{upper},lo,hi);
  auto plan=mem.plan(lp,resident); PresolveStatsGpu stats; PresolveParams params;
  params.record_postsolve_tape_cpu=!resident;
  stats.singleton_row_mask=mem.put<std::uint8_t>({1,1,1});
  stats.singleton_row_col=mem.put<int>({0,0,0});
  stats.singleton_row_val=mem.put<double>({sign,sign,sign});
  apply_rule_singleton_rows(plan,lp,stats,params);
  auto r=mem.record(plan,lp);
  auto post=postsolve_gpu(mem.put<double>({2}),nullptr,mem.put<double>({cost}),r,
                          static_cast<const LPInfoGpu*>(nullptr));
  mem.pointers.insert(mem.pointers.end(),{post.x_org,post.y_org,post.z_org});
  valid(lp,post.x_org,post.y_org,post.z_org);
  const auto y=mem.get(post.y_org,3);
  assert(y[2]==0); // The smallest row ID deterministically owns a tied endpoint.
  const bool retained = cost > 0 ? (inherited & 1) : (inherited & 2);
  assert(mem.get(post.z_org,1)[0]==(retained?cost:0));
  export_recovery_record(r); assert(r.tape.types.size()==3);
}
void singleton_then_elimination(bool resident, double cost, int rule) {
  Memory mem;
  auto lp=mem.model({{1}}, {cost}, {0},{inf},{-inf},{0});
  auto plan=mem.plan(lp,resident); PresolveStatsGpu stats; PresolveParams params;
  params.record_postsolve_tape_cpu=!resident;
  stats.singleton_row_mask=mem.put<std::uint8_t>({1});
  stats.singleton_row_col=mem.put<int>({0}); stats.singleton_row_val=mem.put<double>({1});
  apply_rule_singleton_rows(plan,lp,stats,params);
  if(rule==0) apply_rule_infeasible_and_fixed_variables(plan,lp,params);
  else if(rule==1) {
    stats.empty_col_mask=mem.put<std::uint8_t>({1});
    apply_rule_empty_columns(plan,lp,stats,params);
  } else apply_rule_dual_fix(plan,lp,stats,params);
  auto r=mem.record(plan,lp);
  assert(r.m1==0 && r.n1==0);
  auto post=postsolve_gpu(nullptr,nullptr,nullptr,r,static_cast<const LPInfoGpu*>(nullptr));
  mem.pointers.insert(mem.pointers.end(),{post.x_org,post.y_org,post.z_org});
  valid(lp,post.x_org,post.y_org,post.z_org);
  assert(mem.get(post.y_org,1)[0]==std::min(cost,0.0));
  assert(mem.get(post.z_org,1)[0]==std::max(cost,0.0));
}
void bound_fixed_batch(bool resident,double sign,double scale) {
  Memory mem;
  const double rhs=sign>0?2:1;
  auto lp=mem.model({{scale,scale*sign}},{1,sign},{0,0},{1,1},
                    {scale>0?scale*rhs:-inf},{scale>0?inf:scale*rhs});
  auto plan=mem.plan(lp,resident);PresolveStatsGpu stats;PresolveParams params;
  stats.row_nnz=mem.put<int>({2});
  apply_rule_bound_strengthening(plan,lp,stats,params);
  assert((mem.get(plan.keep_col_mask,2)==std::vector<std::uint8_t>{0,0}));
  auto r=mem.record(plan,lp);
  auto post=postsolve_gpu(nullptr,nullptr,nullptr,r,static_cast<const LPInfoGpu*>(nullptr));
  mem.pointers.insert(mem.pointers.end(),{post.x_org,post.y_org,post.z_org});
  valid(lp,post.x_org,post.y_org,post.z_org);
  // Both input multipliers must be frozen before either source row is lifted.
  assert(std::abs(mem.get(post.y_org,1)[0]-2/scale)<1e-12);
  assert((mem.get(post.z_org,2)==std::vector<double>{-1,-sign}));
}
void bound_sequential_batches(bool resident) {
  Memory mem;
  auto lp=mem.model({{1,-1,0},{0,1,1}},{1,1,0},{0,0,0},{3,3,0},{0,1},{inf,inf});
  auto plan=mem.plan(lp,resident);PresolveStatsGpu stats;PresolveParams params;
  stats.row_nnz=mem.put<int>({2,2});
  apply_rule_bound_strengthening(plan,lp,stats,params);
  assert((mem.get(plan.new_l,3)==std::vector<double>{0,1,0}));
  apply_rule_bound_strengthening(plan,lp,stats,params);
  assert((mem.get(plan.new_l,3)==std::vector<double>{1,1,0}));
  auto r=mem.record(plan,lp);
  auto post=postsolve_gpu(mem.put<double>({1,1,0}),mem.put<double>({0,0}),
      mem.put<double>({1,1,0}),r,static_cast<const LPInfoGpu*>(nullptr));
  mem.pointers.insert(mem.pointers.end(),{post.x_org,post.y_org,post.z_org});
  valid(lp,post.x_org,post.y_org,post.z_org);
  assert((mem.get(post.y_org,2)==std::vector<double>{1,2}));
}
void relaxed_bound_certificate(bool resident) {
  Memory mem;
  auto lp=mem.model({{1,1}},{1,0},{0,0},{inf,0},{0.1},{inf});
  auto plan=mem.plan(lp,resident);PresolveStatsGpu stats;PresolveParams params;
  stats.row_nnz=mem.put<int>({2});
  apply_rule_bound_strengthening(plan,lp,stats,params);
  auto r=mem.record(plan,lp);export_recovery_record(r);
  assert(r.tape.types.size()==3 && r.tape.types[1]==int(PostsolveReductionType::BoundChangeCompact));
  const auto ip=r.tape.index_starts[1];
  const double accepted=mem.get(plan.new_l,2)[0];assert(accepted<0.1);
  assert(r.tape.indices[ip+1]==0 && r.tape.vals==std::vector<double>({1,1}));
  auto post=postsolve_gpu(mem.put<double>({accepted,0}),mem.put<double>({0}),
      mem.put<double>({1,0}),r,static_cast<const LPInfoGpu*>(nullptr));
  mem.pointers.insert(mem.pointers.end(),{post.x_org,post.y_org,post.z_org});
  const auto k=diagnostics::check_kkt(&lp,post.x_org,post.y_org,post.z_org);
  assert(k.dual_domain_valid && k.stationarity_inf==0);
  assert(std::abs(k.complementarity_inf-(0.1-accepted))<1e-15);
  assert(!k.passed(1e-12)); // A numerical source does not certify exact KKT.
}
void parallel_row_endpoints(double scale,double multiplier,bool fixed,bool tie) {
  Memory mem;
  const double kept_l=tie?1:0,kept_u=tie?2:3;
  const double deleted_l=1/scale,deleted_u=(fixed?1:2)/scale;
  const double lo=std::min(deleted_l,deleted_u),hi=std::max(deleted_l,deleted_u);
  auto lp=mem.model({{1},{1/scale}},{multiplier},{-inf},{inf},{kept_l,lo},{kept_u,hi});
  PresolveRecordGpu r;r.m0=2;r.n0=1;r.m1=1;r.n1=1;r.row_red2org={0};r.col_red2org={0};
  r.tape.types={int(PostsolveReductionType::ParallelRow)};r.tape.dual_modes={unsigned(PostsolveDualMode::Minimal)};
  r.tape.index_starts={0,2};r.tape.value_starts={0,5};r.tape.indices={0,1};r.tape.vals={scale,kept_l,kept_u,lo,hi};
  auto post=postsolve_gpu(mem.put<double>({multiplier>0||fixed?1.0:2.0}),mem.put<double>({multiplier}),
      mem.put<double>({0}),r,static_cast<const LPInfoGpu*>(nullptr));
  mem.pointers.insert(mem.pointers.end(),{post.x_org,post.y_org,post.z_org});
  valid(lp,post.x_org,post.y_org,post.z_org);
}
void kkt_errors() {
  Memory mem;
  auto lp=mem.model({{1}},{1},{0},{inf},{1},{inf}); lp.obj_constant=7;
  auto x=mem.put<double>({1}), y=mem.put<double>({1}), z=mem.put<double>({0}); valid(lp,x,y,z);
  auto k=diagnostics::check_kkt(&lp,x,y,z);assert(k.primal_objective==8 && k.dual_objective==8);
  assert(k.objective_l2==1 && k.objective_scale==3 && k.complementarity_l1==0);
  auto bad=diagnostics::check_kkt(&lp,mem.put<double>({2}),y,z);
  assert(!bad.passed(1e-6) && bad.complementarity_inf==1 && bad.stationarity_inf==0);
  bad=diagnostics::check_kkt(&lp,x,mem.put<double>({-1}),mem.put<double>({2}));
  assert(!bad.dual_domain_valid && bad.cone==1 && !bad.passed(1e-6));
  bad=diagnostics::check_kkt(&lp,mem.put<double>({0}),y,z);assert(bad.primal_inf==1 && bad.worst_primal_row==0);
  bad=diagnostics::check_kkt(&lp,x,y,mem.put<double>({1}));assert(bad.stationarity_inf==1 && bad.worst_stationarity_col==0);
  bad=diagnostics::check_kkt(&lp,mem.put<double>({NAN}),y,z);assert(!bad.finite && !bad.passed(1e-6));
}
void replay_preserves_approximate_kkt(bool resident,double sign,bool fixed) {
  Memory mem;
  const double lower=sign>0||fixed?0.0:-inf,upper=sign<0||fixed?0.0:inf;
  auto lp=mem.model({{0}}, {sign},{lower},{upper},{0},{0});
  auto plan=mem.plan(lp,resident);auto record=mem.record(plan,lp);
  auto x=mem.put<double>({sign*2e-7}),y=mem.put<double>({0}),z=mem.put<double>({sign});
  const auto before=diagnostics::check_kkt(&lp,x,y,z);
  assert(before.passed(1e-6));
  auto post=postsolve_gpu(x,y,z,record,&lp);
  mem.pointers.insert(mem.pointers.end(),{post.x_org,post.y_org,post.z_org});
  const auto after=diagnostics::check_kkt(&lp,post.x_org,post.y_org,post.z_org);
  if(!after.passed(1e-6))diagnostics::print_kkt("recovery_regression",after,1e-6);
  assert(after.passed(1e-6));
  assert(after.stationarity_inf==0 && mem.get(post.z_org,1)[0]==sign);
  assert(mem.get(post.x_org,1)[0]==sign*2e-7);
}
void replay_preserves_inaccurate_duals(bool resident,bool invalid_domain) {
  Memory mem;
  const double cost=invalid_domain?-1:1;
  auto lp=mem.model({{0}}, {cost},{0},{inf},{0},{0});
  auto plan=mem.plan(lp,resident);auto record=mem.record(plan,lp);
  auto post=postsolve_gpu(mem.put<double>({invalid_domain?2e-7:0}),mem.put<double>({0}),
      mem.put<double>({invalid_domain?-1.0:0.0}),record,&lp);
  mem.pointers.insert(mem.pointers.end(),{post.x_org,post.y_org,post.z_org});
  const auto after=diagnostics::check_kkt(&lp,post.x_org,post.y_org,post.z_org);
  assert(after.dual_domain_valid==!invalid_domain);
  assert(mem.get(post.z_org,1)[0]==(invalid_domain?-1:0));
  assert(!after.passed(1e-12));
}
void replay_preserves_global_gap(bool resident,double sign) {
  Memory mem;
  const double lower=sign>0?0:-10,upper=sign>0?10:0;
  auto lp=mem.model({{sign,-sign}}, {sign*(1+1.2e-7),sign*(-1-2.8e-7)},
                    {lower,lower},{upper,upper},{0},{0});
  lp.obj_constant=1000;
  auto plan=mem.plan(lp,resident);auto record=mem.record(plan,lp);
  auto x=mem.put<double>({sign*10,sign*10}),y=mem.put<double>({1}),
       z=mem.put<double>({0,-sign*1.8e-7});
  const auto before=diagnostics::check_kkt(&lp,x,y,z);
  assert(before.passed(1e-6));
  const auto unsafe=mem.put<double>({0,-sign*2.8e-7});
  const auto candidate=diagnostics::check_kkt(&lp,x,y,unsafe);
  assert(candidate.dual < before.dual && candidate.gap > 1e-6);
  auto post=postsolve_gpu(x,y,z,record,&lp);
  mem.pointers.insert(mem.pointers.end(),{post.x_org,post.y_org,post.z_org});
  const auto after=diagnostics::check_kkt(&lp,post.x_org,post.y_org,post.z_org);
  assert(after.passed(1e-6) && after.gap==before.gap);
  assert(mem.get(post.z_org,2)==mem.get(z,2));
  assert(mem.get(post.x_org,2)==mem.get(x,2) && mem.get(post.y_org,1)==mem.get(y,1));
}
void kkt_endpoint_errors() {
  Memory mem;
  const auto a=std::vector<std::vector<double>>{{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
  auto lp=mem.model(a,{2,-3,4,-5},{-inf,-inf,1,0},{inf,inf,10,3},
                   {1,0,-inf,-inf},{10,4,inf,inf});
  auto x=mem.put<double>({2,2,2,2}),y=mem.put<double>({2,-3,0,0}),z=mem.put<double>({0,0,4,-5});
  auto k=diagnostics::check_kkt(&lp,x,y,z);
  assert(k.stationarity_inf==0 && k.primal_inf==0 && k.complementarity_l1==17);
  assert(k.row_lower.complementarity_inf==2 && k.row_lower.worst_complementarity_index==0);
  assert(k.row_upper.complementarity_inf==6 && k.row_upper.worst_complementarity_index==1);
  assert(k.col_lower.complementarity_inf==4 && k.col_lower.worst_complementarity_index==2);
  assert(k.col_upper.complementarity_inf==5 && k.col_upper.worst_complementarity_index==3);
  assert(k.dual_domain_valid && !k.passed(1e-6));
  auto free_lp=mem.model(a,{2,-3,4,-5},{-inf,-inf,-inf,-inf},{inf,inf,inf,inf},
                        {-inf,-inf,-inf,-inf},{inf,inf,inf,inf});
  k=diagnostics::check_kkt(&free_lp,x,y,z);
  assert(k.row_lower.cone_inf==2 && k.row_lower.worst_cone_index==0);
  assert(k.row_upper.cone_inf==3 && k.row_upper.worst_cone_index==1);
  assert(k.col_lower.cone_inf==4 && k.col_lower.worst_cone_index==2);
  assert(k.col_upper.cone_inf==5 && k.col_upper.worst_cone_index==3);
  assert(k.complementarity_inf==0 && k.row_lower.worst_complementarity_index==-1);
  assert(!k.dual_domain_valid && !k.passed(1e-6));
  // Equality rows and fixed columns admit either signed multiplier.
  auto fixed=mem.model({{1,0},{0,1}},{-1,1},{2,2},{2,2},{2,2},{2,2});
  k=diagnostics::check_kkt(&fixed,mem.put<double>({2,2}),mem.put<double>({2,-3}),mem.put<double>({-3,4}));
  assert(k.passed(1e-12));
  for(const auto* side:{&k.row_lower,&k.row_upper,&k.col_lower,&k.col_upper})
    assert(side->complementarity_inf==0 && side->cone_inf==0 &&
           side->worst_complementarity_index==-1 && side->worst_cone_index==-1);
  // Tied maxima use the smallest stage-local index.
  auto tied=mem.model({{1,0},{0,1}},{1,1},{0,0},{inf,inf},{0,0},{inf,inf});
  k=diagnostics::check_kkt(&tied,mem.put<double>({1,1}),mem.put<double>({1,1}),mem.put<double>({0,0}));
  assert(k.row_lower.complementarity_inf==1 && k.row_lower.worst_complementarity_index==0);
}
void pdg_assessment() {
  using diagnostics::KktCheck;
  Memory mem;
  const double eps = std::ldexp(1.0, -22), cost = 1.0 / eps;
  auto lp = mem.model({}, {cost,cost}, {0,0}, {inf,inf}, {}, {});
  auto z = mem.put<double>({cost,cost});
  // A tiny bound violation and an interior multiplier cancel in the gap.
  // All three requested residuals pass, while complementarity equals two.
  auto k = diagnostics::check_kkt(&lp,mem.put<double>({-eps,eps}),nullptr,z);
  assert(k.primal == eps && k.dual == 0 && k.gap == 0 && k.complementarity == 2);
  assert(k.passed(1e-6,KktCheck::PrimalDualGap) && !k.passed(1e-6,KktCheck::Full));
  // Ignored ancillary fields must not enter the selected gate indirectly.
  k.complementarity = inf; k.finite = false; k.cone = 1; k.dual_domain_valid = false;
  assert(k.passed(1e-6,KktCheck::PrimalDualGap) && !k.passed(1e-6,KktCheck::Full));
  auto evaluate = [&](std::vector<double> x, double* dual) {
    return diagnostics::check_kkt(&lp,mem.put(x),nullptr,dual).passed(1e-6,KktCheck::PrimalDualGap);
  };
  assert(!evaluate({-0.01,0.01},z)); // Primal failure, zero gap/stationarity.
  assert(!evaluate({0,0},mem.put<double>({0,0}))); // Stationarity failure.
  assert(!evaluate({0,eps},z)); // Gap failure, feasible and stationary.
  assert(!evaluate({NAN,0},z));
  assert(!diagnostics::check_kkt(nullptr,nullptr,nullptr,nullptr).passed(1e-6,KktCheck::PrimalDualGap));
}
void row_sparsification(bool resident, double sign, int chain_limit) {
  Memory mem;
  constexpr int rows = 3, cols = 19;
  std::vector<std::vector<double>> a(rows, std::vector<double>(cols));
  std::vector<double> rhs(rows), cost(cols), expected_y{2, -3, 5};
  const std::vector<double> reduced_y = expected_y;
  expected_y[0] -= sign * reduced_y[1];
  if (chain_limit > 1) expected_y[1] -= sign * reduced_y[2];
  for (int i = 0; i < rows; ++i) {
    const double scale = i == 1 ? sign : 1;
    for (int j = 0; j < 16; ++j) a[i][j] = scale;
    a[i][16 + i] = scale;
    rhs[i] = 17 * scale;
    for (int j = 0; j < cols; ++j) cost[j] += a[i][j] * expected_y[i];
  }
  auto lp = mem.model(a, cost, std::vector<double>(cols, -10),
                      std::vector<double>(cols, 10), rhs, rhs);
  PresolveParams params;
  params.max_iters = 1;
  params.record_postsolve_tape_cpu = !resident;
  params.row_sparsification_max_chain = chain_limit;
  params.enable_column_domination = false;
  params.enable_two_row_variable_elimination = false;
  params.enable_zero_cost_product_variables = false;
  params.enable_piecewise_linear_reformulation_variable_pair = false;
  params.enable_piecewise_linear_reformulation_absolute_value = false;
  assert(setenv("GPUPREPROCESSOR_DEVICE_RECOVERY", resident ? "1" : "0", 1) == 0);
  PresolveBudget budget;
  auto summary = run_gpu_presolve_stage(lp, params, PresolveStage::Startup, budget);
  assert(summary.record.startup_rows && summary.has_change);
  assert(summary.reduced_rows == rows && summary.reduced_cols == cols);
  assert(summary.reduced_nnz == (chain_limit > 1 ? 21 : 36));
  auto* x = mem.put(std::vector<double>(cols, 1));
  auto* y = mem.put(reduced_y);
  auto* z = mem.put(std::vector<double>(cols, 0));
  assert(diagnostics::check_kkt(&summary.reduced_lp, x, y, z).passed(1e-12));
  auto restored = postsolve_gpu(x, y, z, summary.record, &lp);
  mem.pointers.insert(mem.pointers.end(), {restored.x_org, restored.y_org, restored.z_org});
  assert(mem.get(restored.y_org, rows) == expected_y);
  assert(mem.get(y, rows) == reduced_y);
  assert(diagnostics::check_kkt(&lp, restored.x_org, restored.y_org, restored.z_org).passed(1e-12));
  free_gpu_presolve_reduced_lp(summary);
}

int main() {
  pdg_assessment();
  kkt_endpoint_errors();
  kkt_errors();
  for(double scale:{1.0,-2.0,1e-8})for(double y:{3.0,-3.0})for(bool fixed:{false,true})
    for(bool tie:{false,true})parallel_row_endpoints(scale,y,fixed,tie);
  for(bool resident:{false,true}) {
    for (double sign : {1.0, -1.0})
      for (int chain_limit : {1, 2}) row_sparsification(resident, sign, chain_limit);
    for(double sign:{-1.0,1.0})replay_preserves_global_gap(resident,sign);
    for(double sign:{-1.0,1.0})for(bool fixed:{false,true})
      replay_preserves_approximate_kkt(resident,sign,fixed);
    for(bool invalid:{false,true})replay_preserves_inaccurate_duals(resident,invalid);
    bound_sequential_batches(resident);relaxed_bound_certificate(resident);
    for(double sign:{1.0,-1.0}) for(double scale:{1.0,-1.0,1e-8}) bound_fixed_batch(resident,sign,scale);
    for(double cost:{-3.0,3.0}) for(int rule:{0,1,2}) singleton_then_elimination(resident,cost,rule);
    for(double sign:{1.0,-1.0,1e-8,-1e-8}) for(double cost:{-3.0,3.0})
      for(int inherited:{0,1,2,3}) singleton_row_batch(resident,sign,cost,inherited);
    for(bool batch:{false,true}) {
      for(double scale:{1.0,-2.0}) doubleton_support(resident,batch,scale);
      for(double cost:{-2.0,4.0}) fixed_production(resident,batch,cost);
    }
    for(double sign:{1.0,-1.0}) restriction(resident,sign);
  }
  for(double multiplier:{-3.0,3.0}) for(bool lower:{false,true}) fixed_intersection(multiplier,lower);
  std::puts("KKT diagnostics and production doubleton/restriction lifts passed");
}
