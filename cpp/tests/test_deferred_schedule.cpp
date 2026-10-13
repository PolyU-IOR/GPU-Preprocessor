#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace runtime = gpu_presolver::runtime;
using namespace gpu_presolver::presolve;
namespace {
void check(cudaError_t e) { if(e!=cudaSuccess){std::cerr<<cudaGetErrorString(e)<<'\n';std::abort();} }
template<class T> std::vector<T> download(const T* p,int n) {
  std::vector<T> v(n);if(n)check(cudaMemcpy(v.data(),p,sizeof(T)*n,cudaMemcpyDeviceToHost));return v;
}
struct Model {
  LPInfoGpu lp;
  std::vector<void*> storage;
  template<class T> T* upload(const std::vector<T>& v) {
    T* p=nullptr;if(!v.empty()){check(cudaMalloc(&p,sizeof(T)*v.size()));storage.push_back(p);
      check(cudaMemcpy(p,v.data(),sizeof(T)*v.size(),cudaMemcpyHostToDevice));}return p;
  }
  Model(int n,const std::vector<std::vector<double>>& rows,
        const std::vector<double>& al,const std::vector<double>& au,double lower=0,double upper=10) {
    int m=rows.size();std::vector<NnzOffset> rp{0};std::vector<int> ci;std::vector<double> av;
    for(const auto& row:rows){assert(int(row.size())==n);for(int j=0;j<n;++j)if(row[j]!=0){ci.push_back(j);av.push_back(row[j]);}rp.push_back(ci.size());}
    std::vector<NnzOffset> tp(n+1);for(int j:ci)++tp[j+1];for(int j=0;j<n;++j)tp[j+1]+=tp[j];
    auto cursor=tp;std::vector<int> ti(ci.size());std::vector<double> tv(av.size());
    for(int i=0;i<m;++i)for(auto p=rp[i];p<rp[i+1];++p){auto q=cursor[ci[p]]++;ti[q]=i;tv[q]=av[p];}
    lp.A={m,n,NnzOffset(av.size()),upload(rp),upload(ci),upload(av)};
    lp.AT={n,m,NnzOffset(av.size()),upload(tp),upload(ti),upload(tv)};
    lp.AL=upload(al);lp.AU=upload(au);lp.l=upload(std::vector<double>(n,lower));
    lp.u=upload(std::vector<double>(n,upper));lp.c=upload(std::vector<double>(n,0));
  }
  ~Model(){for(auto p:storage)cudaFree(p);}
};
PresolveParams parameters() {
  PresolveParams p;p.enable_startup=false;p.enable_structure_specific_rules=false;
  // Existing fixtures exercise the nnz phase transitions independently of
  // the new bound-only repetition policy. New chain tests set their own L.
  p.bound_strengthening_max_bound_only_rounds=1;
  p.max_iters=12;return p;
}
struct Result {
  GpuPresolveSummary summary;
  std::string trace;
  Result(Model& model,PresolveParams p,bool device) {
    runtime::options().device_recovery = device;
    std::ostringstream stream;auto* prior=std::cerr.rdbuf(stream.rdbuf());
    summary=run_gpu_presolve_with_reduced_lp(model.lp,p);
    std::cerr.rdbuf(prior);trace=stream.str();
    assert(!summary.has_infeasible && !summary.has_unbounded);
  }
  ~Result(){free_gpu_presolve_reduced_lp(summary);}
};
int occurrences(const std::string& text,const std::string& pattern) {
  int count=0;std::size_t pos=0;while((pos=text.find(pattern,pos))!=std::string::npos){++count;pos+=pattern.size();}return count;
}
std::vector<std::string> phases(const std::string& trace) {
  std::vector<std::string> out;std::istringstream stream(trace);std::string line;
  while(std::getline(stream,line))if(line.find("[schedule-phase]")!=std::string::npos)out.push_back(line.substr(line.find("phase=")+6));return out;
}
void compare_recovery(Result& host,Result& device) {
  export_recovery_record(device.summary.record);
  const auto& a=host.summary.record;const auto& b=device.summary.record;
  assert(a.row_red2org==b.row_red2org && a.col_red2org==b.col_red2org);
  assert(a.fixed_idx==b.fixed_idx && a.fixed_val==b.fixed_val);
  assert(a.tape.types==b.tape.types && a.tape.indices==b.tape.indices && a.tape.vals==b.tape.vals);
  assert(a.tape.index_starts==b.tape.index_starts && a.tape.value_starts==b.tape.value_starts);
  assert(a.tape.dual_modes==b.tape.dual_modes);
  assert(host.summary.obj_constant_delta==device.summary.obj_constant_delta);
}
void test_cleanup_follows_startup_at_iteration_limit() {
  Model model(3, {{1,-1,-1}}, {0}, {0}, 0, INFINITY);
  const std::vector<double> cost{1,0,0};
  check(cudaMemcpy(model.lp.c,cost.data(),sizeof(double)*3,cudaMemcpyHostToDevice));
  auto p=parameters();p.enable_startup=true;p.max_iters=1;
  Result host(model,p,false),device(model,p,true);compare_recovery(host,device);
  assert(device.summary.iterations==1);
  // Cleanup repeats after removing the row, even with Startup's budget spent.
  assert(device.summary.reduced_rows==0 && device.summary.reduced_cols==0);
  assert(device.summary.reduced_nnz==0);
  assert((phases(device.trace)==std::vector<std::string>{"cleanup","cleanup","cleanup"}));
  assert(occurrences(device.trace,"boundary=finalize")==1);
  assert(occurrences(device.trace,"[schedule-rule]")==15);
  auto restored=postsolve_gpu(nullptr,nullptr,nullptr,device.summary.record,&model.lp);
  for(double x:download(restored.x_org,3)) assert(x==0);
  cudaFree(restored.x_org);cudaFree(restored.y_org);cudaFree(restored.z_org);
}
void test_order() {
  Model model(3,{{1,2,3},{2,-1,1},{1,1,-2}},{0,0,0},{0,0,0},-INFINITY,INFINITY);
  auto p=parameters();p.max_iters=3;
  Result r(model,p,true);
  assert((phases(r.trace)==std::vector<std::string>{"cleanup","propagation","aggregation","doubleton_equations","parallel"}));
  assert(occurrences(r.trace,"[schedule-apply]")==0);
  assert(r.summary.iterations==3);
  // Parallel naturally stops on the last allowed iteration, without an
  // attempted next stage being blocked by the budget.
  assert(!r.summary.stopped_at_iteration_limit);
  const auto start=r.trace.find("phase=propagation"),stop=r.trace.find("phase=aggregation",start);
  const auto red=r.trace.find("rule=infeasible_and_redundant_rows",start);
  const auto bound=r.trace.find("rule=bound_strengthening",start),dual=r.trace.find("rule=dual_fix",start);
  assert(start<bound && bound<red && red<dual && dual<stop);
}
void test_cleanup_dependency_repeats_without_budget() {
  // A single Cleanup fixes only the first variable. The independent Cleanup
  // phase repeats the dependency chain without spending a main-loop iteration.
  for (int budget : {1,2,3,4}) {
    Model model(3,{{1,0,0},{1,1,0},{0,1,1}},{1,2,2},{1,2,2},0,5);
    auto p=parameters();p.enable_startup=true;p.max_iters=budget;
    p.enable_bound_strengthening=p.enable_infeasible_and_redundant_rows=false;
    p.enable_dual_fix=p.enable_doubleton_equations=false;
    p.enable_parallel_rows=p.enable_parallel_columns=p.enable_singleton_columns=false;
    Result host(model,p,false),device(model,p,true);compare_recovery(host,device);
    const int expected_cols=0;
    const int expected_rows=0;
    assert(device.summary.reduced_cols==expected_cols);
    assert(device.summary.reduced_rows==expected_rows);
    assert(occurrences(device.trace,"rule=infeasible_and_fixed_variables")==5);
    assert(occurrences(device.trace,"rule=singleton_rows")==5);
    std::vector<double> xr(expected_cols,1),yr(expected_rows,0),zr(expected_cols,0);
    double *x=model.upload(xr),*y=model.upload(yr),*z=model.upload(zr);
    auto restored=postsolve_gpu(x,y,z,device.summary.record,&model.lp);
    for(double value:download(restored.x_org,3)) assert(std::fabs(value-1)<1e-12);
    cudaFree(restored.x_org);cudaFree(restored.y_org);cudaFree(restored.z_org);
  }
}
void test_interleaved_progress_flags() {
  // Bound strengthening and dual fixing do nothing. Redundancy alone deletes
  // a row; its change/action flags must survive the later no-op dual rule.
  Model model(2,{{1,1},{1,-1}},{-INFINITY,0},{100,0});
  auto p=parameters();p.max_iters=1;
  p.enable_infeasible_and_fixed_variables=p.enable_empty_columns=p.enable_empty_rows=false;
  p.enable_singleton_rows=p.enable_singleton_columns=false;
  p.bound_strengthening_bound_only_nnz_round_budget=1;
  Result host(model,p,false),device(model,p,true);compare_recovery(host,device);
  assert(device.summary.reduced_rows==1 && device.summary.reduced_cols==2);
  assert(device.summary.reduced_nnz==2);
  assert(occurrences(device.trace,"rule=bound_strengthening changed=0")==1);
  assert(occurrences(device.trace,"rule=infeasible_and_redundant_rows changed=1")==1);
  assert(occurrences(device.trace,"rule=dual_fix changed=0")==1);
  assert(occurrences(device.trace,"boundary=1")==1);
  const auto upper=download(device.summary.reduced_lp.u,2);
  assert(upper[0]==10 && upper[1]==10);
  const auto lower=download(device.summary.reduced_lp.l,2);
  assert(lower[0]==0 && lower[1]==0);
}
void test_stage_progress_advances_after_no_change() {
  // Check the absolute threshold, including its exact boundary. After a large
  // reduction, the no-op must compare against the next invocation's own nnz.
  for (int removed : {99,100,102}) {
    const int pairs=removed%2 ? (removed-3)/2 : removed/2;
    std::vector<std::vector<double>> rows(pairs,std::vector<double>{1,1,0});
    // Redundant-row detection leaves singleton rows to Cleanup, so the odd
    // case uses a three-entry redundant row instead of a singleton.
    if (removed%2) rows.push_back({1,1,1});
    std::vector<double> al(rows.size(),-INFINITY),au(rows.size(),100);
    rows.push_back({1,-1,0});al.push_back(0);au.push_back(0);
    Model model(3,rows,al,au);
    auto p=parameters();p.max_iters=4;
    p.enable_infeasible_and_fixed_variables=p.enable_empty_columns=p.enable_empty_rows=false;
    p.enable_singleton_rows=p.enable_singleton_columns=false;
    p.enable_doubleton_equations=p.enable_parallel_rows=p.enable_parallel_columns=false;
    p.bound_strengthening_bound_only_nnz_round_budget=1;
    Result host(model,p,false),device(model,p,true);compare_recovery(host,device);
    const bool repeats=removed>=100;
    assert(device.summary.iterations==(repeats?4:3) && device.summary.reduced_nnz==2);
    assert(occurrences(device.trace,"[schedule-phase] phase=propagation")== (repeats?2:1));
    const auto first="phase=propagation stage_before="+std::to_string(removed+2)+
        " after=2 next="+(repeats?"cleanup":"aggregation");
    assert(occurrences(device.trace,first)==1);
    assert(occurrences(device.trace,"phase=propagation stage_before=2 after=2 next=aggregation")== (repeats?1:0));
    assert(occurrences(device.trace,"[schedule-phase] phase=aggregation")==1);
    assert(occurrences(device.trace,"[schedule-phase] phase=parallel")==1);
    assert(occurrences(device.trace,"boundary=1")==1);
  }
}
void test_parallel_progress_does_not_credit_earlier_phases() {
  // Each phase removes fewer than 100 entries, but their cumulative reduction
  // exceeds 100. Parallel must stop without crediting the earlier phases.
  std::vector<std::vector<double>> rows(9,std::vector<double>(74,0));
  for(int i=0;i<5;++i)for(int j=0;j<20;++j)rows[i][j]=1+(i*7+j*3+j*i)%19;
  for(int j=20;j<70;++j)rows[5][j]=rows[7][j]=rows[8][j]=1;
  rows[6][70]=rows[6][71]=1;  // Isolated doubleton equality.
  std::vector<double> al(9,-INFINITY),au(9,5);
  au[5]=1000;al[6]=au[6]=1;au[7]=au[8]=15;
  Model model(74,rows,al,au);auto p=parameters();
  p.enable_infeasible_and_fixed_variables=p.enable_empty_columns=p.enable_empty_rows=false;
  p.enable_singleton_rows=p.enable_singleton_columns=false;
  p.enable_bound_strengthening=p.enable_dual_fix=p.enable_parallel_columns=false;
  p.doubleton_equations_min_selected_per_batch=1;
  Result host(model,p,false),device(model,p,true);compare_recovery(host,device);
  assert(model.lp.A.nnz==252 && device.summary.reduced_nnz==150);
  assert(device.summary.iterations==3);
  assert(!device.summary.stopped_at_iteration_limit);
  assert(occurrences(device.trace,"[schedule-phase] phase=propagation")==1);
  assert(occurrences(device.trace,"[schedule-phase] phase=aggregation")==1);
  assert(occurrences(device.trace,"[schedule-phase] phase=parallel")==1);
  assert(device.trace.find("phase=propagation stage_before=252 after=202 next=aggregation")!=std::string::npos);
  assert(device.trace.find("phase=aggregation stage_before=202 after=200 next=parallel")!=std::string::npos);
  assert(device.trace.find("phase=parallel stage_before=200 after=150 next=end")!=std::string::npos);
}
void test_single_pass_row_side_update() {
  // A redundant lower row side is removed, but no row/column is deleted.
  // A single Propagation call must retain this update without repeating.
  Model model(2,{{1,1},{1,-1}},{0,0},{15,0});
  auto p=parameters();p.max_iters=1;
  p.enable_infeasible_and_fixed_variables=p.enable_empty_columns=p.enable_empty_rows=false;
  p.enable_singleton_rows=p.enable_singleton_columns=false;
  p.bound_strengthening_bound_only_nnz_round_budget=1;
  Result host(model,p,false),device(model,p,true);compare_recovery(host,device);
  assert(device.summary.reduced_rows==2 && device.summary.reduced_cols==2);
  assert(device.summary.reduced_nnz==4);
  assert(occurrences(device.trace,"rule=bound_strengthening changed=0")==1);
  assert(occurrences(device.trace,"rule=infeasible_and_redundant_rows changed=1")==1);
  assert(occurrences(device.trace,"rule=dual_fix changed=0")==1);
  const auto lower=download(device.summary.reduced_lp.AL,2);
  assert(lower[0]==-INFINITY && lower[1]==0);
}
void test_carry_and_doubleton_equations(int max_iters) {
  Model model(3,{{1,1,0},{0,1,1},{1,0,1}},{2,2,2},{2,2,2});
  auto p=parameters();p.max_iters=max_iters;

  // One Aggregation call processes one batch and retains recovery.
  p.doubleton_equations_min_selected_per_batch=1;
  Result host(model,p,false),device(model,p,true);compare_recovery(host,device);
  const auto& trace=device.trace;
  // The small batch advances to Parallel. With no later reductions, the
  // partially reduced model and its reconstruction must remain valid.
  const bool budget_spent_after_one_batch = max_iters == 2;
  assert(device.summary.stopped_at_iteration_limit == budget_spent_after_one_batch);
  assert(device.summary.reduced_nnz==4);
  assert(device.summary.reduced_rows==2);
  assert(device.summary.reduced_cols==2);
  assert(occurrences(trace,"phase=doubleton_equations changed=true")>=1);
  assert(occurrences(trace,"[GPU Presolve C++ profile] phase=doubleton_equations ")==
         occurrences(trace,"[schedule-phase] phase=aggregation"));
  const auto aggregation=trace.find("[schedule-phase] phase=aggregation");
  const auto doubleton=trace.find("phase=doubleton_equations changed=true");
  assert(aggregation<doubleton);
  const auto carry=trace.find("phase=aggregation stage_before=6 after=4 next=parallel",aggregation);
  assert(carry!=std::string::npos);
  assert(trace.find("[schedule-phase] phase=cleanup",carry)==std::string::npos);
  const int n=device.summary.reduced_cols;
  auto xr=model.upload(std::vector<double>(n,1)),zr=model.upload(std::vector<double>(n,0));
  auto restored=postsolve_gpu(xr,nullptr,zr,device.summary.record,&model.lp);
  const auto x=download(restored.x_org,3);for(double value:x)assert(std::fabs(value-1)<1e-9);
  cudaFree(restored.x_org);cudaFree(restored.y_org);cudaFree(restored.z_org);
}
void test_aggregation_iteration_limit() {
  Model model(3,{{1,1,0},{0,1,1},{1,0,1}},{2,2,2},{2,2,2});
  auto p=parameters();p.max_iters=1;

  p.doubleton_equations_min_selected_per_batch=1;
  Result host(model,p,false),device(model,p,true);compare_recovery(host,device);
  assert(device.summary.iterations==1);
  assert(device.summary.stopped_at_iteration_limit);
  assert(device.summary.reduced_rows==3 && device.summary.reduced_cols==3);
  assert(device.summary.reduced_nnz==6);
  assert(device.trace.find("[schedule-phase] phase=aggregation")==std::string::npos);
  assert(device.trace.find("phase=doubleton_equations")==std::string::npos);
}
void test_parallel() {
  Model model(3,{{1,1,2},{2,2,4}},{2,4},{8,16});
  auto p=parameters();p.enable_dual_fix=false;p.enable_bound_strengthening=false;

  p.enable_infeasible_and_redundant_rows=false;p.enable_doubleton_equations=false;
  Result host(model,p,false),device(model,p,true);compare_recovery(host,device);
  const auto& trace=device.trace;auto parallel=trace.find("phase=parallel");
  auto rows=trace.find("rule=parallel_rows",parallel),cols=trace.find("rule=parallel_columns",rows);
  auto apply=trace.find("boundary=2",cols);
  assert(parallel<rows && rows<cols && cols<apply);
  assert(trace.find("[schedule-phase] phase=cleanup",apply)==std::string::npos);
  assert(device.summary.reduced_rows==1 && device.summary.reduced_cols==1);
  assert(device.summary.reduced_nnz==1);
}
void test_progress(int residual_nnz) {
  std::vector<std::vector<double>> rows(6,std::vector<double>(6));
  for(int i=0;i<5;++i)for(int j=0;j<5;++j)if(i!=j)rows[i][j]=1+i*5+j;
  rows[0][1]=0;if(residual_nnz==18)rows[0][2]=0;rows[5][5]=1;
  std::vector<double> al(6,0),au(6,100);al[5]=1;au[5]=2;
  Model model(6,rows,al,au);auto p=parameters();

  p.enable_singleton_columns=p.enable_doubleton_equations=p.enable_dual_fix=false;
  p.enable_bound_strengthening=p.enable_infeasible_and_redundant_rows=false;
  p.enable_parallel_rows=p.enable_parallel_columns=false;
  Result r(model,p,true);assert(r.summary.reduced_nnz==residual_nnz);
  assert(occurrences(r.trace,"[schedule-phase] phase=propagation")==1);
  // Both 20 -> 19 (exactly 5%) and 19 -> 18 repeat. Removing the now
  // empty column also repeats despite unchanged nnz, then a no-op stops.
  assert(occurrences(r.trace,"[schedule-phase] phase=cleanup")==3);
  const auto prefix="phase=cleanup stage_before="+std::to_string(residual_nnz)+
                    " after="+std::to_string(residual_nnz);
  assert(r.trace.find(prefix+" next=cleanup")!=std::string::npos);
  assert(r.trace.find(prefix+" next=propagation")!=std::string::npos);
  // Cleanup has finished before taking the first Propagation reference.
  const auto expected="phase=propagation stage_before="+std::to_string(residual_nnz)+" after="+std::to_string(residual_nnz)+" next=aggregation";
  assert(r.trace.find(expected)!=std::string::npos);
  assert(occurrences(r.trace,"[schedule-phase] phase=aggregation")==1);
  assert(r.summary.iterations==3);
}
void test_limits_and_empty() {
  Model model(1,{{1}},{2},{2});auto p=parameters();p.max_iters=0;
  Result r(model,p,true);assert(r.summary.reduced_nnz==1 && r.summary.iterations==0);
  p.max_iters=10;p.max_time=1e-12;Result timed(model,p,true);assert(timed.summary.iterations==0);
  Model empty(0,{}, {}, {});Result e(empty,parameters(),true);assert(e.summary.reduced_nnz==0);
}
void test_small_cleanup_cascade_after_startup_budget() {
  // A single singleton row removes only 1/105 of nnz. Its fixed variable
  // enables the next singleton on the next Cleanup pass, and so on.
  // The 5% policy would leave the chain behind with Startup's budget spent.
  constexpr int background=10, n=13;
  std::vector<std::vector<double>> rows(background,std::vector<double>(n,0));
  for(int i=0;i<background;++i)for(int j=0;j<background;++j)rows[i][j]=1+(i*7+j*3+i*j)%19;
  std::vector<double> al(background,-INFINITY),au(background,INFINITY);
  for(int k=0;k<3;++k) {
    std::vector<double> row(n,0);row[background+k]=1;
    if(k)row[background+k-1]=1;
    rows.push_back(row);al.push_back(k ? 2 : 1);au.push_back(k ? 2 : 1);
  }
  Model model(n,rows,al,au);
  auto p=parameters();p.enable_startup=true;p.max_iters=1;
  p.enable_singleton_columns=false;
  p.bound_strengthening_max_bound_only_rounds=64;
  p.bound_strengthening_bound_only_nnz_round_budget=104*3;
  Result host(model,p,false),device(model,p,true);compare_recovery(host,device);
  assert(model.lp.A.nnz==105 && device.summary.iterations==1);
  assert(device.summary.reduced_rows==background && device.summary.reduced_cols==background);
  assert(device.summary.reduced_nnz==100);
  assert(occurrences(device.trace,"[schedule-phase] phase=cleanup")==5);
  assert(device.trace.find("phase=cleanup stage_before=105 after=104 next=cleanup")!=std::string::npos);
  assert(device.trace.find("phase=cleanup stage_before=100 after=100 next=cleanup")!=std::string::npos);
  assert(device.trace.find("phase=cleanup stage_before=100 after=100 next=propagation")!=std::string::npos);
  const auto budget_position=device.trace.find("[schedule-bound-budget] initial_nnz=105 limit=2");
  assert(budget_position!=std::string::npos);
  assert(budget_position<device.trace.find("[schedule-phase] phase=cleanup"));
  assert(occurrences(device.trace,"[schedule-bound-budget]")==1);
  assert(occurrences(device.trace,"[schedule-phase] phase=propagation")==0);
  auto xr=model.upload(std::vector<double>(background,0));
  auto yr=model.upload(std::vector<double>(background,0));
  auto zr=model.upload(std::vector<double>(background,0));
  auto restored=postsolve_gpu(xr,yr,zr,device.summary.record,&model.lp);
  const auto x=download(restored.x_org,n);
  for(int j=0;j<background;++j)assert(x[j]==0);
  for(int j=background;j<n;++j)assert(std::fabs(x[j]-1)<1e-12);
  cudaFree(restored.x_org);cudaFree(restored.y_org);cudaFree(restored.z_org);
}
void test_wide_singleton_recovery(bool startup, int coefficient_sign, int variable_sign) {
  const int n = 4096;
  // t = sum(x): all x have one redundant bound; t has both bounds redundant.
  // Choosing x by column ID alternates substitution and dual inference, storing
  // the remaining row again each time. Choosing t removes the row in one pass.
  std::vector<double> row(n, -coefficient_sign), cost(n, 0);
  row.back() = coefficient_sign;cost.back() = variable_sign;
  Model model(n, {row}, {0}, {0}, variable_sign > 0 ? 0 : -INFINITY,
              variable_sign > 0 ? INFINITY : 0);
  check(cudaMemcpy(model.lp.c, cost.data(), sizeof(double)*n, cudaMemcpyHostToDevice));
  auto p = parameters();p.enable_startup = startup;
  Result host(model,p,false),device(model,p,true);compare_recovery(host,device);
  assert(device.summary.reduced_rows == 0 && device.summary.reduced_cols == 0);
  const auto& tape = device.summary.record.tape;
  int substitutions = 0;
  for (int type : tape.types) substitutions += type == int(PostsolveReductionType::SubCol);
  assert(substitutions == 1);
  assert(tape.indices.size() < 4*n && tape.vals.size() < 8*n);
  auto restored = postsolve_gpu(nullptr,nullptr,nullptr,device.summary.record);
  for (double x : download(restored.x_org,n)) assert(x == 0);
  assert(device.summary.obj_constant_delta == 0);
  cudaFree(restored.x_org);cudaFree(restored.y_org);cudaFree(restored.z_org);
}
void test_fixed_bound_budget(int limit, bool redundant_rows, bool row_side_updates,
                            bool initial_singleton) {
  // Each Jacobi bound-strengthening call can move the cap x[0] <= 1 by
  // exactly one edge of x[j] <= x[j-1]. Compare actual resulting bounds,
  // not just the scheduler's counters.
  const int n = 8;
  std::vector<std::vector<double>> rows;
  std::vector<double> al, au;
  for (int j=1;j<n;++j) {
    std::vector<double> row(n,0);row[j-1]=-1;row[j]=1;
    rows.push_back(row);al.push_back(row_side_updates ? -100 : -INFINITY);au.push_back(0);
  }
  if (redundant_rows) {
    // Deleting these rows halves nnz. Recomputing L afterwards would wrongly
    // double its value and allow two extra propagation steps for limit=2.
    for (int j=1;j<n;++j) {
      std::vector<double> row(n,0);row[0]=row[j]=1;
      rows.push_back(row);al.push_back(-INFINITY);au.push_back(100);
    }
  }
  if (initial_singleton) {
    std::vector<double> row(n,0);row[0]=1;
    rows.push_back(row);al.push_back(-INFINITY);au.push_back(1);
  }
  Model model(n,rows,al,au);
  if (!initial_singleton) {
    const double cap=1;check(cudaMemcpy(model.lp.u,&cap,sizeof(cap),cudaMemcpyHostToDevice));
  }
  auto p=parameters();
  p.enable_infeasible_and_fixed_variables=p.enable_empty_columns=p.enable_empty_rows=false;
  p.enable_singleton_rows=initial_singleton;p.enable_singleton_columns=false;
  p.enable_dual_fix=p.enable_doubleton_equations=p.enable_parallel_rows=p.enable_parallel_columns=false;
  p.enable_infeasible_and_redundant_rows=redundant_rows || row_side_updates;
  p.bound_strengthening_max_bound_only_rounds=64;
  const int chain_nnz=14*(redundant_rows ? 2 : 1);
  p.bound_strengthening_bound_only_nnz_round_budget=chain_nnz*limit;
  const int initial_nnz=chain_nnz+(initial_singleton ? 1 : 0);
  const int actual_limit=std::max(1,chain_nnz*limit/initial_nnz);
  Result host(model,p,false),device(model,p,true);compare_recovery(host,device);
  const int expected_calls=actual_limit+(redundant_rows ? 1 : 0);
  assert(device.summary.reduced_rows==7 && device.summary.reduced_cols==n);
  assert(device.summary.reduced_nnz==14);
  const auto upper=download(device.summary.reduced_lp.u,n);
  for(int j=0;j<n;++j)assert(upper[j]==(j<=expected_calls ? 1 : 10));
  assert(occurrences(device.trace,"[schedule-phase] phase=propagation")==expected_calls);
  assert(occurrences(device.trace,"[schedule-bound-budget]")==1);
  const auto budget="[schedule-bound-budget] initial_nnz="+std::to_string(initial_nnz)+
                    " limit="+std::to_string(actual_limit);
  assert(device.trace.find(budget)!=std::string::npos);
  // Each successful pure-bound continuation visits Cleanup but must retain
  // the counter. Redundant row-side updates are not structural deletions.
  assert(device.trace.find("bound_only_rounds="+std::to_string(actual_limit)+
                           " limit="+std::to_string(actual_limit)+
                           " nnz_progress=0 bound_progress=0")!=std::string::npos);
  if(row_side_updates) {
    assert(device.trace.find("rule=infeasible_and_redundant_rows changed=1")!=std::string::npos);
    assert(device.trace.find("bounds_changed=1 deleted=0 bound_only_rounds=1")!=std::string::npos);
    for(double lower:download(device.summary.reduced_lp.AL,7))assert(lower==-INFINITY);
  }
  if(redundant_rows)assert(device.trace.find("bounds_changed=1 deleted=1 bound_only_rounds=0")!=std::string::npos);
  if(initial_singleton)assert(device.trace.find("phase=cleanup stage_before=15 after=14 next=cleanup")!=std::string::npos);
}
}
int main() {
  runtime::options().profile_rules = true;
  runtime::options().profile_apply = true;
  test_cleanup_follows_startup_at_iteration_limit();
  test_order();
  test_fixed_bound_budget(1,false,false,false);
  test_fixed_bound_budget(2,false,false,false);
  test_fixed_bound_budget(4,false,false,false);
  test_fixed_bound_budget(2,true,false,false);
  test_fixed_bound_budget(2,false,true,false);
  test_fixed_bound_budget(2,false,false,true);
  test_cleanup_dependency_repeats_without_budget();
  test_interleaved_progress_flags();
  test_stage_progress_advances_after_no_change();
  test_parallel_progress_does_not_credit_earlier_phases();
  test_single_pass_row_side_update();
  test_carry_and_doubleton_equations(12);
  test_carry_and_doubleton_equations(2);
  test_aggregation_iteration_limit();
  test_parallel();
  test_progress(19);test_progress(18);
  test_limits_and_empty();
  test_small_cleanup_cascade_after_startup_budget();
  for (bool startup : {false,true})
    for (int coefficient_sign : {-1,1})
      for (int variable_sign : {-1,1})
        test_wide_singleton_recovery(startup,coefficient_sign,variable_sign);
  std::cout<<"PASS four-phase schedule with budget-free Cleanup, apply boundaries, 100-nnz threshold, carry, per-phase progress, limits and recovery\n";
}
