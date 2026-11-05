// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/gpu_postsolve.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_presolve.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_presolve_kernels.hpp"
#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_infeasible_and_fixed_variables.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_empty_rows_and_columns.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_singleton_rows.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_singleton_columns.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_infeasible_and_redundant_rows.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_bound_strengthening.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_dual_fix.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_parallel_rows.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_parallel_columns.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
using namespace gpu_preprocessor::lp::presolve;
namespace {
void check(cudaError_t e) { if (e != cudaSuccess) { std::cerr << cudaGetErrorString(e) << '\n'; std::abort(); } }
template<class T> std::vector<T> download(const T* p, int n) {
  std::vector<T> v(n); if(n) check(cudaMemcpy(v.data(), p, n*sizeof(T), cudaMemcpyDeviceToHost)); return v;
}
struct Model {
  int m, n;
  std::vector<NnzOffset> ptr;
  std::vector<int> col;
  std::vector<double> val, l, u, c, al, au;
  std::vector<std::uint8_t> rows, cols;
};
struct Fixture {
  LPInfoGpu lp;
  PresolvePlanGpu plan;
  PresolveStatsGpu stats;
  std::vector<void*> allocations;
  std::vector<int> rows, cols;
  template<class T> T* upload(const std::vector<T>& v) {
    T* p = nullptr;
    if (!v.empty()) { check(cudaMalloc(&p, v.size()*sizeof(T))); allocations.push_back(p);
      check(cudaMemcpy(p, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice)); }
    return p;
  }
  template<class T> T* zeros(int n) { return upload(std::vector<T>(n)); }
  Fixture(const Model& h, bool compact, bool resident = false) {
    invalidate_matrix_analysis();
    for(int i=0;i<h.m;++i) if(!compact || h.rows[i]) rows.push_back(i);
    for(int j=0;j<h.n;++j) if(!compact || h.cols[j]) cols.push_back(j);
    std::vector<int> map(h.n,-1);for(int j=0;j<int(cols.size());++j)map[cols[j]]=j;
    std::vector<NnzOffset> ptr{0};std::vector<int> col;std::vector<double> val;
    for(int i:rows) { for(auto p=h.ptr[i];p<h.ptr[i+1];++p)if(map[h.col[p]]>=0){col.push_back(map[h.col[p]]);val.push_back(h.val[p]);}ptr.push_back(col.size()); }
    const int m=rows.size(), n=cols.size();
    std::vector<NnzOffset> tp(n+1);for(int j:col)++tp[j+1];for(int j=0;j<n;++j)tp[j+1]+=tp[j];
    auto cursor=tp;std::vector<int> ti(col.size());std::vector<double> tv(val.size());
    for(int i=0;i<m;++i)for(auto p=ptr[i];p<ptr[i+1];++p){auto q=cursor[col[p]]++;ti[q]=i;tv[q]=val[p];}
    lp.A={m,n,NnzOffset(val.size()),upload(ptr),upload(col),upload(val)};
    lp.AT={n,m,NnzOffset(val.size()),upload(tp),upload(ti),upload(tv)};
    auto select=[](const auto& v,const std::vector<int>& ids) { auto out=v;out.clear();for(int i:ids)out.push_back(v[i]);return out; };
    lp.l=upload(select(h.l,cols));lp.u=upload(select(h.u,cols));lp.c=upload(select(h.c,cols));
    lp.AL=upload(select(h.al,rows));lp.AU=upload(select(h.au,rows));
    plan.new_l=upload(select(h.l,cols));plan.new_u=upload(select(h.u,cols));plan.new_c=upload(select(h.c,cols));
    plan.new_AL=upload(select(h.al,rows));plan.new_AU=upload(select(h.au,rows));
    plan.keep_row_mask=upload(select(h.rows,rows));plan.keep_col_mask=upload(select(h.cols,cols));
    plan.rule_entry_keep_col=upload(select(h.cols,cols));
    plan.propagation_reference_l=lp.l;plan.propagation_reference_u=lp.u;
    plan.tape.resident=resident;
    stats.row_nnz=zeros<int>(m);stats.col_nnz=zeros<int>(n);
    stats.empty_row_mask=zeros<std::uint8_t>(m);stats.empty_col_mask=zeros<std::uint8_t>(n);
    stats.singleton_row_mask=zeros<std::uint8_t>(m);stats.singleton_column_mask=zeros<std::uint8_t>(n);
    stats.singleton_row_col=zeros<int>(m);stats.singleton_column_row=zeros<int>(n);
    stats.singleton_row_val=zeros<double>(m);stats.singleton_column_val=zeros<double>(n);
    refresh_live_stats(stats,lp,plan);stats.live_masks=!compact;
  }
  ~Fixture(){for(void* p:allocations)cudaFree(p);}
  void refresh() {
    refresh_live_stats(stats,lp,plan);
    check(cudaMemcpy(const_cast<std::uint8_t*>(plan.rule_entry_keep_col),plan.keep_col_mask,lp.A.cols,cudaMemcpyDeviceToDevice));
    plan.has_change=plan.has_col_action=plan.has_row_action=false;
  }
};
using Rule=void(*)(PresolvePlanGpu&,const LPInfoGpu&,const PresolveStatsGpu&,const PresolveParams&);
void fixed(PresolvePlanGpu& p,const LPInfoGpu& lp,const PresolveStatsGpu&,const PresolveParams& params) {apply_rule_infeasible_and_fixed_variables(p,lp,params);}
void dual(PresolvePlanGpu& p,const LPInfoGpu& lp,const PresolveStatsGpu& stats,const PresolveParams& params) {apply_rule_dual_fix(p,lp,stats,params);}
void single_col(PresolvePlanGpu& p,const LPInfoGpu& lp,const PresolveStatsGpu& stats,const PresolveParams& params) {apply_rule_singleton_columns(p,lp,stats,params);}
void equal(double x,double y) { if(x!=y) {std::cerr<<"values differ: "<<x<<" / "<<y<<'\n';std::abort();} }
std::vector<int> original_indices(const Fixture& f) {
  auto out=f.plan.tape.indices;
  for(int k=0;k<int(f.plan.tape.types.size());++k) {
    const auto begin=f.plan.tape.index_starts[k],end=f.plan.tape.index_starts[k+1];
    auto col=[&](NnzOffset p){if(out[p]>=0)out[p]=f.cols.at(out[p]);};
    auto row=[&](NnzOffset p){if(out[p]>=0)out[p]=f.rows.at(out[p]);};
    switch(static_cast<PostsolveReductionType>(f.plan.tape.types[k])) {
      case PostsolveReductionType::FixedVariable:
        col(begin);for(auto p=begin+1;p<end;++p)row(p);break;
      case PostsolveReductionType::InfiniteBoundDualFix:
        col(begin+1);for(auto p=begin+3;p<end;){int length=out[p++];for(int j=0;j<length;++j)col(p++);}break;
      case PostsolveReductionType::SingletonColumnSubstitution:
        col(begin);row(begin+1);for(auto p=begin+3;p<end;++p)col(p);break;
      case PostsolveReductionType::ParallelColumn: col(begin);col(begin+1);break;
      case PostsolveReductionType::ParallelRow: row(begin);row(begin+1);break;
      case PostsolveReductionType::DeletedRow:
        row(begin);if(end-begin>1)col(begin+1);break;
      case PostsolveReductionType::BoundSourceRow:
        row(begin);for(auto p=begin+1;p<end;++p)col(p);break;
      case PostsolveReductionType::BoundChangeCompact: col(begin);break;
      case PostsolveReductionType::BoundBatch: break;
      case PostsolveReductionType::BoundChangeSharedRow:
        col(begin);row(begin+1);row(begin+2);break;
      case PostsolveReductionType::RowImpliedBoundChange:
        col(begin);row(begin+1);
        if(end-begin>=5) {row(begin+2);for(auto p=begin+5;p<end;++p) col(p);}
        break;
      case PostsolveReductionType::VariableBoundChange: col(begin);break;
      case PostsolveReductionType::RowLowerBoundChange:
      case PostsolveReductionType::RowUpperBoundChange:
        row(begin);break;
      case PostsolveReductionType::RowDualShift:
        row(begin);for(auto p=begin+2;p<end;++p)col(p);break;
      default: assert(false);
    }
  }
  return out;
}
void compare(Fixture& a,Fixture& b) {
  assert(a.plan.has_infeasible==b.plan.has_infeasible);assert(a.plan.has_unbounded==b.plan.has_unbounded);
  assert(a.plan.has_change==b.plan.has_change);equal(a.plan.obj_constant_delta,b.plan.obj_constant_delta);
  auto ar=download(a.plan.keep_row_mask,a.lp.A.rows),br=download(b.plan.keep_row_mask,b.lp.A.rows);
  auto ac=download(a.plan.keep_col_mask,a.lp.A.cols),bc=download(b.plan.keep_col_mask,b.lp.A.cols);
  for(int i=0;i<int(b.rows.size());++i)assert(ar[b.rows[i]]==br[i]);
  for(int j=0;j<int(b.cols.size());++j)assert(ac[b.cols[j]]==bc[j]);
  for(auto pair:{std::make_pair(a.plan.new_AL,b.plan.new_AL),{a.plan.new_AU,b.plan.new_AU}}) {
    auto x=download(pair.first,a.lp.A.rows),y=download(pair.second,b.lp.A.rows);
    for(int i=0;i<int(b.rows.size());++i)if(br[i])equal(x[b.rows[i]],y[i]);
  }
  for(auto pair:{std::make_pair(a.plan.new_l,b.plan.new_l),{a.plan.new_u,b.plan.new_u},{a.plan.new_c,b.plan.new_c}}) {
    auto x=download(pair.first,a.lp.A.cols),y=download(pair.second,b.lp.A.cols);
    for(int j=0;j<int(b.cols.size());++j)if(bc[j])equal(x[b.cols[j]],y[j]);
  }
  export_local_recovery_tape(a.plan.tape);export_local_recovery_tape(b.plan.tape);
  assert(a.plan.tape.types==b.plan.tape.types);
  assert(a.plan.tape.dual_modes==b.plan.tape.dual_modes);
  assert(a.plan.tape.vals==b.plan.tape.vals);
  assert(a.plan.tape.index_starts==b.plan.tape.index_starts);
  assert(a.plan.tape.value_starts==b.plan.tape.value_starts);
  assert(original_indices(a)==original_indices(b));
}
void infinite_dual_fix_replay() {
  // Three records with two, one and zero supporting rows. Unselected columns
  // separate the records, so integer/value prefix offsets must stay independent.
  const double inf = INFINITY;
  Model h{3,5,{0,2,4,7},{0,1,1,2,0,2,3},{1,2,1,-1,1,2,3},
          {0,1,0,-inf,-inf},{10,inf,10,10,7},{1,0,1,0,0},
          {11,4,-inf},{inf,inf,17},{1,1,1},{1,1,1,1,1}};
  PresolveParams params;
  params.max_iters = 1;
  params.enable_startup = false;
  params.enable_infeasible_and_fixed_variables = false;
  params.enable_empty_rows = false;
  params.enable_empty_columns = false;
  params.enable_singleton_rows = false;
  params.enable_singleton_columns = false;
  params.enable_infeasible_and_redundant_rows = false;
  params.enable_bound_strengthening = false;
  params.enable_doubleton_equations = false;
  params.enable_parallel_rows = false;
  params.enable_parallel_columns = false;
  params.dual_fix_max_nonzero_cost_col_degree = 0;
  const char* saved = std::getenv("GPUPREPROCESSOR_DEVICE_RECOVERY");
  const bool had_option = saved != nullptr;
  const std::string saved_option = saved ? saved : "";
  for (bool resident : {false, true}) {
    setenv("GPUPREPROCESSOR_DEVICE_RECOVERY", resident ? "1" : "0", 1);
    Fixture f(h, false, resident);
    auto summary = run_gpu_presolve_with_record(f.lp, params);
    assert(!summary.has_infeasible && !summary.has_unbounded);
    assert(summary.reduced_rows == 0 && summary.reduced_cols == 2);
    assert(summary.obj_constant_delta == 0);
    // Replay resident records before exporting a host snapshot.
    auto* x_red = f.upload(std::vector<double>{1,2});
    auto* z_red = f.upload(std::vector<double>{0.25,-0.5});
    auto result = postsolve_gpu(x_red, nullptr, z_red, summary.record);
    // x1 = max(1, (11-x0)/2, 4+x2), x3 = min(10, (17-x0-2*x2)/3).
    assert((download(result.x_org, 5) == std::vector<double>{1,6,2,4,7}));
    assert((download(result.y_org, 3) == std::vector<double>{0,0,0}));
    assert((download(result.z_org, 5) == std::vector<double>{0.25,0,-0.5,0,0}));
    cudaFree(result.x_org); cudaFree(result.y_org); cudaFree(result.z_org);
    export_recovery_record(summary.record);
    assert((summary.record.col_red2org == std::vector<int>{0,2}));
    const auto& tape = summary.record.tape;
    assert((tape.types == std::vector<int>(3, int(PostsolveReductionType::InfiniteBoundDualFix))));
    assert((tape.dual_modes == std::vector<std::uint8_t>(3, std::uint8_t(PostsolveDualMode::Exact))));
    assert((tape.index_starts == std::vector<NnzOffset>{0,9,16,19}));
    assert((tape.value_starts == std::vector<NnzOffset>{0,7,12,13}));
    assert((tape.indices == std::vector<int>{1,1,2,2,0,1,2,1,2, -1,3,1,3,0,2,3, -1,4,0}));
    assert((tape.vals == std::vector<double>{1,11,1,2,4,1,-1, 10,17,1,2,3, 7}));
    // The same nonempty row payloads must replay identically in old and mixed
    // layouts, through both host upload and resident segment borrowing.
    for (bool mixed : {false, true}) {
      PresolveRecordGpu compatible;
      compatible.m0 = 3; compatible.n0 = 5; compatible.m1 = 0; compatible.n1 = 2;
      compatible.col_red2org = {0,2}; compatible.col_org2red = {0,-1,1,-1,-1};
      auto& old = compatible.tape;
      old.types = tape.types; old.dual_modes = tape.dual_modes;
      if (mixed) {
        old.index_starts = {0,9,15,17}; old.value_starts = {0,7,13,15};
        old.indices = {1,1,2,2,0,1,2,1,2, -1,3,3,0,2,3, -1,4};
        old.vals = {1,11,1,2,4,1,-1, 1,10,17,1,2,3, 0,7};
      } else {
        old.index_starts = {0,8,14,16}; old.value_starts = old.index_starts;
        old.indices = {1,1,2,0,1,2,1,2, -1,3,3,0,2,3, -1,4};
        old.vals = {2,1,11,1,2,4,1,-1, 1,10,17,1,2,3, 0,7};
      }
      if (resident) {
        compatible.device = std::make_shared<DeviceRecoveryState>();
        auto& d = *compatible.device;
        d.cols.count = 2;
        check(cudaMalloc(&d.cols.data, 2 * sizeof(int)));
        check(cudaMemcpy(d.cols.data, compatible.col_red2org.data(), 2 * sizeof(int), cudaMemcpyHostToDevice));
        d.records = 3; d.indices = old.indices.size(); d.values = old.vals.size();
        flush_host_recovery_segment(old);
        d.segments.push_back(std::move(old.segments.front()));
      }
      auto restored = postsolve_gpu(x_red, nullptr, z_red, compatible);
      assert((download(restored.x_org, 5) == std::vector<double>{1,6,2,4,7}));
      assert((download(restored.y_org, 3) == std::vector<double>{0,0,0}));
      assert((download(restored.z_org, 5) == std::vector<double>{0.25,0,-0.5,0,0}));
      cudaFree(restored.x_org); cudaFree(restored.y_org); cudaFree(restored.z_org);
    }
  }
  if (had_option) setenv("GPUPREPROCESSOR_DEVICE_RECOVERY", saved_option.c_str(), 1);
  else unsetenv("GPUPREPROCESSOR_DEVICE_RECOVERY");
}
void rules() {
  const double inf=INFINITY;
  std::vector<Model> cases{
    // Deleted leading coefficient; unequal physical lengths, equal live rows.
    {3,4,{0,3,5,7},{0,1,2,1,2,0,3},{100,1,1,2,2,7,3},
     {9,0,0,0},{9,10,10,10},{1,1,1,0},{2,4,-inf},{8,12,20},{1,1,0},{0,1,1,1}},
    // A deleted row exposes singleton columns and a lower degree for dual fix.
    {3,3,{0,3,5,7},{0,1,2,0,1,0,2},{3,1,4,1,1,1,1},
     {0,0,0},{10,10,10},{1,0,0},{-inf,-inf,-inf},{30,8,8},{0,1,0},{1,1,1}},
    // A removed column exposes a singleton and an empty row/column.
    {3,4,{0,2,3,5},{0,1,0,2,3},{100,2,5,1,-1},
     {4,0,2,0},{4,10,2,10},{9,2,1,0},{4,0,-2},{4,0,4},{1,1,1},{0,1,1,1}},
    // A stored zero still counts toward live nnz and structural degree.
    {2,3,{0,3,5},{0,1,2,1,2},{3,0,1,1,1},
     {10,0,0},{10,10,10},{0,0,0},{-inf,3},{5,8},{1,1},{0,1,1}}
  };
  // Cancellation in objective reductions with holes in the physical columns.
  Model objective{1,600,{0,0},{},{},std::vector<double>(600,1),std::vector<double>(600,1),
    std::vector<double>(600),{0},{0},{1},std::vector<std::uint8_t>(600)};
  for(int j=0;j<600;++j){objective.cols[j]=j%2==0;objective.c[j]=j%6==0?1e16:j%6==2?-1e16:1;}
  cases.push_back(objective);
  PresolveParams params;params.dual_fix_max_nonzero_cost_col_degree=1;
  const std::vector<Rule> all{fixed,apply_rule_empty_columns,apply_rule_empty_rows,apply_rule_singleton_rows,
    single_col,apply_rule_infeasible_and_redundant_rows,apply_rule_bound_strengthening,
    dual,apply_rule_parallel_rows,apply_rule_parallel_columns};
  for(bool resident:{false,true}) for(int ci=0;ci<int(cases.size());++ci)for(int ri=0;ri<int(all.size());++ri) {
    std::cerr<<"case "<<ci<<" rule "<<ri<<" resident "<<resident<<'\n';
    Fixture a(cases[ci],false,resident),b(cases[ci],true,resident);
    assert(a.stats.live_nnz==b.stats.live_nnz);
    all[ri](a.plan,a.lp,a.stats,params);all[ri](b.plan,b.lp,b.stats,params);compare(a,b);
  }
  // Reuse the same matrix pointers after deleting another column. Verify support
  // and empty flags change, including stored zeros, without a physical apply.
  Fixture a(cases[3],false);
  auto mask=download(a.plan.keep_col_mask,3);mask[2]=0;
  check(cudaMemcpy(a.plan.keep_col_mask,mask.data(),3,cudaMemcpyHostToDevice));a.refresh();
  assert(a.stats.live_nnz==2);
  assert((download(a.stats.row_nnz,2)==std::vector<int>{1,1}));
  assert((download(a.stats.singleton_row_col,2)==std::vector<int>{1,1}));
  assert((download(a.stats.singleton_row_val,2)==std::vector<double>{0,1}));
}
void parallel_chain() {
  Model h{3,4,{0,3,5,7},{0,1,2,1,2,1,2},{100,1,1,2,2,1,1},
          {9,0,0,0},{9,10,10,10},{0,1,1,0},{2,4,0},{8,12,20},{1,1,1},{0,1,1,1}};
  Fixture a(h,false,true);
  PresolveParams params;
  for(Rule rule:{apply_rule_parallel_rows,apply_rule_parallel_columns}) {
    Fixture b(h,true,true);
    a.refresh();a.plan.tape={};a.plan.tape.resident=true;a.plan.obj_constant_delta=0;
    rule(a.plan,a.lp,a.stats,params);rule(b.plan,b.lp,b.stats,params);compare(a,b);
    auto scatter=[](auto& out,auto* device,const std::vector<int>& ids) {
      auto values=download(device,ids.size());for(int k=0;k<int(ids.size());++k)out[ids[k]]=values[k];
    };
    scatter(h.rows,b.plan.keep_row_mask,b.rows);scatter(h.cols,b.plan.keep_col_mask,b.cols);
    scatter(h.al,b.plan.new_AL,b.rows);scatter(h.au,b.plan.new_AU,b.rows);
    scatter(h.l,b.plan.new_l,b.cols);scatter(h.u,b.plan.new_u,b.cols);scatter(h.c,b.plan.new_c,b.cols);
  }
}
void propagation_reference() {
  // Nonzero costs within zero_tol exercise the legacy free-box guard.
  Model h{1,3,{0,3},{0,1,2},{50,1,1},{4,0,0},{4,10,1},{0,1e-12,1e-12},
          {4},{8},{1},{0,1,1}};
  Fixture a(h,false,true),b(h,true,true);
  a.lp.l=a.upload(std::vector<double>(3,-INFINITY));
  a.lp.u=a.upload(std::vector<double>(3,INFINITY));
  PresolveParams params;params.bound_strengthening_preserve_free_zero_cost=true;
  apply_rule_bound_strengthening(a.plan,a.lp,a.stats,params);
  apply_rule_bound_strengthening(b.plan,b.lp,b.stats,params);
  compare(a,b);
  equal(download(a.plan.new_l,3)[1],3);
  equal(download(a.plan.new_u,3)[1],8);
}
void long_masked_rules() {
  for(int n:{33,1025,4099}) {
    Model h{3,n,{0},{},{},std::vector<double>(n,0),std::vector<double>(n,1),
      std::vector<double>(n,0),{-INFINITY,-INFINITY,-INFINITY},{100,100,100},
      {1,1,0},std::vector<std::uint8_t>(n)};
    for(int j=0;j<n;++j) h.cols[j]=j>5 && (unsigned(j)*2654435761u%17)<7;
    for(int i=0;i<3;++i) {
      for(int j=0;j<n;++j){h.col.push_back(j);h.val.push_back((i+1)*(1.0+(j%7)*0.25));}
      h.ptr.push_back(h.col.size());
    }
    for(int hash_mode:{0,1})for(bool resident:{false,true}) {
      Fixture a(h,false,resident),b(h,true,resident);
      a.plan.parallel_row_hash_mode=b.plan.parallel_row_hash_mode=hash_mode;
      apply_rule_parallel_rows(a.plan,a.lp,a.stats,PresolveParams{});
      apply_rule_parallel_rows(b.plan,b.lp,b.stats,PresolveParams{});compare(a,b);
    }
    // Structural stats must also handle long stored rows reduced to one/zero
    // live entries, retaining stored zeros as entries and ignoring deleted rows.
    h.cols.assign(n,0);h.cols[n-1]=1;h.val[n-1]=0;
    Fixture a(h,false);
    assert((download(a.stats.row_nnz,3)==std::vector<int>{1,1,0}));
    assert((download(a.stats.singleton_row_col,3)==std::vector<int>{n-1,n-1,-1}));
    assert((download(a.stats.singleton_row_val,3)==std::vector<double>{0,h.val[2*n-1],0}));
    assert(a.stats.live_nnz==2);
  }
  Model tall{4099,3,{0},{},{},{0,0,0},{1,1,1},{0,0,0},
    std::vector<double>(4099,-INFINITY),std::vector<double>(4099,INFINITY),
    std::vector<std::uint8_t>(4099,0),{1,1,0}};
  tall.rows.back()=1;
  for(int i=0;i<tall.m;++i){
    for(int j=0;j<3;++j){tall.col.push_back(j);tall.val.push_back(j);}
    tall.ptr.push_back(tall.col.size());
  }
  Fixture a(tall,false);
  assert((download(a.stats.col_nnz,3)==std::vector<int>{1,1,0}));
  assert((download(a.stats.singleton_column_row,3)==std::vector<int>{4098,4098,-1}));
  assert((download(a.stats.singleton_column_val,3)==std::vector<double>{0,1,0}));
}

void activity() {
  // Cross both adaptive thresholds after holes disappear; cancellation makes
  // changed per-lane accumulation order visible to this exact comparison.
  for(int live:{7,8,9,1024,1025,1537}) {
    int n=live*2;Model h{1,n,{0,NnzOffset(n)},{},{},std::vector<double>(n,1),std::vector<double>(n,1),
      std::vector<double>(n,0),{-INFINITY},{INFINITY},{1},std::vector<std::uint8_t>(n,0)};
    for(int j=0;j<n;++j){h.col.push_back(j);h.val.push_back(j%6==0?1e16:j%6==2?-1e16:1);h.cols[j]=j%2==0;}
    Fixture a(h,false),b(h,true);
    std::vector<double> outputs;
    for(Fixture* f:{&a,&b}) {
      auto* lo=f->zeros<double>(1);auto* hi=f->zeros<double>(1);auto* ni=f->zeros<int>(1);auto* pi=f->zeros<int>(1);
      compute_row_activity_summary(lo,hi,ni,pi,f->lp.A,f->plan.new_l,f->plan.new_u,1e-9,
        f->plan.keep_row_mask,f->stats.live_masks?f->plan.keep_col_mask:nullptr,
        f->stats.live_masks?f->stats.row_nnz:nullptr);
      outputs.push_back(download(lo,1)[0]);outputs.push_back(download(hi,1)[0]);
    }
    equal(outputs[0],outputs[2]);equal(outputs[1],outputs[3]);
  }
}
}
int main() {
  infinite_dual_fix_replay();
  for(const char* mode:{"0","1"}) {
    for(const char* name:{"GPUPREPROCESSOR_P1_ADAPTIVE","GPUPREPROCESSOR_P1_EXTENDED","GPUPREPROCESSOR_P1_REUSE","GPUPREPROCESSOR_P1_HASH_SCAN"})setenv(name,mode,1);
    MatrixAnalysisScope scope;rules();parallel_chain();propagation_reference();activity();long_masked_rules();
  }
  std::cout<<"PASS deferred rule/compact differential tests, host/device recovery, adaptive activity and mask refresh\n";
}
