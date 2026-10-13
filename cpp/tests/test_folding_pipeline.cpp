#include "gpu_presolver/detail/runtime_options.hpp"
#include "../src/folding/folding_internal.cuh"
#include "gpu_presolver/folding/folding.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace runtime = gpu_presolver::runtime;

using namespace gpu_presolver::folding;

void ck(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void require(bool b, const char* message) { if (!b) throw std::runtime_error(message); }
template<class T> std::vector<T> host(const T* p, std::size_t n) {
  std::vector<T> v(n);
  if (n) ck(cudaMemcpy(v.data(), p, n*sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}
template<class T> T* device(const std::vector<T>& v) {
  T* p=nullptr;
  if (!v.empty()) { ck(cudaMalloc(&p, v.size()*sizeof(T))); ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); }
  return p;
}
template<class T> void equal(const std::vector<T>& a, const std::vector<T>& b) {
  require(a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0), "bitwise output differs");
}

// Independent CPU partition oracle uses the original float signature and
// adjacent tolerance rule; tie order cannot affect a nonnegative-tolerance ID.
std::vector<int> partition(const std::vector<int>& ids,
                           const std::vector<double>& signatures,double tol) {
  std::vector<int> order(ids.size()),result(ids.size());
  for (std::size_t i=0;i<order.size();++i) order[i]=static_cast<int>(i);
  std::sort(order.begin(),order.end(),[&](int a,int b) {
    if (ids[a]!=ids[b]) return ids[a]<ids[b];
    return static_cast<float>(signatures[a])<static_cast<float>(signatures[b]);
  });
  int color=0;
  for (std::size_t i=0;i<order.size();++i) {
    const int member=order[i];
    if (i==0 || ids[member]!=ids[order[i-1]] ||
        static_cast<double>(static_cast<float>(signatures[member]))>
        static_cast<double>(static_cast<float>(signatures[order[i-1]]))+tol) ++color;
    result[member]=color;
  }
  return result;
}

void test_updates(int rows, int cols, double tol) {
  std::mt19937_64 rng(912377);
  std::vector<int> row_ids(rows),col_ids(cols);
  std::vector<double> row_sig(rows),col_sig(cols);
  for (auto* v : {&row_ids,&col_ids}) for (auto& x : *v) x=static_cast<int>(rng()%35)-17;
  for (auto* v : {&row_sig,&col_sig}) for (auto& x : *v) {
    const auto k=rng()%25;
    x=k==0 ? -0.0 : k==1 ? std::numeric_limits<double>::infinity() :
      k==2 ? -std::numeric_limits<double>::infinity() : (static_cast<int>(rng()%41)-20)*0.75e-8;
  }
  auto expected_row=row_ids,expected_col=col_ids;
  for (int round=0;round<3;++round) {
    expected_row=partition(expected_row,row_sig,tol);
    expected_col=partition(expected_col,col_sig,tol);
  }
  std::vector<int> reference_row,reference_col;
  std::vector<double> reference_row_sig,reference_col_sig;
  int ref_rows=0,ref_cols=0;
  for (bool batch : {false,true}) {
    FoldingWorkspace w;
    init_workspace(w,rows,cols);
    ck(cudaMemcpy(w.row_color_id,row_ids.data(),rows*sizeof(int),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(w.col_color_id,col_ids.data(),cols*sizeof(int),cudaMemcpyHostToDevice));
    // Repeated queued updates exercise scratch reuse on both sides, with no
    // intervening host array reads. Same-stream ordering must protect scratch.
    for (int round=1;round<=3;++round) {
      ck(cudaMemcpy(w.row_color_sig,row_sig.data(),rows*sizeof(double),cudaMemcpyHostToDevice));
      ck(cudaMemcpy(w.col_color_sig,col_sig.data(),cols*sizeof(double),cudaMemcpyHostToDevice));
      const bool fuse=batch;
      const int rc=folding_update_color_and_sig(w,true,w.row_color_sig,tol,round,2,batch,fuse);
      if (!batch) w.num_row_color=rc;
      if (!fuse) folding_write_row_color_sig(w,round,2);
      const int cc=folding_update_color_and_sig(w,false,w.col_color_sig,tol,round,1,batch,fuse);
      if (!batch) w.num_col_color=cc;
      if (!fuse) folding_write_col_color_sig(w,round,1);
      if (batch) folding_read_color_counts(w);
    }
    const auto ids_r=host(w.row_color_id,rows),ids_c=host(w.col_color_id,cols);
    const auto sig_r=host(w.row_color_sig,rows),sig_c=host(w.col_color_sig,cols);
    equal(ids_r,expected_row);equal(ids_c,expected_col);
    if (!batch) {
      reference_row=ids_r;reference_col=ids_c;reference_row_sig=sig_r;reference_col_sig=sig_c;
      ref_rows=w.num_row_color;ref_cols=w.num_col_color;
    } else {
      equal(reference_row,ids_r);equal(reference_col,ids_c);
      equal(reference_row_sig,sig_r);equal(reference_col_sig,sig_c);
      require(w.num_row_color==ref_rows && w.num_col_color==ref_cols,"batched counts differ");
    }
    free_workspace(w);
  }
}

void test_shared_map(bool grow) {
  gpu_presolver::presolve::LPInfoGpu lp;
  lp.A={3,4,0,device<NnzOffset>({0,0,0,0}),nullptr,nullptr};
  lp.c=device<double>({2,3,2,3});lp.l=device<double>({0,1,0,1});lp.u=device<double>({10,20,10,20});
  lp.AL=device<double>({1,2,1});lp.AU=device<double>({3,4,3});
  FoldingWorkspace w;init_workspace(w,3,4);
  std::vector<int> ri={1,2,1},ci={1,2,1,2};
  w.num_row_color=2;w.num_col_color=2;
  ck(cudaMemcpy(w.row_color_id,ri.data(),ri.size()*sizeof(int),cudaMemcpyHostToDevice));
  ck(cudaMemcpy(w.col_color_id,ci.data(),ci.size()*sizeof(int),cudaMemcpyHostToDevice));
  auto folded=reduce_size(w,lp,1e-8,nullptr);
  if (grow) {
    free_lp_device(folded);
    ri={1,2,3};ci={1,2,3,4};w.num_row_color=3;w.num_col_color=4;
    ck(cudaMemcpy(w.row_color_id,ri.data(),ri.size()*sizeof(int),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(w.col_color_id,ci.data(),ci.size()*sizeof(int),cudaMemcpyHostToDevice));
    folded=reduce_size(w,lp,1e-8,nullptr);
  }
  auto* row_pointer=w.row_color_id;
  auto* col_pointer=w.col_color_id;
  auto map=build_device_map(w,nullptr);
  require(map.row_color_id==row_pointer && map.col_color_id==col_pointer,"IDs were copied instead of transferred");
  require(!w.row_color_id && !w.col_color_id,"workspace still owns transferred IDs");
  free_workspace(w);
  const auto h=copy_map_to_host(map);
  equal(h.row_color_id,ri);equal(h.col_color_id,ci);
  equal(h.row_scale,grow?std::vector<double>{1,1,1}:std::vector<double>{0.5,1,0.5});
  equal(h.col_scale,grow?std::vector<double>{1,1,1,1}:std::vector<double>{0.5,0.5,0.5,0.5});
  free_folding_map_device(map);free_folding_map_device(map);
  free_lp_device(folded);free_lp_device(lp);
}

int main() {
  try {
    ck(cudaSetDevice(0));
    runtime::options().profile_folding = runtime::Options{}.profile_folding;
    require(folding_batch_refinement(),"normal refinement is not batched");
    runtime::options().profile_folding = true;
    require(!folding_batch_refinement(),"profiling did not restore synchronous count reads");
    runtime::options().profile_folding = runtime::Options{}.profile_folding;
    for (double tol : {0.0,1e-8}) {
      test_updates(0,0,tol);
      test_updates(0,17,tol);
      test_updates(33,0,tol);
      test_updates(33,17,tol);
      test_updates(65537,65537,tol);
      test_updates(131073,64033,tol);
    }
    test_shared_map(false);test_shared_map(true);
    // A map built without reduction statistics must retain its old copy path.
      FoldingWorkspace w;init_workspace(w,1,1);
    const int one=1;
    ck(cudaMemcpy(w.row_color_id,&one,sizeof(one),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(w.col_color_id,&one,sizeof(one),cudaMemcpyHostToDevice));
    auto map=build_device_map(w,nullptr);
    require(map.row_color_id!=w.row_color_id && map.col_color_id!=w.col_color_id,"missing histogram fallback lost ownership");
    free_workspace(w);equal(host(map.row_scale,1),std::vector<double>{1});free_folding_map_device(map);
    std::cout<<"PASS pipeline: CPU partition oracle, bitwise fused signatures/counts, scratch reuse, histogram growth and map lifetime\n";
  } catch (const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<'\n';return 2; }
}
