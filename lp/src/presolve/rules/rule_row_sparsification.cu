// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/rules/rule_row_sparsification.hpp"
#include "gpu_preprocessor/lp/presolve/transpose64.hpp"
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace gpu_preprocessor::lp::presolve {
namespace {
using Clock = std::chrono::steady_clock;
void checked(cudaError_t error) {
  if (error == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> struct Array {
  T* p = nullptr;
  Array() = default;
  explicit Array(std::size_t n) { allocate(n); }
  Array(const Array&) = delete;
  Array& operator=(const Array&) = delete;
  ~Array() { if (p) cudaFreeAsync(p, 0); }
  void allocate(std::size_t n) {
    if (n > SIZE_MAX / sizeof(T)) throw std::bad_alloc();
    if (p) checked(cudaFreeAsync(p, 0));
    p = nullptr;
    if (n) checked(cudaMallocAsync(reinterpret_cast<void**>(&p), n * sizeof(T), 0));
  }
  T* release() { T* result = p; p = nullptr; return result; }
  void get(T* destination, std::size_t n) const {
    checked(cudaMemcpy(destination, p, n * sizeof(T), cudaMemcpyDeviceToHost));
  }
};
template<class T> void allocate(T** output, std::size_t n) {
  if (n > SIZE_MAX / sizeof(T)) throw std::bad_alloc();
  if (n) checked(cudaMallocAsync(reinterpret_cast<void**>(output), n * sizeof(T), 0));
}
template<class T> void duplicate(T** output, const T* input, std::size_t n) {
  allocate(output, n);
  if (n) checked(cudaMemcpyAsync(*output, input, n * sizeof(T), cudaMemcpyDeviceToDevice, 0));
}
constexpr int nt = 256;
int grid(std::int64_t n) { return int(std::min<std::int64_t>(8192, (n + nt - 1) / nt)); }
template<class To, class From> __global__ void convert(const From* a, To* b, std::int64_t n) {
  for (auto k = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; k < n;
       k += std::int64_t(blockDim.x) * gridDim.x) b[k] = To(a[k]);
}
struct Candidate { int row, parent; double rhs, parent_rhs; };
struct Decision { std::int64_t nnz; double rhs; int alpha, valid; };
constexpr int threads = 256;
// Stable merge-path partition. Equal column IDs take A before B.
__device__ std::int64_t partition(std::int64_t diagonal, const int* a,
                                std::int64_t na, const int* b, std::int64_t nb) {
    std::int64_t lo = diagonal > nb ? diagonal - nb : 0;
    std::int64_t hi = diagonal < na ? diagonal : na;
    while (lo <= hi) {
        auto x = lo + (hi-lo)/2, y = diagonal-x;
        if (x>0 && y<nb && a[x-1]>b[y]) hi=x-1;
        else if (y>0 && x<na && b[y-1]>=a[x]) lo=x+1;
        else return x;
    }
    return lo;
}
// Each thread owns a contiguous interval in the stable merged stream.
// Duplicates can straddle intervals: test their global A/B neighbors.
__device__ std::int64_t merge_chunk(const int* ai, const double* av, std::int64_t na,
                                  const int* bi, const double* bv, std::int64_t nb,
                                  int alpha, int& bad, int* out_i, double* out_v,
                                  std::int64_t destination) {
    auto length=na+nb;
    auto begin=length*threadIdx.x/blockDim.x;
    auto end=length*(threadIdx.x+1)/blockDim.x;
    auto p=partition(begin,ai,na,bi,nb), q=begin-p;
    std::int64_t count=0;
    for (auto d=begin; d<end; ++d) {
        int col; double value; bool emit;
        if (p<na && (q==nb || ai[p]<=bi[q])) {
            col=ai[p];value=av[p];
            if (!isfinite(value) || (p>0 && ai[p-1]>=col)) bad=1;
            bool shared=q<nb && bi[q]==col;
            if (shared && value!=alpha*bv[q]) bad=1;
            emit=!shared && value!=0; ++p;
        } else {
            col=bi[q];value=-alpha*bv[q];
            if (!isfinite(bv[q]) || (q>0 && bi[q-1]>=col)) bad=1;
            bool shared=p>0 && ai[p-1]==col;
            emit=!shared && value!=0; ++q;
        }
        if (emit) {
            if (out_i) {out_i[destination+count]=col;out_v[destination+count]=value;}
            ++count;
        }
    }
    return count;
}
__global__ void validate_kernel(int count,int columns,const Candidate* pairs,
                                const std::int64_t* ptr,const int* ids,
                                const double* values,Decision* decisions) {
    int k=blockIdx.x;if(k>=count) return;
    auto pair=pairs[k];
    if(pair.row==pair.parent) {if(threadIdx.x==0) decisions[k]={0,0,0,0};return;}
    auto a=ptr[pair.row],b=ptr[pair.parent];
    auto na=ptr[pair.row+1]-a,nb=ptr[pair.parent+1]-b;
    int alpha=values[a]==values[b]?1:values[a]==-values[b]?-1:0;
    double rhs=0;int bad=alpha==0;
    if(pair.rhs==alpha*pair.parent_rhs) rhs=0;
    else if(pair.parent_rhs==0) rhs=pair.rhs;
    else if(pair.rhs==0) rhs=-alpha*pair.parent_rhs;
    else bad=1;
    // Certify merge-path's ordering precondition independently. A merge of
    // unsorted rows cannot itself be relied on to visit every invalid entry.
    for(auto k=std::int64_t(threadIdx.x);k<na;k+=blockDim.x)
        if(!isfinite(values[a+k]) || ids[a+k]<0 || ids[a+k]>=columns ||
           (k>0 && ids[a+k-1]>=ids[a+k])) bad=1;
    for(auto k=std::int64_t(threadIdx.x);k<nb;k+=blockDim.x)
        if(!isfinite(values[b+k]) || ids[b+k]<0 || ids[b+k]>=columns ||
           (k>0 && ids[b+k-1]>=ids[b+k])) bad=1;
    __shared__ std::int64_t sizes[threads];__shared__ int errors[threads];
    errors[threadIdx.x]=bad;__syncthreads();
    for(int offset=threads/2;offset;offset/=2) {
        if(threadIdx.x<offset) errors[threadIdx.x]|=errors[threadIdx.x+offset];
        __syncthreads();
    }
    if(errors[0]) {
        if(threadIdx.x==0) decisions[k]={0,rhs,alpha,0};
        return;
    }
    __syncthreads(); // All warps consumed errors[0] before it is reused below.
    auto size=merge_chunk(ids+a,values+a,na,ids+b,values+b,nb,alpha,bad,nullptr,nullptr,0);
    sizes[threadIdx.x]=size;errors[threadIdx.x]=bad;__syncthreads();
    for(int offset=threads/2;offset;offset/=2) {
        if(threadIdx.x<offset) {sizes[threadIdx.x]+=sizes[threadIdx.x+offset];errors[threadIdx.x]|=errors[threadIdx.x+offset];}
        __syncthreads();
    }
    if(threadIdx.x==0) decisions[k]={sizes[0],rhs,alpha,
        !errors[0] && sizes[0]>0 && sizes[0]<=na/2};
}
__global__ void copy_unchanged(int rows,const int* parents,const std::int64_t* old_ptr,
                              const int* old_ids,const double* old_values,
                              const std::int64_t* new_ptr,int* ids,double* values) {
    std::int64_t row=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=rows || parents[row]>=0) return;
    auto destination=new_ptr[row];
    for(auto k=old_ptr[row];k<old_ptr[row+1];++k,++destination) {
        ids[destination]=old_ids[k];values[destination]=old_values[k];
    }
}
__global__ void fill_changed(int count,const Candidate* pairs,const Decision* decisions,
                            const int* parents,const std::int64_t* old_ptr,
                            const int* old_ids,const double* old_values,
                            const std::int64_t* new_ptr,int* ids,double* values) {
    int k=blockIdx.x;if(k>=count) return;
    auto pair=pairs[k];if(parents[pair.row]!=pair.parent) return;
    auto a=old_ptr[pair.row],b=old_ptr[pair.parent];
    auto na=old_ptr[pair.row+1]-a,nb=old_ptr[pair.parent+1]-b;
    int bad=0;auto decision=decisions[k];
    auto size=merge_chunk(old_ids+a,old_values+a,na,old_ids+b,old_values+b,nb,
                          decision.alpha,bad,nullptr,nullptr,0);
    __shared__ std::int64_t offsets[threads];offsets[threadIdx.x]=size;__syncthreads();
    if(threadIdx.x==0) {
        std::int64_t running=0;
        for(int t=0;t<threads;++t) {auto value=offsets[t];offsets[t]=running;running+=value;}
    }
    __syncthreads();
    merge_chunk(old_ids+a,old_values+a,na,old_ids+b,old_values+b,nb,
                decision.alpha,bad,ids,values,new_ptr[pair.row]+offsets[threadIdx.x]);
}
enum Gate {Go=0,InvalidOffsets,NoRows,ScreenBudget,InvalidColumn,NoPairs,PairBudget,NoCancellation,LowGain};
struct Counters {int invalid=0,candidates=0,changed=0,max_depth=0,gate=0,valid=0;unsigned long long work=0,saved=0;std::int64_t total=0;};
__global__ void screen_gate(const int* count,int max_candidates,Counters* s) {
    s->gate=s->invalid?InvalidOffsets:*count==0?NoRows:*count>max_candidates?ScreenBudget:Go;
}
__global__ void pair_gate(std::int64_t nnz,std::int64_t limit,Counters* s) {
    auto budget=limit?limit:(nnz>INT64_MAX/4?INT64_MAX:nnz*4);
    s->gate=s->invalid?InvalidColumn:s->candidates==0?NoPairs:s->work>static_cast<unsigned long long>(budget)?PairBudget:Go;
}
__global__ void gain_gate(std::int64_t nnz,double minimum,Counters* s) {
    s->total=nnz-std::int64_t(s->saved);
    s->gate=s->changed==0?NoCancellation:static_cast<double>(s->saved)/nnz<minimum?LowGain:Go;
}
const char* gate_reason(int code) {
    switch(code) {case InvalidOffsets:return "invalid_offsets";case NoRows:return "no_eligible_rows";
    case ScreenBudget:return "screen_budget";case InvalidColumn:return "invalid_column";case NoPairs:return "no_candidates";
    case PairBudget:return "pair_budget";case NoCancellation:return "no_cancellations";case LowGain:return "low_gain";default:return "ready";}
}
__global__ void filter_rows(int m,std::int64_t nnz,const NnzOffset* ptr,const double* al,const double* au,int* flags,Counters* stats) {
    for(auto row=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;row<m;row+=std::int64_t(blockDim.x)*gridDim.x) {
        std::int64_t a=ptr[row],b=ptr[row+1];
        bool valid=a>=0 && b>=a && b<=nnz;
        if(!valid || (row==0 && a!=0) || (row==m-1 && b!=nnz)) atomicExch(&stats->invalid,1);
        flags[row]=valid && b-a>=16 && isfinite(al[row]) && al[row]==au[row];
    }
}
__global__ void make_keys(int count,int n,const int* rows,const std::int64_t* ptr,
                          const int* ids,unsigned long long* keys,Counters* stats) {
    for(auto k=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;k<count;k+=std::int64_t(blockDim.x)*gridDim.x) {
        int row=rows[k],col=ids[ptr[row]];
        if(col<0 || col>=n) atomicExch(&stats->invalid,1);
        keys[k]=(static_cast<unsigned long long>(static_cast<unsigned>(col))<<32)|static_cast<unsigned>(row);
    }
}
__global__ void make_pairs(int count,const unsigned long long* keys,const std::int64_t* ptr,
                           const double* values,const double* rhs,Candidate* pairs,Counters* stats) {
    for(auto k=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;k<count;k+=std::int64_t(blockDim.x)*gridDim.x) {
        int row=int(unsigned(keys[k])),parent=row;
        if(k>0 && (keys[k]>>32)==(keys[k-1]>>32)) {
            int previous=int(unsigned(keys[k-1]));
            double x=values[ptr[row]],y=values[ptr[previous]];
            int alpha=x==y?1:x==-y?-1:0;
            // Cheap exact rejection before the full cooperative merge.
            if(alpha && (rhs[row]==alpha*rhs[previous] || rhs[row]==0 || rhs[previous]==0)) {
                parent=previous;atomicAdd(&stats->candidates,1);
                atomicAdd(&stats->work,static_cast<unsigned long long>(ptr[row+1]-ptr[row]+ptr[parent+1]-ptr[parent]));
            }
        }
        pairs[k]={row,parent,rhs[row],rhs[parent]};
    }
}
__global__ void any_valid_decision(int count,const Decision* decisions,Counters* stats) {
    using Reduce=cub::BlockReduce<int,nt>;
    __shared__ typename Reduce::TempStorage scratch;
    int valid=0;
    for(auto k=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;k<count;k+=std::int64_t(blockDim.x)*gridDim.x)
        valid|=decisions[k].valid;
    int total=Reduce(scratch).Sum(valid);
    if(threadIdx.x==0 && total) atomicExch(&stats->valid,1);
}
__global__ void accepted_gate(Counters* stats) {stats->gate=stats->valid?Go:NoCancellation;}
__global__ void init_lengths(int m,const std::int64_t* ptr,std::int64_t* lengths) {
    for(auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<=m;i+=std::int64_t(blockDim.x)*gridDim.x)
        lengths[i]=i<m?ptr[i+1]-ptr[i]:0;
}
// One thread per independent first-column group, walking at most the bounded
// eligible count. Ordered choices match the CPU reference without host scans.
__global__ void select_chains(int count,const unsigned long long* keys,const Candidate* pairs,
                              const Decision* decisions,const std::int64_t* ptr,int cap,
                              int* parents,int* alpha,std::int64_t* lengths,Counters* stats) {
    for(auto begin=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;begin<count;begin+=std::int64_t(blockDim.x)*gridDim.x) {
        if(begin>0 && (keys[begin]>>32)==(keys[begin-1]>>32)) continue;
        int depth=0;
        for(auto k=begin;k<count && (keys[k]>>32)==(keys[begin]>>32);++k) {
            auto pair=pairs[k];auto d=decisions[k];
            if(pair.parent<pair.row && d.valid && depth<cap) {
                parents[pair.row]=pair.parent;alpha[pair.row]=d.alpha;lengths[pair.row]=d.nnz;++depth;
                atomicAdd(&stats->changed,1);atomicMax(&stats->max_depth,depth);
                atomicAdd(&stats->saved,static_cast<unsigned long long>(ptr[pair.row+1]-ptr[pair.row]-d.nnz));
            } else depth=0;
        }
    }
}
__global__ void replace_rhs(int count,const Candidate* pairs,const Decision* decisions,
                            const int* parents,double* al,double* au) {
    for(auto k=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;k<count;k+=std::int64_t(blockDim.x)*gridDim.x) {
        auto p=pairs[k];if(parents[p.row]==p.parent) al[p.row]=au[p.row]=decisions[k].rhs;
    }
}
std::shared_ptr<StartupRowSparsification> build_rows(
    const LPInfoGpu& input, const PresolveParams& options, RowSparsificationReport& report) {
  const auto& a = input.A;
  const int m = a.rows, n = a.cols;
  if (m <= 0 || m == INT32_MAX || n <= 0 || !a.rowPtr || !a.colVal || !a.nzVal ||
      !input.AL || !input.AU || !input.l || !input.u || !input.c) {
    report.reason = "unsupported_layout"; return {};
  }
  if (options.row_sparsification_max_candidates < 1 || options.row_sparsification_max_chain < 1 ||
      options.row_sparsification_max_pair_entries < 0 ||
      !std::isfinite(options.row_sparsification_min_gain) || options.row_sparsification_min_gain < 0 ||
      options.row_sparsification_min_gain >= 1) {
    report.reason = "invalid_options"; return {};
  }
  Array<int> flags(m), count(1);
  Array<Counters> counters(1);
  checked(cudaMemsetAsync(counters.p, 0, sizeof(Counters), 0));
  filter_rows<<<grid(m),nt>>>(m, a.nnz, a.rowPtr, input.AL, input.AU, flags.p, counters.p);
  checked(cudaGetLastError());
  std::size_t bytes = 0;
  checked(cub::DeviceReduce::Sum(nullptr, bytes, flags.p, count.p, m));
  Array<unsigned char> scratch(bytes);
  checked(cub::DeviceReduce::Sum(scratch.p, bytes, flags.p, count.p, m));
  screen_gate<<<1,1>>>(count.p, options.row_sparsification_max_candidates, counters.p);
  checked(cudaGetLastError());
  int eligible = 0; count.get(&eligible, 1);
  Counters host; counters.get(&host, 1); report.eligible_rows = eligible;
  if (host.gate != Go) { report.reason = gate_reason(host.gate); return {}; }
  // Widen only after the cheap count gate. All arithmetic and scan offsets are
  // int64; the persistent output uses the dependency's selected ABI.
  Array<std::int64_t> wide(std::size_t(m) + 1);
  convert<<<grid(std::int64_t(m)+1),nt>>>(a.rowPtr, wide.p, std::int64_t(m)+1);
  checked(cudaGetLastError());
  const auto* ptr = wide.p;
  Array<int> rows(eligible);
  thrust::counting_iterator<int> ids(0);
  checked(cub::DeviceSelect::Flagged(nullptr, bytes, ids, flags.p, rows.p, count.p, m));
  scratch.allocate(bytes);
  checked(cub::DeviceSelect::Flagged(scratch.p, bytes, ids, flags.p, rows.p, count.p, m));
  flags.allocate(0);
  Array<unsigned long long> keys_in(eligible), keys(eligible);
  make_keys<<<grid(eligible),nt>>>(eligible,n,rows.p,ptr,a.colVal,keys_in.p,counters.p);
  checked(cudaGetLastError());
  checked(cub::DeviceRadixSort::SortKeys(nullptr,bytes,keys_in.p,keys.p,eligible));
  scratch.allocate(bytes);
  checked(cub::DeviceRadixSort::SortKeys(scratch.p,bytes,keys_in.p,keys.p,eligible));
  Array<Candidate> pairs(eligible); Array<Decision> decisions(eligible);
  make_pairs<<<grid(eligible),nt>>>(eligible,keys.p,ptr,a.nzVal,input.AL,pairs.p,counters.p);
  checked(cudaGetLastError());
  pair_gate<<<1,1>>>(a.nnz,options.row_sparsification_max_pair_entries,counters.p);
  checked(cudaGetLastError()); counters.get(&host,1); report.candidates=host.candidates;
  if(host.gate!=Go) { report.reason=gate_reason(host.gate); return {}; }
  validate_kernel<<<eligible,threads>>>(eligible,n,pairs.p,ptr,a.colVal,a.nzVal,decisions.p);
  checked(cudaGetLastError());
  any_valid_decision<<<grid(eligible),nt>>>(eligible,decisions.p,counters.p);
  checked(cudaGetLastError());
  accepted_gate<<<1,1>>>(counters.p);checked(cudaGetLastError());counters.get(&host,1);
  if(host.gate!=Go) { report.reason=gate_reason(host.gate); return {}; }
  Array<int> parents(m),alpha(m);
  Array<std::int64_t> lengths(std::size_t(m)+1),newptr(std::size_t(m)+1);
  checked(cudaMemsetAsync(parents.p,255,std::size_t(m)*sizeof(int),0));
  checked(cudaMemsetAsync(alpha.p,0,std::size_t(m)*sizeof(int),0));
  init_lengths<<<grid(std::int64_t(m)+1),nt>>>(m,ptr,lengths.p);
  checked(cudaGetLastError());
  select_chains<<<grid(eligible),nt>>>(eligible,keys.p,pairs.p,decisions.p,ptr,
      options.row_sparsification_max_chain,parents.p,alpha.p,lengths.p,counters.p);
  checked(cudaGetLastError());
  gain_gate<<<1,1>>>(a.nnz,options.row_sparsification_min_gain,counters.p);
  checked(cudaGetLastError());counters.get(&host,1);
  if(host.gate!=Go) { report.reason=gate_reason(host.gate); return {}; }
  checked(cub::DeviceScan::ExclusiveSum(nullptr,bytes,lengths.p,newptr.p,m+1));
  scratch.allocate(bytes);
  checked(cub::DeviceScan::ExclusiveSum(scratch.p,bytes,lengths.p,newptr.p,m+1));
  auto state=std::make_shared<StartupRowSparsification>();
  auto& out=state->transformed;
  out.obj_constant=input.obj_constant;
  out.A.rows=m;out.A.cols=n;out.A.nnz=static_cast<NnzOffset>(host.total);
  allocate(&out.A.rowPtr,std::size_t(m)+1);
  allocate(&out.A.colVal,host.total);allocate(&out.A.nzVal,host.total);
  convert<<<grid(std::int64_t(m)+1),nt>>>(newptr.p,out.A.rowPtr,std::int64_t(m)+1);
  checked(cudaGetLastError());
  copy_unchanged<<<(std::int64_t(m)+127)/128,128>>>(m,parents.p,ptr,a.colVal,a.nzVal,newptr.p,out.A.colVal,out.A.nzVal);
  checked(cudaGetLastError());
  fill_changed<<<eligible,threads>>>(eligible,pairs.p,decisions.p,parents.p,ptr,a.colVal,a.nzVal,newptr.p,out.A.colVal,out.A.nzVal);
  checked(cudaGetLastError());
  duplicate(&out.AL,input.AL,m);duplicate(&out.AU,input.AU,m);
  duplicate(&out.l,input.l,n);duplicate(&out.u,input.u,n);duplicate(&out.c,input.c,n);
  replace_rhs<<<grid(eligible),nt>>>(eligible,pairs.p,decisions.p,parents.p,out.AL,out.AU);
  checked(cudaGetLastError());
  // The embedding owner can release the large original upload before the
  // replacement transpose is allocated. The remaining Startup rules still
  // receive a complete A/AT pair at their explicit handoff.
  if (!options.defer_startup_row_transpose) out.AT=transpose_csr64(out.A);
  checked(cudaStreamSynchronize(0));
  state->parents=parents.release();state->alpha=alpha.release();
  report.reason="applied";report.changed_rows=host.changed;report.max_depth=host.max_depth;
  report.new_nnz=host.total;
  return state;
}
}  // namespace

StartupRowSparsification::~StartupRowSparsification() {
  // Every producer/consumer uses stream 0, including transpose's async buffers.
  auto release=[](void* p) { if(p) cudaFreeAsync(p,0); };
  release(transformed.A.rowPtr);release(transformed.A.colVal);release(transformed.A.nzVal);
  release(transformed.AT.rowPtr);release(transformed.AT.colVal);release(transformed.AT.nzVal);
  release(transformed.AL);release(transformed.AU);release(transformed.l);release(transformed.u);release(transformed.c);
  release(parents);release(alpha);
}
std::shared_ptr<StartupRowSparsification> run_startup_row_sparsification(
    const LPInfoGpu& input,const PresolveParams& params,RowSparsificationReport& report) {
  report={};report.old_nnz=report.new_nnz=input.A.nnz;
  if(!params.enable_startup || !params.enable_structure_specific_rules ||
      !params.enable_row_sparsification || params.max_iters<=0) return {};
  const auto start=Clock::now();
  std::shared_ptr<StartupRowSparsification> state;
  try { state=build_rows(input,params,report); }
  catch(const std::bad_alloc&) { report.reason="allocation_fallback"; }
  catch(const TransposeOutOfMemory&) { report.reason="allocation_fallback"; }
  if(report.reason==std::string("allocation_fallback")) {
    // Only an allocation failure may fall back. Fatal execution errors propagate.
    if(cudaPeekAtLastError()==cudaErrorMemoryAllocation) cudaGetLastError();
    checked(cudaStreamSynchronize(0));
  }
  report.seconds=std::chrono::duration<double>(Clock::now()-start).count();
  return state;
}
void print_startup_row_sparsification(const RowSparsificationReport& r) {
  const auto flags=std::cout.flags();const auto precision=std::cout.precision();
  std::cout<<std::defaultfloat<<std::setprecision(12)<<"ROW_SPARSIFY_STARTUP reason="<<r.reason
      <<" eligible_rows="<<r.eligible_rows<<" candidates="<<r.candidates
      <<" changed_rows="<<r.changed_rows<<" old_nnz="<<r.old_nnz<<" new_nnz="<<r.new_nnz
      <<" max_chain="<<r.max_depth<<" input_reused="<<(r.changed_rows==0)
      <<" host_matrix_roundtrips=0 host_row_decisions=0 seconds="<<r.seconds<<'\n';
  std::cout.flags(flags);std::cout.precision(precision);
}
}  // namespace gpu_preprocessor::lp::presolve
