// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

// Implementation fragment owned by gpu_postsolve.cu; not a separate build source.
#include <cublas_v2.h>
#include <cub/device/device_select.cuh>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace gpu_preprocessor::lp::presolve {
namespace pair_polish {
constexpr int threads = 256;
constexpr int width = 6;
int blocks(int n) { return int((std::int64_t(n) + threads - 1) / threads); }

struct Unavailable : std::runtime_error { Unavailable():std::runtime_error("pair polish allocation unavailable") {} };
struct Scratch {
  std::vector<void*> allocations;
  cublasHandle_t blas = nullptr;
  ~Scratch() {
    if (blas) cublasDestroy(blas);
    for (void* p : allocations) cudaFreeAsync(p, nullptr);
  }
  template<class T> T* alloc(std::size_t n) {
    T* p = nullptr;
    cudaError_t status=cudaMallocAsync(&p, sizeof(T) * n, nullptr);
    if (status==cudaErrorMemoryAllocation) { (void)cudaGetLastError(); throw Unavailable(); }
    throw_if_cuda_error(status, "pair polish allocation");
    try { allocations.push_back(p); } catch (...) { cudaFreeAsync(p,nullptr); throw; }
    return p;
  }
};
void check_blas(cublasStatus_t s) {
  if (s == CUBLAS_STATUS_ALLOC_FAILED) throw Unavailable();
  if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("pair polish cuBLAS failure");
}
void zero(void* p, std::size_t bytes) {
  throw_if_cuda_error(cudaMemsetAsync(p, 0, bytes), "pair polish clear");
}
void check_launch() { throw_if_cuda_error(cudaGetLastError(), "pair polish kernel"); }

__device__ void max_double(double* p, double value) {
  auto* addr = reinterpret_cast<unsigned long long*>(p);
  auto old = atomicCAS(addr, 0ULL, 0ULL), assumed = old;
  do {
    assumed = old;
    old = atomicCAS(addr, assumed, __double_as_longlong(fmax(value, __longlong_as_double(assumed))));
  } while (old != assumed);
}
__device__ void min_double(double* p, double value) { // also handles negative bounds
  auto* addr = reinterpret_cast<unsigned long long*>(p);
  auto old = atomicCAS(addr, 0ULL, 0ULL), assumed = old;
  do {
    assumed = old;
    old = atomicCAS(addr, assumed, __double_as_longlong(fmin(value, __longlong_as_double(assumed))));
  } while (old != assumed);
}
__global__ void init(int n, double* lo, double* hi) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { lo[i] = -INFINITY; hi[i] = INFINITY; }
}
// Certify that the one recovery group covers exactly the original antipodal
// columns. This excludes epigraph/tail layouts and partially remapped groups.
__global__ void certify_pairs(LPInfoGpu lp, int k, const int* plus, const int* minus,
                             const int* roots, const int* root_minus,
                             const double* shift_plus, const double* shift_minus,
                             const double* x, double* d, int* bad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= k) return;
  int root = roots[i];
  if (plus[i] != i || minus[i] != i+k || root < 0 || root >= k ||
      root_minus[i] != root+k) { atomicExch(bad, 1); return; }
  double v = x[i] - x[i+k], vr = x[root] - x[root+k];
  if (roots[root] != root || !isfinite(x[i]) || !isfinite(x[i+k]) ||
      !isfinite(v) || !isfinite(vr) || fabs(v-vr) > 64*DBL_EPSILON*(1+fabs(vr)) ||
      !isfinite(lp.l[i]) || !isfinite(lp.l[i+k]) ||
      isnan(lp.u[i]) || isnan(lp.u[i+k]) ||
      lp.u[i] < lp.l[i] || lp.u[i+k] < lp.l[i+k] ||
      shift_plus[i] != lp.l[i] || shift_minus[i] != lp.l[i+k] ||
      !(lp.c[i] > 0) || !isfinite(lp.c[i]) || lp.c[i] != lp.c[i+k]) {
    atomicExch(bad, 1); return;
  }
  d[i] = v;
}
__global__ void collect_bounds(LPInfoGpu lp, int k, const int* roots, double* lo, double* hi) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) {
    max_double(lo+roots[i], lp.l[i]-lp.u[i+k]);
    min_double(hi+roots[i], lp.u[i]-lp.l[i+k]);
  }
}
// Same short-row family as shifted_antipodal, but re-certified in the original
// model after every recovery operation. Duplicate root coefficients are summed.
__device__ int quotient_row(LPInfoGpu lp, int row, int k, const int* roots,
                           int* ids, double* vals) {
  NnzOffset begin = lp.A.rowPtr[row], end = lp.A.rowPtr[row+1];
  if (lp.AL[row] != 0 || lp.AU[row] != 0 || end < begin ||
      end-begin > 2*width || (end-begin)%2) return -1;
  int half = int((end-begin)/2), count = 0, previous = -1;
  for (int t=0; t<half; ++t) {
    int col = lp.A.colVal[begin+t];
    double a = lp.A.nzVal[begin+t];
    if (col <= previous || col >= k || lp.A.colVal[begin+half+t] != col+k ||
        !isfinite(a) || lp.A.nzVal[begin+half+t] != -a) return -1;
    previous = col;
    int root = roots[col], j=0;
    while (j<count && ids[j]!=root) ++j;
    if (j==count) { ids[count]=root; vals[count++]=0; }
    vals[j] += a;
  }
  int kept=0;
  for (int t=0; t<count; ++t) if (vals[t]!=0) {
    if (!isfinite(vals[t])) return -1;
    ids[kept]=ids[t]; vals[kept++]=vals[t];
  }
  return kept;
}
__global__ void select_rows(LPInfoGpu lp, int k, const int* roots, int* row_ids, int* bad) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= lp.A.rows) return;
  int ids[width]; double vals[width];
  int count = quotient_row(lp,row,k,roots,ids,vals);
  if (count < 0) atomicExch(bad,1);
  row_ids[row] = count>0 ? row : -1;
}
struct Nonnegative { __host__ __device__ bool operator()(int v) const { return v>=0; } };
__global__ void build_rows(LPInfoGpu lp, int k, const int* roots, const int* selected,
                          int rows, int* indices, double* values, double* diagonal) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row>=rows) return;
  int ids[width]; double vals[width];
  int count = quotient_row(lp,selected[row],k,roots,ids,vals);
  for (int t=0; t<width; ++t) {
    std::size_t at=std::size_t(row)*width+t;
    indices[at]=t<count ? ids[t] : 0;
    values[at]=t<count ? vals[t] : 0;
    if (t<count) atomicAdd(diagonal+ids[t],vals[t]*vals[t]);
  }
}
__global__ void multiply(int rows, const int* ids, const double* a,
                         const double* v, double* out, double sign) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row>=rows) return;
  double sum=0;
  for (int t=0; t<width; ++t) {
    auto at=std::size_t(row)*width+t;
    sum=fma(a[at],v[ids[at]],sum);
  }
  out[row]=sign*sum;
}
__global__ void transpose(int rows, const int* ids, const double* a,
                         const double* v, double* out) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row>=rows) return;
  for (int t=0; t<width; ++t) {
    auto at=std::size_t(row)*width+t;
    if (a[at]!=0) atomicAdd(out+ids[at],a[at]*v[row]);
  }
}
__global__ void precondition(int n, const int* roots, const double* d,
                            const double* lo, const double* hi, const double* diag,
                            const double* g, double* z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<n) {
    // Freeze the current active face; an infeasible candidate is rejected, never
    // clipped; tiny endpoint roundoff is handled only during recovery and
    // followed by a fresh original-matrix residual check.
    bool free = roots[i]==i && d[i]>lo[i]+1e-9 && d[i]<hi[i]-1e-9 && diag[i]>0;
    z[i] = free ? g[i]/diag[i] : 0;
  }
}
// Unfolding and affine pair recovery can put an active endpoint a few ULPs
// outside its box. Snap only roundoff-sized violations, then certify the actual
// original CSR again. Larger violations still reject the whole candidate.
__device__ double snap_bound_roundoff(double value, double lower, double upper, int* count) {
  if (!isfinite(value)) return value;
  double scale=fmax(1.0,fabs(value));
  if (value<lower && isfinite(lower) &&
      lower-value<=64*DBL_EPSILON*fmax(scale,fabs(lower))) {
    atomicAdd(count,1); return lower;
  }
  if (value>upper && isfinite(upper) &&
      value-upper<=64*DBL_EPSILON*fmax(scale,fabs(upper))) {
    atomicAdd(count,1); return upper;
  }
  return value;
}
__global__ void recover(LPInfoGpu lp, int k, const int* roots, const double* d,
                        const double* delta, const double* x, double* candidate,
                        double* cost_old, double* cost_change, int* bad, int* bound_snaps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i>=k) return;
  int root=roots[i];
  double v=d[root]+delta[root], shift=lp.l[i]-lp.l[i+k];
  double p=lp.l[i]+fmax(0.0,v-shift), m=lp.l[i+k]+fmax(0.0,shift-v);
  p=snap_bound_roundoff(p,lp.l[i],lp.u[i],bound_snaps);
  m=snap_bound_roundoff(m,lp.l[i+k],lp.u[i+k],bound_snaps);
  candidate[i]=p; candidate[i+k]=m;
  cost_old[i]=lp.c[i]*x[i]+lp.c[i+k]*x[i+k];
  cost_change[i]=lp.c[i]*(p-x[i])+lp.c[i+k]*(m-x[i+k]);
  if (!isfinite(p) || !isfinite(m) || p<lp.l[i] || p>lp.u[i] ||
      m<lp.l[i+k] || m>lp.u[i+k]) atomicExch(bad,1);
}
__global__ void original_residual(LPInfoGpu lp, const double* x, double* residual) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;
  if (row>=lp.A.rows) return;
  double sum=0;
  for (NnzOffset at=lp.A.rowPtr[row]; at<lp.A.rowPtr[row+1]; ++at)
    sum=fma(lp.A.nzVal[at],x[lp.A.colVal[at]],sum);
  residual[row]=sum; // all rows already certified homogeneous equalities
}
__global__ void ones(int n,double* v) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) v[i]=1;
}

bool run(double* x, const PresolveRecordGpu& record, const LPInfoGpu& lp, double tol) {
  if (!(tol>0) || !std::isfinite(tol) || !x || !record.device || record.startup_rows || record.startup_poisson ||
      record.device->piecewise_linear_reformulation_variable_pair.size()!=1 || lp.A.cols<=0 || lp.A.rows<=0 ||
      lp.A.cols%2 || record.n0!=lp.A.cols || record.m0!=lp.A.rows) return false;
  const auto& group=record.device->piecewise_linear_reformulation_variable_pair[0];
  int k=lp.A.cols/2, m=lp.A.rows;
  if (group.ids.size()!=4 || group.values.size()!=2) return false;
  for (const auto& a:group.ids) if(a.count!=k || !a.data) return false;
  for (const auto& a:group.values) if(a.count!=k || !a.data) return false;
  const int* roots=group.ids[2].data;
  auto start=std::chrono::steady_clock::now();
  Scratch s;
  int* bad=s.alloc<int>(2); zero(bad,2*sizeof(int));
  double* d=s.alloc<double>(k);
  certify_pairs<<<blocks(k),threads>>>(lp,k,group.ids[0].data,group.ids[1].data,roots,
      group.ids[3].data,group.values[0].data,group.values[1].data,x,d,bad);
  check_launch();
  auto invalid=[&]() {int v=0; throw_if_cuda_error(cudaMemcpy(&v,bad,sizeof(v),cudaMemcpyDeviceToHost),"pair polish certificate"); return v!=0;};
  if (invalid()) return false;
  int* all_rows=s.alloc<int>(m);
  select_rows<<<blocks(m),threads>>>(lp,k,roots,all_rows,bad); check_launch();
  if (invalid()) return false;
  check_blas(cublasCreate(&s.blas));
  auto dot=[&](int n,const double* a,const double* b) {double v=0;check_blas(cublasDdot(s.blas,n,a,1,b,1,&v));return v;};
  auto axpy=[&](int n,double alpha,const double* a,double* b) {check_blas(cublasDaxpy(s.blas,n,&alpha,a,1,b,1));};
  double* raw=s.alloc<double>(m);
  original_residual<<<blocks(m),threads>>>(lp,x,raw); check_launch();
  double before=std::sqrt(dot(m,raw,raw));
  if (!std::isfinite(before) || before<=tol) return false;
  int* selected=s.alloc<int>(m); int* selected_count=s.alloc<int>(1);
  std::size_t bytes=0;
  throw_if_cuda_error(cub::DeviceSelect::If(nullptr,bytes,all_rows,selected,selected_count,m,Nonnegative()),"pair polish row selection size");
  void* temp=s.alloc<char>(bytes);
  throw_if_cuda_error(cub::DeviceSelect::If(temp,bytes,all_rows,selected,selected_count,m,Nonnegative()),"pair polish row selection");
  int rows=0; throw_if_cuda_error(cudaMemcpy(&rows,selected_count,sizeof(rows),cudaMemcpyDeviceToHost),"pair polish row count");
  if (!rows) return false;
  int* ids=s.alloc<int>(std::size_t(rows)*width);
  double* values=s.alloc<double>(std::size_t(rows)*width);
  double* diag=s.alloc<double>(k); zero(diag,sizeof(double)*k);
  double* lo=s.alloc<double>(k); double* hi=s.alloc<double>(k);
  init<<<blocks(k),threads>>>(k,lo,hi);
  collect_bounds<<<blocks(k),threads>>>(lp,k,roots,lo,hi);
  build_rows<<<blocks(rows),threads>>>(lp,k,roots,selected,rows,ids,values,diag);
  double* r=s.alloc<double>(rows); double* q=s.alloc<double>(rows);
  double* g=s.alloc<double>(k); double* z=s.alloc<double>(k);
  double* p=s.alloc<double>(k); double* delta=s.alloc<double>(k); zero(delta,sizeof(double)*k);
  multiply<<<blocks(rows),threads>>>(rows,ids,values,d,r,-1);
  auto gradient=[&]() {
    zero(g,sizeof(double)*k);
    transpose<<<blocks(rows),threads>>>(rows,ids,values,r,g);
    precondition<<<blocks(k),threads>>>(k,roots,d,lo,hi,diag,g,z); check_launch();
    return dot(k,g,z);
  };
  double gamma=gradient();
  throw_if_cuda_error(cudaMemcpyAsync(p,z,sizeof(double)*k,cudaMemcpyDeviceToDevice),"pair polish direction");
  int iterations=0;
  double residual=before;
  // PCGLS on the fixed active face; the objective solve, presolve state,
  // and requested stopping tolerance remain unchanged.
  for (; iterations<1000 && gamma>0 && std::isfinite(gamma);) {
    multiply<<<blocks(rows),threads>>>(rows,ids,values,p,q,1); check_launch();
    double denom=dot(rows,q,q);
    if (!(denom>0) || !std::isfinite(denom)) break;
    double alpha=gamma/denom;
    axpy(k,alpha,p,delta); axpy(rows,-alpha,q,r); ++iterations;
    residual=std::sqrt(dot(rows,r,r));
    if (!std::isfinite(residual) || residual<=0.1*tol) break;
    if (iterations%32==0 && std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>5.0) break;
    double next=gradient(), beta=next/gamma;
    if (!std::isfinite(beta)) break;
    check_blas(cublasDscal(s.blas,k,&beta,p,1)); axpy(k,1,z,p); gamma=next;
  }
  double* candidate=s.alloc<double>(lp.A.cols);
  double* costs=s.alloc<double>(k); double* changes=s.alloc<double>(k);
  recover<<<blocks(k),threads>>>(lp,k,roots,d,delta,x,candidate,costs,changes,bad,bad+1); check_launch();
  bool bounds_valid=!invalid();
  bool accepted=bounds_valid;
  int bound_snaps=0;
  throw_if_cuda_error(cudaMemcpy(&bound_snaps,bad+1,sizeof(int),cudaMemcpyDeviceToHost),"pair polish bound snaps");
  original_residual<<<blocks(m),threads>>>(lp,candidate,raw); check_launch();
  double after=std::sqrt(dot(m,raw,raw));
  ones<<<blocks(k),threads>>>(k,z); check_launch();
  double objective=dot(k,costs,z)+lp.obj_constant, change=dot(k,changes,z);
  double norm_delta=std::sqrt(dot(k,delta,delta)), norm_d=std::sqrt(dot(k,d,d));
  accepted=accepted && std::isfinite(after) && after<=0.5*tol &&
      std::isfinite(objective) && std::isfinite(change) &&
      std::abs(change)<=tol*(1+std::abs(objective)) &&
      std::isfinite(norm_delta) && norm_delta<=1e-3*(1+norm_d);
  if (accepted) throw_if_cuda_error(cudaMemcpyAsync(x,candidate,sizeof(double)*lp.A.cols,cudaMemcpyDeviceToDevice),"pair polish accept");
  throw_if_cuda_error(cudaStreamSynchronize(nullptr),"pair polish complete");
  // Use the caller's informational stream so the host solver can filter it.
  std::ostringstream log;
  log << "PAIR_PRIMAL_POSTSOLVE accepted=" << int(accepted) << " iterations=" << iterations
      << std::scientific << std::setprecision(15)
      << " original_row_before=" << before << " original_row_after=" << after
      << " objective_change=" << change << " bounds_valid=" << int(bounds_valid)
      << " bound_snaps=" << bound_snaps << std::fixed << std::setprecision(6)
      << " time=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() << '\n';
  std::cout << log.str();
  return accepted;
}
} // namespace pair_polish

bool polish_variable_pair_primal_gpu(double* x, const PresolveRecordGpu& record,
                                    const LPInfoGpu& original, double tolerance) {
  try { return pair_polish::run(x,record,original,tolerance); }
  catch (const pair_polish::Unavailable&) {
    (void)cudaGetLastError();
    std::fprintf(stderr,"[warn] pair primal postsolve skipped: GPU allocation failed\n");
    return false;
  }
}
} // namespace gpu_preprocessor::lp::presolve
