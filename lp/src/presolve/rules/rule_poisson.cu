// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"
#include <cufft.h>
#include <cstddef>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <cmath>
#include <algorithm>
#include "gpu_preprocessor/lp/presolve/transpose64.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace gpu_preprocessor::lp::presolve {
namespace {
namespace poisson {
namespace kernels {
namespace {
constexpr int threads=256;
__device__ bool boundary(int j,int p) {
    int x=j%p,y=j/p%p,z=j/(p*p);
    return x==0 || y==0 || z==0 || x==p-1 || y==p-1 || z==p-1;
}
template<class Offset>
__global__ void scale_kernel(const Offset* ptr,const int* col,const double* a,
                             int p,double* scale) {
    *scale=0;
    if(ptr[1]-ptr[0]!=8) return;
    for(Offset k=ptr[0];k<ptr[1];++k)
        if(col[k]==p*p+p+1) *scale=-a[k]/6;
}
template<class Offset>
__global__ void rows_kernel(int m,int q,const Offset* ptr,const int* col,
                            const double* a,const double* al,const double* au,
                            const double* scale,int* counts) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=m) return;
    int p=q+2,nt=p*p*p,center=(i/(q*q)+1)*p*p+(i/q%q+1)*p+i%q+1;
    double s=*scale;
    if(!(s>0) || !isfinite(6*s) || al[i]!=0 || au[i]!=0 || ptr[i+1]-ptr[i]!=8) {
        atomicExch(counts,1);return;
    }
    int ids[8]={center,center-1,center+1,center-p,center+p,center-p*p,center+p*p,nt+i};
    unsigned seen=0;
    for(Offset k=ptr[i];k<ptr[i+1];++k) {
        int slot=-1;
        for(int t=0;t<8;++t) if(col[k]==ids[t]) slot=t;
        if(slot<0 || (seen & (1u<<slot)) || a[k]!=(slot==0?-6*s:(slot==7?1:s))) {
            atomicExch(counts,1);return;
        }
        seen|=1u<<slot;
    }
}
__global__ void bounds_kernel(int n,int q,const double* l,const double* u,
                              const double* c,int* counts) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=n) return;
    int p=q+2,nt=p*p*p;
    if(isnan(l[j]) || isnan(u[j]) || l[j]>u[j] || l[j]==INFINITY ||
       u[j]==-INFINITY || !isfinite(c[j])) {atomicExch(counts,1);return;}
    if(j<nt) {
        if(c[j]!=0) {atomicExch(counts,1);return;}
        if(boundary(j,p)) {if(l[j]!=0 || u[j]!=0) atomicExch(counts,1);}
        else if(l[j]==u[j]) atomicAdd(counts+1,1);
        else if(l[j]!=-INFINITY || u[j]!=INFINITY) atomicExch(counts,1);
    } else if(l[j]==u[j]) {
        if(l[j]!=0) atomicExch(counts,1);
    } else atomicAdd(counts+2,1);
}
struct Observation {
    const double *l,*u;int p;
    __device__ bool operator()(int j) const {return !boundary(j,p) && l[j]==u[j];}
};
struct Source {
    const double *l,*u;int nt;
    __device__ bool operator()(int j) const {return l[nt+j]!=u[nt+j];}
};
__global__ void spectrum_kernel(cufftDoubleComplex* a,int s,double scale,bool multiply,
                                std::size_t count) {
    std::size_t k=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(k>=count) return;
    int h=s/2+1,x=k%h,y=k/h%s,z=k/(std::size_t(h)*s);
    // Fold frequencies before sin: symmetric eigenvalues are bit-identical.
    double sx=sin(3.1415926535897932384626433832795*min(x,s-x)/s);
    double sy=sin(3.1415926535897932384626433832795*min(y,s-y)/s);
    double sz=sin(3.1415926535897932384626433832795*min(z,s-z)/s);
    double v=k?1/(4*scale*(sx*sx+sy*sy+sz*sz)*double(s)*s*s):0;
    if(multiply) {a[k].x*=v;a[k].y*=v;} else {a[k].x=v;a[k].y=0;}
}
__global__ void response_kernel(const double* g,int q,const int* obs,const int* src,
                                int m,int n,NnzOffset* ptr,int* col,double* a,
                                const double* ol,const double* ou,const double* oc,
                                double* al,double* au,double* l,double* u,double* c,int* invalid) {
    int k=blockIdx.x*blockDim.x+threadIdx.x,p=q+2,s=2*(q+1),nt=p*p*p;
    if(k<=m) ptr[k]=k*n;
    if(k<n) {int j=nt+src[k];l[k]=ol[j];u[k]=ou[j];c[k]=oc[j];}
    if(k<m) al[k]=au[k]=ol[obs[k]];
    if(k>=m*n) return;
    int i=k/n,j=k%n,o=obs[i],t=src[j];
    int ox=o%p,oy=o/p%p,oz=o/(p*p);
    int sx=t%q+1,sy=t/q%q+1,sz=t/(q*q)+1;
    double v=0;
    for(int mask=0;mask<8;++mask) {
        int ax=(mask&1)?-1:1,ay=(mask&2)?-1:1,az=(mask&4)?-1:1;
        int x=(ox-ax*sx+s)%s,y=(oy-ay*sy+s)%s,z=(oz-az*sz+s)%s;
        v+=ax*ay*az*g[(std::size_t(z)*s+y)*s+x];
    }
    if(!isfinite(v) || v<=0) atomicExch(invalid,1);
    col[k]=j;a[k]=v;
}
__global__ void rhs_kernel(double* rhs,int q,const int* loc,const double* values,
                          int count,bool temperature) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    int p=temperature?q+2:q,s=2*(q+1),v=loc[i];
    int x=v%p+(temperature?0:1),y=v/p%p+(temperature?0:1),z=v/(p*p)+(temperature?0:1);
    for(int mask=0;mask<8;++mask) {
        int ax=(mask&1)?-1:1,ay=(mask&2)?-1:1,az=(mask&4)?-1:1;
        std::size_t k=(std::size_t((az*z+s)%s)*s+(ay*y+s)%s)*s+(ax*x+s)%s;
        // Unique certified coordinates have disjoint odd images.
        rhs[k]=ax*ay*az*values[i];
    }
}
__global__ void extract_kernel(const double* a,int q,double* out,bool temperature) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=q*q*q) return;
    int x=i%q+1,y=i/q%q+1,z=i/(q*q)+1,s=2*(q+1),p=q+2;
    std::size_t target=temperature?(std::size_t(z)*p+y)*p+x:i;
    out[target]=a[(std::size_t(z)*s+y)*s+x];
}
__global__ void scatter_kernel(int nt,const int* ids,const double* in,int n,double* out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[nt+ids[i]]=in[i];
}
template<class Offset, class Index>
__global__ void z_kernel(int n,const Offset* ptr,const Index* col,const double* a,
                         const double* l,const double* u,const double* c,const double* y,double* z) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
    double v=0;
    if(l[j]==u[j]) {v=c[j];for(Offset k=ptr[j];k<ptr[j+1];++k)v-=a[k]*y[col[k]];}
    z[j]=v;
}
}
void certify(const LPInfoGpu& lp,int q,double* scale,int* counts) {
    const auto& a=lp.A;
    scale_kernel<<<1,1>>>(a.rowPtr,a.colVal,a.nzVal,q+2,scale);
    rows_kernel<<<(lp.A.rows+threads-1)/threads,threads>>>(lp.A.rows,q,a.rowPtr,a.colVal,a.nzVal,lp.AL,lp.AU,scale,counts);
    bounds_kernel<<<(std::size_t(lp.A.cols)+threads-1)/threads,threads>>>(lp.A.cols,q,lp.l,lp.u,lp.c,counts);
}

void maps(const LPInfoGpu& lp,int q,int* obs,int* src) {
    int p=q+2,nt=p*p*p;auto begin=thrust::make_counting_iterator(0);
    thrust::copy_if(thrust::cuda::par.on(0),begin,begin+nt,obs,Observation{lp.l,lp.u,p});
    thrust::copy_if(thrust::cuda::par.on(0),begin,begin+lp.A.rows,src,Source{lp.l,lp.u,nt});
}
void spectrum(cufftDoubleComplex* a,int side,double scale,bool multiply) {
    std::size_t count=std::size_t(side)*side*(side/2+1);
    spectrum_kernel<<<(count+threads-1)/threads,threads>>>(a,side,scale,multiply,count);
}
void response(const double* g,int q,const int* obs,const int* src,LPInfoGpu& r,const LPInfoGpu& o,int* invalid) {
    const auto& a=r.A;
    std::size_t count=std::max(std::size_t(r.A.rows)*r.A.cols,std::size_t(r.A.rows)+1);
    response_kernel<<<(count+threads-1)/threads,threads>>>(g,q,obs,src,r.A.rows,r.A.cols,a.rowPtr,a.colVal,a.nzVal,
        o.l,o.u,o.c,r.AL,r.AU,r.l,r.u,r.c,invalid);
}
void odd_rhs(double* a,int q,const int* loc,const double* v,int count,bool temperature) {
    rhs_kernel<<<(count+threads-1)/threads,threads>>>(a,q,loc,v,count,temperature);
}
void extract(const double* a,int q,double* out,bool temperature) {
    extract_kernel<<<(std::size_t(q)*q*q+threads-1)/threads,threads>>>(a,q,out,temperature);
}
void scatter_sources(int nt,const int* ids,const double* in,int n,double* out) {
    scatter_kernel<<<(n+threads-1)/threads,threads>>>(nt,ids,in,n,out);
}
void scatter_z(int nt,const int* ids,const double* in,int n,double* out) {scatter_sources(nt,ids,in,n,out);}
void recover_z(const LPInfoGpu& o,const double* y,double* z) {
    const auto& a=o.AT;auto blocks=(std::size_t(o.A.cols)+threads-1)/threads;
    z_kernel<<<blocks,threads>>>(o.A.cols,a.rowPtr,a.colVal,a.nzVal,o.l,o.u,o.c,y,z);
}
} // namespace kernels

namespace {
void check(cudaError_t e) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string("Poisson CUDA: ")+cudaGetErrorString(e));
}
void fft_check(cufftResult e) {
    if(e!=CUFFT_SUCCESS) throw std::runtime_error("Poisson cuFFT error "+std::to_string(int(e)));
}
template<class T> struct Buffer {
    T* p=nullptr;
    Buffer()=default;
    explicit Buffer(std::size_t n) {allocate(n);}
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    ~Buffer() {if(p)cudaFree(p);}
    void allocate(std::size_t n) {check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    T* release() {T* result=p;p=nullptr;return result;}
};
template<class T> T* allocate(std::size_t n) {Buffer<T> b(n);return b.release();}
struct Plan {
    cufftHandle handle=0;
    Plan() {fft_check(cufftCreate(&handle));}
    ~Plan() {if(handle)cufftDestroy(handle);}
    Plan(const Plan&)=delete;
};
struct Inverse {
    int q,side;
    double scale;
    Plan forward,backward;
    Buffer<double> real;
    Buffer<cufftDoubleComplex> frequency;
    Buffer<char> work;
    std::size_t real_count;
    Inverse(int q_,double scale_):q(q_),side(2*(q+1)),scale(scale_),real_count(std::size_t(side)*side*side) {
        std::size_t fw=0,bw=0,free=0,total=0;
        check(cudaMemGetInfo(&free,&total));
        const std::size_t complex_count=std::size_t(side)*side*(side/2+1);
        const std::size_t bytes=real_count*sizeof(double)+complex_count*sizeof(cufftDoubleComplex);
        // Leave headroom for the original model, recovery vectors and core.
        if(bytes>free/2) throw std::runtime_error("FFT workspace capacity gate");
        fft_check(cufftSetAutoAllocation(forward.handle,0));
        fft_check(cufftSetAutoAllocation(backward.handle,0));
        fft_check(cufftMakePlan3d(forward.handle,side,side,side,CUFFT_D2Z,&fw));
        fft_check(cufftMakePlan3d(backward.handle,side,side,side,CUFFT_Z2D,&bw));
        check(cudaMemGetInfo(&free,&total));
        if(bytes+std::max(fw,bw)>free*3/4) throw std::runtime_error("FFT plan capacity gate");
        real.allocate(real_count);frequency.allocate(complex_count);
        if(std::max(fw,bw))work.allocate(std::max(fw,bw));
        fft_check(cufftSetWorkArea(forward.handle,work.p));
        fft_check(cufftSetWorkArea(backward.handle,work.p));
    }
    void green() {
        kernels::spectrum(frequency.p,side,scale,false);
        check(cudaGetLastError());
        fft_check(cufftExecZ2D(backward.handle,frequency.p,real.p));
    }
    void recover(const int* locations,const double* values,int count,bool temperature_ids,
                 double* output,bool temperature_layout) {
        check(cudaMemset(real.p,0,real_count*sizeof(double)));
        kernels::odd_rhs(real.p,q,locations,values,count,temperature_ids);
        check(cudaGetLastError());
        fft_check(cufftExecD2Z(forward.handle,real.p,frequency.p));
        kernels::spectrum(frequency.p,side,scale,true);
        check(cudaGetLastError());
        fft_check(cufftExecZ2D(backward.handle,frequency.p,real.p));
        kernels::extract(real.p,q,output,temperature_layout);
        check(cudaGetLastError());
    }
};
bool enabled() {
    const char* value=std::getenv("GPUPREPROCESSOR_POISSON_PRESOLVE");
    if(!value || !std::strcmp(value,"1"))return true;
    if(std::strcmp(value,"0"))std::cerr<<"[warn] GPUPREPROCESSOR_POISSON_PRESOLVE expects 0 or 1; stage disabled\n";
    return false;
}
int shape(const LPInfoGpu* o) {
    if(!o || o->A.rows<=0 || o->A.cols<=0 || !std::isfinite(o->obj_constant) ||
       o->A.nnz!=8LL*o->A.rows || !o->A.rowPtr || !o->A.colVal || !o->A.nzVal ||
       !o->AL || !o->AU || !o->l || !o->u || !o->c)return 0;
    int q=std::lround(std::cbrt(double(o->A.rows)));
    if(1LL*q*q*q!=o->A.rows || 1LL*(q+2)*(q+2)*(q+2)+o->A.rows!=o->A.cols)return 0;
    return q;
}
}
// The response, transpose and inverse metadata have one owner. Failed
// certification/allocation never publishes a partially initialized model.
struct Reduction final : StartupPoissonReduction {
    int q=0,m=0,n=0;
    double scale=0;
    Buffer<int> observations,sources;
    ~Reduction() override {
        for(auto* a : {&transformed.A,&transformed.AT}) {
            cudaFree(a->rowPtr);cudaFree(a->colVal);cudaFree(a->nzVal);
        }
        cudaFree(transformed.AL);cudaFree(transformed.AU);cudaFree(transformed.l);
        cudaFree(transformed.u);cudaFree(transformed.c);cudaFree(transformed.AT_slack_after);
    }
    GpuPostsolveResult restore(const GpuPostsolveResult&,const LPInfoGpu&) const override;
};

std::shared_ptr<StartupPoissonReduction> presolve(const LPInfoGpu& input) {
    if(!enabled())return {};
    const auto* original=&input;
    int q=shape(original);if(!q)return {};
    auto start=std::chrono::steady_clock::now();
    try {
        Buffer<int> counts(3);Buffer<double> scale(1);
        check(cudaMemset(counts.p,0,3*sizeof(int)));
        kernels::certify(*original,q,scale.p,counts.p);
        check(cudaGetLastError());
        int h[3]={};double coefficient=0;
        check(cudaMemcpy(h,counts.p,sizeof(h),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&coefficient,scale.p,sizeof(double),cudaMemcpyDeviceToHost));
        if(h[0]) {std::cout<<"POISSON_GPU skip=structure_or_bounds\n";return {};}
        const std::int64_t nnz=std::int64_t(h[1])*h[2];
        // Bound dense growth, CSR int32 arithmetic and optional allocation.
        if(!h[1] || !h[2] || nnz>16000000 || nnz*2>=original->A.nnz) {
            std::cout<<"POISSON_GPU skip=reduction_size\n";return {};
        }
        auto state=std::make_shared<Reduction>();state->q=q;state->m=h[1];state->n=h[2];state->scale=coefficient;
        state->observations.allocate(h[1]);state->sources.allocate(h[2]);
        kernels::maps(*original,q,state->observations.p,state->sources.p);
        check(cudaGetLastError());
        auto& r=state->transformed;r.A.rows=h[1];r.A.cols=h[2];r.A.nnz=nnz;
        r.obj_constant=original->obj_constant;
        r.A.rowPtr=allocate<NnzOffset>(std::size_t(r.A.rows)+1);
        r.A.colVal=allocate<int>(nnz);r.A.nzVal=allocate<double>(nnz);
        r.AL=allocate<double>(r.A.rows);r.AU=allocate<double>(r.A.rows);
        r.l=allocate<double>(r.A.cols);r.u=allocate<double>(r.A.cols);r.c=allocate<double>(r.A.cols);
        {
            Inverse inverse(q,coefficient);inverse.green();
            kernels::response(inverse.real.p,q,state->observations.p,state->sources.p,r,*original,counts.p);
            check(cudaGetLastError());
            check(cudaMemcpy(h,counts.p,sizeof(int),cudaMemcpyDeviceToHost));
            if(h[0]) {std::cout<<"POISSON_GPU skip=nonpositive_or_nonfinite_response\n";return {};}
        }
        r.AT=transpose_csr64(r.A);
        check(cudaStreamSynchronize(0));
        std::cout<<"POISSON_GPU applied=1 q="<<q<<" scale="<<coefficient
                 <<" rows="<<original->A.rows<<"->"<<r.A.rows<<" cols="<<original->A.cols<<"->"<<r.A.cols
                 <<" nnz="<<original->A.nnz<<"->"<<nnz
                 <<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<'\n';
        return state;
    } catch(const std::exception& e) {
        auto status=cudaDeviceSynchronize();
        if(status!=cudaSuccess && status!=cudaErrorMemoryAllocation)throw;
        (void)cudaGetLastError();
        std::cerr<<"[warn] POISSON_GPU fallback=ordinary_presolver reason="<<e.what()<<'\n';
        return {};
    }
}

GpuPostsolveResult Reduction::restore(const GpuPostsolveResult& reduced,
                                     const LPInfoGpu& input) const {
    if(reduced.n0!=n || reduced.m0!=m || !reduced.x_org || !reduced.y_org || !reduced.z_org ||
       input.A.rows!=q*q*q || input.A.cols!=(q+2)*(q+2)*(q+2)+q*q*q ||
       input.AT.rows!=input.A.cols || input.AT.cols!=input.A.rows ||
       !input.AT.rowPtr || !input.AT.colVal || !input.AT.nzVal ||
       !input.l || !input.u || !input.c)
        throw std::invalid_argument("invalid Poisson device recovery input");
    Buffer<double> x(input.A.cols),y(input.A.rows),z(input.A.cols);
    check(cudaMemset(x.p,0,std::size_t(input.A.cols)*sizeof(double)));
    int p=q+2,nt=p*p*p;
    {
        Inverse inverse(q,scale);
        inverse.recover(sources.p,reduced.x_org,n,false,x.p,true);
        inverse.recover(observations.p,reduced.y_org,m,true,y.p,false);
    }
    kernels::scatter_sources(nt,sources.p,reduced.x_org,n,x.p);
    kernels::recover_z(input,y.p,z.p);
    kernels::scatter_z(nt,sources.p,reduced.z_org,n,z.p);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(0));
    std::cout << "POISSON_GPU postsolve=restored_original device_handoff=1\n";
    return {x.release(),y.release(),z.release(),input.A.cols,input.A.rows};
}
} // namespace poisson
} // namespace

std::shared_ptr<StartupPoissonReduction> run_startup_poisson(const LPInfoGpu& input) {
    return poisson::presolve(input);
}
} // namespace gpu_preprocessor::lp::presolve
