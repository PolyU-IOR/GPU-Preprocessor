// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "kernels.cuh"
#include "gpu_preprocessor/lp/presolve/transpose64.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace gpu_preprocessor::lp::presolve {
namespace poisson {
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

std::shared_ptr<StartupPoissonReduction> run_startup_poisson(const LPInfoGpu& input) {
    return poisson::presolve(input);
}
} // namespace gpu_preprocessor::lp::presolve
