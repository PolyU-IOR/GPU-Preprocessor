// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once
#include "gpu_preprocessor/lp.hpp"
#include "gpu_preprocessor/lp/presolve/transpose64.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

namespace poisson_test {
namespace gp=gpu_preprocessor::lp::presolve;
inline void require(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
inline void check(cudaError_t error) {if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
template<class T> T* upload(const std::vector<T>& host) {
    T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),std::max(std::size_t(1),host.size())*sizeof(T)));
    check(cudaMemcpy(p,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice));return p;
}
template<class T> std::vector<T> download(const T* p,std::size_t n) {
    std::vector<T> h(n);check(cudaMemcpy(h.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return h;
}
struct Device {
    gp::LPInfoGpu lp;
    Device()=default;
    Device(const Device&)=delete;
    ~Device() {
        for(auto* a:{&lp.A,&lp.AT}) {cudaFree(a->rowPtr);cudaFree(a->colVal);cudaFree(a->nzVal);}
        cudaFree(lp.AL);cudaFree(lp.AU);cudaFree(lp.l);cudaFree(lp.u);cudaFree(lp.c);
    }
};
struct Point {
    gp::GpuPostsolveResult v;
    Point()=default;
    explicit Point(gp::GpuPostsolveResult value):v(value) {}
    Point(const Point&)=delete;
    Point(const std::vector<double>& x,const std::vector<double>& y,const std::vector<double>& z)
        :v{upload(x),upload(y),upload(z),int(x.size()),int(y.size())} {}
    ~Point() {cudaFree(v.x_org);cudaFree(v.y_org);cudaFree(v.z_org);}
};
inline int temperature(int i,int q) {int p=q+2;return (i/(q*q)+1)*p*p+(i/q%q+1)*p+i%q+1;}
// Independent spectral sum of the finite Dirichlet matrix, without a periodic
// grid, odd images, FFTs, or production code.
inline double inverse_entry(int a,int b,int q,double scale) {
    const double pi=std::acos(-1.0);double result=0;
    int ax=a%q+1,ay=a/q%q+1,az=a/(q*q)+1;
    int bx=b%q+1,by=b/q%q+1,bz=b/(q*q)+1;
    for(int i=1;i<=q;++i)for(int j=1;j<=q;++j)for(int k=1;k<=q;++k) {
        double x=pi*i/(q+1),y=pi*j/(q+1),z=pi*k/(q+1);
        result+=std::sin(ax*x)*std::sin(bx*x)*std::sin(ay*y)*std::sin(by*y)*std::sin(az*z)*std::sin(bz*z)/
            (scale*(6-2*std::cos(x)-2*std::cos(y)-2*std::cos(z)));
    }
    return result*8/std::pow(q+1,3);
}
struct Fixture {
    int q,m,nt,n;double scale=7.25;
    std::vector<int> obs,src;
    std::vector<double> x,y,z,response;
    std::vector<double> al,l,u,c,a;
    std::vector<int> col;
    std::vector<NnzOffset> ptr;
    explicit Fixture(int q_,bool pipeline=false,double coefficient=7.25,double amplitude=1):q(q_),m(q*q*q),nt((q+2)*(q+2)*(q+2)),n(nt+m) {
        scale=coefficient;
        src=q==1?std::vector<int>{0}:std::vector<int>{0,m/3,m-1};
        obs=q==1?std::vector<int>{0}:std::vector<int>{1,m/4,m/2,m-2};
        if (pipeline) {
            // Underdetermined response with permutation-symmetric sources:
            // folding could merge columns. Main remains meaningful without
            // eliminating this complete fixture to 0x0.
            src={0,1,4,16,2,8,32,63};obs={21,42};
        }
        std::sort(obs.begin(),obs.end());obs.erase(std::unique(obs.begin(),obs.end()),obs.end());
        x.resize(src.size());z.resize(src.size());y.resize(obs.size());
        for(int j=0;j<int(src.size());++j) {x[j]=amplitude*(pipeline ? (j==0?1:2) : 1+0.25*j);z[j]=j==0?0.4:-0.3;}
        for(int i=0;i<int(obs.size());++i)y[i]=(i%2?-1:1)*(0.7+0.2*i);
        al.resize(m);l.resize(n);u.resize(n);c.resize(n);ptr.push_back(0);
        for(int i=0;i<m;++i) {
            int t=temperature(i,q),p=q+2;
            std::vector<int> ids{t,t-1,t+1,t-p,t+p,t-p*p,t+p*p,nt+i};
            std::vector<std::pair<int,double>> row;
            for(int k=0;k<8;++k)row.emplace_back(ids[k],k==0?-6*scale:(k==7?1:scale));
            std::sort(row.begin(),row.end());
            for(auto entry:row){col.push_back(entry.first);a.push_back(entry.second);}
            ptr.push_back(a.size());l[t]=-INFINITY;u[t]=INFINITY;
        }
        for(int i=0;i<int(obs.size());++i) {
            double b=0;
            for(int j=0;j<int(src.size());++j) {
                double v=inverse_entry(obs[i],src[j],q,scale);response.push_back(v);b+=v*x[j];c[nt+src[j]]+=v*y[i];
            }
            l[temperature(obs[i],q)]=u[temperature(obs[i],q)]=b;
        }
        for(int j=0;j<int(src.size());++j) {
            l[nt+src[j]]=j==0?x[j]:0;u[nt+src[j]]=j==0?INFINITY:x[j];c[nt+src[j]]+=z[j];
        }
    }
    void upload_to(Device& d,bool transpose=true) const {
        auto& o=d.lp;o.A={m,n,NnzOffset(a.size()),upload(ptr),upload(col),upload(a)};
        o.AL=upload(al);o.AU=upload(al);o.l=upload(l);o.u=upload(u);o.c=upload(c);o.obj_constant=3.5;
        if(transpose)o.AT=gp::transpose_csr64(o.A);
    }
};
} // namespace poisson_test
