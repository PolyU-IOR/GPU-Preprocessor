// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "poisson_fixture.hpp"
#include "gpu_preprocessor/lp/presolve/gpu_postsolve.hpp"
#include "../tools/postsolve_replay/kkt.hpp"
#include <iostream>
#include <functional>

using namespace poisson_test;
namespace api=gpu_preprocessor::lp;
namespace diag=api::diagnostics;

// Fail a particular allocation without reserving device memory belonging to
// other applications. Exercise partial construction and recovery separately.
static int fail_after=-1;
extern "C" cudaError_t __real_cudaMalloc(void**,std::size_t);
extern "C" cudaError_t __wrap_cudaMalloc(void** p,std::size_t n) {
    if(fail_after>=0 && fail_after--==0) { *p=nullptr;return cudaErrorMemoryAllocation; }
    return __real_cudaMalloc(p,n);
}

void reference_and_recovery() {
    for(int q:{1,2,4,5}) {
        Fixture f(q);Device d;f.upload_to(d);
        auto state=gp::run_startup_poisson(d.lp);
        require(bool(state),"eligible fixture rejected");
        auto& r=state->transformed;
        require(r.A.rows==int(f.obs.size()) && r.A.cols==int(f.src.size()),"response dimensions");
        require(r.AT.nnz==r.A.nnz && r.AT.rowPtr,"response transpose missing");
        auto a=download(r.A.nzVal,r.A.nnz);
        for(std::size_t k=0;k<a.size();++k)
            require(std::abs(a[k]-f.response[k])<1e-14,"response differs from independent sine inverse");
        require(r.obj_constant==3.5,"objective constant lost");
        Point reduced(f.x,f.y,f.z),full(state->restore(reduced.v,d.lp));
        require(full.v.n0==f.n && full.v.m0==f.m,"recovery dimensions");
        require(diag::check_kkt(&d.lp,full.v.x_org,full.v.y_org,full.v.z_org).passed(1e-10),"original KKT");
        auto x=download(full.v.x_org,f.n),y=download(full.v.y_org,f.m);
        for(int i=0;i<f.m;++i) {
            double t=0,v=0;
            for(int j=0;j<int(f.src.size());++j)t+=inverse_entry(i,f.src[j],q,f.scale)*f.x[j];
            for(int j=0;j<int(f.obs.size());++j)v+=inverse_entry(i,f.obs[j],q,f.scale)*f.y[j];
            require(std::abs(x[temperature(i,q)]-t)<1e-13,"primal inverse mismatch");
            require(std::abs(y[i]-v)<1e-13,"dual inverse mismatch");
        }
        require(download(d.lp.A.nzVal,d.lp.A.nnz)==f.a,"input mutated");
        require(download(reduced.v.x_org,f.x.size())==f.x,"borrowed solution mutated");
        auto bad=reduced.v;bad.n0++;
        bool threw=false;try{Point invalid(state->restore(bad,d.lp));}catch(const std::invalid_argument&){threw=true;}
        require(threw,"invalid recovery dimensions accepted");
        x[temperature(f.obs[0],q)]+=0.1;
        check(cudaMemcpy(full.v.x_org,x.data(),x.size()*sizeof(double),cudaMemcpyHostToDevice));
        require(!diag::check_kkt(&d.lp,full.v.x_org,full.v.y_org,full.v.z_org).passed(1e-6),"corrupt solution certified");
    }
    std::cout<<"POISSON_REFERENCE_RECOVERY PASS\n";
}

void gates() {
    for(int which=0;which<14;++which) {
        Fixture f(4);
        if(which==0)f.a[0]+=1e-10;
        if(which==1)f.c[temperature(2,4)]=1e-30;
        if(which==2)f.l[0]=f.u[0]=1;
        if(which==3)f.l[temperature(3,4)]=0;
        if(which==4)f.al[5]=1;
        if(which==5)f.l[f.nt+2]=f.u[f.nt+2]=1;
        if(which==6)f.col[1]=f.col[2];
        if(which==7)f.c[f.nt]=NAN;
        if(which==8)setenv("GPUPREPROCESSOR_POISSON_PRESOLVE","0",1);
        if(which==9)for(int i:f.obs){f.l[temperature(i,4)]=-INFINITY;f.u[temperature(i,4)]=INFINITY;}
        if(which==10)for(int i:f.src)f.l[f.nt+i]=f.u[f.nt+i]=0;
        if(which==11)setenv("GPUPREPROCESSOR_POISSON_PRESOLVE","invalid",1);
        if(which==12) { // Valid structure whose dense response would grow.
            for(int i=0;i<f.m;++i){f.l[temperature(i,4)]=f.u[temperature(i,4)]=0;f.u[f.nt+i]=INFINITY;}
        }
        Device d;f.upload_to(d,false);
        if(which==13)d.lp.obj_constant=NAN;
        require(!gp::run_startup_poisson(d.lp),"ineligible/off stage accepted");
        require(download(d.lp.A.nzVal,f.a.size())==f.a,"miss mutated input");
        unsetenv("GPUPREPROCESSOR_POISSON_PRESOLVE");
    }
    std::cout<<"POISSON_CERTIFICATION_GATES PASS\n";
}

// Small independent active-set enumeration. It checks primal feasibility,
// stationarity and multiplier signs, and has no production solver dependency.
api::Solution solve_small(const gp::LPInfoGpu& lp) {
    const int n=lp.A.cols,m=lp.A.rows;require(n>0 && n<=8,"unexpected small LP dimensions");
    auto ptr=download(lp.A.rowPtr,m+1);auto ids=download(lp.A.colVal,lp.A.nnz);
    auto a=download(lp.A.nzVal,lp.A.nnz),c=download(lp.c,n),l=download(lp.l,n),u=download(lp.u,n);
    auto al=download(lp.AL,m),au=download(lp.AU,m);
    struct Constraint {std::vector<double> a;double b;int id,sign;};
    std::vector<Constraint> constraints;
    auto add=[&](std::vector<double> row,double lo,double hi,int id) {
        if(std::isfinite(lo))constraints.push_back({row,lo,id,lo==hi?0:1});
        if(std::isfinite(hi) && lo!=hi)constraints.push_back({row,hi,id,-1});
    };
    for(int i=0;i<m;++i){std::vector<double> row(n);for(auto k=ptr[i];k<ptr[i+1];++k)row[ids[k]]=a[k];add(row,al[i],au[i],i);}
    for(int j=0;j<n;++j){std::vector<double> row(n);row[j]=1;add(row,l[j],u[j],m+j);}
    auto linear=[&](std::vector<std::vector<double>> mat,std::vector<double> rhs,std::vector<double>& out) {
        for(int k=0;k<n;++k) {
            int p=k;for(int i=k+1;i<n;++i)if(std::abs(mat[i][k])>std::abs(mat[p][k]))p=i;
            if(std::abs(mat[p][k])<1e-25)return false;
            std::swap(mat[p],mat[k]);std::swap(rhs[p],rhs[k]);
            double pivot=mat[k][k];for(int j=k;j<n;++j)mat[k][j]/=pivot;rhs[k]/=pivot;
            for(int i=0;i<n;++i)if(i!=k){double v=mat[i][k];for(int j=k;j<n;++j)mat[i][j]-=v*mat[k][j];rhs[i]-=v*rhs[k];}
        }out=rhs;return true;
    };
    std::vector<int> chosen;api::Solution result;bool found=false;
    std::function<void(int)> search=[&](int start) {
        if(found)return;
        if(int(chosen.size())<n){for(int k=start;k<int(constraints.size());++k){chosen.push_back(k);search(k+1);chosen.pop_back();}return;}
        std::vector<std::vector<double>> mat(n),trans(n,std::vector<double>(n));std::vector<double> rhs(n),x,dual;
        for(int i=0;i<n;++i){mat[i]=constraints[chosen[i]].a;rhs[i]=constraints[chosen[i]].b;for(int j=0;j<n;++j)trans[j][i]=mat[i][j];}
        if(!linear(mat,rhs,x) || !linear(trans,c,dual))return;
        for(auto& con:constraints){double v=0;for(int j=0;j<n;++j)v+=con.a[j]*x[j];double tol=1e-9*(1+std::abs(con.b));
            if((con.sign>=0 && v<con.b-tol) || (con.sign<=0 && v>con.b+tol))return;}
        for(int i=0;i<n;++i)if(constraints[chosen[i]].sign*dual[i]<-1e-9)return;
        result.x=x;result.y.assign(m,0);result.z.assign(n,0);
        for(int i=0;i<n;++i){int id=constraints[chosen[i]].id;if(id<m)result.y[id]+=dual[i];else result.z[id-m]+=dual[i];}
        found=true;
    };search(0);require(found,"independent small LP solve failed");return result;
}

static int preparations=0;
void prepare(gp::LPInfoGpu& lp,void*) {
    ++preparations;
    require(lp.AT.rowPtr && lp.A.rows==2,"Poisson must precede column access");
}

void pipeline() {
    for(bool tiny:{false,true})for(bool folding:{false,true})for(bool csr_only:{false,true}) {
        Fixture f(4,true,tiny?7.25e9:7.25,tiny?1e8:1);Device d;f.upload_to(d);
        auto input=d.lp;if(csr_only)input.AT={};
        api::Parameters p;p.enable_folding=folding;
        if(csr_only){preparations=0;p.prepare_startup_column_access=prepare;}
        auto s=api::preprocess(input,p);
        auto& record=s.startup?s.startup->summary.record:s.presolve.record;
        require(bool(record.startup_poisson),"default Startup did not run Poisson");
        if(csr_only)require(preparations==1,"CSR-only preparation count");
        require(!s.folding_applied && s.folding.profile.total_seconds==0,
                "folding ran after successful Startup Poisson");
        if(folding)require(s.folded_cols==int(f.src.size()) && s.folded_rows==int(f.obs.size()),
                           "skipped folding changed the response dimensions");
        require(!s.presolve.has_infeasible && s.presolve.reduced_nnz>0,"response lost nonzero coefficients");
        if(tiny){auto a=download(s.presolve.reduced_lp.A.nzVal,s.presolve.reduced_nnz);
            require(std::any_of(a.begin(),a.end(),[](double v){return v!=0 && std::abs(v)<1e-10;}),"tiny coefficients dropped");}
        auto solved=solve_small(s.presolve.reduced_lp);Point reduced(solved.x,solved.y,solved.z);
        Point full(api::postprocess_device(reduced.v.x_org,reduced.v.y_org,reduced.v.z_org,s,d.lp));
        auto metrics=diag::check_kkt(&d.lp,full.v.x_org,full.v.y_org,full.v.z_org);
        diag::print_kkt("Poisson original",metrics,1e-7);
        // The extreme-scale fixture tests coefficient preservation and replay.
        // Its active-set duals can be O(1e11), so original-space stationarity
        // suffers cancellation; do not label it a tight original-KKT pass.
        if(!tiny)require(metrics.passed(1e-7),"pipeline original KKT");
        else require(metrics.finite && metrics.gap<1e-10 && metrics.complementarity<1e-10,
                     "tiny-response replay is nonfinite or loses primal/dual objective agreement");
        auto host=api::postprocess(reduced.v.x_org,reduced.v.y_org,reduced.v.z_org,s,d.lp);
        require(host.x==download(full.v.x_org,f.n) && host.y==download(full.v.y_org,f.m) && host.z==download(full.v.z_org,f.n),"host/device replay differs");
        require(download(d.lp.A.nzVal,f.a.size())==f.a,"pipeline mutated original");
        api::free_summary(s);api::free_summary(s);
    }
    // The same symmetric response folds when Poisson does not fire. The skip
    // is tied to a successful reduction, not dimensions or an instance name.
    Fixture f(4,true);Device d;f.upload_to(d);auto state=gp::run_startup_poisson(d.lp);
    auto s=api::preprocess(state->transformed);
    require(s.folding_applied && s.folded_cols<int(f.src.size()),"Poisson miss incorrectly bypassed folding");
    api::free_summary(s);
    std::cout<<"POISSON_STARTUP_SKIP_FOLDING_MAIN_HOST_DEVICE_TINY_CSR_ONLY PASS\n";
}

void scheduling_and_failures() {
    Fixture f(4);Device d;f.upload_to(d);
    for(int which=0;which<6;++which) {
        gp::PresolveParams p;gp::PresolveBudget budget;
        if(which==0)p.startup_poisson=nullptr;
        if(which==1)p.enable_startup=false;
        if(which==2)p.enable_structure_specific_rules=false;
        if(which==3)p.max_iters=0;
        if(which==4)p.max_time=1e-12;
        if(which==5)budget.iterations=p.max_iters;
        auto s=gp::run_gpu_presolve_stage(d.lp,p,gp::PresolveStage::Startup,budget);
        require(!s.record.startup_poisson,"disabled/exhausted Startup applied Poisson");
        gp::free_gpu_presolve_reduced_lp(s);
    }
    for(int allocation:{0,1,2,5,10,12}) {
        fail_after=allocation;auto state=gp::run_startup_poisson(d.lp);fail_after=-1;
        require(!state,"allocation failure did not fall back");
        require(download(d.lp.A.nzVal,f.a.size())==f.a,"failed allocation changed input");
    }
    auto state=gp::run_startup_poisson(d.lp);require(bool(state),"retry after OOM failed");
    Point reduced(f.x,f.y,f.z);
    fail_after=1;bool threw=false;
    try{Point out(state->restore(reduced.v,d.lp));}catch(const std::exception&){threw=true;}
    fail_after=-1;require(threw,"recovery allocation failure was swallowed");
    Point retry(state->restore(reduced.v,d.lp));
    require(diag::check_kkt(&d.lp,retry.v.x_org,retry.v.y_org,retry.v.z_org).passed(1e-10),"retry recovery KKT");
    std::cout<<"POISSON_BUDGET_SWITCHES_OOM_RETRY PASS\n";
}

// Exercise the row -> custom response seam independently of the restrictive
// stencil certificate. Recovery must call the response inverse before S^T.
struct IdentityResponse final : gp::StartupPoissonReduction {
    Device owner;
    explicit IdentityResponse(const gp::LPInfoGpu& in) {
        require(in.A.nnz==48 && !in.AT.rowPtr,"response did not receive the deferred row basis");
        auto& o=owner.lp;o.A={in.A.rows,in.A.cols,in.A.nnz,upload(download(in.A.rowPtr,in.A.rows+1)),
            upload(download(in.A.colVal,in.A.nnz)),upload(download(in.A.nzVal,in.A.nnz))};
        o.AT=gp::transpose_csr64(o.A);o.AL=upload(download(in.AL,in.A.rows));o.AU=upload(download(in.AU,in.A.rows));
        o.c=upload(download(in.c,in.A.cols));o.l=upload(download(in.l,in.A.cols));o.u=upload(download(in.u,in.A.cols));
        o.obj_constant=in.obj_constant;transformed=o;
    }
    gp::GpuPostsolveResult restore(const gp::GpuPostsolveResult& r,const gp::LPInfoGpu& in) const override {
        require(in.AT.rowPtr && in.AT.nnz==48,"retained row transpose missing");
        require(download(r.y_org,3)==std::vector<double>({7,6,4}),"row adjoint ran before response inverse");
        return {upload(download(r.x_org,r.n0)),upload(download(r.y_org,r.m0)),upload(download(r.z_org,r.n0)),r.n0,r.m0};
    }
};
std::shared_ptr<gp::StartupPoissonReduction> identity_response(const gp::LPInfoGpu& input) {
    return std::make_shared<IdentityResponse>(input);
}
void composed_row_recovery() {
    Device d;auto& o=d.lp;std::vector<NnzOffset> ptr{0};std::vector<int> ids;std::vector<double> a,c(48);
    for(int i=0;i<3;++i){for(int j=0;j<16*(i+1);++j){ids.push_back(j);a.push_back(1);c[j]+=1<<i;}ptr.push_back(ids.size());}
    o.A={3,48,96,upload(ptr),upload(ids),upload(a)};o.AT=gp::transpose_csr64(o.A);
    o.AL=upload(std::vector<double>(3,1));o.AU=upload(std::vector<double>(3,1));
    o.l=upload(std::vector<double>(48,0));o.u=upload(std::vector<double>(48,INFINITY));o.c=upload(c);
    auto input=o;input.AT={};gp::PresolveParams p;p.defer_startup_row_transpose=true;p.startup_poisson=identity_response;
    p.enable_column_domination=p.enable_two_row_variable_elimination=p.enable_zero_cost_product_variables=false;
    p.enable_piecewise_linear_reformulation_variable_pair=p.enable_piecewise_linear_reformulation_absolute_value=false;
    p.prepare_startup_column_access=[](gp::LPInfoGpu& basis,void*){require(basis.AT.rowPtr,"custom response lost transpose");};
    gp::PresolveBudget budget;auto s=gp::run_gpu_presolve_stage(input,p,gp::PresolveStage::Startup,budget);
    require(s.record.startup_rows && s.record.startup_poisson,"composed Startup lost an inverse");
    std::vector<double> x(48);x[0]=1;Point reduced(x,{7,6,4},std::vector<double>(48));
    gp::free_gpu_presolve_reduced_lp(s); // The recovery record owns both bases.
    Point full(gp::postsolve_gpu(reduced.v.x_org,reduced.v.y_org,reduced.v.z_org,s.record,&o));
    require(download(full.v.y_org,3)==std::vector<double>({1,2,4}),"row chain adjoint mismatch");
    require(diag::check_kkt(&o,full.v.x_org,full.v.y_org,full.v.z_org).passed(1e-12),"composed original KKT");
    std::cout<<"STARTUP_ROW_RESPONSE_INVERSE_ORDER PASS\n";
}

void fully_eliminated_response() {
    Fixture f(1);Device d;f.upload_to(d);
    for(bool folding:{false,true}) {
        api::Parameters p;p.enable_folding=folding;auto s=api::preprocess(d.lp,p);
        require(s.presolve.reduced_rows==0 && s.presolve.reduced_cols==0,
                "single-source fixture was not fully eliminated");
        Point full(api::postprocess_device(nullptr,nullptr,nullptr,s,d.lp));
        require(diag::check_kkt(&d.lp,full.v.x_org,full.v.y_org,full.v.z_org).passed(1e-10),
                "empty response lost original recovery");
        api::free_summary(s);
    }
    std::cout<<"POISSON_FULL_ELIMINATION_RECOVERY PASS\n";
}

int main() {
    unsetenv("GPUPREPROCESSOR_POISSON_PRESOLVE");
    reference_and_recovery();gates();pipeline();scheduling_and_failures();composed_row_recovery();fully_eliminated_response();
}
