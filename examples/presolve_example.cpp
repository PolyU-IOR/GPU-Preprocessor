#include "gpu_presolver/offset.hpp"
#include "gpu_preprocessor/gpu_preprocessor.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace gpp = gpu_preprocessor;
void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<class T> struct DeviceVector {
  T* data = nullptr;
  explicit DeviceVector(const std::vector<T>& v) {
    if (v.empty()) return;
    check(cudaMalloc(&data, v.size()*sizeof(T)));
    auto e = cudaMemcpy(data,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { cudaFree(data); data=nullptr; check(e); }
  }
  ~DeviceVector() { cudaFree(data); }
  DeviceVector(const DeviceVector&) = delete;
  DeviceVector& operator=(const DeviceVector&) = delete;
};
struct PipelineOwner {
  gpp::Summary value;
  ~PipelineOwner() {
    gpp::free_summary(value);
  }
};
bool flag(const char* v) {
  if (std::string(v)=="true" || std::string(v)=="1") return true;
  if (std::string(v)=="false" || std::string(v)=="0") return false;
  throw std::invalid_argument("switch value must be true/false or 1/0");
}
int main(int argc, char** argv) {
  try {
    gpp::Parameters params;
    for(int i=1;i<argc;i+=2) {
      if(i+1>=argc) throw std::invalid_argument("missing switch value");
      if(std::string(argv[i])=="--folding") params.enable_folding=flag(argv[i+1]);
      else if(std::string(argv[i])=="--presolve") params.enable_presolve=flag(argv[i+1]);
      else throw std::invalid_argument("usage: gpu_preprocessor_example [--folding true|false] [--presolve true|false]");
    }
    check(cudaSetDevice(0)); // Select another physical GPU with CUDA_VISIBLE_DEVICES.
    // This example only: min x0+x1, x0>=1, x1>=1, 0<=x0,x1<=10.
    DeviceVector<NnzOffset> rp({0,1,2});
    DeviceVector<std::int32_t> ci({0,1});
    DeviceVector<double> av({1,1}),c({1,1}),al({1,1}),au({INFINITY,INFINITY}),l({0,0}),u({10,10});
    gpp::DeviceLp original;
    original.A={2,2,2,rp.data,ci.data,av.data};
    original.AT=original.A; // The identity is its own transpose; this example owns each allocation once.
    original.c=c.data; original.AL=al.data; original.AU=au.data; original.l=l.data; original.u=u.data;
    PipelineOwner pipeline;
    pipeline.value=gpu_preprocessor::preprocess(original,params);
    auto& p=pipeline.value.presolve;
    if(p.has_infeasible || p.has_unbounded) throw std::runtime_error("unexpected infeasible/unbounded example");
    std::cout<<"original 2 x 2 -> folded "<<pipeline.value.folded_rows<<" x "<<pipeline.value.folded_cols
             <<" -> reduced "<<p.reduced_rows<<" x "<<p.reduced_cols<<"\n";
    // Replace this analytic toy solution with your external LP solver's output.
    // Preserve p.reduced_lp: its arrays may alias original/folded allocations.
    if(p.reduced_rows!=p.reduced_cols || p.reduced_cols>2)
      throw std::runtime_error("unexpected toy reduction; update the analytic example");
    DeviceVector<double> xr(std::vector<double>(p.reduced_cols,1.0));
    DeviceVector<double> yr(std::vector<double>(p.reduced_rows,p.reduced_cols==1?2.0:1.0));
    DeviceVector<double> zr(std::vector<double>(p.reduced_cols,0.0));
    const auto sol=gpu_preprocessor::postprocess(xr.data,yr.data,zr.data,pipeline.value,original);
    if(sol.x.size()!=2 || std::abs(sol.x[0]-1)>1e-8 || std::abs(sol.x[1]-1)>1e-8)
      throw std::runtime_error("original primal recovery failed");
    std::cout<<"original primal: "<<sol.x[0]<<", "<<sol.x[1]<<"; objective: "<<sol.x[0]+sol.x[1]<<"\n";
    std::cout<<"This checks primal recovery for a toy LP; validate original KKT in your solver integration.\n";
    return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
