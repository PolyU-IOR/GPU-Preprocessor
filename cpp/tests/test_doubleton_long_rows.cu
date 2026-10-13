#include "../src/presolve/rules/doubleton_long_rows.cuh"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
using namespace gpu_presolver;
using namespace gpu_presolver::presolve::detail;
static void check(cudaError_t e) { if(e != cudaSuccess) {std::cerr<<cudaGetErrorString(e)<<'\n';std::abort();} }
template<class T> struct Buffer {
  T* p=nullptr; size_t n;
  explicit Buffer(size_t size):n(size){check(cudaMalloc(&p,sizeof(T)*std::max<size_t>(1,n)));}
  explicit Buffer(const std::vector<T>& v):Buffer(v.size()){if(n)check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
  ~Buffer(){cudaFree(p);}
  std::vector<T> read(){std::vector<T> v(n);if(n)check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
// Independent reference: the original serial count/write algorithm, without
// using the new partition or merge helpers.
template<bool Write> __global__ void reference(const NnzOffset* rp,const int* ci,const double* av,
    const NnzOffset* dq,const long long* dk,const double* dv,int n,int m,double tol,
    NnzOffset* counts,const NnzOffset* op=nullptr,int* oc=nullptr,double* ov=nullptr){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=m)return;
 auto p=rp[row],q=dq[row];NnzOffset count=0;
 while(p<rp[row+1]||q<dq[row+1]){
  int a=p<rp[row+1]?ci[p]:INT_MAX,b=q<dq[row+1]?int(dk[q]%n):INT_MAX,c=a<b?a:b;
  double v=0.0;
  while(p<rp[row+1]&&ci[p]==c)v+=av[p++];
  while(q<dq[row+1]&&int(dk[q]%n)==c)v+=dv[q++];
  if(fabs(v)>tol){if constexpr(Write){oc[op[row]+count]=c;ov[op[row]+count]=v;}++count;}
 }
 if constexpr(!Write)counts[row]=count;
}
int main(){
 const int n=2000000;
 const std::vector<int> sizes={0,1,2,31,32,33,255,256,257,4095,4096,4097,65539,503284};
 std::vector<NnzOffset> rp{0},dq{0};std::vector<int> ci;std::vector<long long> dk;std::vector<double> av,dv;
 std::mt19937 rng(738921);
 const double values[]={0.0,-0.0,1e16,1.0,-1e16,1e-12,-1e-12,1e-13,2.0,-2.0,1e-300,std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()};
 for(int size:sizes)for(int mode=0;mode<7;++mode){
  const int row=int(rp.size())-1;
  std::vector<std::pair<int,double>> a,b;
  for(int k=0;k<size;++k){
   const int col=mode==3?17:mode==4?k/300:mode==5?int(rng()%(size+1)):k;
   const double v=values[(k+mode)%14];
   if(mode!=2)a.emplace_back(col,v);
   if(mode!=0){b.emplace_back(mode==6?col+1:col,mode==1?-v:values[(k+3)%14]);if(k%127==0)b.emplace_back(col,1.0);}
  }
  std::stable_sort(a.begin(),a.end(),[](auto x,auto y){return x.first<y.first;});
  std::stable_sort(b.begin(),b.end(),[](auto x,auto y){return x.first<y.first;});
  for(auto [c,v]:a){ci.push_back(c);av.push_back(v);}
  for(auto [c,v]:b){dk.push_back(static_cast<long long>(row)*n+c);dv.push_back(v);}
  rp.push_back(ci.size());dq.push_back(dk.size());
 }
 const int m=rp.size()-1;
 std::vector<int> order(m);std::iota(order.begin(),order.end(),0);std::shuffle(order.begin(),order.end(),rng);
 Buffer<NnzOffset> r(rp),d(dq),cnt(m),refcnt(m),segments(size_t(m)*256);
 Buffer<int> c(ci),queue(order),queue_size(std::vector<int>{m});Buffer<long long> k(dk);Buffer<double> a(av),b(dv);
 for(double tol:{0.0,1e-12,0.5}){
  reference<false><<<(m+255)/256,256>>>(r.p,c.p,a.p,d.p,k.p,b.p,n,m,tol,refcnt.p);
  // Seven blocks intentionally exercise repeated queue iterations and barriers.
  doubleton_merge_long_rows<false><<<7,256>>>(r.p,c.p,a.p,d.p,k.p,b.p,queue.p,queue_size.p,n,tol,cnt.p,segments.p);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  auto expected=refcnt.read(),got=cnt.read();if(expected!=got){std::cerr<<"count mismatch\n";return 1;}
  std::vector<NnzOffset> out(m+1);std::partial_sum(expected.begin(),expected.end(),out.begin()+1);
  Buffer<NnzOffset> op(out);Buffer<int> ec(out.back()),gc(out.back());Buffer<double> ev(out.back()),gv(out.back());
  reference<true><<<(m+255)/256,256>>>(r.p,c.p,a.p,d.p,k.p,b.p,n,m,tol,nullptr,op.p,ec.p,ev.p);
  doubleton_merge_long_rows<true><<<7,256>>>(r.p,c.p,a.p,d.p,k.p,b.p,queue.p,queue_size.p,n,tol,nullptr,segments.p,op.p,gc.p,gv.p);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  auto e=ev.read(),g=gv.read();if(ec.read()!=gc.read()||std::memcmp(e.data(),g.data(),e.size()*sizeof(double))){std::cerr<<"output mismatch\n";return 1;}
  std::cout<<"PASS "<<m<<" rows, tol="<<tol<<", nnz="<<out.back()<<", bitwise identical\n";
 }
}
