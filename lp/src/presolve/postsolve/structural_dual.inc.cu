// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

// Implementation fragment owned by gpu_postsolve.cu; not a separate build source.
#include "gpu_preprocessor/lp/presolve/structural_dual.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gpu_preprocessor::lp::presolve {
namespace structural_dual_detail {
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> T* clone(const T* p, std::size_t n) {
  if(!n || !p) return nullptr;
  T* q=nullptr; check(cudaMalloc(&q,n*sizeof(T)));
  try { check(cudaMemcpy(q,p,n*sizeof(T),cudaMemcpyDeviceToDevice)); }
  catch(...) { cudaFree(q); throw; }
  return q;
}
template<class T> RecoveryArray<T> buffer(int n) {
  RecoveryArray<T> r; r.count=n; if(n) check(cudaMalloc(&r.data,std::size_t(n)*sizeof(T))); return r;
}
template<class T> RecoveryArray<T> copy(const T* p,int n) {
  RecoveryArray<T> r; r.count=n; r.data=clone(p,n); return r;
}
template<class T> RecoveryArray<T> upload(const std::vector<T>& v) {
  auto r=buffer<T>(int(v.size()));
  if(r.count) check(cudaMemcpy(r.data,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return r;
}
void matrix(DeviceCsrMatrix& out,const DeviceCsrMatrix& in) {
  out.rows=in.rows;out.cols=in.cols;out.nnz=in.nnz;
  out.rowPtr=clone(in.rowPtr,std::size_t(in.rows)+1);
  out.colVal=clone(in.colVal,in.nnz);out.nzVal=clone(in.nzVal,in.nnz);
}
DeviceRecoveryGroup copy_group(const DeviceRecoveryGroup& in) {
  DeviceRecoveryGroup out;
  for(const auto& v:in.ids) out.ids.push_back(copy(v.data,v.count));
  for(const auto& v:in.values) out.values.push_back(copy(v.data,v.count));
  return out;
}
DevicePiecewiseLinearReformulationAbsoluteValueStep copy_absolute(const DevicePiecewiseLinearReformulationAbsoluteValueStep& in) {
  DevicePiecewiseLinearReformulationAbsoluteValueStep out;out.pattern=in.pattern;
  out.splits=copy_group(in.splits);out.pairs=copy_group(in.pairs);out.linked=copy_group(in.linked);
  for(const auto& g:in.max_slacks) out.max_slacks.push_back(copy_group(g)); return out;
}
DeviceRecoveryGroup pair_group(const PiecewiseLinearReformulationVariablePairRecoveryStep& in) {
  DeviceRecoveryGroup out;
  out.ids.push_back(upload(in.plus_cols));out.ids.push_back(upload(in.minus_cols));
  out.ids.push_back(upload(in.root_plus_cols));out.ids.push_back(upload(in.root_minus_cols));
  out.values.push_back(upload(in.plus_shift));out.values.push_back(upload(in.minus_shift));return out;
}
DevicePiecewiseLinearReformulationAbsoluteValueStep absolute_group(const PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep& in) {
  DevicePiecewiseLinearReformulationAbsoluteValueStep out;out.pattern=in.pattern;
  std::vector<int> a,b;std::vector<double> v;
  for(const auto& s:in.splits){a.push_back(s.t_col);b.push_back(s.e_col);v.push_back(s.rho);}
  out.splits.ids.push_back(upload(a));out.splits.ids.push_back(upload(b));out.splits.values.push_back(upload(v));
  a.clear();b.clear();v.clear();
  for(const auto& s:in.outer_pairs){a.push_back(s.bound_col);b.push_back(s.free_col);}
  out.pairs.ids.push_back(upload(a));out.pairs.ids.push_back(upload(b));
  a.clear();b.clear();
  for(const auto& s:in.linked_slacks){a.push_back(s.slack_col);b.push_back(s.t_col);v.push_back(s.factor);}
  out.linked.ids.push_back(upload(a));out.linked.ids.push_back(upload(b));out.linked.values.push_back(upload(v));
  for(const auto& s:in.max_slacks){DeviceRecoveryGroup g;g.ids.push_back(upload(std::vector<int>{s.slack_col}));
    g.ids.push_back(upload(s.t_cols));g.values.push_back(upload(s.factors));out.max_slacks.push_back(std::move(g));}
  return out;
}
__device__ double coefficient(DeviceCsrMatrix a,int row,int col) {
  for(auto p=a.rowPtr[row];p<a.rowPtr[row+1];++p) if(a.colVal[p]==col) return a.nzVal[p];
  return 0;
}
__global__ void identity(int* map,int n) {int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)map[i]=i;}
__global__ void pair_maps(int* nodes,int* roots,const int* plus,const int* minus,
                          const int* rootp,const int* rootn,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){nodes[plus[i]]=nodes[minus[i]]=i;roots[plus[i]]=rootp[i];roots[minus[i]]=rootn[i];}
}
__global__ void pair_row_scales(double* scales,DeviceCsrMatrix before,DeviceCsrMatrix after,
                               const int* roots,int m) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m)return;scales[i]=1;
  for(auto p=after.rowPtr[i];p<after.rowPtr[i+1];++p){
    if(after.nzVal[p]==0)continue;int j=after.colVal[p];double sum=0;
    for(auto q=before.rowPtr[i];q<before.rowPtr[i+1];++q)if(roots[before.colVal[q]]==j)sum+=before.nzVal[q];
    if(sum!=0){scales[i]=after.nzVal[p]/sum;break;}
  }
}
__device__ int edge_neighbor(const LPInfoGpu& lp,int row,int node,const int* nodes,
                             const int* plus,const int* minus) {
  auto a=lp.A;auto b=a.rowPtr[row];
  if(a.rowPtr[row+1]-b!=4 || lp.AL[row]!=0 || lp.AU[row]!=0)return -1;
  int other=-1;
  for(auto p=b;p<b+4;++p){int k=nodes[a.colVal[p]];if(k<0)return -1;if(k!=node){if(other>=0&&other!=k)return -1;other=k;}}
  if(other<0)return -1;
  double s=coefficient(a,row,plus[node]);
  if(s==0 || coefficient(a,row,minus[node])!=-s || coefficient(a,row,plus[other])!=-s || coefficient(a,row,minus[other])!=s)return -1;
  return other;
}
__global__ void build_pair_trees(LPInfoGpu lp,const int* nodes,const int* plus,const int* minus,
    const int* roots,int n,int* parent,int* tree_row,int* next,int* head,double* scale,NnzOffset* cursor) {
  int root=blockIdx.x*blockDim.x+threadIdx.x;
  if(root>=n || roots[root]!=plus[root])return;
  int node=root,tail=-1;parent[root]=-1;head[root]=-1;cursor[root]=lp.AT.rowPtr[plus[root]];
  while(node>=0){
    bool descend=false;
    while(cursor[node]<lp.AT.rowPtr[plus[node]+1]){
      int row=lp.AT.colVal[cursor[node]++];
      int child=edge_neighbor(lp,row,node,nodes,plus,minus);
      if(child<0 || parent[child]!=-2)continue;
      parent[child]=node;tree_row[child]=row;scale[child]=coefficient(lp.A,row,plus[child]);
      cursor[child]=lp.AT.rowPtr[plus[child]];node=child;descend=true;break;
    }
    if(descend)continue;
    if(tail<0)head[root]=node;else next[tail]=node;
    next[node]=-1;tail=node;node=parent[node];
  }
}
__global__ void init_parent(int* p,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)p[i]=-2;}
__global__ void gather(double* out,const double* in,const int* map,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=in[map[i]];}
__global__ void scatter(double* out,const double* in,const int* map,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[map[i]]=in[i];}
__global__ void protect(std::uint8_t* mask,const int* map,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)mask[map[i]]=1;}
__global__ void clip_covering(double* y,double* delta,int m){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m)return;
  double before=y[i];y[i]=fmin(before,1.0);delta[i]=before-y[i];
}
__global__ void restore_covering(LPInfoGpu lp,double* x,const double* y,const double* delta,
                                 double* z,const std::uint8_t* keep){
  int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=lp.A.cols)return;
  auto at=lp.AT;double value=keep[j]?z[j]:lp.c[j];
  // The incremental formula equals c-A^T*y for an exact input and preserves
  // its stationarity residual for an approximate input on surviving columns.
  for(auto p=at.rowPtr[j];p<at.rowPtr[j+1];++p)
    value+=at.nzVal[p]*(keep[j]?delta[at.colVal[p]]:-y[at.colVal[p]]);
  z[j]=value;if(lp.c[j]>double(at.rowPtr[j+1]-at.rowPtr[j]))x[j]=0;
}
__global__ void undo_row_scales(double* y,const double* scales,const std::uint8_t* keep,int m){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<m)y[i]=keep[i]?y[i]*scales[i]:0;
}
__device__ int epigraph_row(const LPInfoGpu& lp,int p,int n){
  for(auto q=lp.AT.rowPtr[p];q<lp.AT.rowPtr[p+1];++q){int row=lp.AT.colVal[q];
    if(lp.A.rowPtr[row+1]-lp.A.rowPtr[row]==3 && lp.AT.nzVal[q]<0 && coefficient(lp.A,row,n)==lp.AT.nzVal[q])return row;
  }return -1;
}
__global__ void restore_pair_components(LPInfoGpu lp,double* x,double* y,double* z,
    const int* pc,const int* nc,const int* roots,const double* sp,const double* sn,
    const std::uint8_t* keep,const int* parent,const int* tree_row,const int* next,
    const int* head,const double* scale,double* h,int count){
  int root=blockIdx.x*blockDim.x+threadIdx.x;if(root>=count||roots[root]!=pc[root])return;
  const int gp=epigraph_row(lp,pc[root],nc[root]);const bool epi=gp>=0;
  double lambda=epi?y[gp]:0, zp=z[pc[root]],zn=z[nc[root]],xp=x[pc[root]],xn=x[nc[root]];
  double cap_p=INFINITY,cap_n=INFINITY,rho=0,g=0;
  int owner_p=root,owner_n=root,owner_default=root;
  for(int i=head[root];i>=0;i=next[i]){
    const int p=pc[i],n=nc[i],er=epi?epigraph_row(lp,p,n):-1;
    if(er>=0)y[er]=0;
    double a=lp.u[p]-sp[i],b=lp.u[n]-sn[i];
    if(a<cap_p||(a==cap_p&&p<pc[owner_p])){cap_p=a;owner_p=i;}
    if(b<cap_n||(b==cap_n&&p<pc[owner_n])){cap_n=b;owner_n=i;}
    if(p<pc[owner_default])owner_default=i;
    rho+=lp.c[p];
    for(auto q=lp.AT.rowPtr[p];q<lp.AT.rowPtr[p+1];++q)g+=lp.AT.nzVal[q]*y[lp.AT.colVal[q]];
  }
  // Root slots removed as fixed coordinates have no incoming bound multiplier.
  const double epi_term=epi?-coefficient(lp.A,gp,pc[root])*lambda:rho;
  if(!keep[pc[root]])zp=epi_term-g;
  if(!keep[nc[root]])zn=epi_term+g;
  const int owner=zp<0?owner_p:(zn<0?owner_n:owner_default);
  for(int i=head[root];i>=0;i=next[i]){
    const int p=pc[i],n=nc[i];x[p]=xp+sp[i];x[n]=xn+sn[i];
    if(epi){int er=epigraph_row(lp,p,n);y[er]=i==owner?lambda:0;z[p]=i==owner?zp:0;z[n]=i==owner?zn:0;}
    else if(zp<0){z[p]=i==owner_p?zp:0;z[n]=2*lp.c[p]-z[p];}
    else if(zn<0){z[n]=i==owner_n?zn:0;z[p]=2*lp.c[p]-z[n];}
    else {z[p]=lp.c[p]/rho*zp;z[n]=lp.c[p]/rho*zn;}
    double value=lp.c[p]-z[p];
    for(auto q=lp.AT.rowPtr[p];q<lp.AT.rowPtr[p+1];++q)value-=lp.AT.nzVal[q]*y[lp.AT.colVal[q]];
    h[i]=value;
  }
  // Postorder accumulates the required equality contribution toward the root.
  for(int i=head[root];i>=0;i=next[i])if(parent[i]>=0){y[tree_row[i]]=h[i]/scale[i];h[parent[i]]+=h[i];}
}
__device__ bool zero_side(const LPInfoGpu& lp,int row){
  return (lp.AL[row]==0 && lp.AU[row]==INFINITY)||(lp.AL[row]==-INFINITY&&lp.AU[row]==0);
}
__device__ double orientation(const LPInfoGpu& lp,int row){return lp.AL[row]==0?1.0:-1.0;}
__global__ void restore_split(LPInfoGpu lp,double* x,double* y,double* z,
    const int* t,const int* e,const double* rho,const std::uint8_t* keep,int n){
  int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=n)return;
  int tc=t[k],ec=e[k],minus=-1,plus=-1;double am=0,ap=0,v=0;
  for(auto p=lp.AT.rowPtr[tc];p<lp.AT.rowPtr[tc+1];++p){
    int row=lp.AT.colVal[p];double ae=coefficient(lp.A,row,ec),at=lp.AT.nzVal[p];
    if(lp.A.rowPtr[row+1]-lp.A.rowPtr[row]!=2 || !zero_side(lp,row) || at*orientation(lp,row)<=0 || ae==0)continue;
    if(ae*orientation(lp,row)<0){minus=row;am=at;}else {plus=row;ap=at;}
    if(keep[row])v=y[row];
  }
  if(minus<0||plus<0)return;
  double p=x[tc],q=x[ec],zp=z[tc],zn=z[ec];
  x[tc]=p+q;x[ec]=(p-q)/rho[k];
  y[minus]=zn/(2*am);y[plus]=zp/(2*ap);z[tc]=0;z[ec]=rho[k]*v;
}
__global__ void restore_link(LPInfoGpu lp,double* x,double* y,double* z,
    const int* slack,const int* t,const double* factor,int n){
  int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=n)return;int s=slack[k],q=t[k],link=-1;double coeff=0,value=lp.c[s];
  for(auto p=lp.AT.rowPtr[s];p<lp.AT.rowPtr[s+1];++p){int row=lp.AT.colVal[p];
    if(lp.AL[row]==lp.AU[row]&&lp.A.rowPtr[row+1]-lp.A.rowPtr[row]==2&&coefficient(lp.A,row,q)!=0){link=row;coeff=lp.AT.nzVal[p];}
    else value-=lp.AT.nzVal[p]*y[row];
  }
  x[s]=factor[k]*x[q];z[s]=0;if(link>=0)y[link]=value/coeff;
}
__global__ void restore_maximum(LPInfoGpu lp,double* x,double* y,double* z,const int* slack,
    const int* t,const double* factor,int n){
  if(blockIdx.x||threadIdx.x)return;int s=slack[0];double value=lp.l[s];
  for(int i=0;i<n;++i)value=fmax(value,factor[i]*x[t[i]]);
  x[s]=isfinite(value)?value:0;z[s]=0;
  for(auto p=lp.AT.rowPtr[s];p<lp.AT.rowPtr[s+1];++p)y[lp.AT.colVal[p]]=0;
}
__global__ void restore_outer(LPInfoGpu lp,double* x,double* y,double* z,
    const int* bound,const int* source,const std::uint8_t* keep,int n){
  int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=n)return;int b=bound[k],f=source[k];
  double value=lp.c[f],negative_a=0,positive_a=0;int negative=-1,positive=-1;
  for(auto p=lp.AT.rowPtr[f];p<lp.AT.rowPtr[f+1];++p){int row=lp.AT.colVal[p];double af=lp.AT.nzVal[p];
    // Only removed two-entry rows can belong to the outer block. Retained
    // dense rows contribute through this CSC entry without another CSR scan.
    const bool candidate=!keep[row] && lp.A.rowPtr[row+1]-lp.A.rowPtr[row]==2 && zero_side(lp,row);
    const double ab=candidate?coefficient(lp.A,row,b):0;
    bool outer=candidate && ab!=0;
    if(outer){y[row]=0;double sign=orientation(lp,row);
      if(af*sign<0&&ab*sign>0){negative=row;negative_a=af;}
      else if(af*sign>0&&ab*sign<0){positive=row;positive_a=af;}
    }else value-=af*y[row];
  }
  if(value<0&&negative>=0)y[negative]=value/negative_a;
  if(value>0&&positive>=0)y[positive]=value/positive_a;
  x[f]=x[b];z[f]=0;
}
} // namespace structural_dual_detail

std::size_t StructuralDualRecovery::device_bytes() const {
  auto matrix_bytes=[](const DeviceCsrMatrix& a) {
    return (std::size_t(a.rows)+1)*sizeof(NnzOffset)+std::size_t(a.nnz)*(sizeof(int)+sizeof(double));
  };
  auto group_bytes=[](const DeviceRecoveryGroup& g) {
    std::size_t sum=0;
    for(const auto& v:g.ids)sum+=std::size_t(v.count)*sizeof(int);
    for(const auto& v:g.values)sum+=std::size_t(v.count)*sizeof(double);
    return sum;
  };
  const std::size_t m=model.A.rows,n=model.A.cols;
  std::size_t sum=matrix_bytes(model.A)+matrix_bytes(model.AT)+(2*m+3*n)*sizeof(double);
  sum+=(m+n)*(sizeof(int)+sizeof(std::uint8_t))+std::size_t(row_scale.count)*sizeof(double);
  sum+=group_bytes(pairs)+group_bytes(absolute.splits)+group_bytes(absolute.pairs)+group_bytes(absolute.linked);
  for(const auto& g:absolute.max_slacks)sum+=group_bytes(g);
  for(const auto* a:{&tree_parent,&tree_row,&tree_next,&tree_head})sum+=std::size_t(a->count)*sizeof(int);
  return sum+std::size_t(tree_scale.count)*sizeof(double);
}

StructuralDualRecovery::~StructuralDualRecovery(){
  for(void* p:{static_cast<void*>(model.A.rowPtr),static_cast<void*>(model.A.colVal),static_cast<void*>(model.A.nzVal),
      static_cast<void*>(model.AT.rowPtr),static_cast<void*>(model.AT.colVal),static_cast<void*>(model.AT.nzVal),
      static_cast<void*>(model.c),static_cast<void*>(model.l),static_cast<void*>(model.u),
      static_cast<void*>(model.AL),static_cast<void*>(model.AU),static_cast<void*>(model.AT_slack_after)})cudaFree(p);
}
std::shared_ptr<StructuralDualRecovery> capture_structural_dual(const PresolvePlanGpu& plan,
    const LPInfoGpu& lp,const int* rows,const int* cols){
  using namespace structural_dual_detail;
  if(!(plan.has_piecewise_linear_reformulation_absolute_value_recovery||plan.has_piecewise_linear_reformulation_variable_pair_recovery||plan.has_column_domination_reduction))return {};
  // Descriptor-only legacy tests and old snapshots cannot supply stage data.
  if(!lp.A.rowPtr||!lp.AT.rowPtr||!lp.c||!lp.l||!lp.u||!lp.AL||!lp.AU)return {};
  auto out=std::make_shared<StructuralDualRecovery>();auto& s=*out;
  matrix(s.model.A,lp.A);matrix(s.model.AT,lp.AT);
  int n=lp.A.cols,m=lp.A.rows;
  s.model.c=clone(lp.c,n);s.model.l=clone(lp.l,n);s.model.u=clone(lp.u,n);
  s.model.AL=clone(lp.AL,m);s.model.AU=clone(lp.AU,m);s.model.obj_constant=lp.obj_constant;
  s.rows=copy(rows,m);s.cols=copy(cols,n);
  s.keep_rows=copy(plan.keep_row_mask,m);s.keep_cols=copy(plan.keep_col_mask,n);
  if(plan.has_piecewise_linear_reformulation_absolute_value_recovery)
    s.absolute=plan.piecewise_linear_reformulation_absolute_value_device?copy_absolute(*plan.piecewise_linear_reformulation_absolute_value_device):absolute_group(plan.piecewise_linear_reformulation_absolute_value_recovery);
  if(plan.has_piecewise_linear_reformulation_variable_pair_recovery){
    s.pairs=plan.piecewise_linear_reformulation_variable_pair_device?copy_group(*plan.piecewise_linear_reformulation_variable_pair_device):pair_group(plan.piecewise_linear_reformulation_variable_pair_recovery);
    const auto& p=s.pairs;int count=p.ids[0].count;
    auto nodes=buffer<int>(n),roots=buffer<int>(n);auto cursor=buffer<NnzOffset>(count);
    check(cudaMemset(nodes.data,-1,std::size_t(n)*sizeof(int)));
    identity<<<(n+255)/256,256>>>(roots.data,n);
    pair_maps<<<(count+255)/256,256>>>(nodes.data,roots.data,p.ids[0].data,p.ids[1].data,p.ids[2].data,p.ids[3].data,count);
    s.row_scale=buffer<double>(m);
    pair_row_scales<<<(m+255)/256,256>>>(s.row_scale.data,lp.A,plan.new_A,roots.data,m);
    s.tree_parent=buffer<int>(count);s.tree_row=buffer<int>(count);s.tree_next=buffer<int>(count);s.tree_head=buffer<int>(count);s.tree_scale=buffer<double>(count);
    init_parent<<<(count+255)/256,256>>>(s.tree_parent.data,count);
    check(cudaMemset(s.tree_row.data,-1,std::size_t(count)*sizeof(int)));
    check(cudaMemset(s.tree_head.data,-1,std::size_t(count)*sizeof(int)));
    check(cudaMemset(s.tree_next.data,-1,std::size_t(count)*sizeof(int)));
    check(cudaMemset(s.tree_scale.data,0,std::size_t(count)*sizeof(double)));
    build_pair_trees<<<(count+255)/256,256>>>(s.model,nodes.data,p.ids[0].data,p.ids[1].data,p.ids[2].data,count,
        s.tree_parent.data,s.tree_row.data,s.tree_next.data,s.tree_head.data,s.tree_scale.data,cursor.data);
    check(cudaGetLastError());
  }
  check(cudaGetLastError());return out;
}
std::shared_ptr<StructuralDualRecovery> capture_structural_dual(const PresolvePlanGpu& p,
    const LPInfoGpu& lp,const std::vector<int>& rows,const std::vector<int>& cols){
  using namespace structural_dual_detail;
  if(!(p.has_piecewise_linear_reformulation_absolute_value_recovery||p.has_piecewise_linear_reformulation_variable_pair_recovery||p.has_column_domination_reduction))return {};
  auto r=upload(rows),c=upload(cols);return capture_structural_dual(p,lp,r.data,c.data);
}
void restore_structural_joint(double* x,double* y,double* z,std::uint8_t* mask,
    PrimalRecoveryKind kind,const StructuralDualRecovery& s){
  using namespace structural_dual_detail;
  int m=s.model.A.rows,n=s.model.A.cols;
  auto sx=buffer<double>(n),sy=buffer<double>(m),sz=buffer<double>(n);
  gather<<<(n+255)/256,256>>>(sx.data,x,s.cols.data,n);gather<<<(n+255)/256,256>>>(sz.data,z,s.cols.data,n);
  gather<<<(m+255)/256,256>>>(sy.data,y,s.rows.data,m);
  if(kind==PrimalRecoveryKind::ColumnDomination){
    auto delta=buffer<double>(m);
    clip_covering<<<(m+255)/256,256>>>(sy.data,delta.data,m);
    restore_covering<<<(n+255)/256,256>>>(s.model,sx.data,sy.data,delta.data,sz.data,s.keep_cols.data);
  }else if(kind==PrimalRecoveryKind::PiecewiseLinearReformulationVariablePair){
    const auto& p=s.pairs;int count=p.ids[0].count;auto h=buffer<double>(count);
    undo_row_scales<<<(m+255)/256,256>>>(sy.data,s.row_scale.data,s.keep_rows.data,m);
    restore_pair_components<<<(count+255)/256,256>>>(s.model,sx.data,sy.data,sz.data,
        p.ids[0].data,p.ids[1].data,p.ids[2].data,p.values[0].data,p.values[1].data,s.keep_cols.data,
        s.tree_parent.data,s.tree_row.data,s.tree_next.data,s.tree_head.data,s.tree_scale.data,h.data,count);
    check(cudaGetLastError());
  }else {
    const auto& a=s.absolute;int count=a.splits.ids[0].count;
    if(count)restore_split<<<(count+255)/256,256>>>(s.model,sx.data,sy.data,sz.data,
        a.splits.ids[0].data,a.splits.ids[1].data,a.splits.values[0].data,s.keep_rows.data,count);
    count=a.linked.ids[0].count;
    if(count)restore_link<<<(count+255)/256,256>>>(s.model,sx.data,sy.data,sz.data,
        a.linked.ids[0].data,a.linked.ids[1].data,a.linked.values[0].data,count);
    for(const auto& g:a.max_slacks)restore_maximum<<<1,1>>>(s.model,sx.data,sy.data,sz.data,g.ids[0].data,g.ids[1].data,g.values[0].data,g.ids[1].count);
    count=a.pairs.ids[0].count;
    if(count)restore_outer<<<(count+255)/256,256>>>(s.model,sx.data,sy.data,sz.data,a.pairs.ids[0].data,a.pairs.ids[1].data,s.keep_rows.data,count);
  }
  scatter<<<(n+255)/256,256>>>(x,sx.data,s.cols.data,n);scatter<<<(n+255)/256,256>>>(z,sz.data,s.cols.data,n);
  scatter<<<(m+255)/256,256>>>(y,sy.data,s.rows.data,m);
  if(mask)protect<<<(n+255)/256,256>>>(mask,s.cols.data,n);
  check(cudaGetLastError());
}
} // namespace gpu_preprocessor::lp::presolve
