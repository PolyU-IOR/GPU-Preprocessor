#include "gpu_presolver/presolve/transpose64.hpp"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>

namespace gpu_presolver::presolve {
namespace {
void ck(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void ck(cusparseStatus_t e) { if(e != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("64-bit transpose cuSPARSE error " + std::to_string(e)); }
struct Buffer {
    void* p=nullptr; size_t capacity=0;
    void reserve(size_t n) { if(n>capacity) { void* q=nullptr; ck(cudaMallocAsync(&q,n,0)); ck(cudaFreeAsync(p,0)); p=q; capacity=n; } }
    ~Buffer() { if(p) cudaFree(p); }
};
struct Workspace {
    int device=-1; cusparseHandle_t handle=nullptr;
    Buffer rows, cols, scratch;
    Workspace() { ck(cudaGetDevice(&device)); ck(cusparseCreate(&handle)); }
    ~Workspace() { int old=-1; cudaGetDevice(&old); cudaSetDevice(device);
        // Free while the owning device is selected.
        cudaFree(rows.p); rows.p=nullptr; cudaFree(cols.p); cols.p=nullptr;
        cudaFree(scratch.p); scratch.p=nullptr; cusparseDestroy(handle); cudaSetDevice(old); }
};
__global__ void narrow_ptr(const NnzOffset* src,int* dst,NnzOffset n,NnzOffset base) {
    for(std::int64_t i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::int64_t(gridDim.x)*blockDim.x) dst[i]=int(src[i]-base);
}
__global__ void widen_ptr(const int* src,NnzOffset* dst,NnzOffset n) {
    for(std::int64_t i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::int64_t(gridDim.x)*blockDim.x) dst[i]=src[i];
}
__global__ void histogram(const int* cols,NnzOffset nnz,NnzOffset* ptr) {
    for(std::int64_t p=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;p<nnz;p+=std::int64_t(gridDim.x)*blockDim.x) atomicAdd(ptr+cols[p]+1,NnzOffset(1));
}
__global__ void append_chunk(const int* ptr,const int* rows,const double* vals,
    NnzOffset* next,int* out_rows,double* out_vals,int n,int first_row) {
    for(std::int64_t col=blockIdx.x;col<n;col+=gridDim.x) {
        NnzOffset base=next[col];
        for(std::int64_t p=std::int64_t(ptr[col])+threadIdx.x;p<ptr[col+1];p+=blockDim.x) {
            NnzOffset q=base+p-ptr[col]; out_rows[q]=rows[p]+first_row;out_vals[q]=vals[p];
        }
        __syncthreads();
        if(threadIdx.x==0) next[col]=base+NnzOffset(ptr[col+1])-ptr[col];
        __syncthreads();
    }
}
int blocks(std::int64_t n) { return int(std::min<std::int64_t>(65535,(n+255)/256)); }
void transpose_chunk(Workspace& w,int m,int n,NnzOffset nnz,const NnzOffset* ptr,NnzOffset base,
 const int* cols,const double* vals,int* out_rows,double* out_vals) {
    if(nnz>INT32_MAX) throw std::length_error("transpose chunk exceeds int32");
    w.rows.reserve(checked_nnz_bytes(NnzOffset(m)+1,sizeof(int)));
    w.cols.reserve(checked_nnz_bytes(NnzOffset(n)+1,sizeof(int)));
    auto* rp=static_cast<int*>(w.rows.p); auto* cp=static_cast<int*>(w.cols.p);
    narrow_ptr<<<blocks(NnzOffset(m)+1),256>>>(ptr,rp,NnzOffset(m)+1,base); ck(cudaGetLastError());
    size_t bytes=0;
    ck(cusparseCsr2cscEx2_bufferSize(w.handle,m,n,int(nnz),vals,rp,cols,out_vals,cp,out_rows,
       CUDA_R_64F,CUSPARSE_ACTION_NUMERIC,CUSPARSE_INDEX_BASE_ZERO,CUSPARSE_CSR2CSC_ALG1,&bytes));
    w.scratch.reserve(bytes);
    ck(cusparseCsr2cscEx2(w.handle,m,n,int(nnz),vals,rp,cols,out_vals,cp,out_rows,
       CUDA_R_64F,CUSPARSE_ACTION_NUMERIC,CUSPARSE_INDEX_BASE_ZERO,CUSPARSE_CSR2CSC_ALG1,w.scratch.p));
}
}
DeviceCsrMatrix transpose_csr64(const DeviceCsrMatrix& a) {
    static thread_local std::unique_ptr<Workspace> cached;
    int device; ck(cudaGetDevice(&device));
    if(!cached || cached->device!=device) cached=std::make_unique<Workspace>();
    Workspace& w=*cached;
    DeviceCsrMatrix out; out.rows=a.cols;out.cols=a.rows;out.nnz=a.nnz;
    try {
        ck(cudaMallocAsync(&out.rowPtr,checked_nnz_bytes(NnzOffset(out.rows)+1,sizeof(NnzOffset)),0));
        if(a.nnz==0) { ck(cudaMemsetAsync(out.rowPtr,0,checked_nnz_bytes(NnzOffset(out.rows)+1,sizeof(NnzOffset)),0));return out; }
        ck(cudaMallocAsync(&out.colVal,checked_nnz_bytes(a.nnz,sizeof(int)),0));
        ck(cudaMallocAsync(&out.nzVal,checked_nnz_bytes(a.nnz,sizeof(double)),0));
        // Keep library calls far from its signed-count boundary, even below INT_MAX.
        constexpr NnzOffset chunk_limit= NnzOffset(1)<<27;
        if(a.nnz<=chunk_limit) {
            transpose_chunk(w,a.rows,a.cols,a.nnz,a.rowPtr,0,a.colVal,a.nzVal,out.colVal,out.nzVal);
            widen_ptr<<<blocks(NnzOffset(out.rows)+1),256>>>(static_cast<int*>(w.cols.p),out.rowPtr,NnzOffset(out.rows)+1);
        } else {
            std::vector<NnzOffset> host_ptr(size_t(a.rows)+1);
            ck(cudaMemcpy(host_ptr.data(),a.rowPtr,host_ptr.size()*sizeof(NnzOffset),cudaMemcpyDeviceToHost));
            if(host_ptr.front()!=0 || host_ptr.back()!=a.nnz || !std::is_sorted(host_ptr.begin(),host_ptr.end()))
                throw std::invalid_argument("invalid CSR offsets for transpose");
            ck(cudaMemsetAsync(out.rowPtr,0,checked_nnz_bytes(NnzOffset(out.rows)+1,sizeof(NnzOffset)),0));
            histogram<<<blocks(a.nnz),256>>>(a.colVal,a.nnz,out.rowPtr);
            Buffer scan;size_t bytes=0;
            ck(cub::DeviceScan::InclusiveSum(nullptr,bytes,out.rowPtr,out.rowPtr,NnzOffset(out.rows)+1));
            scan.reserve(bytes); ck(cub::DeviceScan::InclusiveSum(scan.p,bytes,out.rowPtr,out.rowPtr,NnzOffset(out.rows)+1));
            Buffer cursor,chunk_rows,chunk_values;
            cursor.reserve(checked_nnz_bytes(a.cols,sizeof(NnzOffset)));
            ck(cudaMemcpyAsync(cursor.p,out.rowPtr,checked_nnz_bytes(a.cols,sizeof(NnzOffset)),cudaMemcpyDeviceToDevice,0));
            for(int first=0;first<a.rows;) {
                auto it=std::upper_bound(host_ptr.begin()+first+1,host_ptr.end(),std::int64_t(host_ptr[first])+chunk_limit);
                int end=int(it-host_ptr.begin())-1;
                if(end==first) end=first+1;
                const NnzOffset begin=host_ptr[first], count=host_ptr[end]-begin;
                if(count>INT32_MAX) throw std::length_error("single CSR row exceeds int32 dimension bound");
                if(count) {
                    chunk_rows.reserve(checked_nnz_bytes(count,sizeof(int)));
                    chunk_values.reserve(checked_nnz_bytes(count,sizeof(double)));
                    transpose_chunk(w,end-first,a.cols,count,a.rowPtr+first,begin,a.colVal+begin,a.nzVal+begin,
                        static_cast<int*>(chunk_rows.p),static_cast<double*>(chunk_values.p));
                    append_chunk<<<std::min(a.cols,65535),256>>>(static_cast<int*>(w.cols.p),static_cast<int*>(chunk_rows.p),
                        static_cast<double*>(chunk_values.p),static_cast<NnzOffset*>(cursor.p),out.colVal,out.nzVal,a.cols,first);
                }
                first=end;
            }
        }
        ck(cudaGetLastError());return out;
    } catch(...) { cudaFree(out.rowPtr);cudaFree(out.colVal);cudaFree(out.nzVal);throw; }
}
}
