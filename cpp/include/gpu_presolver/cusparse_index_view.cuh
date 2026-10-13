#pragma once
#include "gpu_presolver/offset.hpp"
#include <cusparse.h>
#include <cuda_runtime.h>
#include <stdexcept>
// CUDA 12.8 rejects mixed CSR index widths. Presolve/custom kernels keep 64/32
// storage; only the library descriptor receives a same-width compatibility view.
// A 64-bit descriptor does not remove CUDA 12.8 operation-specific nnz limits;
// callers must check SpMV/SpMM return codes (large SpMV can return INTERNAL_ERROR).
template<class D,class S> __global__ void convert_sparse_indices(D* dst,const S* src,NnzOffset n) {
    for(NnzOffset p=NnzOffset(blockIdx.x)*blockDim.x+threadIdx.x;p<n;p+=NnzOffset(gridDim.x)*blockDim.x) dst[p]=static_cast<D>(src[p]);
}
inline void create_cusparse_index_view(int rows, int columns, NnzOffset nnz, NnzOffset* row_ptr, int* col_idx, double* values, cusparseSpMatDescr_t* descr, void** owned) {
#if !GPUPRESOLVER_USE_INT64_NNZ
    // Native int32 mode: no compatibility buffer or conversion kernel.
    *owned=nullptr;
    const auto status=cusparseCreateCsr(descr,rows,columns,nnz,row_ptr,col_idx,values,
        CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
    if(status!=CUSPARSE_STATUS_SUCCESS)throw std::runtime_error("cannot create cuSPARSE CSR descriptor");
#else
    void* ptr=row_ptr;void* cols=col_idx;cusparseIndexType_t type;
    const bool small=nnz<=INT32_MAX;
    const NnzOffset count=small ? NnzOffset(rows)+1 : nnz;
    auto e=cudaMalloc(owned,checked_nnz_bytes(count,small?sizeof(int):sizeof(NnzOffset)));
    if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
    if(small) {
        convert_sparse_indices<<<4096,256>>>(static_cast<int*>(*owned),row_ptr,count);
        ptr=*owned;type=CUSPARSE_INDEX_32I;
    } else {
        convert_sparse_indices<<<4096,256>>>(static_cast<NnzOffset*>(*owned),col_idx,count);
        cols=*owned;type=CUSPARSE_INDEX_64I;
    }
    e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
    const auto status=cusparseCreateCsr(descr,rows,columns,nnz,ptr,cols,values,type,type,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
    if(status!=CUSPARSE_STATUS_SUCCESS)throw std::runtime_error("cannot create cuSPARSE index view");
#endif
}
