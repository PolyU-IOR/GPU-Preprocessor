#include "cuda_common.cuh"

TransferInfo *&cuda_transfer_counter() {
    static thread_local TransferInfo *value = nullptr;
    return value;
}

__global__ static void matvec(int rows, const int *ptr, const int *col, const double *val,
                              const double *x, double *y) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= rows)
        return;
    double sum = 0.;
    for (int k = ptr[i]; k < ptr[i + 1]; k++)
        sum += val[k] * x[col[k]];
    y[i] = sum;
}
__global__ static void transpose_product(int rows, const int *ptr, const int *col,
                                         const double *val, const double *x, double *y) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= rows)
        return;
    for (int k = ptr[i]; k < ptr[i + 1]; k++)
        atomicAdd(&y[col[k]], val[k] * x[i]);
}
extern "C" int cuda_available(int device) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || device < 0 || device >= count)
        return 0;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, device) != cudaSuccess)
        return 0;
    return p.major >= 6; // double-precision atomicAdd in original-space KKT products
}
extern "C" Status cuda_products(const Problem *p, int device, const double *x,
                                       const double *y, double *Ax, double *Qx, double *ATy) {
    DeviceGuard guard(device);
    GPU_CHECK(guard.status);
    DeviceMatrix A, Q;
    Buffer<double> dx, dy, ax, qx, aty;
    int m = p->A.rows, n = p->A.cols;
    GPU_CHECK(A.upload(p->A));
    GPU_CHECK(Q.upload(p->Q));
    GPU_CHECK(dx.upload(x, n));
    GPU_CHECK(dy.upload(y, m));
    GPU_CHECK(ax.reserve(m ? m : 1));
    GPU_CHECK(qx.reserve(n ? n : 1));
    GPU_CHECK(aty.reserve(n ? n : 1));
    GPU_CHECK(cudaMemset(aty.data, 0, (size_t)(n ? n : 1) * sizeof(double)));
    if (m) {
        matvec<<<((unsigned)m + 255u) / 256u, 256>>>(m, A.ptr.data, A.col.data, A.val.data, dx.data,
                                                     ax.data);
        transpose_product<<<((unsigned)m + 255u) / 256u, 256>>>(m, A.ptr.data, A.col.data,
                                                                A.val.data, dy.data, aty.data);
    }
    if (n)
        matvec<<<((unsigned)n + 255u) / 256u, 256>>>(n, Q.ptr.data, Q.col.data, Q.val.data, dx.data,
                                                     qx.data);
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
    GPU_CHECK(ax.download(Ax, m));
    GPU_CHECK(qx.download(Qx, n));
    GPU_CHECK(aty.download(ATy, n));
    return OK;
}
