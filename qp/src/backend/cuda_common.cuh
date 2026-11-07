#ifndef CUDA_COMMON_CUH
#define CUDA_COMMON_CUH
#include "internal.h"
#include <cuda_runtime.h>
#include <new>
#include <utility>
/* One scope per synchronous presolve operation. Thread-local so independent
 * calls/devices cannot charge another state's traffic. */
TransferInfo *&cuda_transfer_counter();
struct TransferScope {
    TransferInfo *previous;
    explicit TransferScope(TransferInfo *p) : previous(cuda_transfer_counter()) {
        cuda_transfer_counter() = p;
    }
    ~TransferScope() {
        cuda_transfer_counter() = previous;
    }
};
inline cudaError_t cuda_memcpy(void *to, const void *from, size_t bytes, cudaMemcpyKind kind) {
    if (!bytes)
        return cudaSuccess;
    cudaError_t code = cudaMemcpy(to, from, bytes, kind);
    TransferInfo *p = cuda_transfer_counter();
    if (code == cudaSuccess && p && bytes) {
        if (kind == cudaMemcpyHostToDevice) {
            p->host_to_device_bytes += bytes;
            ++p->host_to_device_calls;
        } else if (kind == cudaMemcpyDeviceToHost) {
            p->device_to_host_bytes += bytes;
            ++p->device_to_host_calls;
        }
    }
    return code;
}
#define GPU_CHECK(call)                                                                            \
    do {                                                                                           \
        if ((call) != cudaSuccess)                                                                 \
            return CUDA_ERROR;                                                                 \
    } while (0)
struct DeviceGuard {
    int previous = -1;
    cudaError_t status;
    explicit DeviceGuard(int device) {
        status = cudaGetDevice(&previous);
        if (status == cudaSuccess)
            status = cudaSetDevice(device);
    }
    ~DeviceGuard() {
        if (previous >= 0)
            cudaSetDevice(previous);
    }
};
template <class T> struct Buffer {
    T *data = nullptr;
    size_t capacity = 0;
    Buffer() = default;
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    ~Buffer() {
        cudaFree(data);
    }
    void swap(Buffer &other) {
        std::swap(data, other.data);
        std::swap(capacity, other.capacity);
    }
    cudaError_t reserve(size_t n) {
        if (n <= capacity)
            return cudaSuccess;
        if (n > SIZE_MAX / sizeof(T))
            return cudaErrorMemoryAllocation;
        T *p = nullptr;
        cudaError_t e = cudaMalloc((void **)&p, n * sizeof(T));
        if (e != cudaSuccess)
            return e;
        cudaFree(data);
        data = p;
        capacity = n;
        return cudaSuccess;
    }
    cudaError_t upload(const T *p, size_t n) {
        cudaError_t e = reserve(n ? n : 1);
        return e != cudaSuccess || !n
                   ? e
                   : cuda_memcpy(data, p, n * sizeof(T), cudaMemcpyHostToDevice);
    }
    cudaError_t download(T *p, size_t n) const {
        return n ? cuda_memcpy(p, data, n * sizeof(T), cudaMemcpyDeviceToHost) : cudaSuccess;
    }
};
struct DeviceMatrix {
    Buffer<int> ptr, col;
    Buffer<double> val;
    cudaError_t upload(const Matrix &a) {
        cudaError_t e = ptr.upload(a.row_ptr, (size_t)a.rows + 1);
        if (e == cudaSuccess)
            e = col.upload(a.col_ind, a.nnz);
        if (e == cudaSuccess)
            e = val.upload(a.values, a.nnz);
        return e;
    }
};
struct View {
    int rows, cols;
    const int *ptr, *col;
    const double *val;
    __device__ double get(int i, int j) const {
        int l = ptr[i], r = ptr[i + 1];
        while (l < r) {
            int m = l + (r - l) / 2;
            if (col[m] < j)
                l = m + 1;
            else
                r = m;
        }
        return l < ptr[i + 1] && col[l] == j ? val[l] : 0.;
    }
};
inline View view(const DeviceMatrix &a, int m, int n) {
    return View{m, n, a.ptr.data, a.col.data, a.val.data};
}
__device__ inline bool gpu_product(const double *f, int count, double &value) {
    double mantissa = 1.;
    int exponent = 0;
    for (int k = 0; k < count; ++k) {
        if (!isfinite(f[k]))
            return false;
        if (f[k] == 0.) {
            value = 0.;
            return true;
        }
        int e;
        mantissa *= frexp(f[k], &e);
        exponent += e;
        mantissa = frexp(mantissa, &e);
        exponent += e;
    }
    value = ldexp(mantissa, exponent);
    return isfinite(value) && value != 0.;
}
__device__ inline double gpu_product2(double a, double b, int *error) {
    double f[] = {a, b}, value;
    if (!gpu_product(f, 2, value)) {
        atomicExch(error, (int)NUMERICAL_ERROR);
        return 0.;
    }
    return value;
}
struct Sum {
    double sum = 0., correction = 0.;
    __device__ void add(double v) {
        double next = sum + v;
        correction += fabs(sum) >= fabs(v) ? (sum - next) + v : (v - next) + sum;
        sum = next;
    }
    __device__ double value(int *error) const {
        double v = sum + correction;
        if (!isfinite(v))
            atomicExch(error, (int)NUMERICAL_ERROR);
        return v;
    }
};
#endif
