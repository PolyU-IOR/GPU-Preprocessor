#ifndef CUDA_SOC_DOWNLOAD_CUH
#define CUDA_SOC_DOWNLOAD_CUH
#include "backend/cuda_common.cuh"
/* Pin large malloc-owned outputs for DMA without a CPU page-touch worker pool.
 * Registration is temporary: callers retain ordinary free() ownership. If host
 * registration is unavailable, CUDA's synchronous pageable transfer still works. */
static cudaError_t download_host_output(void *host, const void *device, size_t bytes) {
    bool pinned = false;
    if (bytes >= (size_t)64 * 1024 * 1024) {
        cudaError_t code = cudaHostRegister(host, bytes, cudaHostRegisterDefault);
        pinned = code == cudaSuccess;
        if (!pinned) cudaGetLastError();
    }
    cudaError_t code = cuda_memcpy(host, device, bytes, cudaMemcpyDeviceToHost);
    if (pinned) {
        cudaError_t release = cudaHostUnregister(host);
        if (code == cudaSuccess) code = release;
    }
    return code;
}
#endif
