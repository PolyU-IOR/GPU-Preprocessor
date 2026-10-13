#pragma once
#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>

// The CMake target propagates this ABI setting to all consumers. Header-only
// or manually compiled consumers default to int32 and must match their library.
#ifndef GPUPRESOLVER_USE_INT64_NNZ
#define GPUPRESOLVER_USE_INT64_NNZ 0
#endif
#if GPUPRESOLVER_USE_INT64_NNZ
using NnzOffset = std::int64_t;
#else
using NnzOffset = std::int32_t;
#endif
inline NnzOffset checked_nnz_offset(std::int64_t count) {
    if (count < 0 || count > std::numeric_limits<NnzOffset>::max())
        throw std::length_error("nnz exceeds this build's index range; rebuild with GPUPRESOLVER_USE_INT64_NNZ=ON for large models");
    return static_cast<NnzOffset>(count);
}
inline std::size_t checked_nnz_bytes(NnzOffset count, std::size_t element_size) {
    if (count < 0 || static_cast<std::uint64_t>(count) >
        std::numeric_limits<std::size_t>::max() / element_size)
        throw std::length_error("sparse allocation size overflow");
    return static_cast<std::size_t>(count) * element_size;
}
#if defined(__CUDACC__) && GPUPRESOLVER_USE_INT64_NNZ
#include <cuda_runtime.h>
__device__ inline NnzOffset atomicAdd(NnzOffset* address, NnzOffset value) {
    return static_cast<NnzOffset>(atomicAdd(
        reinterpret_cast<unsigned long long*>(address),
        static_cast<unsigned long long>(value)));
}
#endif
