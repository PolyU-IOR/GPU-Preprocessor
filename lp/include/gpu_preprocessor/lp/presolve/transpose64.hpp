// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once
#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"
#include <stdexcept>
namespace gpu_preprocessor::lp::presolve {
// Let a presolve caller reclaim idle rule caches and retry after exception
// cleanup. Other errors must retain their original failure behavior.
class TransposeOutOfMemory : public std::runtime_error {
 public:
  TransposeOutOfMemory() : std::runtime_error("out of memory") {}
};
// Sorted CSR transpose. Uses bounded 32-bit library calls with 64-bit global offsets.
DeviceCsrMatrix transpose_csr64(const DeviceCsrMatrix& source);
}
