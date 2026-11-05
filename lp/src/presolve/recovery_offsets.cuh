// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once
#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"
#include <cuda_runtime.h>

namespace gpu_preprocessor::lp::presolve::detail {
// Partition without truncating record IDs or overflowing count*worker.
inline NnzOffset recovery_partition(NnzOffset count, unsigned worker, unsigned workers) {
  return NnzOffset((std::int64_t(count) / workers) * worker +
                   (std::int64_t(count) % workers) * worker / workers);
}

__device__ inline void simple_pack_row(int *types, unsigned char *modes, NnzOffset *starts, NnzOffset *vstarts,
                            int *indices, double *vals, const NnzOffset *scan, const unsigned char *a,
                            const unsigned char *b, int i, int kind, int ni, int nv,
                            const int *second, const double *lo, const double *hi,
                            const double *extra) {
  NnzOffset begin = i ? scan[i - 1] : 0, end = scan[i];
  for (NnzOffset k = begin; k < end; ++k) {
    auto ip = NnzOffset(ni) * k, vp = NnzOffset(nv) * k;
    modes[k] = unsigned(PostsolveDualMode::Minimal);
    starts[k + 1] = ip + ni;
    vstarts[k + 1] = vp + nv;
    indices[ip] = i;
    if (kind == 0) {
      types[k] = int(PostsolveReductionType::FixedVariable);
      vals[vp] = lo[i];
      vals[vp + 1] = hi[i];
    } else if (kind == 1 || kind == 2) {
      types[k] = int(PostsolveReductionType::DeletedRow);
      vals[vp] = lo[i];
      vals[vp + 1] = hi[i];
      if (kind == 1) {
        indices[ip + 1] = second[i];
        vals[vp + 2] = extra[i];
      }
    } else if (kind == 3) {
      bool lower = a[i] && k == begin;
      types[k] = int(lower ? PostsolveReductionType::RowLowerBoundChange : PostsolveReductionType::RowUpperBoundChange);
      vals[vp] = lo[i];
      vals[vp + 1] = hi[i];
      vals[vp + 2] = lower ? -INFINITY : lo[i];
      vals[vp + 3] = lower ? hi[i] : INFINITY;
    } else {
      types[k] = int(PostsolveReductionType::VariableBoundChange);
      vals[vp] = lo[i];
      vals[vp + 1] = hi[i];
      vals[vp + 2] = a[i] ? -INFINITY : lo[i];
      vals[vp + 3] = b[i] ? INFINITY : hi[i];
    }
  }
}
} // namespace gpu_preprocessor::lp::presolve::detail
