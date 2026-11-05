// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once
#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"

namespace gpu_preprocessor::lp::diagnostics {
// Optional synchronous diagnostic callback. Unset during production and timed
// replay. The callback borrows stage data and must not modify it or retain it.
using PostsolveObserver = void (*)(const char*, const presolve::LPInfoGpu*,
                                  const double*, const double*, const double*, void*);
inline thread_local PostsolveObserver postsolve_observer = nullptr;
inline thread_local void* postsolve_observer_context = nullptr;
inline void observe_postsolve(const char* stage, const presolve::LPInfoGpu* model,
                              const double* x, const double* y, const double* z) {
  if (postsolve_observer) postsolve_observer(stage, model, x, y, z, postsolve_observer_context);
}
}  // namespace gpu_preprocessor::lp::diagnostics
