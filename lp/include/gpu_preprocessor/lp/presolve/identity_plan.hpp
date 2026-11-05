// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"

namespace gpu_preprocessor::lp::presolve::detail {

// Nonzero mask entries mean keep. No identity permutation is materialized.
bool plan_keeps_all_indices(const PresolvePlanGpu& plan, std::int32_t m, std::int32_t n);

// Caller owns lp's vectors and has already committed recovery using the old lp.
// Missing plan vectors leave their destinations untouched. This performs only
// bit-preserving assignments, never arithmetic or a change to either matrix.
void commit_identity_plan_vectors(const PresolvePlanGpu& plan, LPInfoGpu& lp);

}  // namespace gpu_preprocessor::lp::presolve::detail
