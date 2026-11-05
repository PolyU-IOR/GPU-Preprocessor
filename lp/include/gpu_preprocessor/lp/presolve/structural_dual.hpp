// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once
#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"

namespace gpu_preprocessor::lp::presolve {
// Immutable sparse stage data. Coordinates are local to this checkpoint;
// maps connect them to the outer recovery vectors. No full solution is saved.
struct StructuralDualRecovery {
  LPInfoGpu model;
  RecoveryArray<int> rows, cols;
  RecoveryArray<std::uint8_t> keep_rows, keep_cols;
  RecoveryArray<double> row_scale;
  DeviceRecoveryGroup pairs;
  DevicePiecewiseLinearReformulationAbsoluteValueStep absolute;
  RecoveryArray<int> tree_parent, tree_row, tree_next, tree_head;
  RecoveryArray<double> tree_scale;
  StructuralDualRecovery() = default;
  StructuralDualRecovery(const StructuralDualRecovery&) = delete;
  StructuralDualRecovery& operator=(const StructuralDualRecovery&) = delete;
  ~StructuralDualRecovery();
  std::size_t device_bytes() const;
};
std::shared_ptr<StructuralDualRecovery> capture_structural_dual(
    const PresolvePlanGpu&, const LPInfoGpu&, const int* rows, const int* cols);
std::shared_ptr<StructuralDualRecovery> capture_structural_dual(
    const PresolvePlanGpu&, const LPInfoGpu&, const std::vector<int>& rows,
    const std::vector<int>& cols);
void restore_structural_joint(double* x, double* y, double* z,
    std::uint8_t* protected_cols, PrimalRecoveryKind, const StructuralDualRecovery&);
} // namespace gpu_preprocessor::lp::presolve
