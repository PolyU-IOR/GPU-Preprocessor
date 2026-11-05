// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/folding/folding.hpp"
#include <memory>
#include <string>

namespace gpu_preprocessor::lp::diagnostics {

struct PostsolveSnapshot {
  folding::FoldingPipelineSummary pipeline;
  presolve::LPInfoGpu original;
  double* x = nullptr;
  double* y = nullptr;
  double* z = nullptr;
  double primal_tolerance = 0;
  unsigned format_version = 7;
  PostsolveSnapshot() = default;
  PostsolveSnapshot(const PostsolveSnapshot&) = delete;
  PostsolveSnapshot& operator=(const PostsolveSnapshot&) = delete;
  ~PostsolveSnapshot();
};

void save_snapshot(const std::string& path, const double* x, const double* y,
                   const double* z, const folding::FoldingPipelineSummary& pipeline,
                   const presolve::LPInfoGpu& original, double primal_tolerance);
std::unique_ptr<PostsolveSnapshot> load_snapshot(const std::string& path);
void save_reference(const std::string& path, const folding::UnfoldedSolutionHost& reference);
folding::UnfoldedSolutionHost load_reference(const std::string& path);
void print_snapshot_statistics(const PostsolveSnapshot& snapshot);

}  // namespace gpu_preprocessor::lp::diagnostics
