// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "gpu_preprocessor/lp.hpp"
#include "../tools/gpu_preprocessor_tool_common.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <type_traits>

static_assert(!std::is_copy_constructible_v<gpu_preprocessor::lp::tools::DeviceLpOwner>);
static_assert(std::is_nothrow_move_constructible_v<gpu_preprocessor::lp::tools::DeviceLpOwner>);

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

int main() {
  // Bound propagation tightens both upper bounds without deleting rows/columns.
  std::istringstream input(
      "NAME BOUNDS\nROWS\n N OBJ\n E EQ\nCOLUMNS\n"
      " X OBJ 1 EQ 1\n Y OBJ 2 EQ 1\nRHS\n R EQ 2\n"
      "BOUNDS\n UP B X 10\n UP B Y 10\nENDATA\n");
  auto model = cpu_presolve::read_mps(input);
  auto owned = gpu_preprocessor::lp::tools::upload_lp(model.lp);
  auto moved = std::move(owned);
  require(owned.lp.A.rowPtr == nullptr, "upload owner did not transfer ownership");
  gpu_preprocessor::lp::Parameters p;
  p.enable_folding = false;
  p.enable_presolve = false;
  auto identity = gpu_preprocessor::lp::preprocess(moved.lp, p);
  require(identity.preprocessing_seconds == 0.0, "disabled stages must report zero");
  require(!identity.presolve.has_change, "identity incorrectly reports a change");
  require(identity.presolve.reduced_lp.A.rowPtr == moved.lp.A.rowPtr, "identity must borrow input");
  gpu_preprocessor::lp::free_summary(identity);
  require(moved.lp.A.rowPtr != nullptr, "free_summary freed caller-owned input");

  p.enable_presolve = true;
  p.enable_startup = false;
  p.enable_structure_specific_rules = false;
  p.enable_infeasible_and_fixed_variables = false;
  p.enable_empty_rows = p.enable_empty_columns = false;
  p.enable_singleton_rows = p.enable_singleton_columns = false;
  p.enable_infeasible_and_redundant_rows = false;
  p.enable_doubleton_equations = p.enable_dual_fix = false;
  p.enable_parallel_rows = p.enable_parallel_columns = false;
  auto result = gpu_preprocessor::lp::preprocess(moved.lp, p);
  require(result.presolve.has_change, "bound-only change was not reported");
  require(result.presolve.reduced_rows == 1 && result.presolve.reduced_cols == 2,
          "bound-only test unexpectedly changed dimensions");
  const auto bounds = gpu_preprocessor::lp::tools::copy_to_host(result.presolve.reduced_lp.u, 2);
  require(bounds[0] < 2.0001 && bounds[1] < 2.0001, "upper bounds were not tightened");
  const auto original = gpu_preprocessor::lp::tools::copy_to_host(moved.lp.u, 2);
  require(original[0] == 10.0 && original[1] == 10.0, "input was modified");
  require(result.preprocessing_seconds >= result.presolve.elapsed_seconds,
          "core timer exceeds preprocessing wall time");
  require(result.preprocessing_seconds >= result.workspace_setup_seconds + result.workspace_release_seconds,
          "workspace timers exceed preprocessing wall time");
  // The reduced optimum is x=(2,0), y=1, z=(0,1). Check recovery through the
  // public host/device APIs, original KKT, repeated replay and borrowed inputs.
  namespace tools = gpu_preprocessor::lp::tools;
  auto* x = tools::copy_to_device(std::vector<double>{2.0, 0.0});
  auto* y = tools::copy_to_device(std::vector<double>{1.0});
  auto* z = tools::copy_to_device(std::vector<double>{0.0, 1.0});
  const auto host = gpu_preprocessor::lp::postprocess(x, y, z, result, moved.lp);
  const auto device = gpu_preprocessor::lp::postprocess_device(x, y, z, result, moved.lp);
  require(host.x == tools::copy_to_host(device.x_org, device.n0) &&
          host.y == tools::copy_to_host(device.y_org, device.m0) &&
          host.z == tools::copy_to_host(device.z_org, device.n0), "host/device recovery differs");
  require(host.x.size() == 2 && host.y.size() == 1 && host.z.size() == 2,
          "recovered point has wrong dimensions");
  require(std::abs(host.x[0] + host.x[1] - 2.0) < 1e-12, "original row infeasible");
  for (int j = 0; j < 2; ++j) {
    require(host.x[j] >= 0.0 && host.x[j] <= 10.0, "original bound infeasible");
    require(std::abs((j + 1.0) - host.y[0] - host.z[j]) < 1e-12, "original stationarity failed");
    require(host.z[j] >= 0.0 && std::abs(host.x[j] * host.z[j]) < 1e-12,
            "original complementarity failed");
  }
  require(tools::copy_to_host(x, 2) == std::vector<double>({2.0, 0.0}) &&
          tools::copy_to_host(y, 1) == std::vector<double>({1.0}) &&
          tools::copy_to_host(z, 2) == std::vector<double>({0.0, 1.0}), "recovery changed borrowed inputs");
  cudaFree(device.x_org); cudaFree(device.y_org); cudaFree(device.z_org);
  cudaFree(x); cudaFree(y); cudaFree(z);
  gpu_preprocessor::lp::free_summary(result);
  gpu_preprocessor::lp::free_summary(result);
}
