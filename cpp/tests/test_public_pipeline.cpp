#include "gpu_preprocessor/gpu_preprocessor.hpp"
#include "../tools/gpu_presolver_tool_common.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <type_traits>

static_assert(!std::is_copy_constructible_v<gpu_presolver::tools::DeviceLpOwner>);
static_assert(std::is_nothrow_move_constructible_v<gpu_presolver::tools::DeviceLpOwner>);

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
  auto owned = gpu_presolver::tools::upload_lp(model.lp);
  auto moved = std::move(owned);
  require(owned.lp.A.rowPtr == nullptr, "upload owner did not transfer ownership");
  gpu_preprocessor::Parameters p;
  p.enable_folding = false;
  p.enable_presolve = false;
  auto identity = gpu_preprocessor::preprocess(moved.lp, p);
  require(identity.preprocessing_seconds == 0.0, "disabled stages must report zero");
  require(!identity.presolve.has_change, "identity incorrectly reports a change");
  require(identity.presolve.reduced_lp.A.rowPtr == moved.lp.A.rowPtr, "identity must borrow input");
  gpu_preprocessor::free_summary(identity);
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
  auto result = gpu_preprocessor::preprocess(moved.lp, p);
  require(result.presolve.has_change, "bound-only change was not reported");
  require(result.presolve.reduced_rows == 1 && result.presolve.reduced_cols == 2,
          "bound-only test unexpectedly changed dimensions");
  const auto bounds = gpu_presolver::tools::copy_to_host(result.presolve.reduced_lp.u, 2);
  require(bounds[0] < 2.0001 && bounds[1] < 2.0001, "upper bounds were not tightened");
  const auto original = gpu_presolver::tools::copy_to_host(moved.lp.u, 2);
  require(original[0] == 10.0 && original[1] == 10.0, "input was modified");
  require(result.preprocessing_seconds >= result.presolve.elapsed_seconds,
          "core timer exceeds preprocessing wall time");
  require(result.preprocessing_seconds >= result.workspace_setup_seconds + result.workspace_release_seconds,
          "workspace timers exceed preprocessing wall time");
  gpu_preprocessor::free_summary(result);
  gpu_preprocessor::free_summary(result);
}
