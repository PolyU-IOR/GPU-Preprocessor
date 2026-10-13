#include "gpu_presolver_tool_common.hpp"
#include "gpu_presolver/presolve/device_recovery.hpp"
#include <chrono>
#include <iomanip>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
}

int main(int argc, char** argv) {
  try {
    namespace tools = gpu_presolver::tools;
    const auto options = tools::parse_command_line(argc, argv, 1);
    if (options.help) {
      tools::print_usage(std::cout, "gpu_preprocessor_mps", "<model.mps|->");
      return 0;
    }
    const auto& params = options.params;
    const auto parse_start = Clock::now();
    auto model = tools::read_mps_file(options.positional[0]);
    const double parse_seconds = elapsed(parse_start);
    auto device = tools::upload_lp(model.lp);
    tools::FoldingPipelineOwner pipeline;
    pipeline.summary = gpu_presolver::folding::run_gpu_presolve_with_folding(
        device.lp, params, true, false);
    // Explicit diagnostics are outside preprocessing and core timing.
    const auto diagnostic_start = Clock::now();
    gpu_presolver::presolve::export_recovery_record(pipeline.summary.presolve.record);
    auto& summary = pipeline.summary.presolve;
    auto fixed = summary.record.fixed_idx;
    std::sort(fixed.begin(), fixed.end());
    fixed.erase(std::unique(fixed.begin(), fixed.end()), fixed.end());
    const double diagnostic_seconds = elapsed(diagnostic_start);
    std::cout << std::setprecision(12);
    std::cout << "model " << model.name << '\n';
    std::cout << "status " << (summary.has_infeasible ? "infeasible" : summary.has_unbounded ? "unbounded" : "ok") << '\n';
    std::cout << "changed " << (summary.has_change || pipeline.summary.folding_applied) << '\n';
    std::cout << "original_rows " << device.lp.A.rows << '\n';
    std::cout << "original_cols " << device.lp.A.cols << '\n';
    std::cout << "original_nnz " << device.lp.A.nnz << '\n';
    std::cout << "presolve_enabled " << params.enable_presolve << '\n';
    std::cout << "folding_enabled " << params.enable_folding << '\n';
    std::cout << "structure_specific_enabled " << params.enable_structure_specific_rules << '\n';
    std::cout << "folding_applied " << pipeline.summary.folding_applied << '\n';
    std::cout << "folded_rows " << pipeline.summary.folded_rows << '\n';
    std::cout << "folded_cols " << pipeline.summary.folded_cols << '\n';
    std::cout << "reduced_rows " << summary.reduced_rows << '\n';
    std::cout << "reduced_cols " << summary.reduced_cols << '\n';
    std::cout << "reduced_nnz " << summary.reduced_nnz << '\n';
    std::cout << "fixed_columns " << fixed.size() << '\n';
    std::cout << "objective_shift " << summary.obj_constant_delta << '\n';
    std::cout << "iterations " << summary.iterations << '\n';
    std::cout << "iteration_limit_reached " << summary.stopped_at_iteration_limit << '\n';
    std::cout << "parse_seconds " << parse_seconds << '\n';
    std::cout << "folding_seconds " << pipeline.summary.folding.profile.total_seconds << '\n';
    std::cout << "presolve_seconds " << summary.elapsed_seconds << '\n';
    std::cout << "workspace_setup_seconds " << pipeline.summary.workspace_setup_seconds << '\n';
    std::cout << "workspace_release_seconds " << pipeline.summary.workspace_release_seconds << '\n';
    std::cout << "preprocessing_seconds " << pipeline.summary.preprocessing_seconds << '\n';
    std::cout << "diagnostic_export_seconds " << diagnostic_seconds << '\n';
    return summary.has_infeasible || summary.has_unbounded ? 1 : 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
