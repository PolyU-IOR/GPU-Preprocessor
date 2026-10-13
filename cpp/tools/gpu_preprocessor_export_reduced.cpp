#include "gpu_presolver/presolve/device_recovery.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver_tool_common.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstdlib>

namespace {

void write_lp_export(const std::filesystem::path& out_dir,
                     const std::string& model_name,
                     const gpu_presolver::folding::FoldingPipelineSummary& pipeline) {
  const auto& summary = pipeline.presolve;
  namespace tools = gpu_presolver::tools;
  const gpu_presolver::presolve::LPInfoGpu& lp = summary.reduced_lp;
  std::filesystem::create_directories(out_dir);

  std::ofstream meta(out_dir / "meta.txt");
  if (!meta.is_open()) {
    throw std::runtime_error("failed to open export meta");
  }
  meta << std::setprecision(17);
  meta << (sizeof(NnzOffset) == 8 ? "format gpu_presolver_reduced_lp_v2\n" : "format gpu_presolver_reduced_lp_v1\n");
  meta << "offset_bits " << sizeof(NnzOffset)*8 << "\n";
  meta << "model " << model_name << "\n";
  meta << "status " << (summary.has_infeasible ? "infeasible" : (summary.has_unbounded ? "unbounded" : "ok")) << "\n";
  meta << "original_rows " << pipeline.original_rows << "\n";
  meta << "original_cols " << pipeline.original_cols << "\n";
  meta << "folding_applied " << pipeline.folding_applied << "\n";
  meta << "folded_rows " << pipeline.folded_rows << "\n";
  meta << "folded_cols " << pipeline.folded_cols << "\n";
  meta << "reduced_rows " << summary.reduced_rows << "\n";
  meta << "reduced_cols " << summary.reduced_cols << "\n";
  meta << "reduced_nnz " << summary.reduced_nnz << "\n";
  meta << "obj_constant " << lp.obj_constant << "\n";
  meta << "objective_shift " << summary.obj_constant_delta << "\n";
  meta << "preprocessing_seconds " << pipeline.preprocessing_seconds << "\n";
  meta << "presolve_seconds " << summary.elapsed_seconds << "\n";
  meta << "folding_seconds " << pipeline.folding.profile.total_seconds << "\n";
  meta << "iterations " << summary.iterations << "\n";

  tools::write_binary_vector(out_dir / (sizeof(NnzOffset) == 8 ? "A_rowPtr_i64.bin" : "A_rowPtr_i32.bin"),
                             lp.A.rows == 0 ? std::vector<NnzOffset>{0} :
                             tools::copy_to_host(lp.A.rowPtr, static_cast<std::size_t>(lp.A.rows) + 1));
  tools::write_binary_vector(out_dir / "A_colVal_i32.bin",
                             tools::copy_to_host(lp.A.colVal, static_cast<std::size_t>(lp.A.nnz)));
  tools::write_binary_vector(out_dir / "A_nzVal_f64.bin",
                             tools::copy_to_host(lp.A.nzVal, static_cast<std::size_t>(lp.A.nnz)));
  tools::write_binary_vector(out_dir / "c_f64.bin",
                             tools::copy_to_host(lp.c, static_cast<std::size_t>(lp.A.cols)));
  tools::write_binary_vector(out_dir / "l_f64.bin",
                             tools::copy_to_host(lp.l, static_cast<std::size_t>(lp.A.cols)));
  tools::write_binary_vector(out_dir / "u_f64.bin",
                             tools::copy_to_host(lp.u, static_cast<std::size_t>(lp.A.cols)));
  tools::write_binary_vector(out_dir / "AL_f64.bin",
                             tools::copy_to_host(lp.AL, static_cast<std::size_t>(lp.A.rows)));
  tools::write_binary_vector(out_dir / "AU_f64.bin",
                             tools::copy_to_host(lp.AU, static_cast<std::size_t>(lp.A.rows)));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = gpu_presolver::tools::parse_command_line(argc, argv, 2);
    if (options.help) {
      gpu_presolver::tools::print_usage(std::cout, "gpu_preprocessor_export_reduced",
                                       "<model.mps|-> <out_dir>");
      return 0;
    }
    const std::string input_path = options.positional[0];
    const std::filesystem::path out_dir = options.positional[1];
    const auto& params = options.params;
    cpu_presolve::MpsModel model = gpu_presolver::tools::read_mps_file(input_path);
    gpu_presolver::tools::DeviceLpOwner device_lp = gpu_presolver::tools::upload_lp(model.lp);

    gpu_presolver::tools::FoldingPipelineOwner pipeline;
    pipeline.summary = gpu_presolver::folding::run_gpu_presolve_with_folding(
        device_lp.lp, params, true, false);
    gpu_presolver::presolve::export_recovery_record(pipeline.summary.presolve.record);
    const auto& summary = pipeline.summary.presolve;

    if (summary.has_infeasible || summary.has_unbounded) {
      std::cout << "status " << (summary.has_infeasible ? "infeasible" : "unbounded") << '\n';
      return 1;
    }
    write_lp_export(out_dir, model.name, pipeline.summary);
    std::cout << "status " << (summary.has_infeasible ? "infeasible" : (summary.has_unbounded ? "unbounded" : "ok")) << "\n";
    std::cout << "reduced_rows " << summary.reduced_rows << "\n";
    std::cout << "reduced_cols " << summary.reduced_cols << "\n";
    std::cout << "reduced_nnz " << summary.reduced_nnz << "\n";
    std::cout << "out_dir " << out_dir.string() << "\n";
    return summary.has_infeasible || summary.has_unbounded ? 1 : 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 2;
  }
}
