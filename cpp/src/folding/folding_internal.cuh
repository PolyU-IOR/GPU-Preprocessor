#pragma once

#include "gpu_presolver/detail/runtime_options.hpp"

#include "gpu_presolver/folding/folding_structs.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::folding {

// The validated default pipeline is the only production implementation.
// Detailed profiling restores stage barriers and synchronous count reads.
inline bool folding_phase_sync_required() {
  return runtime::options().profile_folding;
}

inline bool folding_batch_refinement() {
  return !folding_phase_sync_required();
}

struct ColorPair {
  std::int32_t color_id = 0;
  float color_sig = 0.0f;
};

struct FoldingScratchBlock {
  char* pointer = nullptr;
  std::size_t bytes = 0;
  bool in_use = false;
};

struct FoldingWorkspace {
  std::vector<FoldingScratchBlock> color_scratch;
  std::int32_t num_row = 0;
  std::int32_t num_col = 0;
  std::int32_t num_row_color = 1;
  std::int32_t num_col_color = 1;
  ColorPair* row_color = nullptr;
  ColorPair* col_color = nullptr;
  std::int32_t* row_color_id = nullptr;
  std::int32_t* col_color_id = nullptr;
  double* row_color_sig = nullptr;
  double* col_color_sig = nullptr;
  std::int32_t* row_color_start = nullptr;
  std::int32_t* col_color_start = nullptr;
  std::int32_t* row_perm = nullptr;
  std::int32_t* col_perm = nullptr;
  std::int32_t* device_color_counts = nullptr;
  std::int32_t* row_color_count = nullptr;
  std::int32_t* col_color_count = nullptr;
  std::int32_t row_count_capacity = 0;
  std::int32_t col_count_capacity = 0;
};

void init_workspace(FoldingWorkspace& workspace, std::int32_t num_row, std::int32_t num_col);
void free_workspace(FoldingWorkspace& workspace);
std::int32_t folding_update_row_color_id(FoldingWorkspace& workspace,
                                         const double* color_sig,
                                         double tol);
std::int32_t folding_update_col_color_id(FoldingWorkspace& workspace,
                                         const double* color_sig,
                                         double tol);
void folding_write_row_color_sig(FoldingWorkspace& workspace, std::int32_t round, std::uint64_t side);
void folding_write_col_color_sig(FoldingWorkspace& workspace, std::int32_t round, std::uint64_t side);
std::int32_t folding_update_color_and_sig(FoldingWorkspace& workspace, bool row,
                                         const double* sig, double tol,
                                         std::int32_t round, std::uint64_t side,
                                         bool defer_count, bool write_sig);
void folding_read_color_counts(FoldingWorkspace& workspace);

bool refine_color(const presolve::LPInfoGpu& device_model,
                  FoldingWorkspace& workspace,
                  double tolerance,
                  bool verbose,
                  FoldingRefinementProfile* profile);

presolve::LPInfoGpu reduce_size(FoldingWorkspace& workspace,
                                const presolve::LPInfoGpu& model,
                                double tolerance,
                                FoldingReduceProfile* profile);

FoldingMapDevice build_device_map(FoldingWorkspace& workspace,
                                  FoldingMapProfile* profile);

void free_folding_map_device(FoldingMapDevice& map);

UnfoldedSolutionHost unfold_solution_device_to_host(const FoldingMapDevice& map,
                                                    const double* x_red_device,
                                                    const double* y_red_device,
                                                    const double* z_red_device);

void free_lp_device(presolve::LPInfoGpu& lp);

}  // namespace gpu_presolver::folding
