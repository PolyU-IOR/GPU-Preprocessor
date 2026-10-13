#pragma once

#include "gpu_presolver/offset.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {

enum class AbsoluteValueEpigraphResidualBoundMode : std::uint8_t {
  // Preserve every finite side of the original residual-variable box by
  // adding the exact ranged row l_e <= p - n <= u_e after the split.
  PreserveFinite = 0,
  // Apply the split only when the residual variable is genuinely free.
  // This is useful as a conservative diagnostic mode.
  StrictFreeOnly = 1,
  // Compatibility-only mode for the historical heuristic that treated a
  // sufficiently wide finite box as free.  This is not an exact reduction
  // and therefore must be selected explicitly.
  LegacyWideAsFree = 2,
};

struct PresolveParams {
  int max_iters = 512;
  double max_time = __builtin_huge_val();
  bool verbose = false;
  bool record_postsolve_tape = true;
  bool record_postsolve_tape_cpu = false;

  // Default pipeline: folding -> this presolver. Use the combined API in
  // gpu_presolver/folding/folding.hpp; raw presolve entrypoints do not fold.
  bool enable_presolve = true;  // Combined pipeline stage switch.
  bool enable_folding = true;
  double folding_tolerance = 1.0e-8;

  double feasibility_tol = 1.0e-6;
  double bound_tol = 1.0e-6;
  double zero_tol = 1.0e-10;
  // Outward margin for accepted, nonintegral implied upper bounds.
  // Lower-bound margins and near-fixing checks retain their existing policy.
  double bound_strengthening_upper_relaxation_factor = 1.0;
  // One GPU kernel round sees only its entry bounds; bound dependency chains
  // therefore need bounded fixed-point rounds.
  int bound_strengthening_max_rounds = 64;
  // Upper limit L for consecutive Propagation executions without deletion.
  // L is fixed after Startup, before the first Cleanup.
  int bound_strengthening_max_bound_only_rounds = 64;
  // A wide, redundant box added to an otherwise free zero-cost auxiliary can
  // severely degrade HPR scaling.  Keep such columns free unless propagation
  // nearly fixes them; one-sided and nonzero-cost tightening is unchanged.
  bool bound_strengthening_preserve_free_zero_cost = true;
  // Actual fill-in cap; the GPU path rebuilds sparse storage after every
  // accepted batch.
  int doubleton_equations_max_fill_in_proxy = 96;
  bool doubleton_equations_scan = true;
  int doubleton_equations_min_selected_per_batch = 256;
  double doubleton_equations_min_selected_ratio = 0.005;
  // Legacy batch-loop controls; the single-batch implementation ignores them.
  int doubleton_equations_max_batch_rounds = 0;
  double doubleton_equations_max_time = 0.0;

  bool enable_infeasible_and_fixed_variables = true;
  bool enable_empty_rows = true;
  bool enable_singleton_rows = true;
  bool enable_infeasible_and_redundant_rows = true;
  bool enable_bound_strengthening = true;
  bool enable_parallel_rows = true;
  bool enable_empty_columns = true;
  bool enable_singleton_columns = true;
  bool enable_doubleton_equations = true;
  bool enable_dual_fix = true;
  // Solver-aware numerical guard: nonzero-cost lock fixing of very dense
  // columns can remove useful HPR slack directions.  Zero-cost candidates
  // remain unrestricted; nonzero-cost candidates above this degree are kept.
  int dual_fix_max_nonzero_cost_col_degree = 128;
  bool enable_parallel_columns = true;
  // Symmetric ratio guard for A_k = ratio * A_j.  Only moderate substitutions
  // 1/R <= |ratio| <= R are accepted; more aggressive scaling can damage the
  // conditioning seen by the first-order solver.
  double parallel_columns_max_abs_ratio = 2.0;
  // Activate the ratio guard only for a widespread family of small scaled
  // groups.  This targets the ill-conditioned replicated-box pattern without
  // changing isolated or large aggregation groups.
  int parallel_columns_scale_guard_max_group_size = 4;
  int parallel_columns_scale_guard_min_groups = 128;
  // Master switch for the five structure-specific rules: column domination,
  // two-row variable elimination, McCormick variable elimination, variable-pair
  // aggregation, and absolute-value epigraph reduction.
  bool enable_structure_specific_rules = true;
  // Strict, no-fill projection for a zero-cost finite-bounded column that
  // occurs in exactly two short one-sided rows.  The two row envelopes and
  // the column box must have a positive interval-certificate margin before
  // the column and both rows are removed.
  bool enable_two_row_variable_elimination = true;
  int two_row_variable_elimination_max_row_nnz = 8;
  double two_row_variable_elimination_safety_tol = 1.0e-8;
  int two_row_variable_elimination_min_candidates = 128;
  double two_row_variable_elimination_min_candidate_ratio = 0.01;
  // Strict Fortet/McCormick auxiliary detector.  It accepts only the exact
  // binary-box convex envelope z >= x+y-1, z <= x, z <= y with c_z = 0.
  bool enable_mccormick_variable_elimination = true;
  int mccormick_variable_elimination_min_candidates = 128;
  double mccormick_variable_elimination_min_candidate_ratio = 0.01;
  // Variable-pair aggregation: epigraph case, with the strict layout
  // [p_0...p_{k-1}, n_0...n_{k-1}, T_0...T_{g-1}].  Accepted models are
  // fully verified before equality-connected p_i-n_i components are merged.
  bool enable_variable_pair_aggregation = true;
  int variable_pair_epigraph_probe_pairs = 256;
  int variable_pair_epigraph_min_pairs = 4096;
  double variable_pair_epigraph_min_potential_gain = 0.20;
  double variable_pair_epigraph_min_actual_gain = 0.20;
  int variable_pair_epigraph_max_tail_cols = 8;
  int variable_pair_epigraph_max_quotient_terms_per_row = 5;
  // Detect strict half-offset (+/-) column pairs, collapse equality-connected
  // pair components, and rebuild the LP in shifted quotient coordinates. The
  // sampled probe only rejects; accepted models are fully verified on GPU.
  int variable_pair_shifted_antipodal_probe_pairs = 256;
  int variable_pair_shifted_antipodal_min_pairs = 4096;
  double variable_pair_shifted_antipodal_min_potential_gain = 0.20;
  double variable_pair_shifted_antipodal_min_actual_gain = 0.20;
  int variable_pair_shifted_antipodal_max_quotient_terms_per_row = 6;
  // Strict unit set-covering detector.  A model is changed only after a full
  // GPU certificate proves rows Ax>=1, A in {0,1}, x in [0,1], positive
  // integer costs, and a cost-one witness for every row.  Columns with
  // c_j > nnz(A[:,j]) are then fixed to zero.
  bool enable_column_domination = true;
  int column_domination_probe_rows = 64;
  int column_domination_probe_cols = 256;
  int column_domination_probe_nnz = 256;
  int column_domination_min_fixed_cols = 4096;
  double column_domination_min_candidate_ratio = 0.20;
  bool enable_absolute_value_epigraph_reduction = true;
  AbsoluteValueEpigraphResidualBoundMode absolute_value_epigraph_residual_bound_mode =
      AbsoluteValueEpigraphResidualBoundMode::PreserveFinite;
  // Used only when absolute_value_epigraph_residual_bound_mode is LegacyWideAsFree.
  double absolute_value_epigraph_residual_bound_as_free_min = 1.0e8;
  // Mode for explicit apply_rule_redundant_variable_bounds calls. This rule is not
  // part of the production schedule: false removes redundant zero-cost boxes;
  // true also considers redundant one-sided bounds.
  bool enable_redundant_variable_bounds = false;

  bool enable_startup = true;

  // Compute L once from live nnz after Startup, before the first Cleanup:
  // min(max_bound_only_rounds, max(1, budget / initial_live_nnz)).
  // Actual row/column deletions reset the consecutive-round counter; Cleanup
  // without deletion preserves it. Entering Aggregation resets it. Later nnz
  // changes never recompute L. A nonpositive value disables the nnz work cap.
  std::int64_t bound_strengthening_bound_only_nnz_round_budget = 8000000;
};

struct DeviceCsrMatrix {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  NnzOffset nnz = 0;
  NnzOffset* rowPtr = nullptr;
  std::int32_t* colVal = nullptr;
  double* nzVal = nullptr;
};

struct LPInfoGpu {
  DeviceCsrMatrix A;
  DeviceCsrMatrix AT;
  double* c = nullptr;
  double* AL = nullptr;
  double* AU = nullptr;
  double* l = nullptr;
  double* u = nullptr;
  double obj_constant = 0.0;
  NnzOffset AT_leading_slack = 0;
  NnzOffset* AT_slack_after = nullptr;
};

struct PresolveStatsGpu {
  // Set by refresh_live_stats after mask changes; all indices remain physical.
  bool live_masks = false;
  NnzOffset live_nnz = 0;
  void* contiguous_storage = nullptr;
  std::int32_t* row_nnz = nullptr;
  std::uint8_t* empty_row_mask = nullptr;
  std::uint8_t* singleton_row_mask = nullptr;
  std::int32_t* singleton_row_col = nullptr;
  double* singleton_row_val = nullptr;

  std::int32_t* col_nnz = nullptr;
  std::uint8_t* empty_col_mask = nullptr;
  std::uint8_t* singleton_column_mask = nullptr;
  std::int32_t* singleton_column_row = nullptr;
  double* singleton_column_val = nullptr;

  bool row_nnz_valid = false;
  bool col_nnz_valid = false;
  std::int32_t* absolute_value_epigraph_screen_counts = nullptr;
  std::uint8_t* absolute_value_epigraph_screen_flag = nullptr;
  std::int32_t absolute_value_epigraph_screen_pattern_code = -1;
  bool absolute_value_epigraph_screen_valid = false;
};

struct AbsoluteValueEpigraphSplitRecovery {
  std::int32_t t_col = -1;
  std::int32_t e_col = -1;
  double rho = 1.0;
};

struct AbsoluteValueOuterPairRecovery {
  std::int32_t bound_col = -1;
  std::int32_t free_col = -1;
};

struct AbsoluteValueLinkedSlackRecovery {
  std::int32_t slack_col = -1;
  std::int32_t t_col = -1;
  double factor = 0.0;
};

struct AbsoluteValueMaxSlackRecovery {
  std::int32_t slack_col = -1;
  std::vector<std::int32_t> t_cols;
  std::vector<double> factors;
};

struct AbsoluteValueEpigraphPrimalRecoveryStep {
  std::string pattern;
  std::vector<AbsoluteValueEpigraphSplitRecovery> splits;
  std::vector<AbsoluteValueOuterPairRecovery> outer_pairs;
  std::vector<AbsoluteValueLinkedSlackRecovery> linked_slacks;
  std::vector<AbsoluteValueMaxSlackRecovery> max_slacks;
};

// Primal recovery for both variable-pair aggregation cases. Root slots hold
// P_C/N_C; recover p_i = P_C + s_p_i and n_i = N_C + s_n_i. Shifts are zero
// for the epigraph case and original lower bounds for shifted antipodal pairs.
// Deleted equality-row duals
// are intentionally not advertised as recoverable.
struct VariablePairAggregationRecoveryStep {
  std::vector<std::int32_t> plus_cols;
  std::vector<std::int32_t> minus_cols;
  std::vector<std::int32_t> root_plus_cols;
  std::vector<std::int32_t> root_minus_cols;
  std::vector<double> plus_shift;
  std::vector<double> minus_shift;
};

// Primal-only transformations share the same chronological position space as
// PostsolveTape records.  A checkpoint is appended when the forward
// transformation is committed and stores the number of ordinary tape records
// that preceded it.  Postsolve walks these checkpoints backwards, replaying
// each intervening tape range before expanding the corresponding payload.
enum class PrimalRecoveryKind : std::uint8_t {
  AbsoluteValueEpigraph = 0,
  VariablePairAggregation = 1,
};

struct PrimalRecoveryCheckpoint {
  PrimalRecoveryKind kind = PrimalRecoveryKind::AbsoluteValueEpigraph;
  NnzOffset payload_index = -1;
  NnzOffset tape_position = 0;
};

enum class PostsolveReductionType : std::int32_t {
  FixedCol = 0,
  FixedColInf = 1,
  SubCol = 2,
  ParallelColumn = 3,
  ParallelRow = 4,
  DeletedRow = 5,
  AddedRow = 6,
  AddedRows = 7,
  LhsChange = 8,
  RhsChange = 9,
  EqToIneq = 10,
  BoundChangeNoRow = 11,
  BoundChangeTheRow = 12,
  DoubletonEquation = 13,
  FmeCol = 14,
};

enum class PostsolveDualMode : std::uint8_t {
  None = 0,
  Exact = 1,
  Minimal = 2,
};

// Owning device segment of the chronological recovery tape. Segment offsets
// are local; full presolve retains these segments through postsolve.
struct PostsolveTapeGpu {
  PostsolveTapeGpu() = default;
  ~PostsolveTapeGpu();

  PostsolveTapeGpu(const PostsolveTapeGpu&) = delete;
  PostsolveTapeGpu& operator=(const PostsolveTapeGpu&) = delete;
  PostsolveTapeGpu(PostsolveTapeGpu&& other) noexcept;
  PostsolveTapeGpu& operator=(PostsolveTapeGpu&& other) noexcept;

  void reset() noexcept;

  std::int32_t* types = nullptr;
  NnzOffset* index_starts = nullptr;
  NnzOffset* value_starts = nullptr;
  std::uint8_t* dual_modes = nullptr;
  std::int32_t* indices = nullptr;
  double* vals = nullptr;
  NnzOffset record_count = 0;
  NnzOffset index_count = 0;
  NnzOffset value_count = 0;
  NnzOffset record_capacity = 0;
  NnzOffset index_capacity = 0;
  NnzOffset value_capacity = 0;
};

struct PostsolveTape {
  PostsolveTape() = default;
  PostsolveTape(const PostsolveTape &);
  PostsolveTape &operator=(const PostsolveTape &);
  PostsolveTape(PostsolveTape &&) noexcept = default;
  PostsolveTape &operator=(PostsolveTape &&) noexcept = default;
  // Public rule calls keep the host oracle; full presolve enables device segments.
  bool resident = false;
  std::vector<PostsolveTapeGpu> segments;
  std::vector<std::int32_t> types;
  std::vector<NnzOffset> index_starts{0};
  std::vector<NnzOffset> value_starts{0};
  std::vector<std::uint8_t> dual_modes;
  std::vector<std::int32_t> indices;
  std::vector<double> vals;
};

struct DeviceRecoveryState;
struct DeviceRecoveryGroup;
struct DeviceAbsoluteValueEpigraphStep;

struct PresolveRecordGpu {
  std::shared_ptr<DeviceRecoveryState> device;
  std::int32_t m0 = 0;
  std::int32_t n0 = 0;
  std::int32_t m1 = 0;
  std::int32_t n1 = 0;

  // Legacy/oracle and explicitly exported diagnostic views. These stay empty
  // during normal GPU-resident presolve; postsolve borrows device storage.
  std::vector<std::int32_t> row_org2red;
  std::vector<std::int32_t> row_red2org;
  std::vector<std::int32_t> col_org2red;
  std::vector<std::int32_t> col_red2org;

  std::vector<std::int32_t> fixed_idx;
  std::vector<double> fixed_val;
  std::vector<std::int32_t> removed_row_idx;
  std::vector<std::int32_t> removed_col_idx;

  double obj_constant_old = 0.0;
  double obj_constant_new = 0.0;

  std::vector<AbsoluteValueEpigraphPrimalRecoveryStep> absolute_value_epigraph_recoveries;
  std::vector<VariablePairAggregationRecoveryStep> variable_pair_aggregation_recoveries;
  std::vector<PrimalRecoveryCheckpoint> primal_recovery_timeline;
  bool has_primal_only_variable_pair_aggregation_reduction = false;
  bool has_primal_only_column_domination_reduction = false;
  bool has_primal_only_projected_auxiliary_reduction = false;
  PostsolveTape tape;
  PostsolveTapeGpu tape_gpu;
};

struct GpuPostsolveResult {
  double* x_org = nullptr;
  double* y_org = nullptr;
  double* z_org = nullptr;
  std::int32_t n0 = 0;
  std::int32_t m0 = 0;
};

struct PresolvePlanGpu {
  // Optional immutable snapshot of keep_col_mask at rule entry (not an alias).
  // Preserves the objective reduction domain across deferred matrix compaction.
  const std::uint8_t* rule_entry_keep_col = nullptr;
  // Optional immutable bound snapshots at propagation entry, valid across its rounds.
  // These borrowed pointers must be cleared/rebound when physical indices change.
  const double* propagation_reference_l = nullptr;
  const double* propagation_reference_u = nullptr;
  void* contiguous_storage = nullptr;
  std::uint8_t* keep_row_mask = nullptr;
  std::uint8_t* keep_col_mask = nullptr;

  DeviceCsrMatrix new_A;
  bool has_new_A = false;
  NnzOffset new_AT_leading_slack = 0;
  NnzOffset* new_AT_slack_after = nullptr;

  double* new_c = nullptr;
  double* new_AL = nullptr;
  double* new_AU = nullptr;
  double* new_l = nullptr;
  double* new_u = nullptr;
  double obj_constant_delta = 0.0;
  // Borrowed stage-entry mask for deferred trivial cleanup. It preserves the
  // compact objective reduction domain and prefix-before-tail fixed records.
  // Borrowed masks for a cleanup epoch before physical compaction.
  const std::uint8_t* cleanup_entry_keep_col = nullptr;
  bool cleanup_recovery_captured = false;
  const std::uint8_t* cleanup_prefix_keep_col = nullptr;
  const std::uint8_t* cleanup_dual_keep_col = nullptr;
  // Borrowed compact logical-column launch domain, used only by fused cleanup.
  const std::int32_t* cleanup_active_cols = nullptr;
  std::int32_t cleanup_active_count = 0;
  bool cleanup_requires_materialization = false;
  // -1 uses physical dimensions; startup batching preserves the live-domain hash choice.
  int parallel_row_hash_mode = -1;

  bool has_change = false;
  bool has_row_action = false;
  bool has_col_action = false;
  // Redundant-row detection also treats a relaxed row side as a row action.
  // Track full-row deletion separately for the Propagation round budget.
  bool has_redundant_row_deletion = false;
  // Full-row deletions reported by RR, empty rows and singleton rows.
  // Other compaction actions that delete a column use has_col_action.
  bool has_row_deletion = false;
  bool has_infeasible = false;
  bool has_unbounded = false;

  bool has_absolute_value_epigraph_recovery = false;
  AbsoluteValueEpigraphPrimalRecoveryStep absolute_value_epigraph_recovery;
  std::shared_ptr<DeviceAbsoluteValueEpigraphStep> absolute_value_epigraph_device;
  bool has_variable_pair_aggregation_recovery = false;
  VariablePairAggregationRecoveryStep variable_pair_aggregation_recovery;
  std::shared_ptr<DeviceRecoveryGroup> variable_pair_aggregation_device;
  bool has_column_domination_reduction = false;
  bool has_projected_auxiliary_reduction = false;
  // Legacy transient mirror. GPU-resident rules append to tape.segments.
  bool tape_gpu_mirrors_cpu = false;
  PostsolveTape tape;
  PostsolveTapeGpu tape_gpu;
};

}  // namespace gpu_presolver::presolve
