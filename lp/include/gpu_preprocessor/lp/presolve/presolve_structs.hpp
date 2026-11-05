// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/offset.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gpu_preprocessor::lp::presolve {

enum class PiecewiseLinearReformulationAbsoluteValueResidualBoundMode : std::uint8_t {
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

struct LPInfoGpu;
struct StartupPoissonReduction;

struct PresolveParams {
  int max_iters = 512;
  double max_time = __builtin_huge_val();
  bool verbose = false;
  bool record_postsolve_tape = true;
  bool record_postsolve_tape_cpu = false;

  // The combined API defaults to Startup -> folding -> Main. See
  // gpu_preprocessor/lp/folding/folding.hpp; raw presolve entrypoints do not fold.
  bool enable_presolve = true;  // Combined pipeline stage switch.
  bool enable_folding = true;
  double folding_tolerance = 1.0e-8;

  double feasibility_tol = 1.0e-6;
  double bound_tol = 1.0e-6;
  double zero_tol = 1.0e-10;
  // Outward margin for accepted, nonintegral implied upper bounds.
  // Lower-bound margins and near-fixing checks retain their existing policy.
  double bound_strengthening_upper_relaxation_factor = 1.0;
  // Compatibility slot; ignored. Main controls repeated propagation through
  // bound_strengthening_max_bound_only_rounds and its nnz work budget.
  int bound_strengthening_max_rounds = 64;
  // Upper limit L for consecutive Propagation executions without deletion.
  // L is fixed after Startup, before the first Cleanup.
  int bound_strengthening_max_bound_only_rounds = 64;
  // Exactly zero-cost columns always retain both bounds in this integration.
  // This older setting additionally avoids a wide box on initially free
  // columns with tiny nonzero cost (abs(c) <= zero_tol), except near-fixing.
  bool bound_strengthening_preserve_free_zero_cost = true;
  // Fill-in cap for doubleton-equation elimination; the GPU path rebuilds
  // sparse storage after every accepted batch.
  int doubleton_equations_max_fill_in_proxy = 96;
  bool doubleton_equations_scan = true;
  int doubleton_equations_min_selected_per_batch = 256;
  double doubleton_equations_min_selected_ratio = 0.005;
  // Compatibility slots; ignored by the single-batch doubleton implementation.
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
  // Numerical guard: nonzero-cost lock fixing of very dense
  // columns can remove useful slack directions. Zero-cost candidates
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
  // Master switch for column domination, two-row variable elimination,
  // zero-cost product variables, and both piecewise-linear reformulation
  // branches. The Startup row/Poisson prefix also observes this switch.
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
  // Zero-cost product variables: accept only the exact Fortet/McCormick
  // binary-box envelope z >= x+y-1, z <= x, z <= y with c_z = 0.
  bool enable_zero_cost_product_variables = true;
  int zero_cost_product_variables_min_candidates = 128;
  double zero_cost_product_variables_min_candidate_ratio = 0.01;
  // Piecewise-linear reformulation, variable-pair branch: epigraph case with
  // strict layout [p_0...p_{k-1}, n_0...n_{k-1}, T_0...T_{g-1}]. Accepted models are
  // fully verified before equality-connected p_i-n_i components are merged.
  bool enable_piecewise_linear_reformulation_variable_pair = true;
  int piecewise_linear_reformulation_variable_pair_epigraph_probe_pairs = 256;
  int piecewise_linear_reformulation_variable_pair_epigraph_min_pairs = 4096;
  double piecewise_linear_reformulation_variable_pair_epigraph_min_potential_gain = 0.20;
  double piecewise_linear_reformulation_variable_pair_epigraph_min_actual_gain = 0.20;
  int piecewise_linear_reformulation_variable_pair_epigraph_max_tail_cols = 8;
  int piecewise_linear_reformulation_variable_pair_epigraph_max_quotient_terms_per_row = 5;
  // Detect strict half-offset (+/-) column pairs, collapse equality-connected
  // pair components, and rebuild the LP in shifted quotient coordinates. The
  // sampled probe only rejects; accepted models are fully verified on GPU.
  int piecewise_linear_reformulation_variable_pair_shifted_antipodal_probe_pairs = 256;
  int piecewise_linear_reformulation_variable_pair_shifted_antipodal_min_pairs = 4096;
  double piecewise_linear_reformulation_variable_pair_shifted_antipodal_min_potential_gain = 0.20;
  double piecewise_linear_reformulation_variable_pair_shifted_antipodal_min_actual_gain = 0.20;
  int piecewise_linear_reformulation_variable_pair_shifted_antipodal_max_quotient_terms_per_row = 6;
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
  // Piecewise-linear reformulation, absolute-value branch. This remains
  // independently configurable from the variable-pair branch above.
  bool enable_piecewise_linear_reformulation_absolute_value = true;
  PiecewiseLinearReformulationAbsoluteValueResidualBoundMode
      piecewise_linear_reformulation_absolute_value_residual_bound_mode =
      PiecewiseLinearReformulationAbsoluteValueResidualBoundMode::PreserveFinite;
  // Used only when the absolute-value residual-bound mode is LegacyWideAsFree.
  double piecewise_linear_reformulation_absolute_value_residual_bound_as_free_min = 1.0e8;
  // Compatibility mode for the four-argument redundant-bound rule. This is
  // not an enable switch and does not schedule the rule. New direct callers
  // pass RedundantVariableBoundsMode explicitly; false here means zero-cost
  // boxes only, true also considers one-sided bounds. Keep this ABI slot.
  bool enable_redundant_variable_bounds = false;

  bool enable_startup = true;
  // Optional owner-managed CSR-only entry. Called after the row/Poisson
  // prefix (or when that prefix is skipped), before column-oriented rules.
  // The owner supplies a valid transpose and may retire the original input
  // after a successful replacement. Borrowed input metadata must stay valid.
  void (*prepare_startup_column_access)(LPInfoGpu&, void*) = nullptr;
  void* startup_column_access_context = nullptr;
  bool defer_startup_row_transpose = false;
  // Adapter-supplied native Poisson strategy, executed after row sparsification.
  // Null in standalone builds. Miss leaves the input and recovery maps intact.
  std::shared_ptr<StartupPoissonReduction> (*startup_poisson)(const LPInfoGpu&) = nullptr;
  // Exact equality-row differences, once at the start of Startup. Detection,
  // certification, materialization and the adjoint inverse run on the GPU.
  bool enable_row_sparsification = true;
  int row_sparsification_max_candidates = 1000000;
  int row_sparsification_max_chain = INT32_MAX;
  std::int64_t row_sparsification_max_pair_entries = 0; // 0: 4 * input nnz.
  double row_sparsification_min_gain = 0.01;
  bool row_sparsification_log = false;

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
  std::int32_t* piecewise_linear_reformulation_absolute_value_screen_counts = nullptr;
  std::uint8_t* piecewise_linear_reformulation_absolute_value_screen_flag = nullptr;
  std::int32_t piecewise_linear_reformulation_absolute_value_screen_pattern_code = -1;
  bool piecewise_linear_reformulation_absolute_value_screen_valid = false;
};

struct PiecewiseLinearReformulationAbsoluteValueSplitRecovery {
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

struct PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep {
  std::string pattern;
  std::vector<PiecewiseLinearReformulationAbsoluteValueSplitRecovery> splits;
  std::vector<AbsoluteValueOuterPairRecovery> outer_pairs;
  std::vector<AbsoluteValueLinkedSlackRecovery> linked_slacks;
  std::vector<AbsoluteValueMaxSlackRecovery> max_slacks;
};

// Primal recovery for both variable-pair reformulation cases. Root slots hold
// P_C/N_C; recover p_i = P_C + s_p_i and n_i = N_C + s_n_i. Shifts are zero
// for the epigraph case and original lower bounds for shifted antipodal pairs.
// This payload stores primal data only; joint x/y/z recovery uses the
// checkpoint's separate StructuralDualRecovery stage.
struct PiecewiseLinearReformulationVariablePairRecoveryStep {
  std::vector<std::int32_t> plus_cols;
  std::vector<std::int32_t> minus_cols;
  std::vector<std::int32_t> root_plus_cols;
  std::vector<std::int32_t> root_minus_cols;
  std::vector<double> plus_shift;
  std::vector<double> minus_shift;
};

// Structural transformations share the same chronological position space as
// PostsolveTape records.  A checkpoint is appended when the forward
// transformation is committed and stores the number of ordinary tape records
// that preceded it.  Postsolve walks these checkpoints backwards, replaying
// each intervening tape range before expanding the corresponding payload.
enum class PrimalRecoveryKind : std::uint8_t {
  PiecewiseLinearReformulationAbsoluteValue = 0,
  PiecewiseLinearReformulationVariablePair = 1,
  ColumnDomination = 2,
};

struct PrimalRecoveryCheckpoint {
  PrimalRecoveryKind kind = PrimalRecoveryKind::PiecewiseLinearReformulationAbsoluteValue;
  NnzOffset payload_index = -1;
  NnzOffset tape_position = 0;
  NnzOffset dual_index = -1; // Immutable stage data; -1 denotes a legacy primal payload.
};

enum class PostsolveReductionType : std::int32_t {
  FixedVariable = 0,
  InfiniteBoundDualFix = 1,
  SingletonColumnSubstitution = 2,
  ParallelColumn = 3,
  ParallelRow = 4,
  DeletedRow = 5,
  RowLowerBoundChange = 8,
  RowUpperBoundChange = 9,
  RowDualShift = 10,
  VariableBoundChange = 11,
  RowImpliedBoundChange = 12,
  DoubletonEquation = 13,
  FmeCol = 14,
  BoundChangeSharedRow = 15,
  BoundSourceRow = 16,
  BoundBatch = 17,
  BoundChangeCompact = 18,
  BoundPackedBatch = 19,

  // Historical names and reserved record IDs retained for source/snapshot compatibility.
  FixedCol = FixedVariable,
  FixedColInf = InfiniteBoundDualFix,
  SubCol = SingletonColumnSubstitution,
  LhsChange = RowLowerBoundChange,
  RhsChange = RowUpperBoundChange,
  EqToIneq = RowDualShift,
  BoundChangeNoRow = VariableBoundChange,
  BoundChangeTheRow = RowImpliedBoundChange,
  AddedRow = 6,  // Reserved; no producer or replay action.
  AddedRows = 7, // Reserved; no producer or replay action.
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

struct StartupRowSparsification;
struct DeviceRecoveryState;
struct DeviceRecoveryGroup;
struct DevicePiecewiseLinearReformulationAbsoluteValueStep;
struct StructuralDualRecovery;

struct PresolveRecordGpu {
  // Outer basis change; ordinary tape/checkpoints refer to transformed, not
  // to the caller's pre-startup A. Replay it last, after unfolding.
  std::shared_ptr<StartupRowSparsification> startup_rows;
  // Ordinary maps refer to the response LP when this dimension change applies.
  std::shared_ptr<StartupPoissonReduction> startup_poisson;
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

  std::vector<PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep>
      piecewise_linear_reformulation_absolute_value_recoveries;
  std::vector<PiecewiseLinearReformulationVariablePairRecoveryStep>
      piecewise_linear_reformulation_variable_pair_recoveries;
  std::vector<PrimalRecoveryCheckpoint> primal_recovery_timeline;
  std::vector<std::shared_ptr<StructuralDualRecovery>> structural_dual_recoveries;
  bool has_primal_only_piecewise_linear_reformulation_variable_pair_reduction = false;
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

// Owns the immutable response LP and the native inverse without introducing
// application model types or FFT dependencies into standalone presolve.
struct StartupPoissonReduction {
  LPInfoGpu transformed;
  virtual ~StartupPoissonReduction() = default;
  // Returns owning x/y/z device vectors in the strategy input coordinates.
  // Borrow reduced and input; the caller releases reduced after this returns.
  // This inverse runs before Startup's row-sparsification adjoint.
  virtual GpuPostsolveResult restore(const GpuPostsolveResult& reduced,
                                    const LPInfoGpu& input) const = 0;
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

  bool has_piecewise_linear_reformulation_absolute_value_recovery = false;
  PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep
      piecewise_linear_reformulation_absolute_value_recovery;
  std::shared_ptr<DevicePiecewiseLinearReformulationAbsoluteValueStep>
      piecewise_linear_reformulation_absolute_value_device;
  bool has_piecewise_linear_reformulation_variable_pair_recovery = false;
  PiecewiseLinearReformulationVariablePairRecoveryStep
      piecewise_linear_reformulation_variable_pair_recovery;
  std::shared_ptr<DeviceRecoveryGroup>
      piecewise_linear_reformulation_variable_pair_device;
  bool has_column_domination_reduction = false;
  bool has_projected_auxiliary_reduction = false;
  // Retain the legacy warning only when a projection was accepted without a tape.
  bool has_unrecorded_projected_auxiliary_reduction = false;
  // Legacy transient mirror. GPU-resident rules append to tape.segments.
  bool tape_gpu_mirrors_cpu = false;
  PostsolveTape tape;
  PostsolveTapeGpu tape_gpu;
};

}  // namespace gpu_preprocessor::lp::presolve
