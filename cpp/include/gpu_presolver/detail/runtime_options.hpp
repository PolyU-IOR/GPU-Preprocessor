#pragma once

#include <string>

namespace gpu_presolver::runtime {

// Implementation settings, not public solver parameters. Defaults preserve the
// production paths; diagnostics remain off. Regression tests can override them
// on their calling thread without changing the process environment.
enum class SignatureMode { Disabled, Enabled, Automatic };

struct Options {
  bool adaptive_activity = true;
  bool extended_row_kernels = true;
  bool gather_live_rows = true;
  bool hash_scan = true;
  bool reuse_row_groups = true;
  bool reuse_sortedness = true;
  bool direct_fixed_values = true;
  bool parallel_tape_build = true;
  bool reserve_tape_headroom = true;
  bool chunked_tape = false;
  bool parallel_mapping = true;
  bool warp_compaction = true;
  bool long_row_compaction = true;
  bool parallel_column_prefilter = true;
  bool device_recovery = true;
  bool reuse_cleanup_workspace = true;
  bool incremental_live_stats = false;
  bool doubleton_long_rows = true;
  bool parallel_column_preverify = false;
  bool profile_allocations = false;
  bool profile_apply = false;
  bool profile_details = false;
  bool profile_records = false;
  bool profile_rules = false;
  bool profile_live_stats = false;
  bool profile_doubleton = false;
  bool profile_parallel_columns = false;
  bool profile_parallel_rows = false;
  bool profile_absolute_value = false;
  bool profile_variable_pair_epigraph = false;
  bool profile_shifted_antipodal = false;
  bool record_removed_indices = false;
  bool recovery_stats = false;
  bool audit_row_signatures = false;
  bool audit_column_signatures = false;
  bool profile_folding = false;
  bool trace_cleanup_epochs = false;
  bool trace_recovery = false;
  SignatureMode row_signatures = SignatureMode::Enabled;
  SignatureMode column_signatures = SignatureMode::Enabled;
  std::string snapshot_directory;
};

inline Options& options() {
  static thread_local Options current;
  return current;
}

}  // namespace gpu_presolver::runtime
