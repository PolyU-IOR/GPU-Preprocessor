#pragma once
#include "gpu_presolver/presolve/presolve_structs.hpp"
#include <utility>

namespace gpu_presolver::presolve {
void recovery_device_free(void *) noexcept;
template <class T> struct RecoveryArray {
  T *data = nullptr;
  std::int32_t count = 0;
  RecoveryArray() = default;
  ~RecoveryArray() { recovery_device_free(data); }
  RecoveryArray(const RecoveryArray &) = delete;
  RecoveryArray &operator=(const RecoveryArray &) = delete;
  RecoveryArray(RecoveryArray &&x) noexcept
      : data(std::exchange(x.data, nullptr)), count(std::exchange(x.count, 0)) {}
  RecoveryArray &operator=(RecoveryArray &&x) noexcept {
    if (this != &x) {
      recovery_device_free(data);
      data = std::exchange(x.data, nullptr);
      count = std::exchange(x.count, 0);
    }
    return *this;
  }
};
struct DeviceFixedSegment {
  RecoveryArray<std::int32_t> indices;
  RecoveryArray<double> values;
};
struct DeviceRecoveryGroup {
  std::vector<RecoveryArray<std::int32_t>> ids;
  std::vector<RecoveryArray<double>> values;
};
struct DeviceAbsoluteValueEpigraphStep {
  std::string pattern;
  DeviceRecoveryGroup splits, pairs, linked;
  std::vector<DeviceRecoveryGroup> max_slacks;
};
struct DeviceRecoveryState {
  std::vector<DeviceAbsoluteValueEpigraphStep> absolute_value_epigraph;
  std::vector<DeviceRecoveryGroup> variable_pair_aggregation;
  RecoveryArray<std::int32_t> rows, cols;
  std::vector<PostsolveTapeGpu> segments;
  std::vector<DeviceFixedSegment> fixed;
  std::vector<RecoveryArray<std::int32_t>> removed_rows, removed_cols;
  NnzOffset records = 0, indices = 0, values = 0;
  std::size_t host_staged_bytes = 0;
};
enum class SimpleRecovery { FixedCol, SingletonRow, DeletedRow, RowSides, ColumnBounds };
void append_simple_device_recovery(PostsolveTape &, SimpleRecovery, const std::uint8_t *,
                                   const std::uint8_t *, std::int32_t, const std::int32_t *,
                                   const double *, const double *, const double *);
void append_projection_device_recovery(PostsolveTape &, const LPInfoGpu &, const std::uint8_t *,
                                       int, const std::int32_t*, const std::uint8_t*);
void append_infinite_device_recovery(PostsolveTape &, const std::uint8_t *, const std::uint8_t *,
                                     const double *, const std::uint8_t *, const double *,
                                     const double *, const double *, const double *,
                                     const DeviceCsrMatrix &, const DeviceCsrMatrix &, int,
                                     const std::uint8_t* entry_cols = nullptr);
bool device_recovery_enabled();
// Capture local structural payloads before rule scratch is released. Commit
// globalizes their indices on the GPU and transfers ownership to the record.
void capture_variable_pair_recovery_device(PresolvePlanGpu&, const int* parent, int pairs,
                                          const double* plus_shift = nullptr,
                                          const double* minus_shift = nullptr);
void capture_absolute_value_splits_device(PresolvePlanGpu&, const int* t, const int* e,
                                         const double* rho, int count, const char* pattern);
void capture_absolute_value_outer_device(PresolvePlanGpu&, const int* bound, const int* free,
                                        int pairs, const int* slack, const int* t,
                                        const double* factor, int blocks);
void capture_absolute_value_max_slacks_device(PresolvePlanGpu&, const int* t,
    const int* row_block, const int* row_slack, const double* row_factor,
    const std::uint8_t* row_removable, int rows, int cols);
PostsolveTapeGpu allocate_recovery_segment(NnzOffset records, NnzOffset indices, NnzOffset values);
// Explicit legacy import. A resident plan rejects host-built ordinary records.
void flush_host_recovery_segment(PostsolveTape &);
void append_device_recovery_segment(PostsolveTape &, PostsolveTapeGpu &&);
void export_local_recovery_tape(PostsolveTape &);
void initialize_device_recovery(PresolveRecordGpu &);
void update_device_recovery(PresolveRecordGpu &, PresolvePlanGpu &, const LPInfoGpu &, std::int32_t,
                            std::int32_t, double, const std::int32_t *, const std::int32_t *);
// Capture one successful logical cleanup epoch without compacting maps/A/AT.
void capture_cleanup_recovery(PresolveRecordGpu&, PresolvePlanGpu&, const LPInfoGpu&, double);
void trace_cleanup_epoch(const PresolveRecordGpu&, const PresolvePlanGpu&, const LPInfoGpu&, double);
// Explicit diagnostic export only; postsolve never calls this.
void export_recovery_record(PresolveRecordGpu &);
void validate_device_recovery(const PresolveRecordGpu &);
void trace_recovery_plan(const PresolvePlanGpu &, const LPInfoGpu &);
} // namespace gpu_presolver::presolve
