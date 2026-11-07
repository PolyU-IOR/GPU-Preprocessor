# QP presolve implementation

`presolve.c` owns the public API, rule order, iteration/time budgets and stage
dispatch. `gpu_presolve.cu` assembles the private CUDA implementation. Rule
selection continues to use `Config.rules`; this layout adds no public interface.

| Strategy | Implementation in `rules/` |
| --- | --- |
| `EMPTY_ROWS` | `rule_empty_rows.cuh` |
| `SINGLETON_ROWS` | `rule_singleton_rows.cuh` |
| `ACTIVITY` | `rule_activity.cuh` |
| `PROPAGATION` | `rule_propagation.cuh` |
| `PARALLEL_ROWS` | `rule_parallel_rows.cuh` |
| `FIXED_COLS` | `rule_fixed_cols.cuh` |
| `EMPTY_COLS` | `rule_empty_cols.cuh` |
| `SINGLETON_DUAL` | `rule_singleton_dual.cuh` |
| `SINGLETON_EQ` | `rule_singleton_eq.cuh` |
| `DOUBLETON_EQ` | `rule_doubleton_eq.cuh` |
| `DUAL_FIX` | `rule_dual_fix.cuh` |
| `PARALLEL_COLS` | `rule_parallel_cols.cuh` |
| `FOLDING` | `rule_folding.cuh` |

Each strategy owns its candidate logic and private kernels. Propagation owns
its active queue and bound-update kernels; parallel columns own grouped bound
aggregation; doubletons own their sparse streaming path; folding owns its
partition refinement, matrix construction and class capture.

The `.cuh` implementations are included once by `gpu_presolve.cu`, keeping
internal linkage and device inlining. The existing row/column detector kernels
dispatch to the strategy helpers. Splitting files therefore does not introduce
one launch per strategy helper, a device-link boundary or a host handoff.

Shared code stays outside `rules/`:

| File | Responsibility |
| --- | --- |
| `cuda_plan.cuh` | Candidate/plan ownership, transpose, commit and rejection |
| `cuda_rule_common.cuh` | Exact coefficient checks, interval arithmetic, shared equality/dual-fixing algorithms and signatures |
| `cuda_row_activity.cuh` | Cached row activity used by activity and propagation |
| `cuda_detection.cuh` | Existing detector kernels and candidate compaction |
| `cuda_batch.cuh` | Batch selection masks and shared bound intersections |
| `cuda_matrix_transform.cuh`, `cuda_transform.cuh` | Sparse affine A/Q/c transformation and validation |
| `cuda_row_transform.cuh` | Row-only compaction preserving the objective |
| `cuda_presolve_resident.cuh` | Compatibility selection, record capture, resident execution and boundary export |

All presolve rules execute on CUDA. Device buffers, transfer accounting, tape
types and GPU postsolve live in `../backend/`, where presolve and recovery share
them. Input/output handling and independent KKT checks use host code.

Validation uses CUDA regression suites, Compute Sanitizer and original-space
KKT checks. See [project validation](../../../docs/validation.md).
