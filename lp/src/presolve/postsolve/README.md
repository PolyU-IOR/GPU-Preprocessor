# Postsolve implementation

```text
original LP -> Startup -> folding -> Main -> reduced LP
reduced point -> Main recovery -> unfold -> Startup recovery -> original point
```

This is the default pipeline. Folding is enabled unless `enable_folding` is
set to `false` (`--folding false` in the CLI).

`../../folding/folding.cu` composes the stages; `../../folding/folding_map.cu`
owns solution expansion. `../gpu_presolve.cu` schedules forward rules and commits
their recovery records. Record producers, index mapping and shared layout
decoders live in the parent `presolve/` directory.

`../gpu_postsolve.cu` owns recovery-vector allocation, index mapping, joint
reverse x/y/z replay, parallel-path selection and Startup coordinate restoration.
Ordinary ranges alternate with structural checkpoints in reverse chronological
order. Each range restores x/y/z jointly before its preceding structural
checkpoint is reversed, using the saved stage model and coordinate maps.

The consumer must check feasibility, stationarity and gap against the original
LP; a reduced-model stopping criterion does not certify the recovered point.
No global column-dual projection is applied. Optional primal polish is disabled
when `primal_tolerance` is zero. A positive value permits only certified pair
corrections that pass constraint, bound, objective and correction-size checks.

| File stem | Responsibility |
| --- | --- |
| `postsolve_parallel_rows` | Split merged row multipliers using endpoint ownership |
| `postsolve_parallel_columns` | Split aggregated primal variables and reduced costs |
| `postsolve_doubleton_equations` | Undo equality substitution and its dual lift |
| `postsolve_singleton_columns` | Undo singleton-column substitution |
| `postsolve_singleton_rows` | Restore a deleted row and its bound multiplier |
| `postsolve_row_sparsification` | Apply the Startup row transform's adjoint to y; x/z are unchanged |
| `postsolve_bound_strengthening` | Replay packed/shared bound batches and legacy row-implied bounds |
| `postsolve_fixed_variables` | Recover fixed variables and infinite-bound dual fixing |
| `postsolve_projected_variables` | Shared inverse for the two projected-auxiliary rules |
| `postsolve_piecewise_linear_reformulation_absolute_value` | Absolute-value primal payloads and kernels |
| `postsolve_piecewise_linear_reformulation_variable_pair` | Variable-pair primal payloads and kernels |
| `postsolve_common` | Shared bounds, record checks, row-dual shifts and host utilities |
| `parallel_fixed_postsolve` | Certify and replay distinct fixed-variable targets |
| `parallel_bound_postsolve` | Ordered source-row transfers and target-column subtractions |
| `parallel_subcol_postsolve` | Certify independent singleton-column substitutions |
| `parallel_components_postsolve` | Group records by x/y/z read/write conflicts |
| `structural_dual.inc.cu` | Immutable structural stages and joint structural recovery |
| `pair_primal_polish.inc.cu` | Accept guarded original-space pair corrections |

The `.cuh` handlers are inlined into the replay kernels. The `.cu` files own
structural primal kernels and the row-sparsification adjoint. Both `.inc.cu`
fragments are included by `../gpu_postsolve.cu`, never compiled separately.

Startup recovery first invokes an optional application-owned Poisson inverse,
then applies the row-sparsification adjoint `y = S^T y'`. Standalone builds
provide the Poisson callback contract, not a native implementation; the callback
is null by default. A provider must supply both the forward transform and its
x/y/z inverse.

## Parallel replay and diagnostics

Parallel paths certify independent work, preserve reverse order within dependent
components and retain scalar arithmetic within each record. Unsupported layouts
and supported scratch-allocation failures fall back to serial replay.

| Environment option | Meaning |
| --- | --- |
| `GPUPREPROCESSOR_DEVICE_RECOVERY=0` | Construct recovery records on the host |
| `GPUPREPROCESSOR_PARALLEL_FIXED_COLS=0` | Disable fixed-variable parallel replay |
| `GPUPREPROCESSOR_PARALLEL_BOUND=0` | Disable shared-bound parallel replay |
| `GPUPREPROCESSOR_PARALLEL_SUBCOL=0` | Disable singleton-column parallel replay |
| `GPUPREPROCESSOR_PARALLEL_COMPONENTS=0` | Disable dependency-component replay |
| `GPUPREPROCESSOR_PACKED_BOUND_RECORDS=0` | Use compact per-column bound records |
| `GPUPREPROCESSOR_DISABLE_BOUND_STRENGTHENING=1` | Disable scheduled bound strengthening |
| `GPUPREPROCESSOR_BOUND_STRENGTHENING_FIX_ONLY=1` | Keep only finite zero-width fixings from bound batches |

Bound strengthening preserves exactly-zero-cost bounds. Fixing-only mode excludes
tolerance-only near-fixing. The legacy round/time fields and redundant-bound
mode do not add rules to Main; see `PresolveParams` for their contracts.

`gpu_preprocessor_postsolve_replay --help` describes snapshot replay and KKT
checks. Snapshot writers use version 7; readers accept versions 1 through 7;
reference files use version 6. Comparisons require matching sparse-offset width
and compatible recovery semantics. Older projection-enabled references require
numerical reassessment. Serial/parallel vector equality alone is not an
optimality certificate; check the recovered point against the original LP.

See [Tests](../../../../README.md#tests) for build commands. Regression coverage
includes joint/structural/projected KKT, parallel replay, packed/shared records,
snapshots, Startup/folding handoffs, row-sparsification chains, ownership and the
public interface, including cleanup after allocation and transfer failures.
