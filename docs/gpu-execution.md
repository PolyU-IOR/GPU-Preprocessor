# GPU-resident execution

CUDA presolve keeps the model, candidates, compatibility masks, stable IDs and
owning recovery records on the device. All twelve rule families use device
numerical decisions and sparse transformations. The CPU launches work, checks
status/count metadata, applies scheduling limits and formats diagnostics. CUDA
execution never dispatches a scalar CPU presolve rule.

## Data ownership and transfers

| Data | CUDA owner | Host boundary |
| --- | --- | --- |
| A, Q, AT, objective and bounds | Owning device plan | One initial host upload, or device-to-device input copy |
| Candidates and compatible selections | Device buffers | Counts/status only |
| Survivor maps | Device buffers | Explicit diagnostic export only |
| Stage Hessian/constraint snapshots | Segmented device tape | Explicit diagnostic export only |
| Final reduced model | Device plan | One export for ordinary host-CSR APIs; none for resident APIs |
| Recovery workspace and dependency groups | Device buffers | Counts/status only |
| Solution vectors | Device buffers for device API | Host vector APIs upload/download at their boundaries |

There is no per-batch host CSR mirror, selected-index upload, candidate download,
bound-pair download or host recovery-record assembly. The final plan releases
candidate, signature, queue and batch scratch buffers before it is published.
The state keeps only the reduced device model, stable maps and recovery tape.

Input reading remains a CPU boundary operation. The QP host API validates its
host input on CPU. Device inputs are validated
on GPU, including CSR structure, finite values,
Hessian symmetry and box/row bounds. Input buffers are borrowed and unchanged;
the plan owns its device copies and constructs its own transpose.
CSR validation of device input cooperates across each row's entries, followed
by vector and Hessian-symmetry checks after storage has been validated. AT is
constructed when a rule first needs it. Pure row reductions and a fixed-column
pass with no fixed columns do not build it. A public `device_problem_view`
request still returns a complete AT, building it on first request if needed.

## APIs

The ordinary `presolve` and `presolve_with_options` APIs preserve their
host-CSR output contract. CUDA execution exports the final model exactly once;
its recovery state and tape remain on the device.

Use `presolve_resident` to avoid that export:

```c
Config cfg = default_config();
cfg.backend = CUDA;
Result result = {0};
Status status = presolve_resident(&problem, &cfg, NULL, &result);
if (status == OK || status == TIME_LIMIT) {
    DeviceProblemView view;
    device_problem_view(result.state, &view);
    /* Pass view's device pointers to a compatible GPU consumer. */
}
result_free(&result);
```

Until `result_download(&result)` is explicitly called, `result.reduced`
contains dimensions and nonzero counts, with NULL host array pointers. Do not
pass that metadata-only model to a host solver or `validate`.
`result_download` is idempotent and performs no copy after the first export.

`presolve_device(&device_input, &cfg, &options, &result)` accepts device CSR
and vector pointers directly, with no H2D model transfer. `cfg.device` must match
the input view. CSR offsets/indices are int32, values are float64 and Q is the
full symmetric Hessian. The input AT view is not used; AT is built on device.
All device views remain valid only until their owning state is freed.

Every presolve entry point requires CUDA. The default backend is `CUDA`;
`AUTO` also selects CUDA, and `CPU` is rejected. Unavailable CUDA and CUDA runtime
failures are reported as `CUDA_ERROR`.

## Batch transactions and numerical checks

Detection and stable compaction run on GPU. Simple independent families select
candidates in parallel. Structural families use a deterministic device greedy
ownership pass, preserving the reference compatibility policy and record order.
The latter pass is sequential on GPU and can remain a bottleneck for very large
conflicting candidate sets; GPU residency does not imply that every operation
is parallel.

Device code builds survivor masks, the affine map T, CSR row counts/prefix sums,
`A*T`, `T^T*Q*T`, objective/row-bound shifts, and the next stable maps. Stable COO
reduction and compensated sums retain the arithmetic order; only the upper
Hessian triangle is reduced, then mirrored. No nonzero objective coefficient is
dropped by a tolerance. The objective constant is accumulated on GPU with a
compensated sum, replacing the previous host long-double accumulation.

Fill estimates screen pivots before selection. Actual combined A/Q fill and
finite/symmetry checks precede commit. Rejected combined batches shrink against
the same source device model. A fill-blocked singleton is left in the model;
unrepresentable arithmetic reports `NUMERICAL_ERROR`. Neither case invokes
a CPU rule. Accepted owning tape payloads are captured before source buffers
are freed, and their indices are globalized on GPU using stable original IDs.

Propagation keeps a resident active-column queue. The first pass considers every
column; bound changes enqueue variables sharing constraint rows. Deferred
proposals remain queued. Singleton dual row-bound changes invalidate affected
variables; topology changes start a fresh queue. Epoch markers deduplicate the
queue, and device sorting preserves ascending variable order. Only accepted
bounds are updated; complete bound arrays are not cloned per propagation batch.

For rows with at least 32 entries, propagation caches outward-rounded finite
activity sums and separate infinite-contribution counts for each bound snapshot.
Removing the target's same rounded product with outward subtraction gives a
conservative excluded-variable activity; infinite terms are handled by counts,
without infinity-minus-infinity. Short rows retain the direct scan. The long-row
index list is built once per topology and the sums are refreshed before each
detection pass. This replaces repeated full-row scans for every incident column
by one cooperative row scan and constant-work queries. Free rows are skipped.
On matrices with at least 32 entries per column on average, a warp cooperates
on each active column's contiguous AT entries. A ballot selects the earliest
eligible row, preserving the scalar detector's choice and recovery chronology.
Affected-row marking likewise reads AT cooperatively. Validated independent
candidates use a direct forward/reverse index fill instead of a second filter.

Bound changes first mark and deduplicate affected rows on device. A compact
device row queue then expands each affected row once in parallel, rather than
once for every changed variable incident to that row. These scratch buffers
are released with the other presolve scratch; recovery tapes are unchanged.

For a single doubleton with an empty pivot Q row, zero pivot linear cost, and
pivot/survivor column degrees at most 4096, a device-computed specification drives
a direct CSR/CSC rewrite. It preserves compensated coefficient arithmetic and
checks the two views agree. No pivot descriptor, changed row, objective vector
or matrix crosses to CPU. Other pivots use the general checked GPU transform.
This is a single-pivot optimization, not composition of an entire chain.

Bound intersections use stable per-target prefix scans. Parallel-column bound
addition preserves chronological order inside a cooperative group. Recovery
records save the preceding and following bounds on GPU. Conservative directed
interval arithmetic is unchanged. Snapshot-based parallel propagation can differ
from sequential updates within `bound_tol`; validate recovered solutions and
objective/duals in the original model when comparing batch sizes.

## Device recovery

The tape uses owning segments rather than repeatedly doubling a multi-gigabyte
allocation. Segments are immutable after presolve and shared by independent CUDA
workspaces. Stage snapshots never point into a live model or temporary scratch.

For each segment, GPU code builds reverse read/write dependency levels over
joint x/y/z accesses, then constructs the replay order and level starts on GPU.
Independent segments build these levels concurrently on separate SMs. Each has
private read/write history, so batching preserves exactly the same per-segment
levels and replay order. GPU job descriptors contain only pointers/counts set by
kernel arguments; no tape is downloaded or uploaded for planning.

Transient planning scratch targets at most one eighth of currently free device
memory and 4 GiB, with a one-segment minimum and at most 128 concurrent segments.
Allocation pressure halves the batch until it fits; scratch is per workspace
and released after preparation. The single-segment minimum can exceed that soft
budget on exceptionally large models. Repeated recovery uses only the retained
plan and solution buffers.

Only two scalar group statistics per segment are read by the host. Segments replay in reverse
chronological order. Wide, shallow levels use parallel kernels; deep levels use
a cooperative block to avoid thousands of host kernel launches. This preparation
and cooperative traversal can still cost time for very long dependency chains.

The LP reference motivates parallelizing independent recovery work. Its general
connected-component interpreter is less suitable for large coupled QP bound
tapes: transitive conflicts can collapse a segment into a serial component.
This implementation retains finer dependency levels and parallelizes their
preparation across segments instead.

Use `postsolve_device` with a CUDA workspace and six device vector pointers:

```c
PostsolveWorkspace *workspace = NULL;
postsolve_workspace_create(result.state, &workspace);
status = postsolve_device(result.state, workspace,
    xr_device, reduced_n, yr_device, reduced_m, zr_device, reduced_n,
    x_device, y_device, z_device);
postsolve_workspace_free(&workspace);
```

The workspace must use the state's GPU. Device vector inputs/outputs are copied
only device-to-device; status scalars return to CPU. No host solution scratch is
allocated for this device-only path. Host-pointer APIs remain available and
allocate their host scratch on first use. Repeat calls reuse their workspace.

An explicit CPU workspace builds its own independent host snapshot when needed.
Tests can inspect private maps and records through `state_export_maps` and
`state_export_host`, declared in `qp/src/internal.h`. Call them before sharing a
state across threads. These diagnostic exports preserve the resident tape;
CUDA recovery never calls them.

## Scheduling

The default is the original fixed rule order and batch limit 65536. Staged
scheduling defaults to three cleanup rounds and global period three.
Opt into staged scheduling without changing
`Config`:

```c
ExecutionOptions options = default_execution_options();
options.schedule = SCHEDULE_TIERED;
options.batch_limit = 65536;
```

Staged scheduling repeats cheap cleanup, visits light rules, and visits global
rules periodically, when progress stops or on the final iteration. On CUDA models
with at least 4096 remaining columns, each propagation visit yields after 32
successful batches. Doubleton visits yield after 16 consecutive batches smaller
than `max(4, n/65536)` reductions. Later passes may continue. These work limits can
change the valid partial reduction; default fixed visits retain their old limit.
`ALL_RULES` selects the twelve reduction rules and the CUDA-only `FOLDING` stage.
Folding runs once on the input model before the main scheduler; ordinary cleanup
remains in main presolve and does not depend on the folding switch. Its device
signature refinement and exact block-sum certification include `A`, `A^T` and `Q`; all partitions, matrices and
recovery records remain on device. Only counts and status reach the host.
See [QP folding](qp-folding.md) for numerical guards and the stage boundary.

## Transfer diagnostics and integration

`transfer_info` counts explicit presolve H2D/D2H payload copies, including the
initial host upload and any compatibility final-model export. It excludes
CUDA/Thrust internal copies, solver and postsolve traffic.
`residency_info` reports an immutable `iterations` snapshot taken before any
final host export, plus explicit state `model_export_bytes`, `tape_export_bytes`
and allocated device tape bytes. Iteration statistics include initial input
preparation: H2D is one host model upload, or zero for a device input; D2H consists
only of status/count metadata. Candidate downloads and CPU fallback counts are
zero. Explicit CPU-workspace reference copies are outside these presolve counters.
Use a CUDA trace to measure runtime/library transfers independently.

Consumers that accept host CSR can use the final reduced-model export and the
host-vector recovery API. These boundary copies are outside resident presolve
iterations. GPU consumers can use the device model and device postsolve APIs
to keep model and solution arrays on device across the solver handoff.

## Validation and profiling

Select an idle physical card with `tools/gpu_idle_guard.sh` before GPU execution.
The public device index is relative to `CUDA_VISIBLE_DEVICES`.

```bash
source tools/gpu_idle_guard.sh
select_idle_gpu 0
ctest --test-dir build --output-on-failure
build/bin/device_resident_tests
```

The device contract test checks initial-upload-only traffic, zero candidate/tape
exports, device-input presolve without H2D, device pointer types, device recovery,
invalid host pointers, zero dimensions, original KKT, one-time compatibility
exports and workspace lifetime. Its default profiler API range covers host-input
resident presolve. `--profile-device-input` captures presolve from device buffers;
`--profile-postsolve` captures device workspace preparation and recovery.
The numerical and batch suites also exercise GPU-resident execution and
independent CPU recovery. See [validation](validation.md) for test commands.
