# Native C API and ownership

The QP public API is `qp/include/gpupresolve_qp.h` in `gpupresolve_qp`.
Link `GPUPreprocessor::qp`; existing C consumers can use
`GPUPreprocessorQP::gpupresolve_qp`. The core has no external solver dependency.

## Model

`Problem` owns `Q`, `A`, `c`, variable bounds, row bounds and an objective
constant. `Matrix` uses 32-bit signed indices and sorted, duplicate-free,
zero-based CSR. Duplicate triplets and objective coefficients use compensated summation. `Q` must contain both triangles with identical coefficients.
`matrix_from_triplets` sorts and sums duplicates, retaining every nonzero
coefficient, including small positive curvature. Bounds may be infinite;
coefficients, objective values and objective constants must be finite.

Initialize objects to zero. `problem_alloc` allocates empty matrices and
free bounds. `problem_clone` makes an independent copy. Use `matrix_free` for a
standalone matrix built with `matrix_from_triplets`; `problem_free`
releases both matrices and all vectors of a problem. MPS input uses the
MPS default lower bound zero and upper bound infinity. The matrix helper and
reader replace an existing initialized model only after successful construction.

`validate` checks dimensions, CSR invariants, coefficients and symmetry.
It does not certify positive semidefiniteness. Contradictory finite bounds are
reported as `INFEASIBLE` by presolve.

## Presolve

`presolve` reads a caller-owned problem and writes a fresh `Result`.
Free a previous result before using that result object again. The input is never
mutated. `OK` and `TIME_LIMIT` contain a usable equivalent reduced model
and reusable recovery state. Terminal infeasible/unbounded results and errors
have no reduced model or recovery state. `OK` means reduction succeeded;
it is not a declaration that the QP has been solved or fully reduced.

Every successful reduction records its reverse operation. Equality elimination
uses `x_j=alpha+gamma*x_remaining`, transforms both Hessian triangles and the
objective constant, and shifts all remaining constraints. If an eliminated
variable has finite bounds, the pivot row is replaced by its transformed bounds.
Fill limits conservatively skip a substitution; they never discard Hessian fill.

Ordinary presolve APIs return host-owned matrices and vectors. CUDA keeps all
rule data, compatible selections, stable maps and owning tape records on device;
it exports the final host model once. CUDA calls restore the caller's device and
never reset its context. `presolve_resident` omits this final export;
`result.reduced` then contains dimensions/nnz only, with NULL host pointers until
`result_download` is called. `device_problem_view` returns borrowed device
CSR/vector pointers. `presolve_device` accepts device inputs, copies them
only device-to-device and builds AT itself. See [GPU execution](gpu-execution.md).

All presolve entry points require CUDA. `default_config()` selects `CUDA`;
`AUTO` is accepted as an alias for CUDA, and `CPU` returns `INVALID_ARGUMENT`.
Unavailable CUDA or a CUDA runtime failure returns `CUDA_ERROR`. Rejected combined plans shrink and retry on device. A fill-blocked
single pivot stays in the model; unrepresentable arithmetic reports a numerical
error. No CUDA rule invokes a scalar CPU fallback. Existing fallback diagnostic
fields retain their layouts and report zero.

`presolve_with_options` accepts a separate `ExecutionOptions`, preserving
existing `Config` layouts. Defaults use fixed scheduling and batches up to
65536 candidates. `SCHEDULE_TIERED` enables cleanup/light/global staging.
Options must have a valid schedule and positive batch limit, cleanup rounds and
global period.

`device` is a visible CUDA index, so expose a physical GPU through
`CUDA_VISIBLE_DEVICES` and then select visible index zero.

## Recovery and KKT

Call `postsolve` with the unscaled reduced primal, row dual and bound dual
vectors and their explicit dimensions. Allocate outputs using the original model
dimensions. Null vectors are allowed only for zero dimensions. Inputs and replay
records are not consumed. Repeat calls are supported. These vector APIs use host
memory; `postsolve_device` uses six device vector pointers and a CUDA workspace
on the state's GPU, without host solution scratch or tape staging.
Recovery state owns its snapshots and can outlive the input and reduced models.
Original model storage is not duplicated inside the state.

`postsolve_workspace_create` allocates reusable scratch storage; initialize
its output pointer to NULL. The default workspace uses the state's presolve
backend/device. `postsolve_workspace_create_backend` selects CPU or CUDA
explicitly. Repeated `postsolve_with_workspace` calls on one state allocate
nothing after first use; switching states can rebuild dependency scratch.
Resident CUDA workspaces share the immutable device tape. An explicit CPU
workspace instead makes an independent host reference snapshot when needed.
Each concurrent call needs its own workspace; the immutable state may be shared.
A workspace can outlive its creator state and serve another state that fits its
row/column capacities. Free it separately with `postsolve_workspace_free`.
`postsolve` remains the convenience entry point using a temporary workspace.
Inputs may alias outputs, but output arrays must be distinct from one another.
A failed recovery leaves outputs unchanged and the workspace reusable.
`postsolve_info` provides tape dimensions, record and payload counts, and
owned requested bytes. See [recovery design](postsolve.md) for the indexing,
byte-count scope and joint primal/dual dependencies.

The stationarity convention is `Q*x+c-A'*y-z=0`. Lower-side duals are positive,
upper-side duals negative, equality multipliers unrestricted. Undo solver scaling
and convert its signs before recovery. Equality duals are recovered using the
saved objective gradient at each elimination stage, preserving quadratic cross
terms across substitution chains. Bound-tightening replay transfers multipliers
back to the inducing rows and adjusts the other stationarity components.

`compute_kkt` uses original-space CPU products;
`compute_kkt_backend` also supports CUDA sparse products. Metrics follow the
previous raw infinity-norm convention: primal and dual objectives, normalized
primal infeasibility, normalized stationarity/domain infeasibility and relative
duality gap. They do not project or repair supplied duals. Nonfinite vectors
produce infinite residuals. Overflow of finite-input products returns `NUMERICAL_ERROR` with infinite residuals. Array lengths for this utility are determined by
the model; callers must provide correctly sized arrays.

`state_free(&result.state)` releases only recovery state and clears the
pointer. `result_free` also releases the reduced problem. Free functions
accept zeroed objects and may be repeated. Shallow copies of owning structs or
use of a stale copy after free are invalid C ownership operations.

## MPS and CLI

`read_mps` accepts plain/gzip MPS, free whitespace records and standard fixed
records with ordinary names, RHS, RANGES, LO/UP/FX/FR/MI/PL bounds, QUADOBJ,
QMATRIX and named objective QSECTION sections. A QSECTION attached to a constraint is rejected. Continuation lines can omit
a column/RHS name. It selects the first RHS/range/bound vector. Names containing
embedded whitespace and nonstandard vendor extensions are unsupported and
rejected. MAX, integer, semicontinuous and unknown constraint sections are
rejected. One triangle is mirrored; conflicting full Hessian entries are errors.
Objective RHS is negated to obtain the objective constant.

`write_mps` creates a new MPS file exclusively. It never replaces an existing
file. Reduced files can be read by an external solver using the same `0.5*x'Q*x`
convention. The reader does not implement every historical MPS dialect.

Gzip trailers are checked before returning a model; CRC errors and truncated
streams return `IO_ERROR`. Continuations retain the active RHS/range vector
selection, including when an unselected vector is being skipped. Unrepresentable
range widths are rejected before an output file is created.

Static CUDA installations export their CUDA runtime and C++ support-library
dependencies. A C-only downstream project can use `find_package(GPUPreprocessorQP CONFIG)`;
supply `CUDAToolkit_ROOT` for the toolkit version used to build the archive.
The installed example locates its MPS beside the installed data, so a relocated
installation does not depend on the original source checkout.

## Presolve transfer diagnostics

`transfer_info(const State *, TransferInfo *)` exposes presolve traffic counters.
`TransferInfo` contains uint64 counters
`host_to_device_bytes`, `device_to_host_bytes`, `host_to_device_calls`,
`device_to_host_calls`, `model_uploads`, `resident_commits`, `bound_updates`,
`device_candidates`, and `downloaded_candidates`. It covers explicit presolve
payload copies only; CUDA/Thrust internals and solver/postsolve traffic are not
included. A null state/output is
`INVALID_ARGUMENT`. See [GPU execution](gpu-execution.md).

`batch_info(const State *, BatchInfo *)` exposes batch diagnostics with uint64
`retries`, `fill_rejections`, `numerical_rejections`, and `cpu_fallback_phases`.
The retained `cpu_fallback_phases` field is always zero.
Null state/output arguments return `INVALID_ARGUMENT`. Queries read the
completed state's host counters and perform no device synchronization or copy.
Fill rejection does not imply CPU execution: compatible batches shrink and retry
without reuploading the resident source model.

`rule_info(state, rule, &info)` returns `RuleInfo` with call count,
accepted reduction count and inclusive wall time. Supply exactly one rule bit;
zero, combinations and unknown bits return `INVALID_ARGUMENT`. Disabled
rules return zero counters. Only the thirteen bits in `ALL_RULES` are valid, including `FOLDING`.

`residency_info` separates the pre-export
iteration counters from `model_export_bytes`, explicit state `tape_export_bytes`
and device tape storage. Device-input iterations have zero H2D payload; host
inputs upload once. Only status/count scalars return during presolve. Counters
exclude CUDA/Thrust internal copies, diagnostic workspace exports and solver
traffic. The private state-inspection helpers used by tests are described in
[GPU execution](gpu-execution.md).
