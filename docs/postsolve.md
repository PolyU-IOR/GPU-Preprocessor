# Recovery design

The reduced solver supplies unscaled vectors `(x_r,y_r,z_r)` under
`Q*x+c-A^T*y-z=0`. Recovery returns all three vectors in original coordinates.
An external solver must undo its own scaling and convert multiplier signs first.

## Record order and coordinates

Presolve changes the compact model after each reduction. The state maintains
compact-to-original maps for surviving rows and columns. CUDA captures owning
records, maps indices and compacts survivor maps entirely on device. Indices are
integers, never encoded in doubles.

Recovery scatters the reduced vectors through the final survivor maps into
original-sized scratch arrays. It then visits records in reverse chronological
order. Each record restores values directly into its stable slots. There are no
array insertions or index renumberings during replay. For a tape with `R` records
and `P` saved sparse entries, replay costs `O(n_original+m_original+R+P)` and uses
`2*n_original+m_original` scratch doubles. Structural presolve still rebuilds
sparse matrices and compacts its index maps; this change does not make presolve
linear-time.

Primal and dual replay must stay in the same reverse traversal. Grouping all
primal operations ahead of dual operations can read values from the wrong
reduction stage. In particular, do not sort records by type or original index.

## QP snapshots and dual transport

For fixed variables and affine substitutions, the record owns the eliminated
variable's Hessian row, linear coefficient and constraint column at the moment
of elimination. Substitution additionally saves `x_j=alpha+gamma*x_remaining`
and the pivot coefficient. After later reductions are undone, these snapshots
compute the stage gradient including quadratic cross terms. A snapshot does not
point into the input, reduced model, or temporary rule buffers.

| Record | Reverse action |
| --- | --- |
| Fixed column | Restore its value; recover its bound dual from the saved gradient and row multipliers |
| Affine substitution | Restore its value; recover the equality multiplier, including a transformed-bound multiplier when the row was retained |
| Removed row | Restore its zero multiplier |
| Singleton row / tightened bound | Transfer an incompatible bound multiplier back to its inducing row; adjust the other affected bound multipliers |
| Parallel rows | Place the merged multiplier on an originally active side, accounting for the proportionality ratio |
| Parallel columns | Split the aggregate value within the original boxes; map its bound multiplier using the ratio |
| Folding class | Copy class primal values; divide row/bound duals by their class sizes |
| Equality row combination | Restore the source multiplier by `y_source -= ratio*y_target` |

Bound records retain both old and new bounds. A multiplier on an unchanged
bound side stays on that variable, even when a solver's primal error exceeds the
replay activity tolerance. Otherwise approximate solutions could transfer an
upper-bound multiplier to an inducing lower-side constraint. Bound records cache
the pivot coefficient at commit, avoiding a coefficient
search on every recovery. Parallel-column records are constructed directly;
they do not allocate and discard fixed-column gradient snapshots.

When a bound multiplier is transferred, its pivot slot is set exactly to zero
and only the other columns receive coefficient updates. Computing that zero as
`z - pivot*(z/pivot)` can leave a rounding residue. Even a tiny residue is an
invalid multiplier on an originally infinite bound and gives an infinite dual
support value. CPU and CUDA replay use the same exact-zero action.

The state stores original dimensions rather than a complete duplicate of the
original model. Keep the caller-owned original model if original-space KKT
validation is needed. `compute_kkt[_backend]` performs that separate check.

## Workspace, ownership and errors

`postsolve` retains its original signature and creates a temporary workspace.
For repeated recovery, create a workspace once:

```c
PostsolveWorkspace *workspace = NULL;
Status status = postsolve_workspace_create(result.state, &workspace);
if (status == OK) {
    /* Repeat with fresh reduced vectors as needed. Outputs have original sizes. */
    status = postsolve_with_workspace(result.state, workspace,
                                          xr, nr, yr, mr, zr, nr, x, y, z);
}
postsolve_workspace_free(&workspace);
```

The create output pointer must initially be NULL. Workspace replay allocates no
memory for repeat calls with the same state. CUDA state defaults to a device
workspace; `postsolve_workspace_create_backend` can select CPU explicitly.
Changing states can rebuild the device cache. A workspace can be reused for a different state whose original row and
column counts fit its capacities. Its storage is independent of state lifetime.
Use a separate workspace for each concurrent call; immutable states can be
shared. Caller vectors and output buffers still require normal synchronization.

Inputs may share storage with outputs. Output arrays must be distinct from each
other and correctly sized. Dimension errors and nonfinite reduced inputs return
`INVALID_ARGUMENT`; nonfinite intermediate/final results return
`NUMERICAL_ERROR`. Outputs are copied only after successful replay, and a
workspace remains reusable after a failed call.

`postsolve_info` reports dimensions, record count, saved sparse entries, and
owned requested bytes. CUDA states return cached device segment and map
allocation bytes. Diagnostic host copies, allocator overhead, the reduced model
and workspace are excluded. These are diagnostic
counts, not a serialized state format or a stable private record ABI.

## Design and scope

QP recovery uses stable mapping, immutable recovery payloads and joint reverse
replay, as in the project's [LP recovery design](../lp/src/presolve/postsolve/README.md).
It adds stage Hessian/gradient transport and QP numerical safeguards.

CUDA states own a segmented device tape and stable maps; CUDA workspaces share
it immutably and build dependency levels/order on GPU.
Preparation batches independent segments across SMs with private access history
and a memory-bounded scratch allocation. It does not change the joint x/y/z
hazards, stable order within a level, or reverse barriers between segments.
Only group/status scalars are read by the host. An explicit CPU workspace exports
its own independent snapshot. `postsolve_device` keeps reduced and original
solution vectors on device, with no host solution scratch. Ordinary vector APIs
retain their host input/output boundary copies. See [GPU execution](gpu-execution.md).

## Folding boundary

CUDA folding records own class membership in stable original coordinates. Each
class retains one representative slot for the folded model. A separate immutable
tape segment preserves the stage boundary: reverse main reductions, including
their saved quadratic cross terms, then unfold to input coordinates. Normal
GPU replay uses one cooperative block per class, caching the representative
before any member (including that representative) is written. Diagnostic CPU
replay and host-tape CUDA replay implement the same formulas. See
[QP folding](qp-folding.md).

Resident folding segments bypass dependency preparation entirely. The plan vector
retains a null entry at each folding tape position, so reverse traversal still
executes main recovery followed by unfolding. Only
ordinary reduction segments contribute to dependency scratch sizing, job batches,
level/order construction and statistics downloads. A pure folding tape needs no
dependency scratch. Explicit host-tape replay still computes its generic record
dependencies, since it does not carry resident segment boundaries.
