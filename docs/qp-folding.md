# GPU QP folding and unfolding

`FOLDING` reduces an equitable row/column partition of the current convex QP.
It is included in `ALL_RULES`. The original twelve rules remain independently
selectable. All presolve execution requires CUDA.

```c
Config config = default_config();
config.backend = CUDA;         /* Includes folding. */
config.rules = FOLDING;         /* Optional: folding alone, without cleanup/main rules. */
/* config.rules = ALL_RULES & ~FOLDING;  To disable folding. */
```

The presolve CLI accepts `--folding false` to disable folding. For example, after
selecting an idle physical GPU with `tools/gpu_idle_guard.sh`:

```bash
build/bin/gpu_preprocessor_qp --gpu 0 qp/examples/qp_folding.mps
build/bin/gpu_preprocessor_qp --gpu 0 --folding false qp/examples/qp_folding.mps
```

This example folds from 4 rows/4 columns to 2/2, with nonproportional original
Hessian columns. Its original optimum is `(1,1,1,1)`, with objective `17.25`.
`rule_info(state, FOLDING, &info)` reports one stage call, elapsed time and the
number of removed rows plus columns. The CLI prints the latter two metrics.
`max_iterations=0`, `rules=0` and the existing time-limit contract still apply.

## Partition and model

Rows initially share a class only if their lower/upper bounds are exactly equal.
Columns initially share a class only if their costs and lower/upper bounds are
exactly equal, including matching infinite endpoints. Every refinement preserves
old class membership and compares:

1. For columns: the sums over each row class in `A^T`, and over each **old**
   column class in `Q`.
2. For rows: the sums over each **updated** column class in `A`.

Both partitions must stop changing. Hessian refinement can continue while row
classes stay unchanged, so both counts participate in the stopping test. The
round bound remains `1 + m - initial_row_classes + n - initial_column_classes`;
no smaller cap or minimum reduction threshold is used. An 80% class-count cutoff
as in the LP reference skips useful folds on this QP set, and a 32-round cap also
loses a large fold. Empty-sided models bypass folding. Counts that already reach
both full dimensions exit immediately.

For row class B and column classes C,D, the reduced data are

```text
Abar[B,C] = sum(A[i,j], i in B, j in C) / |B|
Qbar[C,D] = sum(Q[j,k], j in C, k in D)       = S^T Q S
cbar[C]   = sum(c[j], j in C)
```

Bounds are copied from the common class data; the objective offset is unchanged.
Both triangles and all cross terms of the full symmetric Hessian are included.
No nonzero entry is discarded by a tolerance. With PSD Q and the block-sum
identities, copying a folded feasible point preserves its objective. Forward
averaging of an arbitrary feasible original point can decrease its quadratic
objective; folding does not assert that every original objective is preserved.

## Device implementation and numerical acceptance

`qp/src/presolve/rules/rule_folding.cuh` follows the LP reference's product-based
refinement, extended to `A^T`, `Q` and `A`. Two salted modular projections encode
binary64 coefficients in the prime field `2^31-1`. Powers of two (including
subnormals) have field representations, so equal exact block-sum vectors always
give equal projections, independently of accumulation order. Each product uses
one cooperative warp per CSR row. Matrix encodings and double-buffered partition
storage are allocated once per folding stage and reused across rounds. Class
sorting uses stable integer radix passes on the signature and old label, avoiding
indirect sparse-vector comparisons inside a comparison sort.

Projected matches are candidates only: after both partitions stop changing,
the complete `A^T`, `Q` and `A` block-sum vectors are constructed on GPU and
compared within every class. Hash matches are resolved by full-vector comparison.
A projection collision or failed numerical check rejects the entire optional
stage. Thus no probabilistic assumption, approximate grouping or solver tolerance
authorizes a merge. Exact sparse sorting/aggregation is performed once for the
certificate and folded matrices, rather than three times in every round.

Block sums use error-free two-sum operations and a checked two-component
expansion. Accepted sums are exact sums of the binary64 input coefficients;
comparisons include the low component. If an addition would require a third
component or overflow, the entire optional folding transaction is discarded.
This can skip a valid reduction on extreme data, but does not accept an
uncertified partition. Rounded-equal sums such as `1e16+1` and `1e16+2` remain
distinct. No feasibility/bound/zero tolerance licenses a merge.

Reduced matrix values and costs are stored as binary64, with ordinary rounding;
A blocks also undergo row-size division. Nonfinite or underflowed nonzero
transforms are rejected. A rejected transaction leaves the model and recovery
tape unchanged and increments `batch_info().numerical_rejections`. Original-space
KKT checking is still needed after an approximate solver finishes.

Partitions, sparse matrices, stable IDs and owning tape payloads stay on GPU.
Only counts, error flags and time-budget decisions reach the host.
Refinement writes both class counts on device and reads them together once per
complete row/column round. The row update consumes the new column labels without
waiting for a host column count. Initial grouping and final certification retain
their boundary checks. Host input is
uploaded once; `presolve_device` performs no explicit host-to-device payload
copies. `presolve_resident` exports neither the reduced model nor the tape.
Ordinary `presolve` retains its documented final host-model export for consumers
that accept host CSR data.

## Recovery and stage order

The stage order is folding -> main rules. Folding receives the input model
without preceding cleanup. Empty-row, singleton-row, fixed-column and empty-column
rules remain in main presolve, with either folding setting. Main preserves the
selected fixed/tiered scheduling policy; `cleanup_rounds` controls only tiered
main scheduling. Recovery reverses main first, then unfolds:

```text
x[j] = xbar[C]
z[j] = zbar[C] / |C|
y[i] = ybar[B] / |B|
```

Class records own their original stable member IDs and the surviving class
representative. They occupy a separate immutable tape segment; later captures
cannot append to it. CUDA uses a cooperative block per class, caching the source
value/multiplier before writing members. This includes the representative and
avoids in-place scaling races. Resident folding segments require no dependency
levels or ordering arrays: preparation excludes them from scratch sizing and
planning, retains their tape positions, and replay launches the unfolding kernel
at the original reverse-stage boundary. Pure folding tapes allocate no dependency
scratch and download no dependency statistics. Workspace reuse, independent CPU diagnostic
replay, explicit host-tape CUDA replay, and device-vector postsolve are supported.
No inverse Hessian or positive slack is needed, so zero/singular Q, fixed/free
bounds and nonunique multipliers follow the same map. Ray and infeasibility
certificate unfolding are outside the existing solution-recovery API.

## References and verification

The QP implementation extends the project's LP folding design to a quadratic
objective. See [LP folding](../lp/src/folding/) for refinement, reduction and
mapping code. QP uses its own model, state ownership and CUDA utilities.

`qp/tests/test_folding.cu` checks full Q/A/c aggregation, unequal class sizes,
nonproportional columns, signed cross terms, singular/zero Q, lower/upper/fixed/free
intervals, nonzero duals, the six-vertex path refinement counterexample, exact
initial-data distinctions, tiny hidden asymmetries, deliberately colliding modular signatures, long
refinement paths beyond 32 rounds, small exact folds, transaction rejection,
folding before ordinary cleanup, subsequent quadratic-gradient recovery, device input,
resident transfer counters, repeated CPU/GPU/device replay, and empty/disabled
stages. Small reduced/original optima come from an independent active-face oracle;
recovered primal/dual feasibility, complementarity and objective are checked in
the original model. A 2048-variable cycle also exercises cooperative unfolding.

## Optional profiling

`GPUPRESOLVE_FOLD_PROFILE=1` prints class counts, refinement rounds, exit reason
and synchronized phase timings to stderr. It adds diagnostic synchronizations;
leave it unset for performance measurements. Normal execution introduces no
extra profiling synchronization. `small_reduction` indicates that both partitions are already singletons, so the
stage is the identity. `certificate_rejected` reports a conservative numerical
rejection before model or tape mutation.
