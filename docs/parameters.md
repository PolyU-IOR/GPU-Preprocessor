# C presolve configuration

Start with `Config c=default_config()` and change named fields.

| Field | Default | Meaning |
| --- | --- | --- |
| `backend` | `CUDA` | CUDA execution; `AUTO` also requires CUDA; CPU presolve is rejected |
| `device` | 0 | Visible CUDA device index |
| `rules` | `ALL_RULES` | Bit mask; twelve rules plus CUDA folding by default |
| `max_iterations` | 10 | Maximum complete visits to the rule sequence |
| `max_seconds` | 1000 | Finite, nonnegative soft limit, checked between actions |
| `feasibility_tol` | 1e-6 | Activity infeasibility comparisons |
| `bound_tol` | 1e-6 | Bound-propagation progress and replay activity tolerance |
| `zero_tol` | 1e-10 | Screening of substitution/bound-propagation pivots |
| `max_substitution_support` | 20 | Maximum affine support; negative disables limit |
| `max_q_fill` | 128 | Maximum Hessian nnz growth per pivot; negative disables limit |
| `max_q_fill_ratio` | 2 | Maximum new/old Hessian nnz ratio; infinity disables |
| `monotone_empty_columns` | 0 | Enable box-gradient fixing of coupled A-empty columns |

Select rules by OR-ing their constants, e.g. `c.rules=FIXED_COLS` for the
analytic example. Set `rules=0` or `max_iterations=0` for an identity transform.
A zero time limit returns the unchanged model and `TIME_LIMIT`.

The fixed rule order is empty rows, singleton rows, activity checks, primal
propagation, parallel rows, fixed columns, empty columns, singleton dual
inference, singleton equality substitution, doubleton equality substitution,
dual fixing and parallel columns. Rules execute snapshot-based compatible
batches with a bounded per-visit work count before moving to the next rule. `ExecutionOptions` adds an
optional cleanup/light/global scheduler and batch limit without extending
`Config`. See [GPU execution](gpu-execution.md) for the policy and device-only CUDA execution. The default schedule remains fixed.

CUDA screens each equality pivot with a conservative structural fill
bound before selecting a batch. It credits only the pivot's removed diagonal;
potentially useful pivots can be deferred when the bound is conservative. The
batch absolute allowance is the sum of its pivot allowances, rather than one
128-entry allowance for an entire parallel batch. The exact resulting batch's
`max_q_fill_ratio` is still checked. Rejected batches shrink on GPU without
publishing a model or recovery records. Numerical and exact-coefficient guards
remain in force.

Parallel rows and columns require exact proportionality using a fused residual
check. Approximate matches never justify objective aggregation. Empty-column
minimization tests curvature against zero, retaining positive values below
`zero_tol`. Objective and Hessian updates are never thresholded by `zero_tol`.
Unrepresentable finite minimizers report `NUMERICAL_ERROR`, not unboundedness.

All rules preserve primal/dual recovery. Redundant-row deletion has zero replay
multipliers; singleton rows and propagation retain bound provenance; equality
substitutions retain quadratic-gradient data; compatible parallel columns keep
bound decomposition records. A skipped rule affects reduction strength, not
validity of the model returned.

Fixed-column elimination requires equal finite bounds. Near-equal bounds remain
intervals: replacing them by a midpoint can make a feasible chain of equalities
appear inconsistent after rounding. `bound_tol` controls propagation progress,
not permission to collapse a nonzero-width domain. Derived propagation conflicts
within the relative `feasibility_tol` are deferred after affine roundoff; existing
bounds are not widened and zero tolerance retains strict rejection.

`ALL_RULES` selects the twelve rules above plus `FOLDING` (mask 8191).
On CUDA, folding runs once on the input model before the main rule sequence.
Ordinary cleanup rules run only in main presolve, with either folding setting.
Use `c.rules &= ~FOLDING` to
disable it, or `c.rules = FOLDING` to run folding alone. The presolve CLI accepts
`--folding false`; `--no-folding` remains an alias. See [QP folding](qp-folding.md)
for partition conditions, numerical guards and recovery.
Unknown rule bits are rejected. `rule_info` reports calls, reductions and time
for any one supported rule, including a disabled rule with zero counters.
