# Presolve implementation

`gpu_presolve.cu` owns scheduling, including the epigraph-first variable-pair
dispatch. Public rule declarations remain under `include/gpu_preprocessor/lp/presolve/rules/`.

Each ordinary or Startup strategy has one implementation file in `rules/`.
Rule-specific kernels and helpers live in that file. Empty rows and empty
columns have separate entries. Piecewise-linear types share one directory:

```text
rules/
  rule_poisson.cu
  rule_empty_rows.cu
  rule_empty_columns.cu
  rule_doubleton_equations.cu
  rule_parallel_rows.cu
  rule_parallel_columns.cu
  rule_*.cu
  piecewise_linear/
    absolute_value.cu
    epigraph.cu
    epigraph.hpp
    shifted_antipodal.cu
    shifted_antipodal.hpp
```

The piecewise-linear headers declare analysis state shared with dispatch and
tests; each type's implementation stays in its corresponding `.cu` file.
`parallel_signature_common.cuh` and `projection_support.cuh` in this directory
are shared support for multiple strategies. Other model, plan and recovery
support stays here too. See [postsolve](postsolve/README.md) for inverse replay.

## Certified Poisson Startup reduction

[`rules/rule_poisson.cu`](rules/rule_poisson.cu) owns the complete certificate,
device kernels, observation/source maps, FFT plans, response LP and original-space
inverse. These implementation details are private to its translation unit;
public headers expose no cuFFT types.

The accepted layout is a cubic interior grid with zero Dirichlet boundary and
one source column per interior node. Every row must contain exactly the seven
stencil coefficients and its unit source coefficient, with one positive common
scale and zero right-hand side. Boundary temperatures are fixed to zero.
Interior temperatures are either free or fixed observations, and all
temperature costs are zero. Fixed sources must be zero; active sources retain
their original bounds and costs. Duplicate stencil entries, perturbed
coefficients, unsupported bounds and nonfinite data reject the specialization.

The response has one row per observation and one column per active source.
The objective constant is preserved. Acceptance requires positive finite Green
entries, at most 16 million response entries and fewer than half the original
nonzeros. FFT buffers and plans must fit the original capacity gates. An
unsupported structure or recoverable allocation/FFT failure leaves the input
unchanged and continues ordinary presolve.

The default schedule is row sparsification, Poisson, remaining Startup rules,
then Main. Successful Poisson bypasses folding: small Green entries can collide
under ordinary folding signatures. Main retains all nonzero response entries.
Poisson misses use the ordinary Startup/folding/Main schedule. The shared
iteration/time budget, Startup switch and structure-specific master switch
still apply. `startup_poisson = nullptr` disables this strategy; custom callbacks
remain supported. `GPUPREPROCESSOR_POISSON_PRESOLVE` controls only the built-in
strategy (`0`: off; unset or `1`: on; other values warn and disable it).

Recovery applies Main and ordinary Startup inverses first. The retained Poisson
inverse solves for original interior temperatures and row duals, scatters active
source values and duals, and computes fixed-variable duals as `c - A^T y`.
Boundary temperatures remain zero. A preceding row transform finally applies
`S^T` to row duals. Returned device vectors are independently owned; input and
reduced vectors remain borrowed. Host postprocess wraps the same device path.
The caller must retain the original A/AT and vectors until recovery completes;
CSR-only owners can still use the explicit column-access callback.

`test_poisson_startup` covers an independent finite sine-series reference,
original KKT, rejection gates, default dispatch and disabling, budget exhaustion,
folding bypass and miss behavior, CSR-only entry, host/device replay, allocation
failure and retry, tiny coefficient preservation, and row/response inverse
composition, including recovery when Main eliminates the entire response.
Run it in both sparse-offset builds. The extreme-scale fixture
checks retained coefficients, finite replay and objective agreement; it does
not claim tight original-space KKT, which can amplify cancellation in large
reduced duals. External solver users should check their recovered original KKT.
