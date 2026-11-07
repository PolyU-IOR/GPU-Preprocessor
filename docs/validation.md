# C/CUDA validation

```bash
make test GPU=0
```

For an existing build, use `ctest --test-dir BUILD_DIR --output-on-failure`.
`tools/test.sh PHYSICAL_GPU BUILD_DIR` selects an idle device by UUID and runs
the full suite, including host parser and API checks. CUDA tests fail if CUDA is
unavailable. The idle check does not reserve the selected GPU.
`qp_cuda_required` deliberately hides all GPUs and verifies that default/AUTO
presolve fails instead of selecting CPU, and that CPU presolve is rejected.

## Regression coverage

The core tests cover all twelve rules plus CUDA QP folding, analytic and randomized convex QPs,
objective identities, original-space feasibility and primal/dual KKT recovery.
They exercise tiny positive curvature, exact proportionality, underflow and
overflow guards, fill limits, infeasibility, unboundedness and time limits.
MPS tests include gzip integrity, ranges, continuations, malformed input and
exclusive output creation.

Recovery tests cover stable survivor maps, bound-dual transport, workspace reuse,
aliasing, state/workspace lifetime, failed-call isolation and concurrent replay.
GPU tests also check capped batches, resident retries, shared support, coupled
chains, zero candidate/tape downloads and device-vector recovery. Explicit CPU
replay of the same GPU tape provides an independent recovery check.

Device-resident tests compare cached excluded-variable activities with an
independent host long-double sum, covering 32/33-entry threshold cases,
infinite endpoints, cancellation, overflow and subnormal products, plus a
100003-entry row. A long-row propagation fixture checks bound-dual recovery
and original-model KKT. Deferred queues, singleton-dual invalidation and
coupled propagation chains exercise cache refresh and affected-row expansion.
Additional checks cover the first valid propagation row at AT positions
31/32/64/96, host-validation ordering violations at parallel partition boundaries,
and a download exceeding 64 MiB into freshly committed host memory. Frozen
before/after model hashes can check complete CSR, objective, bounds and cone
arrays when benchmarking the same case.

`gpu_folding` checks equitable A/Q partitions, complete Hessian aggregation,
unequal class sizes, signed multipliers, modular-signature collision rejection,
long refinement paths, numerical rejection, folding before ordinary cleanup,
and reverse folding/main composition. It also checks resident transfers and a
2048-variable device-input case. See [QP folding](qp-folding.md).

QP batch tests verify bitwise preservation of Q, c and the objective constant
under empty-row, singleton-row, activity and parallel-row reductions.

On a selected idle GPU, NVIDIA Compute Sanitizer can check `BUILD_DIR/bin/presolve_batch_tests --gpu`,
`BUILD_DIR/bin/device_resident_tests` and `BUILD_DIR/bin/folding_tests`.
Use the Compute Sanitizer supplied by the configured CUDA toolkit.

## Installed package

```bash
cmake --install build --prefix "$PWD/install"
cmake -S tests/consumer -B build-consumer -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build-consumer -j4
build-consumer/consumer qp/examples/qp_fixed.mps
```

The C consumer uses the installed QP header, library and legacy CMake package.
It checks ownership, diagnostics and original-space recovery using an analytic solution.
It runs presolve and recovery on CUDA. The standalone
`qp/examples/` project also uses the installed package.

For an installation containing LP and QP, the C++ consumer checks their
public headers and targets together without executing GPU kernels:

```bash
cmake -S tests/components -B logs/installed-components -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build logs/installed-components -j4
logs/installed-components/components
```

For module isolation, enable exactly one of `GPUPREPROCESSOR_BUILD_LP` and
`GPUPREPROCESSOR_BUILD_QP`. Both modules require CUDA.
Enable tests with `GPUPREPROCESSOR_BUILD_TESTS=ON` (or `BUILD_TESTING=ON` on first
configuration). The installed package exports the selected components only.

## Independent numerical check

The optional NumPy/SciPy test uses the small active-face oracle shared with the C
suite. The oracle equilibrates KKT rows and certifies returned stationarity.
It is neither installed nor called by production presolve.

```bash
python qp/tests/differential.py build/lib/libgpupresolve_qp.so NEW_TEST_DIRECTORY \
  --cases 1000 --oracle build/lib/libtest_oracle.so
```

The test rejects an existing output directory and distinguishes certified,
unresolved and failed cases. This coverage does not replace the positive
semidefiniteness precondition of the QP API.
