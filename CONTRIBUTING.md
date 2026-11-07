# Development

Reuse CUDA build directories during development; store run logs under `logs/`.
Machine-specific
compiler, dependency and GPU settings belong in local CMake arguments.

```bash
make
make test GPU=1
```

`BUILD_DIR`, `BUILD_JOBS`, `CUDA_ARCH`, `CMAKE`, `CTEST` and `CMAKE_ARGS` are
configurable. GPU tests use `tools/test.sh` to check that the selected physical
GPU is idle and expose it as device zero. This check is not a scheduler lock.

For a library-only build:

```bash
cmake -S . -B build-library \
  -DBUILD_TESTING=OFF -DBUILD_TOOLS=OFF -DBUILD_EXAMPLES=OFF
cmake --build build-library -j4
```

Root CMake owns project options; `lp/` and `qp/` own their libraries, tests and
examples. Keep native
reduction and recovery algorithms in the corresponding module.
`cmake/Install.cmake` exports the unified `GPUPreprocessor` package and the
compatible `GPUPreprocessorQP` package for existing C callers.

Read the [QP source map](qp/src/presolve/README.md) and
[recovery invariants](docs/postsolve.md) before changing a reduction.

| Test source | Responsibility |
| --- | --- |
| `lp/tests/`, `lp/mps_reader/tests/` | LP rules, folding, recovery, GPU transfers and MPS parsing |
| `qp/tests/test_presolve.c` | Rule families, random convex QPs, affine identities, MPS and original-space KKT |
| `qp/tests/test_edge_cases.c` | Numerical, parser and rule-mask edge cases |
| `qp/tests/test_presolve_batches.c` | Batch independence, shared support, capped plans and resident retries |
| `qp/tests/test_folding.cu` | Equitable QP classes, Hessian/cost aggregation, stage composition and device unfolding |
| `qp/tests/test_device_resident.cu` | Device pointers, transfer invariants and explicit diagnostic exports |
| `qp/tests/test_postsolve.c` | Workspace ownership, aliasing, failure isolation and concurrent replay |
| `qp/tests/support/qp_oracle.c` | Independent small-QP active-face solver |
| `tests/consumer/` | Installed C header/library/package integration |
| `tests/component_link.cpp`, `tests/components/` | Both public modules linked together, from the build tree and installed package |

Rule changes need an original-model solution check, including duals when
available. A reduced objective alone cannot validate dual transport. Run the
relevant CUDA tests and Compute Sanitizer checks; see [validation](docs/validation.md).
Benchmarks and numerical logs are local artifacts, excluded from source sync.

Implement presolve rules directly in CUDA. Keep detection, sparse transformations
and recovery metadata on device;
use analytic fixtures and original-space KKT to validate new GPU rules. Do not
add a CPU fallback or stage model/candidate arrays through host memory. Document
any necessary host boundary and check transfer counters in regression tests.

Name files for the operation they implement. CUDA helpers use `cuda_` followed
by that operation; keep public declarations in each module’s `include/` and private
contracts in its `src/`. External applications can use the public model and
recovery APIs to integrate a solver without adding a solver dependency here.

QP GPU strategy files live in `src/presolve/rules/rule_*.cuh`.
Keep strategy-specific device helpers with their strategy, shared batch/model
operations in `src/presolve/`. These private headers are included by `gpu_presolve.cu`;
do not compile each header separately or add a kernel launch merely to cross a
file boundary. Public selection remains `Config.rules`.
