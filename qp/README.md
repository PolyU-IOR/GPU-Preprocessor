# Quadratic programming (QP)

Run the commands in this guide from the repository root.

The QP module provides CUDA folding, presolve and solution recovery for continuous
convex quadratic programs. It reduces a QP before an external solver is called,
then recovers primal and dual solutions in the original model space.

**The QP presolver is currently experimental and requires further improvement.**

The default pipeline runs **folding**, then **presolve**. Reductions update the
Hessian, linear objective, objective constant and constraints together. After
solving, recovery reverses the presolve and folding operations. See the
[recovery guide](../docs/postsolve.md) for the primal/dual conventions and
recovery-state ownership.

The module currently supports continuous convex quadratic programming problems
with a quadratic objective, linear constraints and variable bounds:

```text
minimize    0.5 xᵀQx + cᵀx + c0
subject to  AL ≤ Ax ≤ AU
             l ≤ x  ≤ u
```

All variables must be continuous, and `Q` must be symmetric positive semidefinite.
Linear equalities and inequalities are represented by the row bounds `AL` and
`AU`. **Quadratically constrained quadratic programs (QCQPs) are not supported,
including convex QCQPs:** quadratic terms are allowed only in the objective,
not in the constraints. Nonconvex QPs and mixed-integer QPs are also outside the
supported scope.

Input validation checks symmetry and storage consistency; it does not certify
positive semidefiniteness.
A zero Hessian permits LP input through the QP API. The dedicated [LP module](../lp/README.md)
has its own implementation and API.

## Installation

Requires Linux, Make, CMake 3.24+, C11 and C++17 compilers, CUDA Toolkit 12+,
zlib and an NVIDIA GPU. Run from the project root:

```sh
make
```

This builds the LP and QP libraries, command-line tools and examples, selecting
the local GPU architecture automatically. For a QP-only build, pass
`CMAKE_ARGS='-DGPUPREPROCESSOR_BUILD_LP=OFF'` to `make`.

Include `gpupresolve_qp.h` and link `GPUPreprocessor::qp`. QP presolve requires
CUDA and reports an error if the selected GPU is unavailable. The core module
has no dependency on LP or an external solver.

## Usage

**Command line:**

After building, run the commands below from the project root. Replace `model.mps`
with a continuous minimization QP in MPS format. Plain `.mps` and compressed
`.mps.gz` files are supported; integer and semicontinuous inputs are rejected.

```sh
# Run folding and presolve; print status, dimensions and timing
./build/bin/gpu_preprocessor_qp model.mps

# Run presolve only
./build/bin/gpu_preprocessor_qp model.mps --folding false

# Export the reduced QP for an external solver
./build/bin/gpu_preprocessor_qp model.mps reduced.mps
```

The exporter writes the reduced constraints, quadratic objective and bounds to
a new MPS file. It refuses to overwrite an existing file. File export does not
serialize recovery state; use the C API for a complete presolve/solve/postsolve
pipeline.

LP and QP both default to visible GPU 0. Use `--gpu N` to select another visible
device; CUDA device numbering respects `CUDA_VISIBLE_DEVICES`.
Options may appear before or after the model path. Use `--help` to list them.

**C:** pass a host-CSR `Problem` to `presolve`, solve the reduced QP, then call
`postsolve` to recover the original primal variables and row/bound duals:

```c
#include "gpupresolve_qp.h"

Config config = default_config();
Result result = {0};
Status status = presolve(&problem, &config, &result);
if (status == OK || status == TIME_LIMIT) {
    /* Solve result.reduced to obtain host arrays x_red, y_red, z_red.
       Allocate x, y, z with the original problem dimensions. */
    status = postsolve(result.state,
                      x_red, result.reduced.A.cols,
                      y_red, result.reduced.A.rows,
                      z_red, result.reduced.A.cols,
                      x, y, z);
}
result_free(&result);
```

The input model is borrowed and unchanged. `Result` owns its reduced model and
recovery state; keep it alive until recovery finishes, then call `result_free`.
Matrices use zero-based CSR with 32-bit indices and double-precision values.
The C API requires both triangles of `Q`.

Presolve runs on GPU; the ordinary API exports the reduced model to host memory
once. For GPU consumers, `presolve_resident`, `presolve_device` and
`postsolve_device` keep model or solution data on device. See the
[API guide](../docs/api.md) and [GPU execution guide](../docs/gpu-execution.md)
for ownership and transfer details. Check recovered feasibility, stationarity
and objective values against the original QP; `compute_kkt` provides these
original-space checks.

To link from a C/C++ project:

```cmake
add_subdirectory(path/to/GPU-Preprocessor gpu-preprocessor-build EXCLUDE_FROM_ALL)
target_link_libraries(YOUR_TARGET PRIVATE GPUPreprocessor::qp)
```

For an installed package, use `find_package(GPUPreprocessor CONFIG REQUIRED COMPONENTS qp)`
and link the same `GPUPreprocessor::qp` target. The existing
`GPUPreprocessorQP::gpupresolve_qp` target remains available.

**Example:** [presolve_example.c](examples/presolve_example.c) demonstrates
preprocessing and primal/dual recovery using an analytically known solution.
It is built with the main project:

```sh
./build/bin/presolve_example
```

For the MPS tool, [qp_pipeline.mps](examples/qp_pipeline.mps) provides a small
convex QP. Use the C API to pass the reduced model to a solver and recover its
solution in the original model space.

## Parameters

Initialize `Config` with `default_config()` and change the fields you need:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `device` | `0` | Visible CUDA device index |
| `rules` | `ALL_RULES` | Enable folding and all twelve presolve rule families |
| `max_iterations` | `10` | Maximum visits to the presolve rule sequence |
| `max_seconds` | `1000` | Soft time limit, checked between actions |
| `feasibility_tol` | `1e-6` | Tolerance for activity infeasibility checks |
| `bound_tol` | `1e-6` | Bound-propagation progress tolerance |
| `zero_tol` | `1e-10` | Screening tolerance for substitution pivots |
| `max_q_fill_ratio` | `2` | Limit on Hessian nonzero growth per candidate batch |

Set `config.rules &= ~FOLDING` to run presolve only, or
`config.rules = FOLDING` to run folding alone. Set `rules = 0` for an identity
transform. The default backend is CUDA.

`presolve_with_options` additionally accepts `ExecutionOptions` for the rule
schedule and batch size. See [parameters](../docs/parameters.md) for all fields,
[folding](../docs/qp-folding.md) for partition conditions, and the
[presolve source map](src/presolve/README.md) for the rule implementations.

For command-line overrides:

```sh
./build/bin/gpu_preprocessor_qp model.mps --max-iters 20 --max-time 60
```

`presolve_seconds` covers the folding and presolve stages, excluding MPS parsing
and the initial model upload. An `OK` status means preprocessing succeeded;
the reduced problem still needs to be solved. `TIME_LIMIT` also returns a usable
reduced model and recovery state.

## Tests

```sh
make test GPU=0
```

This builds and runs the enabled test suite on the selected idle physical GPU.
QP tests cover reductions, objective preservation, original-space primal/dual
recovery, numerical edge cases, GPU-resident execution and command-line input.
See [validation](../docs/validation.md) for individual tests and installed-package
checks. Use separate build directories for different CUDA toolkits.

## License

[MIT](../LICENSE). See [NOTICE.md](../NOTICE.md) for source attribution.

## Citation

For the project's LP paper and citation details, see the
[LP guide](../lp/README.md#citation).
