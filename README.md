# GPU-Preprocessor

GPU preprocessing and original-space solution recovery for continuous LPs and
convex QPs. The two modules have independent APIs, rule implementations and
recovery state. Both can be used without an external solver.

**The QP presolver is currently experimental and requires further improvement.**

| Component | Public header | CMake target | Guide |
| --- | --- | --- | --- |
| LP | `gpu_preprocessor/lp.hpp` | `GPUPreprocessor::lp` | [LP usage and parameters](lp/README.md) |
| QP | `gpupresolve_qp.h` | `GPUPreprocessor::qp` | [QP usage and parameters](qp/README.md) |

The LP pipeline runs Startup presolve, folding and Main presolve, then reverses
these stages during recovery. QP runs CUDA folding followed by its rule sequence,
updating the Hessian, objective and recovery tape along with the constraints.
QP accepts `min 0.5*x'*Q*x + c'*x + c0` with linear row and variable bounds;
Q must be positive semidefinite. The QP input validator does not certify PSD.
Quadratic constraints (QCQP), nonconvex QPs and mixed-integer QPs are not supported.

## Installation

Builds require Linux, Make, CMake 3.24+, C/C++ compilers, CUDA Toolkit 12+ and an NVIDIA
GPU. LP uses cuSPARSE, cuBLAS and cuFFT; QP also requires zlib. Both modules
run presolve on CUDA. Run commands from the repository root.

```sh
# Build both LP and QP; choose the local GPU architecture automatically.
make
```

The default build directory is `build/`. Select modules independently with
`GPUPREPROCESSOR_BUILD_LP` and `GPUPREPROCESSOR_BUILD_QP`:

```sh
# LP only
cmake -S . -B build-lp -DGPUPREPROCESSOR_BUILD_QP=OFF
cmake --build build-lp --parallel 4
# QP only
cmake -S . -B build-qp -DGPUPREPROCESSOR_BUILD_LP=OFF
cmake --build build-qp --parallel 4
```

The existing LP options `GPUPREPROCESSOR_BUILD_TOOLS`,
`GPUPREPROCESSOR_BUILD_EXAMPLES`, `GPUPREPROCESSOR_BUILD_TESTS` and
`GPUPREPROCESSOR_USE_INT64_NNZ` remain available. Tests are opt-in; tools and
examples default to ON for a top-level build. Native QP callers can also supply
`BUILD_TOOLS`, `BUILD_EXAMPLES` and `BUILD_TESTING` on initial configuration.
The int64 option applies to LP sparse offsets; QP uses int32 sparse indices.

## Usage

```sh
# LP
build/bin/gpu_preprocessor_lp lp/examples/model.mps
# QP
build/bin/gpu_preprocessor_qp qp/examples/qp_pipeline.mps
```

Both commands default to visible GPU 0 and accept the same common options:
`--gpu N`, `--folding true|false`, `--max-iters N`, `--max-time SECONDS` and
`--help`. Options may appear before or after the model path. CUDA device
numbering respects `CUDA_VISIBLE_DEVICES`.

From another CMake project:

```cmake
add_subdirectory(path/to/GPU-Preprocessor gpu-preprocessor-build EXCLUDE_FROM_ALL)
target_link_libraries(YOUR_TARGET PRIVATE GPUPreprocessor::lp GPUPreprocessor::qp)
```

Or install and use `find_package(GPUPreprocessor CONFIG REQUIRED COMPONENTS lp qp)`.
Link only the components you use. Existing native QP consumers can continue to
use the `GPUPreprocessorQP` package and `GPUPreprocessorQP::gpupresolve_qp` target.
The [API guide](docs/api.md) documents ownership and recovery contracts.

## Layout

- `lp/`: LP implementation, tests, examples and tools.
- `qp/`: QP implementation, tests, examples and tools.
- `cmake/`, `tests/`, `tools/`: build, package checks and development helpers.
- `docs/`: API, algorithms, recovery and validation guides.

Copyright and attribution are retained in [LICENSE](LICENSE), [AUTHORS](AUTHORS)
and [NOTICE.md](NOTICE.md).
