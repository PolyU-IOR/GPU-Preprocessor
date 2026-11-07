# Documentation

Module guides: [LP](../lp/README.md), [QP](../qp/README.md).

- [API and ownership](api.md): model conventions, object lifetime, backend and I/O contracts.
- [Parameters](parameters.md): rule selection, tolerances and fill limits.
- [GPU execution](gpu-execution.md): batches, device boundaries and scheduling.
- [QP folding](qp-folding.md): GPU equitable refinement, Hessian aggregation and unfolding.
- [Recovery design](postsolve.md): stable indices, snapshots and primal/dual replay.
- [Validation](validation.md): regression coverage and independent numerical checks.
- [QP source map](../qp/src/presolve/README.md): rule implementations and shared CUDA helpers.
- [Development](../CONTRIBUTING.md): build targets, naming and test responsibilities.
