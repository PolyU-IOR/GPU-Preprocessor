# MPS reader

The bundled continuous-LP MPS reader needs no external download and is covered
by the repository's MIT license. The internal `cpu_presolve` namespace is
retained for compatibility; this directory contains no CPU presolve algorithm.

Its parser tests can run without CUDA:

```sh
cmake -S cpp/mps_reader -B build-reader
cmake --build build-reader
ctest --test-dir build-reader --output-on-failure
```
