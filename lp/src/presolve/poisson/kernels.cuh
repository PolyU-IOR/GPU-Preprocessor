// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once
#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"
#include <cufft.h>
#include <cstddef>

namespace gpu_preprocessor::lp::presolve::poisson::kernels {
// Counts are [invalid, observations, active sources]. All outputs on device.
void certify(const LPInfoGpu&, int q, double* scale, int* counts);
void maps(const LPInfoGpu&, int q, int* observations, int* sources);
void spectrum(cufftDoubleComplex*, int side, double scale, bool multiply);
void response(const double* green, int q, const int* observations,
              const int* sources, LPInfoGpu&, const LPInfoGpu&, int* invalid);
void odd_rhs(double*, int q, const int* locations, const double* values,
             int count, bool temperature_ids);
void extract(const double*, int q, double* output, bool temperature_layout);
void scatter_sources(int nt, const int*, const double*, int count, double* x);
void recover_z(const LPInfoGpu&, const double* y, double* z);
void scatter_z(int nt, const int*, const double*, int count, double* z);
}
