// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_preprocessor/lp/offset.hpp"
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstdint>

namespace gpu_preprocessor::lp::presolve::detail {

// Sparse-storage positions use the selected offset ABI; row IDs remain int32.
__device__ inline std::int32_t _doubleton_equations_next_live_col_row(const NnzOffset* row_ptr,
                                                        const std::int32_t* col_val,
                                                        const double* nz_val,
                                                        const std::uint8_t* keep_row,
                                                        double zero_tol,
                                                        std::int32_t col,
                                                        NnzOffset* pos) {
  const NnzOffset stop = row_ptr[col + 1];
  while (*pos < stop) {
    const std::int32_t row = col_val[*pos];
    const double val = nz_val[*pos];
    if (keep_row[row] != std::uint8_t{0} && fabs(val) > zero_tol) {
      return row;
    }
    ++(*pos);
  }
  return INT_MAX;
}

__device__ inline std::int32_t _doubleton_equations_fill_in_proxy_device_live(const NnzOffset* row_ptr,
                                                                const std::int32_t* col_val,
                                                                const double* nz_val,
                                                                const std::uint8_t* keep_row,
                                                                double zero_tol,
                                                                std::int32_t keep_col,
                                                                std::int32_t elim_col) {
  std::int32_t fill_in = -1;
  NnzOffset jj = row_ptr[keep_col];
  const NnzOffset keep_stop = row_ptr[keep_col + 1];
  NnzOffset kk = row_ptr[elim_col];
  const NnzOffset elim_stop = row_ptr[elim_col + 1];
  if (keep_stop <= jj || elim_stop <= kk) {
    return fill_in;
  }

  while (jj < keep_stop && kk < elim_stop) {
    NnzOffset jj_live = jj;
    NnzOffset kk_live = kk;
    const std::int32_t keep_row_j =
        _doubleton_equations_next_live_col_row(row_ptr, col_val, nz_val, keep_row, zero_tol, keep_col, &jj_live);
    const std::int32_t elim_row_k =
        _doubleton_equations_next_live_col_row(row_ptr, col_val, nz_val, keep_row, zero_tol, elim_col, &kk_live);
    if (keep_row_j == INT_MAX || elim_row_k == INT_MAX) {
      break;
    }
    if (keep_row_j == elim_row_k) {
      jj = jj_live + 1;
      kk = kk_live + 1;
    } else if (elim_row_k < keep_row_j) {
      kk = kk_live + 1;
      ++fill_in;
    } else {
      jj = jj_live + 1;
    }
  }
  while (kk < elim_stop) {
    NnzOffset kk_live = kk;
    const std::int32_t elim_row_k =
        _doubleton_equations_next_live_col_row(row_ptr, col_val, nz_val, keep_row, zero_tol, elim_col, &kk_live);
    if (elim_row_k == INT_MAX) {
      break;
    }
    ++fill_in;
    kk = kk_live + 1;
  }
  return fill_in;
}

}  // namespace gpu_preprocessor::lp::presolve::detail
