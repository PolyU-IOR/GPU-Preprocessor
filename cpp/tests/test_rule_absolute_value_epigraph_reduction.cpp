#include "gpu_presolver/detail/runtime_options.hpp"
#include "gpu_presolver/offset.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver/presolve/rules/rule_absolute_value_epigraph_reduction.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <sstream>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace runtime = gpu_presolver::runtime;

#ifdef NDEBUG
#error "test_rule_absolute_value_epigraph_reduction requires active assert() checks"
#endif

namespace {

void check(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

template <class T>
T* copy_to_device(const std::vector<T>& values) {
  T* device = nullptr;
  check(cudaMalloc(&device, sizeof(T) * values.size()), "cudaMalloc");
  check(cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy H2D");
  return device;
}

template <class T>
std::vector<T> copy_to_host(T* device, std::size_t size) {
  std::vector<T> values(size);
  check(cudaMemcpy(values.data(), device, sizeof(T) * size, cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H");
  return values;
}

struct HostCsr {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  std::vector<NnzOffset> row_ptr;
  std::vector<std::int32_t> col_val;
  std::vector<double> nz_val;
};

HostCsr transpose(const HostCsr& matrix) {
  HostCsr result;
  result.rows = matrix.cols;
  result.cols = matrix.rows;
  result.row_ptr.assign(static_cast<std::size_t>(result.rows + 1), 0);
  result.col_val.resize(matrix.col_val.size());
  result.nz_val.resize(matrix.nz_val.size());
  for (const std::int32_t col : matrix.col_val) {
    ++result.row_ptr[static_cast<std::size_t>(col + 1)];
  }
  for (std::int32_t row = 0; row < result.rows; ++row) {
    result.row_ptr[static_cast<std::size_t>(row + 1)] +=
        result.row_ptr[static_cast<std::size_t>(row)];
  }
  std::vector<NnzOffset> next = result.row_ptr;
  for (std::int32_t row = 0; row < matrix.rows; ++row) {
    for (NnzOffset p = matrix.row_ptr[static_cast<std::size_t>(row)];
         p < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++p) {
      const std::int32_t col = matrix.col_val[static_cast<std::size_t>(p)];
      const std::int32_t dst = next[static_cast<std::size_t>(col)]++;
      result.col_val[static_cast<std::size_t>(dst)] = row;
      result.nz_val[static_cast<std::size_t>(dst)] = matrix.nz_val[static_cast<std::size_t>(p)];
    }
  }
  return result;
}

gpu_presolver::presolve::DeviceCsrMatrix upload(const HostCsr& matrix) {
  return {matrix.rows, matrix.cols, static_cast<std::int32_t>(matrix.col_val.size()),
          copy_to_device(matrix.row_ptr), copy_to_device(matrix.col_val),
          copy_to_device(matrix.nz_val)};
}

void free_matrix(gpu_presolver::presolve::DeviceCsrMatrix& matrix) {
  cudaFree(matrix.rowPtr);
  cudaFree(matrix.colVal);
  cudaFree(matrix.nzVal);
  matrix.rowPtr = nullptr;
  matrix.colVal = nullptr;
  matrix.nzVal = nullptr;
}

enum class OuterFixtureVariant {
  kSharedBound,
  kTwoBlockValid,
  kDuplicateFree,
  kBoundFreeOverlap,
  kOuterBlockRoleConflict,
  kBlockAuxLocalXConflict,
  kDenseOwnQConflict,
  kDenseOwnSConflict,
  kDenseCrossQConflict,
  kDenseCrossEConflict,
  kDenseCrossSConflict,
  kDenseDuplicateE,
  kDenseSharedBoundMappedDuplicate,
};

struct OuterFixture {
  using LPInfoGpu = gpu_presolver::presolve::LPInfoGpu;
  using PresolvePlanGpu = gpu_presolver::presolve::PresolvePlanGpu;

  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> cost = {1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<double> lower = {0.0, -inf, -inf, -inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> upper = {10.0, inf, inf, inf, inf, inf, inf, inf, inf, inf};
  std::vector<double> row_lower = std::vector<double>(14, 0.0);
  std::vector<double> row_upper = {
      inf, inf, inf, inf, inf, inf, 0.0, inf, inf, 0.0, inf, inf, inf, inf};
  LPInfoGpu lp;
  PresolvePlanGpu plan;

  explicit OuterFixture(OuterFixtureVariant variant) {
    // Variables: a=0, b=1, c=2, e=3, q=4, s=5, x1..x4=6..9.
    // The first two three-row groups impose b=a and c=a.  Both pairs sharing
    // a is valid and exercises deterministic accumulation into one objective
    // coefficient.  Rows 6..13 form one linked L1 block.
    HostCsr matrix;
    matrix.rows = 14;
    matrix.cols = 10;
    matrix.row_ptr = {0, 2, 4, 6, 8, 10, 12, 13, 15, 17, 19, 21, 23, 25, 27};
    matrix.col_val = {
        0, 1,  0, 1,  0, 1,
        0, 2,  0, 2,  0, 2,
        3,
        3, 4,  3, 4,
        4, 5,
        5, 6,  5, 7,  5, 8,  5, 9};
    matrix.nz_val = {
        1.0, -1.0,  1.0, 1.0,  -1.0, 1.0,
        1.0, -1.0,  1.0, 1.0,  -1.0, 1.0,
        1.0,
        -1.0, 1.0,  1.0, 1.0,
        1.0, -1.0,
        -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0};

    const auto append_to_first_dense_row = [&](std::int32_t col, double val) {
      const NnzOffset insert_at = matrix.row_ptr[7];
      matrix.col_val.insert(matrix.col_val.begin() + insert_at, col);
      matrix.nz_val.insert(matrix.nz_val.begin() + insert_at, val);
      for (std::size_t r = 7; r < matrix.row_ptr.size(); ++r) {
        ++matrix.row_ptr[r];
      }
    };

    const auto append_valid_second_block = [&]() {
      // Second block: e=10, q=11, s=12.  Reusing x1..x4 is legal; none of
      // those columns is an auxiliary role owned by either block.
      matrix.rows = 22;
      matrix.cols = 13;
      matrix.row_ptr.insert(matrix.row_ptr.end(), {28, 30, 32, 34, 36, 38, 40, 42});
      matrix.col_val.insert(matrix.col_val.end(), {
          10,
          10, 11,  10, 11,
          11, 12,
          12, 6,  12, 7,  12, 8,  12, 9});
      matrix.nz_val.insert(matrix.nz_val.end(), {
          1.0,
          -1.0, 1.0,  1.0, 1.0,
          1.0, -1.0,
          -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0});
      cost.resize(13, 0.0);
      lower.resize(13, 0.0);
      upper.resize(13, inf);
      lower[10] = -inf;
      row_lower.resize(22, 0.0);
      row_upper.insert(row_upper.end(), {0.0, inf, inf, 0.0, inf, inf, inf, inf});
    };

    if (variant == OuterFixtureVariant::kTwoBlockValid) {
      append_valid_second_block();
    } else if (variant == OuterFixtureVariant::kDuplicateFree) {
      matrix.col_val[7] = 1;
      matrix.col_val[9] = 1;
      matrix.col_val[11] = 1;
    } else if (variant == OuterFixtureVariant::kBoundFreeOverlap) {
      // The second pair uses b as its bound even though b is the first pair's
      // free variable.  Its incompatible bound roles must fail closed.
      matrix.col_val[6] = 1;
      matrix.col_val[8] = 1;
      matrix.col_val[10] = 1;
    } else if (variant == OuterFixtureVariant::kOuterBlockRoleConflict) {
      // Reuse outer free variable b as the block's e variable.
      matrix.col_val[12] = 1;
      matrix.col_val[13] = 1;
      matrix.col_val[15] = 1;
    } else if (variant == OuterFixtureVariant::kDenseOwnQConflict) {
      append_to_first_dense_row(4, 2.0);
    } else if (variant == OuterFixtureVariant::kDenseOwnSConflict) {
      append_to_first_dense_row(5, 2.0);
    } else if (variant == OuterFixtureVariant::kDenseDuplicateE) {
      append_to_first_dense_row(3, 2.0);
    } else if (variant ==
               OuterFixtureVariant::kDenseSharedBoundMappedDuplicate) {
      // Both free columns 1 and 2 map to the shared bound column 0.  They are
      // distinct and sorted in the source dense row, but collide after the
      // candidate rewrite and must reject the whole outer transformation.
      const NnzOffset insert_at = matrix.row_ptr[6];
      matrix.col_val.insert(matrix.col_val.begin() + insert_at, {1, 2});
      matrix.nz_val.insert(matrix.nz_val.begin() + insert_at, {2.0, 3.0});
      for (std::size_t r = 7; r < matrix.row_ptr.size(); ++r) {
        matrix.row_ptr[r] += 2;
      }
    } else if (variant == OuterFixtureVariant::kDenseCrossQConflict ||
               variant == OuterFixtureVariant::kDenseCrossEConflict ||
               variant == OuterFixtureVariant::kDenseCrossSConflict) {
      append_valid_second_block();
      const std::int32_t cross_role =
          variant == OuterFixtureVariant::kDenseCrossEConflict
              ? 10
              : (variant == OuterFixtureVariant::kDenseCrossQConflict ? 11 : 12);
      append_to_first_dense_row(cross_role, 2.0);
    } else if (variant == OuterFixtureVariant::kBlockAuxLocalXConflict) {
      // Add a second linked block which incorrectly reuses the first block's
      // q column (4) as one of its original x columns.
      matrix.rows = 22;
      matrix.cols = 16;
      matrix.row_ptr.insert(matrix.row_ptr.end(), {28, 30, 32, 34, 36, 38, 40, 42});
      matrix.col_val.insert(matrix.col_val.end(), {
          10,
          10, 11,  10, 11,
          11, 12,
          12, 4,  12, 13,  12, 14,  12, 15});
      matrix.nz_val.insert(matrix.nz_val.end(), {
          1.0,
          -1.0, 1.0,  1.0, 1.0,
          1.0, -1.0,
          -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0});
      cost = {1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 1.0,
              1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
      lower = {0.0, -inf, -inf, -inf, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, -inf, 0.0, 0.0, 0.0, 0.0, 0.0};
      upper = {10.0, inf, inf, inf, inf, inf, inf, inf,
               inf, inf, inf, inf, inf, inf, inf, inf};
      row_lower.resize(22, 0.0);
      row_upper.insert(row_upper.end(), {0.0, inf, inf, 0.0, inf, inf, inf, inf});
    }

    lp.A = upload(matrix);
    lp.AT = upload(transpose(matrix));
    plan.keep_row_mask = copy_to_device<std::uint8_t>(
        std::vector<std::uint8_t>(static_cast<std::size_t>(matrix.rows), 1));
    plan.keep_col_mask = copy_to_device<std::uint8_t>(
        std::vector<std::uint8_t>(static_cast<std::size_t>(matrix.cols), 1));
    plan.new_c = copy_to_device(cost);
    plan.new_l = copy_to_device(lower);
    plan.new_u = copy_to_device(upper);
    plan.new_AL = copy_to_device(row_lower);
    plan.new_AU = copy_to_device(row_upper);
  }

  OuterFixture(const OuterFixture&) = delete;
  OuterFixture& operator=(const OuterFixture&) = delete;

  ~OuterFixture() {
    free_matrix(lp.A);
    free_matrix(lp.AT);
    cudaFree(plan.keep_row_mask);
    cudaFree(plan.keep_col_mask);
    cudaFree(plan.new_c);
    cudaFree(plan.new_l);
    cudaFree(plan.new_u);
    cudaFree(plan.new_AL);
    cudaFree(plan.new_AU);
    free_matrix(plan.new_A);
  }
};

struct OuterMappedDenseSortFixture {
  using LPInfoGpu = gpu_presolver::presolve::LPInfoGpu;
  using PresolvePlanGpu = gpu_presolver::presolve::PresolvePlanGpu;

  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> cost = {
      1.0, 0.5, 0.25, 0.75, 1.0, 1.0, 1.0, 1.0, 3.0, 2.0};
  std::vector<double> lower = {
      0.0, -inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf};
  std::vector<double> upper = {
      10.0, inf, inf, inf, inf, inf, inf, inf, inf, inf};
  std::vector<double> row_lower =
      {0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> row_upper =
      {inf, inf, inf, 7.0, inf, inf, 0.0, inf, inf, inf, inf};
  LPInfoGpu lp;
  PresolvePlanGpu plan;

  OuterMappedDenseSortFixture() {
    // Outer pair: bound=0, free=9.  Block roles: e=1, q=2, s=3,
    // local x=4..7.  The retained dense equality is initially sorted as
    // [e=1, z=8, free=9].  Rewriting e emits [1,2], while mapping free 9 to
    // bound 0 appends a trailing 0, so the conditional CSR sort is required.
    HostCsr matrix;
    matrix.rows = 11;
    matrix.cols = 10;
    matrix.row_ptr = {0, 2, 4, 6, 9, 11, 13, 15, 17, 19, 21, 23};
    matrix.col_val = {
        0, 9,  0, 9,  0, 9,
        1, 8, 9,
        1, 2,  1, 2,
        2, 3,
        3, 4,  3, 5,  3, 6,  3, 7};
    matrix.nz_val = {
        1.0, -1.0,  1.0, 1.0,  -1.0, 1.0,
        1.0, 5.0, 2.0,
        -1.0, 1.0,  1.0, 1.0,
        1.0, -1.0,
        -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0};

    lp.A = upload(matrix);
    lp.AT = upload(transpose(matrix));
    plan.keep_row_mask = copy_to_device<std::uint8_t>(
        std::vector<std::uint8_t>(static_cast<std::size_t>(matrix.rows), 1));
    plan.keep_col_mask = copy_to_device<std::uint8_t>(
        std::vector<std::uint8_t>(static_cast<std::size_t>(matrix.cols), 1));
    plan.new_c = copy_to_device(cost);
    plan.new_l = copy_to_device(lower);
    plan.new_u = copy_to_device(upper);
    plan.new_AL = copy_to_device(row_lower);
    plan.new_AU = copy_to_device(row_upper);
  }

  OuterMappedDenseSortFixture(const OuterMappedDenseSortFixture&) = delete;
  OuterMappedDenseSortFixture& operator=(const OuterMappedDenseSortFixture&) = delete;

  ~OuterMappedDenseSortFixture() {
    free_matrix(lp.A);
    free_matrix(lp.AT);
    cudaFree(plan.keep_row_mask);
    cudaFree(plan.keep_col_mask);
    cudaFree(plan.new_c);
    cudaFree(plan.new_l);
    cudaFree(plan.new_u);
    cudaFree(plan.new_AL);
    cudaFree(plan.new_AU);
    free_matrix(plan.new_A);
  }
};

enum class L1SplitFixtureVariant {
  kValid,
  kTrailingEmptyRow,
  kTInEquality,
  kSharedRoles,
  kNearEpigraphCoefficient,
  kTinyNonzeroEpigraphRhs,
  kWrongSignInfinityRowBound,
  kDuplicateEInEquality,
  kReverseRoleColumnOrder,
  kTwoIndependentSecondInvalidTBound,
};

struct L1SplitFixture {
  using LPInfoGpu = gpu_presolver::presolve::LPInfoGpu;
  using PresolvePlanGpu = gpu_presolver::presolve::PresolvePlanGpu;

  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> cost;
  std::vector<double> lower;
  std::vector<double> upper;
  std::vector<double> row_lower;
  std::vector<double> row_upper;
  LPInfoGpu lp;
  PresolvePlanGpu plan;

  L1SplitFixture(L1SplitFixtureVariant variant,
                 double residual_lower,
                 double residual_upper) {
    // Valid block, with t=0, e=1 and an unrelated original variable x=2:
    //   e + 2x = 7,  t - e >= 0,  t + e >= 0.
    // The specialized substitution uses p=t_col and n=e_col, so that
    // t_old=p+n and e_old=p-n.
    HostCsr matrix;
    if (variant ==
        L1SplitFixtureVariant::kTwoIndependentSecondInvalidTBound) {
      // Two disjoint blocks pass extraction and global role ownership.  Only
      // block 1 fails validation because its t lower bound is nonzero, even
      // though it differs from zero by less than the old matching tolerance.
      matrix.rows = 6;
      matrix.cols = 6;
      matrix.row_ptr = {0, 2, 4, 6, 8, 10, 12};
      matrix.col_val = {
          1, 4,  0, 1,  0, 1,
          3, 5,  2, 3,  2, 3};
      matrix.nz_val = {
          1.0, 2.0,  1.0, -1.0,  1.0, 1.0,
          1.0, 3.0,  1.0, -1.0,  1.0, 1.0};
      cost = {2.0, 0.5, 4.0, 0.75, 3.0, 5.0};
      lower = {0.0, residual_lower, 5.0e-13, residual_lower, -inf, -inf};
      upper = {inf, residual_upper, inf, residual_upper, inf, inf};
      row_lower = {7.0, 0.0, 0.0, 9.0, 0.0, 0.0};
      row_upper = {7.0, inf, inf, 9.0, inf, inf};
    } else if (variant == L1SplitFixtureVariant::kSharedRoles) {
      // A second otherwise-valid block illegally reuses the first block's
      // t/e columns.  Global role ownership must reject both blocks before
      // any plan array is modified.
      matrix.rows = 6;
      matrix.cols = 4;
      matrix.row_ptr = {0, 2, 4, 6, 8, 10, 12};
      matrix.col_val = {
          1, 2,  0, 1,  0, 1,
          1, 3,  0, 1,  0, 1};
      matrix.nz_val = {
          1.0, 2.0,  1.0, -1.0,  1.0, 1.0,
          1.0, 3.0,  1.0, -1.0,  1.0, 1.0};
      cost = {2.0, 0.5, 3.0, 4.0};
      lower = {0.0, residual_lower, -inf, -inf};
      upper = {inf, residual_upper, inf, inf};
      row_lower = {7.0, 0.0, 0.0, 9.0, 0.0, 0.0};
      row_upper = {7.0, inf, inf, 9.0, inf, inf};
    } else if (variant == L1SplitFixtureVariant::kReverseRoleColumnOrder) {
      // Same valid block with e=0, x=1, t=2.  Replacing e in the equality
      // used to emit [t,e,x] = [2,0,1], while the retained residual-bound
      // row used to emit [t,e] = [2,0].
      matrix.rows = 3;
      matrix.cols = 3;
      matrix.row_ptr = {0, 2, 4, 6};
      matrix.col_val = {0, 1,  0, 2,  0, 2};
      matrix.nz_val = {1.0, 2.0,  -1.0, 1.0,  1.0, 1.0};
      cost = {0.5, 3.0, 2.0};
      lower = {residual_lower, -inf, 0.0};
      upper = {residual_upper, inf, inf};
      row_lower = {7.0, 0.0, 0.0};
      row_upper = {7.0, inf, inf};
    } else {
      matrix.rows = 3;
      matrix.cols = 3;
      if (variant == L1SplitFixtureVariant::kTInEquality) {
        // The epigraph rows still identify t/e, but t has an extra nonzero in
        // the equality.  The exact AT incidence certificate must reject it.
        matrix.row_ptr = {0, 3, 5, 7};
        matrix.col_val = {0, 1, 2,  0, 1,  0, 1};
        matrix.nz_val = {0.25, 1.0, 2.0,  1.0, -1.0,  1.0, 1.0};
      } else if (variant == L1SplitFixtureVariant::kDuplicateEInEquality) {
        // Algebraically the two 0.5e entries sum to e, but the specialized
        // fill requires one physical e occurrence and must reject duplicates
        // before allocating old_nnz+1 slots for the equality row.
        matrix.row_ptr = {0, 3, 5, 7};
        matrix.col_val = {1, 1, 2,  0, 1,  0, 1};
        matrix.nz_val = {0.5, 0.5, 2.0,  1.0, -1.0,  1.0, 1.0};
      } else {
        matrix.row_ptr = {0, 2, 4, 6};
        matrix.col_val = {1, 2,  0, 1,  0, 1};
        matrix.nz_val = {1.0, 2.0,  1.0, -1.0,  1.0, 1.0};
        if (variant == L1SplitFixtureVariant::kNearEpigraphCoefficient) {
          // The detector's historical tolerance accepts this, but the fast
          // split hard-codes exact +/-1 coefficients and must fail closed.
          matrix.nz_val[3] = -1.0 + 5.0e-13;
        }
      }
      cost = {2.0, 0.5, 3.0};
      lower = {0.0, residual_lower, -inf};
      upper = {inf, residual_upper, inf};
      row_lower = {7.0, 0.0, 0.0};
      row_upper = {7.0, inf, inf};
      if (variant == L1SplitFixtureVariant::kTinyNonzeroEpigraphRhs) {
        row_lower[1] = 5.0e-13;
      } else if (variant == L1SplitFixtureVariant::kWrongSignInfinityRowBound) {
        row_upper[1] = -inf;
      }
    }

    if (variant == L1SplitFixtureVariant::kTrailingEmptyRow) {
      ++matrix.rows;
      matrix.row_ptr.push_back(matrix.row_ptr.back());
      row_lower.push_back(0.0);
      row_upper.push_back(inf);
    }

    lp.A = upload(matrix);
    lp.AT = upload(transpose(matrix));
    plan.keep_row_mask = copy_to_device<std::uint8_t>(
        std::vector<std::uint8_t>(static_cast<std::size_t>(matrix.rows), 1));
    plan.keep_col_mask = copy_to_device<std::uint8_t>(
        std::vector<std::uint8_t>(static_cast<std::size_t>(matrix.cols), 1));
    plan.new_c = copy_to_device(cost);
    plan.new_l = copy_to_device(lower);
    plan.new_u = copy_to_device(upper);
    plan.new_AL = copy_to_device(row_lower);
    plan.new_AU = copy_to_device(row_upper);
  }

  L1SplitFixture(const L1SplitFixture&) = delete;
  L1SplitFixture& operator=(const L1SplitFixture&) = delete;

  ~L1SplitFixture() {
    free_matrix(lp.A);
    free_matrix(lp.AT);
    cudaFree(plan.keep_row_mask);
    cudaFree(plan.keep_col_mask);
    cudaFree(plan.new_c);
    cudaFree(plan.new_l);
    cudaFree(plan.new_u);
    cudaFree(plan.new_AL);
    cudaFree(plan.new_AU);
    free_matrix(plan.new_A);
  }
};

struct TwoGraphBlockFixture {
  using LPInfoGpu = gpu_presolver::presolve::LPInfoGpu;
  using PresolvePlanGpu = gpu_presolver::presolve::PresolvePlanGpu;

  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> cost = {0.5, 2.0, 0.0, 0.75, 1.25, 0.0};
  std::vector<double> lower = {0.0, -inf, 0.0, 0.0, -5.0, 0.0};
  std::vector<double> upper = {inf, inf, inf, inf, 5.0, inf};
  std::vector<double> row_lower = {7.0, 0.0, 0.0, 0.0,
                                   11.0, 0.0, 0.0, 0.0};
  std::vector<double> row_upper = {7.0, inf, inf, inf,
                                   11.0, inf, inf, inf};
  LPInfoGpu lp;
  PresolvePlanGpu plan;

  TwoGraphBlockFixture() {
    // Block 0 is legal: (t,e,s)=(0,1,2).  Block 1 has the same legal row
    // structure with (t,e,s)=(3,4,5), but e=4 has a finite box.  Validation
    // must reject the whole graph match before committing block 0.
    HostCsr matrix;
    matrix.rows = 8;
    matrix.cols = 6;
    matrix.row_ptr = {0, 1, 3, 5, 7, 8, 10, 12, 14};
    matrix.col_val = {
        1,
        0, 1,  0, 1,  0, 2,
        4,
        3, 4,  3, 4,  3, 5};
    matrix.nz_val = {
        3.0,
        1.0, -1.0,  1.0, 1.0,  -2.0, 1.0,
        4.0,
        1.0, -1.0,  1.0, 1.0,  -3.0, 1.0};

    lp.A = upload(matrix);
    lp.AT = upload(transpose(matrix));
    plan.keep_row_mask = copy_to_device<std::uint8_t>(
        std::vector<std::uint8_t>(static_cast<std::size_t>(matrix.rows), 1));
    plan.keep_col_mask = copy_to_device<std::uint8_t>(
        std::vector<std::uint8_t>(static_cast<std::size_t>(matrix.cols), 1));
    plan.new_c = copy_to_device(cost);
    plan.new_l = copy_to_device(lower);
    plan.new_u = copy_to_device(upper);
    plan.new_AL = copy_to_device(row_lower);
    plan.new_AU = copy_to_device(row_upper);
  }

  TwoGraphBlockFixture(const TwoGraphBlockFixture&) = delete;
  TwoGraphBlockFixture& operator=(const TwoGraphBlockFixture&) = delete;

  ~TwoGraphBlockFixture() {
    free_matrix(lp.A);
    free_matrix(lp.AT);
    cudaFree(plan.keep_row_mask);
    cudaFree(plan.keep_col_mask);
    cudaFree(plan.new_c);
    cudaFree(plan.new_l);
    cudaFree(plan.new_u);
    cudaFree(plan.new_AL);
    cudaFree(plan.new_AU);
    free_matrix(plan.new_A);
  }
};

void apply_l1_split_fixture(
    L1SplitFixture& fixture,
    const gpu_presolver::presolve::PresolveParams& params) {
  gpu_presolver::presolve::PresolveStatsGpu stats;
  gpu_presolver::presolve::apply_rule_absolute_value_epigraph_reduction(
      fixture.plan, fixture.lp, stats, params);
}

void assert_l1_split_recovery(const L1SplitFixture& fixture) {
  assert(fixture.plan.has_absolute_value_epigraph_recovery);
  assert(fixture.plan.absolute_value_epigraph_recovery.pattern == "l1_split_3row");
  assert(fixture.plan.absolute_value_epigraph_recovery.splits.size() == 1);
  assert(fixture.plan.absolute_value_epigraph_recovery.splits[0].t_col == 0);
  assert(fixture.plan.absolute_value_epigraph_recovery.splits[0].e_col == 1);
  assert(std::fabs(fixture.plan.absolute_value_epigraph_recovery.splits[0].rho - 1.0) <
         1.0e-12);
}

void assert_valid_l1_transformed_columns(const L1SplitFixture& fixture) {
  assert((copy_to_host(fixture.plan.keep_col_mask, 3) ==
          std::vector<std::uint8_t>{1, 1, 1}));
  assert((copy_to_host(fixture.plan.new_c, 3) ==
          std::vector<double>{2.5, 1.5, 3.0}));
  const std::vector<double> l = copy_to_host(fixture.plan.new_l, 3);
  const std::vector<double> u = copy_to_host(fixture.plan.new_u, 3);
  assert(l[0] == 0.0);
  assert(l[1] == 0.0);
  assert(std::isinf(l[2]) && l[2] < 0.0);
  assert(std::isinf(u[0]) && u[0] > 0.0);
  assert(std::isinf(u[1]) && u[1] > 0.0);
  assert(std::isinf(u[2]) && u[2] > 0.0);
}

void assert_l1_exact_matrix(const L1SplitFixture& fixture, bool preserve_bound_row) {
  assert(fixture.plan.has_new_A);
  assert(fixture.plan.new_A.rows == 3);
  assert(fixture.plan.new_A.cols == 3);
  const std::vector<NnzOffset> expected_row_ptr =
      preserve_bound_row ? std::vector<NnzOffset>{0, 3, 5, 5}
                         : std::vector<NnzOffset>{0, 3, 3, 3};
  const std::vector<std::int32_t> expected_col_val =
      preserve_bound_row ? std::vector<std::int32_t>{0, 1, 2, 0, 1}
                         : std::vector<std::int32_t>{0, 1, 2};
  const std::vector<double> expected_nz_val =
      preserve_bound_row ? std::vector<double>{1.0, -1.0, 2.0, 1.0, -1.0}
                         : std::vector<double>{1.0, -1.0, 2.0};
  assert(fixture.plan.new_A.nnz == static_cast<std::int32_t>(expected_col_val.size()));
  assert(copy_to_host(fixture.plan.new_A.rowPtr, 4) == expected_row_ptr);
  assert(copy_to_host(fixture.plan.new_A.colVal, expected_col_val.size()) ==
         expected_col_val);
  assert(copy_to_host(fixture.plan.new_A.nzVal, expected_nz_val.size()) ==
         expected_nz_val);
}

void assert_l1_match_fails_closed(const L1SplitFixture& fixture) {
  assert(!fixture.plan.has_change);
  assert(!fixture.plan.has_row_action);
  assert(!fixture.plan.has_col_action);
  assert(!fixture.plan.has_new_A);
  assert(!fixture.plan.has_absolute_value_epigraph_recovery);
  assert((copy_to_host(fixture.plan.keep_row_mask, fixture.row_lower.size()) ==
          std::vector<std::uint8_t>(fixture.row_lower.size(), std::uint8_t{1})));
  assert((copy_to_host(fixture.plan.keep_col_mask, fixture.cost.size()) ==
          std::vector<std::uint8_t>(fixture.cost.size(), std::uint8_t{1})));
  assert(copy_to_host(fixture.plan.new_c, fixture.cost.size()) == fixture.cost);
  assert(copy_to_host(fixture.plan.new_l, fixture.lower.size()) == fixture.lower);
  assert(copy_to_host(fixture.plan.new_u, fixture.upper.size()) == fixture.upper);
  assert(copy_to_host(fixture.plan.new_AL, fixture.row_lower.size()) == fixture.row_lower);
  assert(copy_to_host(fixture.plan.new_AU, fixture.row_upper.size()) == fixture.row_upper);
}

void assert_two_graph_block_match_fails_closed(const TwoGraphBlockFixture& fixture) {
  assert(!fixture.plan.has_change);
  assert(!fixture.plan.has_row_action);
  assert(!fixture.plan.has_col_action);
  assert(!fixture.plan.has_new_A);
  assert(!fixture.plan.has_absolute_value_epigraph_recovery);
  assert((copy_to_host(fixture.plan.keep_row_mask, fixture.row_lower.size()) ==
          std::vector<std::uint8_t>(fixture.row_lower.size(), std::uint8_t{1})));
  assert((copy_to_host(fixture.plan.keep_col_mask, fixture.cost.size()) ==
          std::vector<std::uint8_t>(fixture.cost.size(), std::uint8_t{1})));
  assert(copy_to_host(fixture.plan.new_c, fixture.cost.size()) == fixture.cost);
  assert(copy_to_host(fixture.plan.new_l, fixture.lower.size()) == fixture.lower);
  assert(copy_to_host(fixture.plan.new_u, fixture.upper.size()) == fixture.upper);
  assert(copy_to_host(fixture.plan.new_AL, fixture.row_lower.size()) == fixture.row_lower);
  assert(copy_to_host(fixture.plan.new_AU, fixture.row_upper.size()) == fixture.row_upper);
}

void assert_csr_rows_strictly_sorted(
    const gpu_presolver::presolve::DeviceCsrMatrix& matrix) {
  const std::vector<NnzOffset> row_ptr =
      copy_to_host(matrix.rowPtr, static_cast<std::size_t>(matrix.rows + 1));
  const std::vector<std::int32_t> col_val =
      copy_to_host(matrix.colVal, static_cast<std::size_t>(matrix.nnz));
  for (std::int32_t row = 0; row < matrix.rows; ++row) {
    for (NnzOffset p = row_ptr[static_cast<std::size_t>(row)] + 1;
         p < row_ptr[static_cast<std::size_t>(row + 1)]; ++p) {
      assert(col_val[static_cast<std::size_t>(p - 1)] <
             col_val[static_cast<std::size_t>(p)]);
    }
  }
}

void test_l1_split_default_preserves_finite_residual_bounds_exactly() {
  using gpu_presolver::presolve::PresolveParams;

  L1SplitFixture fixture(L1SplitFixtureVariant::kValid, -2.0, 2.0);
  PresolveParams params;
  apply_l1_split_fixture(fixture, params);

  assert(fixture.plan.has_change);
  assert(fixture.plan.has_row_action);
  assert(fixture.plan.has_col_action);
  assert_l1_split_recovery(fixture);
  assert_valid_l1_transformed_columns(fixture);
  assert((copy_to_host(fixture.plan.keep_row_mask, 3) ==
          std::vector<std::uint8_t>{1, 1, 0}));
  assert((copy_to_host(fixture.plan.new_AL, 3) ==
          std::vector<double>{7.0, -2.0, 0.0}));
  const std::vector<double> au = copy_to_host(fixture.plan.new_AU, 3);
  assert(au[0] == 7.0);
  assert(au[1] == 2.0);
  assert(std::isinf(au[2]) && au[2] > 0.0);
  assert_l1_exact_matrix(fixture, true);

  // Postsolve must invert p/n back to the original t/e coordinates without
  // losing the finite residual-box semantics retained by p-n in row 1.
  double* x_org = copy_to_device<double>({3.0, 1.0, 2.5});
  double* original_l = copy_to_device(fixture.lower);
  gpu_presolver::presolve::postsolve_restore_absolute_value_epigraph_gpu(
      x_org,
      std::vector<gpu_presolver::presolve::AbsoluteValueEpigraphPrimalRecoveryStep>{
          fixture.plan.absolute_value_epigraph_recovery},
      original_l);
  const std::vector<double> restored = copy_to_host(x_org, 3);
  assert((restored == std::vector<double>{4.0, 2.0, 2.5}));
  cudaFree(x_org);
  cudaFree(original_l);
}

void test_l1_split_true_free_residual_uses_three_to_one_reduction() {
  using gpu_presolver::presolve::PresolveParams;

  const double inf = std::numeric_limits<double>::infinity();
  L1SplitFixture fixture(L1SplitFixtureVariant::kValid, -inf, inf);
  PresolveParams params;
  apply_l1_split_fixture(fixture, params);

  assert(fixture.plan.has_change);
  assert_l1_split_recovery(fixture);
  assert_valid_l1_transformed_columns(fixture);
  assert((copy_to_host(fixture.plan.keep_row_mask, 3) ==
          std::vector<std::uint8_t>{1, 0, 0}));
  assert(copy_to_host(fixture.plan.new_AL, 3) == fixture.row_lower);
  assert(copy_to_host(fixture.plan.new_AU, 3) == fixture.row_upper);
  assert_l1_exact_matrix(fixture, false);
}

void test_l1_split_default_preserves_one_sided_residual_bound_exactly() {
  using gpu_presolver::presolve::PresolveParams;

  const double inf = std::numeric_limits<double>::infinity();
  L1SplitFixture fixture(L1SplitFixtureVariant::kValid, -inf, 5.0);
  PresolveParams params;
  apply_l1_split_fixture(fixture, params);

  assert(fixture.plan.has_change);
  assert_l1_split_recovery(fixture);
  assert_valid_l1_transformed_columns(fixture);
  assert((copy_to_host(fixture.plan.keep_row_mask, 3) ==
          std::vector<std::uint8_t>{1, 1, 0}));
  const std::vector<double> al = copy_to_host(fixture.plan.new_AL, 3);
  const std::vector<double> au = copy_to_host(fixture.plan.new_AU, 3);
  assert(al[0] == 7.0);
  assert(std::isinf(al[1]) && al[1] < 0.0);
  assert(al[2] == 0.0);
  assert(au[0] == 7.0);
  assert(au[1] == 5.0);
  assert(std::isinf(au[2]) && au[2] > 0.0);
  assert_l1_exact_matrix(fixture, true);
}

void test_l1_split_strict_free_only_rejects_finite_bounds_without_mutation() {
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::AbsoluteValueEpigraphResidualBoundMode;

  L1SplitFixture fixture(L1SplitFixtureVariant::kValid, -2.0, 2.0);
  PresolveParams params;
  params.absolute_value_epigraph_residual_bound_mode =
      AbsoluteValueEpigraphResidualBoundMode::StrictFreeOnly;
  apply_l1_split_fixture(fixture, params);

  assert_l1_match_fails_closed(fixture);
}

void test_l1_split_legacy_wide_mode_retains_old_three_to_one_behavior() {
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::AbsoluteValueEpigraphResidualBoundMode;

  L1SplitFixture fixture(L1SplitFixtureVariant::kValid, -1.0e9, 1.0e9);
  PresolveParams params;
  params.absolute_value_epigraph_residual_bound_mode =
      AbsoluteValueEpigraphResidualBoundMode::LegacyWideAsFree;
  params.absolute_value_epigraph_residual_bound_as_free_min = 1.0e8;
  apply_l1_split_fixture(fixture, params);

  assert(fixture.plan.has_change);
  assert_l1_split_recovery(fixture);
  assert_valid_l1_transformed_columns(fixture);
  assert((copy_to_host(fixture.plan.keep_row_mask, 3) ==
          std::vector<std::uint8_t>{1, 0, 0}));
  assert_l1_exact_matrix(fixture, false);
}

void test_l1_split_illegal_t_incidence_fails_closed() {
  using gpu_presolver::presolve::PresolveParams;

  L1SplitFixture fixture(L1SplitFixtureVariant::kTInEquality, -2.0, 2.0);
  PresolveParams params;
  apply_l1_split_fixture(fixture, params);

  assert_l1_match_fails_closed(fixture);
}

void test_l1_split_shared_roles_across_blocks_fail_closed() {
  using gpu_presolver::presolve::PresolveParams;

  L1SplitFixture fixture(L1SplitFixtureVariant::kSharedRoles, -2.0, 2.0);
  PresolveParams params;
  apply_l1_split_fixture(fixture, params);

  assert_l1_match_fails_closed(fixture);
}

void test_l1_split_mixed_validation_fails_before_any_block_commit() {
  using gpu_presolver::presolve::PresolveParams;

  L1SplitFixture fixture(
      L1SplitFixtureVariant::kTwoIndependentSecondInvalidTBound, -2.0, 2.0);
  PresolveParams params;
  apply_l1_split_fixture(fixture, params);

  // Block 0 is fully legal.  Block 1 reaches validation with independent
  // roles but has l_t=5e-13 instead of exactly zero.  No block may commit.
  assert_l1_match_fails_closed(fixture);
}

void test_l1_split_near_epigraph_coefficient_fails_closed() {
  using gpu_presolver::presolve::PresolveParams;

  L1SplitFixture fixture(
      L1SplitFixtureVariant::kNearEpigraphCoefficient, -2.0, 2.0);
  PresolveParams params;
  apply_l1_split_fixture(fixture, params);

  assert_l1_match_fails_closed(fixture);
}

void test_l1_split_tiny_nonzero_epigraph_rhs_fails_closed() {
  using gpu_presolver::presolve::PresolveParams;

  L1SplitFixture fixture(
      L1SplitFixtureVariant::kTinyNonzeroEpigraphRhs, -2.0, 2.0);
  PresolveParams params;
  apply_l1_split_fixture(fixture, params);

  assert_l1_match_fails_closed(fixture);
}

void test_l1_split_wrong_sign_infinity_row_bound_fails_closed() {
  using gpu_presolver::presolve::PresolveParams;

  L1SplitFixture fixture(
      L1SplitFixtureVariant::kWrongSignInfinityRowBound, -2.0, 2.0);
  PresolveParams params;
  apply_l1_split_fixture(fixture, params);

  assert_l1_match_fails_closed(fixture);
}

void test_l1_split_duplicate_e_in_equality_fails_closed() {
  using gpu_presolver::presolve::PresolveParams;

  L1SplitFixture fixture(
      L1SplitFixtureVariant::kDuplicateEInEquality, -2.0, 2.0);
  PresolveParams params;
  apply_l1_split_fixture(fixture, params);

  assert_l1_match_fails_closed(fixture);
}

void test_l1_split_a_side_certificate_rejects_forged_legal_at() {
  using gpu_presolver::presolve::PresolveParams;

  L1SplitFixture fixture(L1SplitFixtureVariant::kTInEquality, -2.0, 2.0);

  // A contains the illegal 0.25*t equality term, but replace AT with the
  // transpose of an otherwise-identical legal A which omits it.  Validation
  // must certify the source CSR itself rather than trusting this forged AT.
  HostCsr forged_at_source;
  forged_at_source.rows = 3;
  forged_at_source.cols = 3;
  forged_at_source.row_ptr = {0, 2, 4, 6};
  forged_at_source.col_val = {1, 2,  0, 1,  0, 1};
  forged_at_source.nz_val = {1.0, 2.0,  1.0, -1.0,  1.0, 1.0};
  free_matrix(fixture.lp.AT);
  fixture.lp.AT = upload(transpose(forged_at_source));

  PresolveParams params;
  apply_l1_split_fixture(fixture, params);

  assert_l1_match_fails_closed(fixture);
}

void test_l1_split_preserves_nonzero_fixed_residual_exactly() {
  using gpu_presolver::presolve::PresolveParams;

  L1SplitFixture fixture(L1SplitFixtureVariant::kValid, 1.25, 1.25);
  PresolveParams params;
  apply_l1_split_fixture(fixture, params);

  assert(fixture.plan.has_change);
  assert_l1_split_recovery(fixture);
  assert_valid_l1_transformed_columns(fixture);
  assert((copy_to_host(fixture.plan.keep_row_mask, 3) ==
          std::vector<std::uint8_t>{1, 1, 0}));
  assert((copy_to_host(fixture.plan.new_AL, 3) ==
          std::vector<double>{7.0, 1.25, 0.0}));
  const std::vector<double> au = copy_to_host(fixture.plan.new_AU, 3);
  assert(au[0] == 7.0);
  assert(au[1] == 1.25);
  assert(std::isinf(au[2]) && au[2] > 0.0);
  assert_l1_exact_matrix(fixture, true);
}

void test_l1_split_strict_free_only_accepts_true_free_residual() {
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::AbsoluteValueEpigraphResidualBoundMode;

  const double inf = std::numeric_limits<double>::infinity();
  L1SplitFixture fixture(L1SplitFixtureVariant::kValid, -inf, inf);
  PresolveParams params;
  params.absolute_value_epigraph_residual_bound_mode =
      AbsoluteValueEpigraphResidualBoundMode::StrictFreeOnly;
  apply_l1_split_fixture(fixture, params);

  assert(fixture.plan.has_change);
  assert_l1_split_recovery(fixture);
  assert_valid_l1_transformed_columns(fixture);
  assert((copy_to_host(fixture.plan.keep_row_mask, 3) ==
          std::vector<std::uint8_t>{1, 0, 0}));
  assert_l1_exact_matrix(fixture, false);
}

void test_l1_split_invalid_residual_bound_mode_fails_closed() {
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::AbsoluteValueEpigraphResidualBoundMode;

  const double inf = std::numeric_limits<double>::infinity();
  L1SplitFixture fixture(L1SplitFixtureVariant::kValid, -inf, inf);
  PresolveParams params;
  params.absolute_value_epigraph_residual_bound_mode =
      static_cast<AbsoluteValueEpigraphResidualBoundMode>(255);
  apply_l1_split_fixture(fixture, params);

  assert_l1_match_fails_closed(fixture);
}

void test_l1_split_reverse_role_order_emits_strictly_sorted_csr() {
  using gpu_presolver::presolve::PresolveParams;

  L1SplitFixture fixture(
      L1SplitFixtureVariant::kReverseRoleColumnOrder, -2.0, 2.0);
  PresolveParams params;
  apply_l1_split_fixture(fixture, params);

  assert(fixture.plan.has_change);
  assert(fixture.plan.has_new_A);
  assert(fixture.plan.has_absolute_value_epigraph_recovery);
  assert(fixture.plan.absolute_value_epigraph_recovery.pattern == "l1_split_3row");
  assert(fixture.plan.absolute_value_epigraph_recovery.splits.size() == 1);
  assert(fixture.plan.absolute_value_epigraph_recovery.splits[0].t_col == 2);
  assert(fixture.plan.absolute_value_epigraph_recovery.splits[0].e_col == 0);
  assert((copy_to_host(fixture.plan.keep_row_mask, 3) ==
          std::vector<std::uint8_t>{1, 1, 0}));
  assert((copy_to_host(fixture.plan.keep_col_mask, 3) ==
          std::vector<std::uint8_t>{1, 1, 1}));
  assert((copy_to_host(fixture.plan.new_c, 3) ==
          std::vector<double>{1.5, 3.0, 2.5}));
  const std::vector<double> l = copy_to_host(fixture.plan.new_l, 3);
  const std::vector<double> u = copy_to_host(fixture.plan.new_u, 3);
  assert(l[0] == 0.0);
  assert(std::isinf(l[1]) && l[1] < 0.0);
  assert(l[2] == 0.0);
  assert(std::isinf(u[0]) && u[0] > 0.0);
  assert(std::isinf(u[1]) && u[1] > 0.0);
  assert(std::isinf(u[2]) && u[2] > 0.0);
  assert((copy_to_host(fixture.plan.new_AL, 3) ==
          std::vector<double>{7.0, -2.0, 0.0}));
  const std::vector<double> au = copy_to_host(fixture.plan.new_AU, 3);
  assert(au[0] == 7.0);
  assert(au[1] == 2.0);
  assert(std::isinf(au[2]) && au[2] > 0.0);

  assert(fixture.plan.new_A.rows == 3);
  assert(fixture.plan.new_A.cols == 3);
  assert(fixture.plan.new_A.nnz == 5);
  assert((copy_to_host(fixture.plan.new_A.rowPtr, 4) ==
          std::vector<NnzOffset>{0, 3, 5, 5}));
  assert((copy_to_host(fixture.plan.new_A.colVal, 5) ==
          std::vector<std::int32_t>{0, 1, 2, 0, 2}));
  assert((copy_to_host(fixture.plan.new_A.nzVal, 5) ==
          std::vector<double>{-1.0, 2.0, 1.0, -1.0, 1.0}));
  assert_csr_rows_strictly_sorted(fixture.plan.new_A);
}

void test_graph_l1_mixed_validity_fails_closed_before_any_block_commit() {
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolveStatsGpu;

  TwoGraphBlockFixture fixture;
  PresolveStatsGpu stats;
  PresolveParams params;
  gpu_presolver::presolve::apply_rule_absolute_value_epigraph_reduction(
      fixture.plan, fixture.lp, stats, params);

  assert_two_graph_block_match_fails_closed(fixture);
}

void test_graph_l1_forged_at_missing_required_r3_fails_closed() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  const double inf = std::numeric_limits<double>::infinity();
  const std::vector<double> cost = {0.5, 2.0, 0.0};
  const std::vector<double> lower = {0.0, -inf, 0.0};
  const std::vector<double> upper = {inf, inf, inf};
  const std::vector<double> row_lower = {7.0, 0.0, 0.0, 0.0};
  const std::vector<double> row_upper = {7.0, inf, inf, inf};

  LPInfoGpu lp;
  lp.A = {4, 3, 7,
          copy_to_device<NnzOffset>({0, 1, 3, 5, 7}),
          copy_to_device<std::int32_t>({1, 0, 1, 0, 1, 0, 2}),
          copy_to_device<double>({3.0, 1.0, -1.0, 1.0, 1.0, -2.0, 1.0})};
  // A is the legal graph block above, but this forged AT omits t=0 in r3.
  // It retains t in r1/r2, all three required e incidences, and s in r3, so
  // the missing required-r3 certificate is the only structural defect.
  lp.AT = {3, 4, 6,
           copy_to_device<NnzOffset>({0, 2, 5, 6}),
           copy_to_device<std::int32_t>({1, 2, 0, 1, 2, 3}),
           copy_to_device<double>({1.0, 1.0, 3.0, -1.0, 1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1, 1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1, 1});
  plan.new_c = copy_to_device(cost);
  plan.new_l = copy_to_device(lower);
  plan.new_u = copy_to_device(upper);
  plan.new_AL = copy_to_device(row_lower);
  plan.new_AU = copy_to_device(row_upper);

  PresolveStatsGpu stats;
  PresolveParams params;
  gpu_presolver::presolve::apply_rule_absolute_value_epigraph_reduction(
      plan, lp, stats, params);

  assert(!plan.has_change);
  assert(!plan.has_row_action);
  assert(!plan.has_col_action);
  assert(!plan.has_new_A);
  assert(!plan.has_absolute_value_epigraph_recovery);
  assert((copy_to_host(plan.keep_row_mask, 4) ==
          std::vector<std::uint8_t>{1, 1, 1, 1}));
  assert((copy_to_host(plan.keep_col_mask, 3) ==
          std::vector<std::uint8_t>{1, 1, 1}));
  assert(copy_to_host(plan.new_c, 3) == cost);
  assert(copy_to_host(plan.new_l, 3) == lower);
  assert(copy_to_host(plan.new_u, 3) == upper);
  assert(copy_to_host(plan.new_AL, 4) == row_lower);
  assert(copy_to_host(plan.new_AU, 4) == row_upper);

  free_matrix(lp.A);
  free_matrix(lp.AT);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  free_matrix(plan.new_A);
}

void test_graph_l1_rewrites_four_row_block_and_records_postsolve() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {4, 3, 7,
          copy_to_device<NnzOffset>({0, 1, 3, 5, 7}),
          copy_to_device<std::int32_t>({1, 0, 1, 0, 1, 0, 2}),
          copy_to_device<double>({3.0, 1.0, -1.0, 1.0, 1.0, -2.0, 1.0})};
  lp.AT = {3, 4, 7,
           copy_to_device<NnzOffset>({0, 3, 6, 7}),
           copy_to_device<std::int32_t>({1, 2, 3, 0, 1, 2, 3}),
           copy_to_device<double>({1.0, 1.0, -2.0, 3.0, -1.0, 1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1, 1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1, 1});
  plan.new_c = copy_to_device<double>({0.5, 2.0, 0.0});
  plan.new_l = copy_to_device<double>({0.0, -inf, 0.0});
  plan.new_u = copy_to_device<double>({inf, inf, inf});
  plan.new_AL = copy_to_device<double>({7.0, 0.0, 0.0, 0.0});
  plan.new_AU = copy_to_device<double>({7.0, inf, inf, inf});

  PresolveStatsGpu stats;
  PresolveParams params;
  gpu_presolver::presolve::apply_rule_absolute_value_epigraph_reduction(plan, lp, stats, params);

  assert(plan.has_change);
  assert(plan.has_row_action);
  assert(plan.has_col_action);
  assert(plan.has_new_A);
  assert(plan.new_A.rows == 4);
  assert(plan.new_A.cols == 3);
  assert(plan.new_A.nnz == 2);

  const std::vector<std::uint8_t> keep_row = copy_to_host(plan.keep_row_mask, 4);
  const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, 3);
  const std::vector<double> c = copy_to_host(plan.new_c, 3);
  const std::vector<double> l = copy_to_host(plan.new_l, 3);
  const std::vector<double> u = copy_to_host(plan.new_u, 3);
  const std::vector<NnzOffset> row_ptr = copy_to_host(plan.new_A.rowPtr, 5);
  const std::vector<std::int32_t> col_val = copy_to_host(plan.new_A.colVal, 2);
  const std::vector<double> nz_val = copy_to_host(plan.new_A.nzVal, 2);

  assert((keep_row == std::vector<std::uint8_t>{1, 0, 0, 0}));
  assert((keep_col == std::vector<std::uint8_t>{1, 1, 0}));
  assert(std::fabs(c[0] - 2.5) < 1.0e-12);
  assert(std::fabs(c[1] - (-1.5)) < 1.0e-12);
  assert(l[0] == 0.0);
  assert(l[1] == 0.0);
  assert(std::isinf(u[0]));
  assert(std::isinf(u[1]));
  assert((row_ptr == std::vector<NnzOffset>{0, 2, 2, 2, 2}));
  assert((col_val == std::vector<std::int32_t>{0, 1}));
  assert(std::fabs(nz_val[0] - 3.0) < 1.0e-12);
  assert(std::fabs(nz_val[1] - (-3.0)) < 1.0e-12);

  assert(plan.has_absolute_value_epigraph_recovery);
  assert(plan.absolute_value_epigraph_recovery.pattern == "graph_l1_substitution");
  assert(plan.absolute_value_epigraph_recovery.splits.size() == 1);
  assert(plan.absolute_value_epigraph_recovery.max_slacks.size() == 1);
  assert(plan.absolute_value_epigraph_recovery.max_slacks[0].slack_col == 2);

  double* x_org = copy_to_device<double>({4.0, 1.0, 0.0});
  double* original_l = copy_to_device<double>({0.0, -inf, 0.0});
  gpu_presolver::presolve::postsolve_restore_absolute_value_epigraph_gpu(
      x_org, std::vector<gpu_presolver::presolve::AbsoluteValueEpigraphPrimalRecoveryStep>{plan.absolute_value_epigraph_recovery},
      original_l);
  const std::vector<double> x = copy_to_host(x_org, 3);
  assert(std::fabs(x[0] - 5.0) < 1.0e-12);
  assert(std::fabs(x[1] - 3.0) < 1.0e-12);
  assert(std::fabs(x[2] - 10.0) < 1.0e-12);

  cudaFree(x_org);
  cudaFree(original_l);
  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_A.rowPtr);
  cudaFree(plan.new_A.colVal);
  cudaFree(plan.new_A.nzVal);
}

void test_graph_l1_shared_auxiliary_uses_targeted_dual_fix() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  const double inf = std::numeric_limits<double>::infinity();
  // Two independent three-row epigraph blocks share s=4 in their fourth rows:
  // s >= 2*t0 and s >= 3*t2.  Only s is dual-fixable.  No other variable is
  // eligible for the targeted operation, even if an ordinary dual_fix would
  // find unrelated candidates elsewhere in the LP.
  HostCsr matrix;
  matrix.rows = 8;
  matrix.cols = 5;
  matrix.row_ptr = {0, 1, 3, 5, 7, 8, 10, 12, 14};
  matrix.col_val = {1, 0, 1, 0, 1, 0, 4, 3, 2, 3, 2, 3, 2, 4};
  matrix.nz_val = {3.0, 1.0, -1.0, 1.0, 1.0, -2.0, 1.0,
                   4.0, 1.0, -1.0, 1.0, 1.0, -3.0, 1.0};

  for (int invalid = 0; invalid < 5; ++invalid) {
    LPInfoGpu lp;
    lp.A = upload(matrix);
    HostCsr at = transpose(matrix);
    if (invalid == 3) {
      // Forged AT omits the second required shared-auxiliary incidence.
      at.col_val.pop_back();
      at.nz_val.pop_back();
      --at.row_ptr.back();
    } else if (invalid == 4) {
      // A forged extra CSC incidence outside the fourth rows must be rejected.
      at.col_val.push_back(0);
      at.nz_val.push_back(1.0);
      ++at.row_ptr.back();
    }
    lp.AT = upload(at);
    PresolvePlanGpu plan;
    plan.keep_row_mask = copy_to_device<std::uint8_t>(std::vector<std::uint8_t>(8, 1));
    plan.keep_col_mask = copy_to_device<std::uint8_t>(std::vector<std::uint8_t>(5, 1));
    plan.new_c = copy_to_device<double>({0.5, 2.0, 0.75, 1.25,
                                         invalid == 1 ? 1.0 : 0.0});
    plan.new_l = copy_to_device<double>({0.0, -inf, 0.0, -inf, 0.0});
    plan.new_u = copy_to_device<double>({inf, inf, inf, inf,
                                         invalid == 2 ? 10.0 : inf});
    plan.new_AL = copy_to_device<double>({7.0, 0.0, 0.0, 0.0,
                                          11.0, 0.0, 0.0, 0.0});
    plan.new_AU = copy_to_device<double>({7.0, inf, inf, inf,
                                          11.0, inf, inf, inf});

    PresolveStatsGpu stats;
    PresolveParams params;
    gpu_presolver::presolve::apply_rule_absolute_value_epigraph_reduction(
        plan, lp, stats, params);
    if (invalid != 0) {
      assert(!plan.has_change);
      assert(!plan.has_new_A);
      assert((copy_to_host(plan.keep_row_mask, 8) ==
              std::vector<std::uint8_t>(8, 1)));
      assert((copy_to_host(plan.keep_col_mask, 5) ==
              std::vector<std::uint8_t>(5, 1)));
    } else {
      assert(plan.has_change && plan.has_new_A);
      assert((copy_to_host(plan.keep_row_mask, 8) ==
              std::vector<std::uint8_t>{1, 0, 0, 0, 1, 0, 0, 0}));
      assert((copy_to_host(plan.keep_col_mask, 5) ==
              std::vector<std::uint8_t>{1, 1, 1, 1, 0}));
      assert(plan.new_A.nnz == 4);
      assert((copy_to_host(plan.new_A.rowPtr, 9) ==
              std::vector<NnzOffset>{0, 2, 2, 2, 2, 4, 4, 4, 4}));
      assert((copy_to_host(plan.new_A.colVal, 4) ==
              std::vector<std::int32_t>{0, 1, 2, 3}));
      assert((copy_to_host(plan.new_A.nzVal, 4) ==
              std::vector<double>{3.0, -3.0, 4.0, -4.0}));
      assert((copy_to_host(plan.new_c, 5) ==
              std::vector<double>{2.5, -1.5, 2.0, -0.5, 0.0}));
      assert(plan.absolute_value_epigraph_recovery.pattern == "graph_l1_substitution");
      assert(plan.absolute_value_epigraph_recovery.max_slacks.size() == 1);
      const auto& recovery = plan.absolute_value_epigraph_recovery.max_slacks[0];
      assert(recovery.slack_col == 4);
      assert((recovery.t_cols == std::vector<std::int32_t>{0, 2}));
      assert((recovery.factors == std::vector<double>{2.0, 3.0}));
      double* x = copy_to_device<double>({4.0, 1.0, 2.0, 1.0, 0.0});
      double* original_l = copy_to_device<double>({0.0, -inf, 0.0, -inf, 0.0});
      gpu_presolver::presolve::postsolve_restore_absolute_value_epigraph_gpu(
          x, {plan.absolute_value_epigraph_recovery}, original_l);
      assert((copy_to_host(x, 5) ==
              std::vector<double>{5.0, 3.0, 3.0, 1.0, 10.0}));
      cudaFree(x);
      cudaFree(original_l);
    }

    free_matrix(lp.A);
    free_matrix(lp.AT);
    cudaFree(plan.keep_row_mask);
    cudaFree(plan.keep_col_mask);
    cudaFree(plan.new_c);
    cudaFree(plan.new_l);
    cudaFree(plan.new_u);
    cudaFree(plan.new_AL);
    cudaFree(plan.new_AU);
    free_matrix(plan.new_A);
  }
}

void test_outer_shared_bound_accumulates_objective_and_restores_primal() {
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolveStatsGpu;

  OuterFixture fixture(OuterFixtureVariant::kSharedBound);
  PresolveStatsGpu stats;
  PresolveParams params;
  gpu_presolver::presolve::apply_rule_absolute_value_epigraph_reduction(
      fixture.plan, fixture.lp, stats, params);

  assert(fixture.plan.has_change);
  assert(fixture.plan.has_new_A);
  assert(fixture.plan.has_absolute_value_epigraph_recovery);
  assert(fixture.plan.absolute_value_epigraph_recovery.pattern == "outer_pair_linked_l1");
  assert(fixture.plan.absolute_value_epigraph_recovery.outer_pairs.size() == 2);
  assert(fixture.plan.absolute_value_epigraph_recovery.outer_pairs[0].bound_col == 0);
  assert(fixture.plan.absolute_value_epigraph_recovery.outer_pairs[0].free_col == 1);
  assert(fixture.plan.absolute_value_epigraph_recovery.outer_pairs[1].bound_col == 0);
  assert(fixture.plan.absolute_value_epigraph_recovery.outer_pairs[1].free_col == 2);

  const std::vector<double> reduced_cost = copy_to_host(fixture.plan.new_c, fixture.cost.size());
  const std::vector<std::uint8_t> keep_col = copy_to_host(fixture.plan.keep_col_mask, 10);
  assert(std::fabs(reduced_cost[0] - 6.0) < 1.0e-12);
  assert(keep_col[0] == std::uint8_t{1});
  assert(keep_col[1] == std::uint8_t{0});
  assert(keep_col[2] == std::uint8_t{0});

  assert(fixture.plan.new_A.rows == 14);
  assert(fixture.plan.new_A.cols == 10);
  assert(fixture.plan.new_A.nnz == 14);
  const std::vector<NnzOffset> row_ptr = copy_to_host(fixture.plan.new_A.rowPtr, 15);
  const std::vector<std::int32_t> col_val = copy_to_host(fixture.plan.new_A.colVal, 14);
  const std::vector<double> nz_val = copy_to_host(fixture.plan.new_A.nzVal, 14);
  assert((row_ptr == std::vector<NnzOffset>{
                         0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 5, 8, 11, 14}));
  assert((col_val == std::vector<std::int32_t>{
                         3, 4,  3, 4, 6,  3, 4, 7,
                         3, 4, 8,  3, 4, 9}));
  assert((nz_val == std::vector<double>{
                        -1.0, 1.0,
                        -1.0, -1.0, 1.0,
                        -1.0, -1.0, 1.0,
                        -1.0, -1.0, 1.0,
                        -1.0, -1.0, 1.0}));

  const double a_value = 2.25;
  const double original_pair_objective =
      fixture.cost[0] * a_value + fixture.cost[1] * a_value + fixture.cost[2] * a_value;
  const double reduced_pair_objective = reduced_cost[0] * a_value;
  assert(std::fabs(original_pair_objective - reduced_pair_objective) < 1.0e-12);

  std::vector<double> reduced_primal(10, 0.0);
  reduced_primal[0] = a_value;
  double* x_org = copy_to_device(reduced_primal);
  double* original_l = copy_to_device(fixture.lower);
  gpu_presolver::presolve::postsolve_restore_absolute_value_epigraph_gpu(
      x_org,
      std::vector<gpu_presolver::presolve::AbsoluteValueEpigraphPrimalRecoveryStep>{
          fixture.plan.absolute_value_epigraph_recovery},
      original_l);
  const std::vector<double> restored = copy_to_host(x_org, reduced_primal.size());
  assert(std::fabs(restored[0] - a_value) < 1.0e-12);
  assert(std::fabs(restored[1] - a_value) < 1.0e-12);
  assert(std::fabs(restored[2] - a_value) < 1.0e-12);
  cudaFree(x_org);
  cudaFree(original_l);
}

void test_outer_mapped_dense_row_is_sorted_with_exact_coefficients() {
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolveStatsGpu;

  OuterMappedDenseSortFixture fixture;
  PresolveStatsGpu stats;
  PresolveParams params;
  gpu_presolver::presolve::apply_rule_absolute_value_epigraph_reduction(
      fixture.plan, fixture.lp, stats, params);

  assert(fixture.plan.has_change);
  assert(fixture.plan.has_row_action);
  assert(fixture.plan.has_col_action);
  assert(fixture.plan.has_new_A);
  assert(fixture.plan.has_absolute_value_epigraph_recovery);
  assert(fixture.plan.absolute_value_epigraph_recovery.pattern ==
         "outer_pair_linked_l1");
  assert(fixture.plan.absolute_value_epigraph_recovery.outer_pairs.size() == 1);
  assert(fixture.plan.absolute_value_epigraph_recovery.outer_pairs[0].bound_col == 0);
  assert(fixture.plan.absolute_value_epigraph_recovery.outer_pairs[0].free_col == 9);
  assert(fixture.plan.absolute_value_epigraph_recovery.splits.size() == 1);
  assert(fixture.plan.absolute_value_epigraph_recovery.splits[0].t_col == 2);
  assert(fixture.plan.absolute_value_epigraph_recovery.splits[0].e_col == 1);

  assert((copy_to_host(fixture.plan.keep_row_mask, 11) ==
          std::vector<std::uint8_t>{0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}));
  assert((copy_to_host(fixture.plan.keep_col_mask, 10) ==
          std::vector<std::uint8_t>{1, 1, 1, 0, 1, 1, 1, 1, 1, 0}));
  assert((copy_to_host(fixture.plan.new_c, 10) ==
          std::vector<double>{3.0, 0.5, 1.5, 0.75, 1.0,
                              1.0, 1.0, 1.0, 3.0, 2.0}));

  assert(fixture.plan.new_A.rows == 11);
  assert(fixture.plan.new_A.cols == 10);
  assert(fixture.plan.new_A.nnz == 16);
  assert((copy_to_host(fixture.plan.new_A.rowPtr, 12) ==
          std::vector<NnzOffset>{0, 0, 0, 0, 4, 4, 4, 4, 7, 10, 13, 16}));
  assert((copy_to_host(fixture.plan.new_A.colVal, 16) ==
          std::vector<std::int32_t>{
              0, 1, 2, 8,
              1, 2, 4,
              1, 2, 5,
              1, 2, 6,
              1, 2, 7}));
  assert((copy_to_host(fixture.plan.new_A.nzVal, 16) ==
          std::vector<double>{
              2.0, -1.0, 1.0, 5.0,
              -1.0, -1.0, 1.0,
              -1.0, -1.0, 1.0,
              -1.0, -1.0, 1.0,
              -1.0, -1.0, 1.0}));
  assert_csr_rows_strictly_sorted(fixture.plan.new_A);
}

void test_outer_two_block_control_is_accepted() {
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolveStatsGpu;

  OuterFixture fixture(OuterFixtureVariant::kTwoBlockValid);
  PresolveStatsGpu stats;
  PresolveParams params;
  gpu_presolver::presolve::apply_rule_absolute_value_epigraph_reduction(
      fixture.plan, fixture.lp, stats, params);

  assert(fixture.plan.has_change);
  assert(fixture.plan.has_new_A);
  assert(fixture.plan.has_absolute_value_epigraph_recovery);
  assert(fixture.plan.absolute_value_epigraph_recovery.splits.size() == 2);
  assert(fixture.plan.absolute_value_epigraph_recovery.linked_slacks.size() == 2);
}

void assert_outer_match_fails_closed(OuterFixtureVariant variant) {
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolveStatsGpu;

  OuterFixture fixture(variant);
  PresolveStatsGpu stats;
  PresolveParams params;
  gpu_presolver::presolve::apply_rule_absolute_value_epigraph_reduction(
      fixture.plan, fixture.lp, stats, params);

  assert(!fixture.plan.has_change);
  assert(!fixture.plan.has_row_action);
  assert(!fixture.plan.has_col_action);
  assert(!fixture.plan.has_new_A);
  assert(!fixture.plan.has_absolute_value_epigraph_recovery);
  assert((copy_to_host(fixture.plan.keep_row_mask, fixture.lp.A.rows) ==
          std::vector<std::uint8_t>(static_cast<std::size_t>(fixture.lp.A.rows),
                                    std::uint8_t{1})));
  assert((copy_to_host(fixture.plan.keep_col_mask, fixture.lp.A.cols) ==
          std::vector<std::uint8_t>(static_cast<std::size_t>(fixture.lp.A.cols),
                                    std::uint8_t{1})));
  assert(copy_to_host(fixture.plan.new_c, fixture.cost.size()) == fixture.cost);
  assert(copy_to_host(fixture.plan.new_l, fixture.lower.size()) == fixture.lower);
  assert(copy_to_host(fixture.plan.new_u, fixture.upper.size()) == fixture.upper);
  assert(copy_to_host(fixture.plan.new_AL, fixture.row_lower.size()) == fixture.row_lower);
  assert(copy_to_host(fixture.plan.new_AU, fixture.row_upper.size()) == fixture.row_upper);
}

void test_outer_shared_bound_dense_mapping_duplicate_fails_closed() {
  assert_outer_match_fails_closed(
      OuterFixtureVariant::kDenseSharedBoundMappedDuplicate);
}

void test_outer_invalid_variable_ownership_fails_closed() {
  assert_outer_match_fails_closed(OuterFixtureVariant::kDuplicateFree);
  assert_outer_match_fails_closed(OuterFixtureVariant::kBoundFreeOverlap);
  assert_outer_match_fails_closed(OuterFixtureVariant::kOuterBlockRoleConflict);
  assert_outer_match_fails_closed(OuterFixtureVariant::kBlockAuxLocalXConflict);
  assert_outer_match_fails_closed(OuterFixtureVariant::kDenseOwnQConflict);
  assert_outer_match_fails_closed(OuterFixtureVariant::kDenseOwnSConflict);
  assert_outer_match_fails_closed(OuterFixtureVariant::kDenseCrossQConflict);
  assert_outer_match_fails_closed(OuterFixtureVariant::kDenseCrossEConflict);
  assert_outer_match_fails_closed(OuterFixtureVariant::kDenseCrossSConflict);
  assert_outer_match_fails_closed(OuterFixtureVariant::kDenseDuplicateE);
}

// Startup recognizes an initially valid pattern. A pattern exposed only by
// later cleanup is intentionally not retried by this experimental schedule.
void test_absolute_value_schedule(bool needs_cleanup, int max_iters,
                                  bool enable_startup = true) {
  using namespace gpu_presolver::presolve;
  L1SplitFixture fixture(needs_cleanup ? L1SplitFixtureVariant::kTrailingEmptyRow :
                                       L1SplitFixtureVariant::kValid, -5.0, 5.0);
  // Rule fixtures own their input vectors through plan; the full pipeline
  // reads them through LPInfoGpu and keeps the borrowed input unchanged.
  fixture.lp.c = fixture.plan.new_c;
  fixture.lp.l = fixture.plan.new_l;
  fixture.lp.u = fixture.plan.new_u;
  fixture.lp.AL = fixture.plan.new_AL;
  fixture.lp.AU = fixture.plan.new_AU;
  PresolveParams params;
  params.max_iters = max_iters;
  params.enable_startup = enable_startup;
  params.enable_infeasible_and_redundant_rows = false;
  params.enable_bound_strengthening = false;
  params.enable_dual_fix = false;
  params.enable_doubleton_equations = false;
  params.enable_parallel_rows = false;
  params.enable_column_domination = false;
  params.enable_variable_pair_aggregation = false;
  params.enable_two_row_variable_elimination = false;
  params.enable_mccormick_variable_elimination = false;
  params.enable_singleton_columns = false;
  params.enable_parallel_columns = false;

  const auto saved = runtime::options().profile_rules;

  runtime::options().profile_rules = true;
  std::ostringstream trace;
  auto* previous = std::cerr.rdbuf(trace.rdbuf());
  auto result = run_gpu_presolve_with_record(fixture.lp, params);
  std::cerr.rdbuf(previous);
  runtime::options().profile_rules = saved;
  assert(!result.has_infeasible && !result.has_unbounded);
  const std::string log = trace.str();
  assert(log.find("phase=startup_rows") == std::string::npos);
  const auto startup = log.find("phase=startup_absolute_value_epigraph changed=true");
  if (enable_startup && !needs_cleanup) {
    assert(startup != std::string::npos);
    assert(!result.record.absolute_value_epigraph_recoveries.empty());
    const auto propagation = log.find("phase=propagation");
    const auto parallel = log.find("phase=parallel");
    assert((propagation == std::string::npos || startup < propagation) &&
           (parallel == std::string::npos || startup < parallel));
    if (max_iters == 1) {
      assert(log.find("[schedule-phase] phase=cleanup") != std::string::npos);
      assert(result.iterations == 1);
    }
  } else {
    assert(startup == std::string::npos);
    assert(result.record.absolute_value_epigraph_recoveries.empty());
    assert(log.find("absolute_value_epigraph_retry") == std::string::npos);
    if (max_iters == 1) {
      assert(log.find("[schedule-phase] phase=cleanup") != std::string::npos);
      assert(result.iterations == 1);
    }
  }
}

void test_absolute_value_startup_schedule() {
  using namespace gpu_presolver::presolve;
  OuterFixture fixture(OuterFixtureVariant::kSharedBound);
  fixture.lp.c = fixture.plan.new_c;
  fixture.lp.l = fixture.plan.new_l;
  fixture.lp.u = fixture.plan.new_u;
  fixture.lp.AL = fixture.plan.new_AL;
  fixture.lp.AU = fixture.plan.new_AU;
  assert(absolute_value_epigraph_prefix_screen_passes(fixture.lp));
  PresolveParams params;
  params.max_iters = 1;
  params.enable_column_domination = false;
  params.enable_variable_pair_aggregation = false;
  params.enable_two_row_variable_elimination = false;
  params.enable_mccormick_variable_elimination = false;

  const auto saved = runtime::options().profile_rules;

  runtime::options().profile_rules = true;
  std::ostringstream trace;
  auto* previous = std::cerr.rdbuf(trace.rdbuf());
  auto result = run_gpu_presolve_with_record(fixture.lp, params);
  std::cerr.rdbuf(previous);
  runtime::options().profile_rules = saved;
  assert(!result.has_infeasible && !result.has_unbounded);
  assert(result.iterations == 1);
  assert(!result.record.absolute_value_epigraph_recoveries.empty());
  assert(trace.str().find("phase=startup_absolute_value_epigraph changed=true") !=
         std::string::npos);
  assert(trace.str().find("phase=parallel") == std::string::npos);
}

}  // namespace

int main() {
  test_absolute_value_startup_schedule();
  test_absolute_value_schedule(false, 1);
  test_absolute_value_schedule(false, 12);
  test_absolute_value_schedule(true, 12);
  test_absolute_value_schedule(false, 12, false);
  test_l1_split_default_preserves_finite_residual_bounds_exactly();
  test_l1_split_true_free_residual_uses_three_to_one_reduction();
  test_l1_split_default_preserves_one_sided_residual_bound_exactly();
  test_l1_split_strict_free_only_rejects_finite_bounds_without_mutation();
  test_l1_split_legacy_wide_mode_retains_old_three_to_one_behavior();
  test_l1_split_illegal_t_incidence_fails_closed();
  test_l1_split_shared_roles_across_blocks_fail_closed();
  test_l1_split_mixed_validation_fails_before_any_block_commit();
  test_l1_split_near_epigraph_coefficient_fails_closed();
  test_l1_split_tiny_nonzero_epigraph_rhs_fails_closed();
  test_l1_split_wrong_sign_infinity_row_bound_fails_closed();
  test_l1_split_duplicate_e_in_equality_fails_closed();
  test_l1_split_a_side_certificate_rejects_forged_legal_at();
  test_l1_split_preserves_nonzero_fixed_residual_exactly();
  test_l1_split_strict_free_only_accepts_true_free_residual();
  test_l1_split_invalid_residual_bound_mode_fails_closed();
  test_l1_split_reverse_role_order_emits_strictly_sorted_csr();
  test_graph_l1_mixed_validity_fails_closed_before_any_block_commit();
  test_graph_l1_forged_at_missing_required_r3_fails_closed();
  test_graph_l1_rewrites_four_row_block_and_records_postsolve();
  test_graph_l1_shared_auxiliary_uses_targeted_dual_fix();
  test_outer_shared_bound_accumulates_objective_and_restores_primal();
  test_outer_mapped_dense_row_is_sorted_with_exact_coefficients();
  test_outer_two_block_control_is_accepted();
  test_outer_shared_bound_dense_mapping_duplicate_fails_closed();
  test_outer_invalid_variable_ownership_fails_closed();
  std::cout << "test_rule_absolute_value_epigraph_reduction passed\n";
  return 0;
}
