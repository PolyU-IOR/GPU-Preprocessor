// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include "postsolve_common.cuh"
#include "gpu_preprocessor/lp/presolve/compact_bound_record.hpp"
#include "gpu_preprocessor/lp/presolve/packed_bound_record.hpp"

namespace gpu_preprocessor::lp::presolve::postsolve {

__device__ __forceinline__ void replay_row_implied_bound_change(
    double* x_org,
    double* y_org,
    double* z_org,
    std::uint8_t* z_retrieved,
    const std::int32_t* tape_idx,
    const double* tape_val,
    const NnzOffset* A_rowPtr,
    const std::int32_t* A_colVal,
    const double* A_nzVal,
    double tol,
    NnzOffset idx0,
    NnzOffset val0) {
  const std::int32_t col = tape_idx[idx0];
  const std::int32_t row = tape_idx[idx0 + 1];
  if (z_retrieved != nullptr && z_retrieved[col] == std::uint8_t{0}) {
    return;
  }
  const double old_l = tape_val[val0];
  const double old_u = tape_val[val0 + 1];
  const double new_l = tape_val[val0 + 2];
  const double new_u = tape_val[val0 + 3];
  const bool lower_changed = new_l > old_l + tol;
  const bool upper_changed = new_u < old_u - tol;
  if (lower_changed == upper_changed) {
    return;
  }
  const double implied_bound = lower_changed ? new_l : new_u;
  const double original_other_bound = lower_changed ? old_u : old_l;
  const bool original_other_is_lower = !lower_changed;
  const double xj = x_org[col];
  const double zj = z_org[col];
  if (_postsolve_is_finite(original_other_bound) && fabs(xj - original_other_bound) <= tol) {
    const bool sign_ok = original_other_is_lower ? (zj >= -tol) : (zj <= tol);
    if (sign_ok) {
      return;
    }
  }
  if (fabs(xj - implied_bound) > tol) {
    return;
  }
  double aij = 0.0;
  for (NnzOffset p = A_rowPtr[row]; p < A_rowPtr[row + 1]; ++p) {
    if (A_colVal[p] == col) {
      aij = A_nzVal[p];
      break;
    }
  }
  if (fabs(aij) <= tol) {
    return;
  }
  const double delta_y = zj / aij;
  if (y_org != nullptr) {
    y_org[row] += delta_y;
  }
  for (NnzOffset p = A_rowPtr[row]; p < A_rowPtr[row + 1]; ++p) {
    const std::int32_t kcol = A_colVal[p];
    if (kcol == col || (z_retrieved != nullptr && z_retrieved[kcol] == std::uint8_t{0})) {
      continue;
    }
    z_org[kcol] -= (A_nzVal[p] / aij) * zj;
  }
  z_org[col] = 0.0;
  if (z_retrieved != nullptr) {
    z_retrieved[col] = std::uint8_t{1};
  }
}

__device__ __forceinline__ bool replay_bound_strengthening_batch(
    double* y_org,
    double* z_org,
    double* bound_delta,
    std::uint8_t* bound_side,
    double* bound_row_delta,
    double* bound_row_error,
    const NnzOffset* idx_starts,
    const NnzOffset* val_starts,
    const std::int32_t* tape_idx,
    const double* tape_val,
    NnzOffset record_begin,
    std::int32_t replay_dual,
    NnzOffset idx0,
    NnzOffset idx1,
    NnzOffset val0,
    NnzOffset val1,
    NnzOffset& k,
    std::int32_t reduction_type) {
    if(reduction_type==int(PostsolveReductionType::BoundPackedBatch)) {
      if(replay_dual) {
        const PackedBoundView b{tape_idx+idx0};
        for(int r=0;r<b.rows();++r)bound_row_delta[b.row(r)]=bound_row_error[b.row(r)]=0.0;
        for(int t=0;t<b.transfers();++t) {
          const auto lo=b.reference(t,0),up=b.reference(t,1),any=lo>=0?lo:up;
          if(any<0)continue;
          const double incoming=z_org[b.column(any)];const auto ref=incoming>0.0?lo:up;
          if(ref<0)continue;const double coefficient=tape_val[val0+ref];if(coefficient==0.0)continue;
          const int row=b.row(b.owner(ref));
          const double delta=incoming/coefficient,old=bound_row_delta[row],sum=old+delta;
          bound_row_error[row]+=fabs(old)>=fabs(delta)?(old-sum)+delta:(delta-sum)+old;
          bound_row_delta[row]=sum;
        }
        for(int r=0;r<b.rows();++r) {
          const int row=b.row(r);const double delta=bound_row_delta[row]+bound_row_error[row];
          if(delta==0.0)continue;if(y_org)y_org[row]+=delta;
          for(auto q=b.start(r);q<b.start(r+1);++q)z_org[b.column(q)]-=tape_val[val0+q]*delta;
        }
      }
      return true;
    }
    if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::BoundBatch)) {
      const int rows=tape_idx[idx0],columns=tape_idx[idx0+1];
      const NnzOffset first=k-columns,sources=first-rows,base=val_starts[sources];
      if(replay_dual!=0) {
        for(NnzOffset t=sources;t<first;++t) {
          const int row=tape_idx[idx_starts[t]];
          bound_row_delta[row]=bound_row_error[row]=0.0;
        }
        for(NnzOffset t=first;t<k;++t) {
          const int* ids=tape_idx+idx_starts[t];
          const double incoming=z_org[ids[0]];const int side=incoming>0.0?0:1;
          const int slot=ids[1+side];const auto ref=compact_bound_reference(ids,side);
          if(slot<0 || ref<0) continue;
          const double coefficient=tape_val[base+ref];
          if(coefficient==0.0) continue;
          const int row=tape_idx[idx_starts[sources+slot]];
          const double delta=incoming/coefficient,old=bound_row_delta[row],sum=old+delta;
          bound_row_error[row]+=fabs(old)>=fabs(delta)?(old-sum)+delta:(delta-sum)+old;
          bound_row_delta[row]=sum;
        }
        for(NnzOffset t=sources;t<first;++t) {
          const auto ip=idx_starts[t],vp=val_starts[t];
          const int row=tape_idx[ip];const double delta=bound_row_delta[row]+bound_row_error[row];
          if(delta==0.0)continue;
          if(y_org)y_org[row]+=delta;
          for(NnzOffset q=0;q<idx_starts[t+1]-ip-1;++q)
            z_org[tape_idx[ip+1+q]]-=tape_val[vp+q]*delta;
        }
      }
      k=sources;
      return true;
    }
    if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::BoundChangeSharedRow)) {
      const NnzOffset first=k-static_cast<NnzOffset>(tape_val[val0+8])+1;
      const NnzOffset sources=first-static_cast<NnzOffset>(tape_val[val0+11]);
      if (replay_dual != 0) {
        for(NnzOffset t=sources;t<first;++t) {
          const int row=tape_idx[idx_starts[t]];
          bound_row_delta[row]=bound_row_error[row]=0.0;
        }
        // Read every incoming z before changing any column. Compensated row
        // sums avoid amplification from cancellation in a shared long row.
        for(NnzOffset t=first;t<=k;++t) {
          const auto ip=idx_starts[t],vp=val_starts[t];
          const double incoming=z_org[tape_idx[ip]];
          const int side=incoming>0.0?0:1;
          const double coefficient=tape_val[vp+6+side];
          if(tape_val[vp+4+side]==0.0 || coefficient==0.0) continue;
          const int row=tape_idx[ip+1+side];
          const double delta=incoming/coefficient, old=bound_row_delta[row], sum=old+delta;
          bound_row_error[row]+=fabs(old)>=fabs(delta)?(old-sum)+delta:(delta-sum)+old;
          bound_row_delta[row]=sum;
        }
        for(NnzOffset t=sources;t<first;++t) {
          const auto ip=idx_starts[t],vp=val_starts[t];
          const int row=tape_idx[ip];
          const double delta=bound_row_delta[row]+bound_row_error[row];
          if(delta==0.0) continue;
          if(y_org) y_org[row]+=delta;
          for(NnzOffset q=0;q<idx_starts[t+1]-ip-1;++q)
            z_org[tape_idx[ip+1+q]]-=tape_val[vp+q]*delta;
        }
      }
      k=sources;
      return true;
    }
    if (reduction_type == static_cast<std::int32_t>(PostsolveReductionType::RowImpliedBoundChange) &&
        idx1 >= idx0+4 && val1 >= val0+8) {
      if (replay_dual == 0) return true;
      // Adjacent host batches have explicit first-record markers. Resident
      // segments retain the same marker, so both paths use the same joint lift.
      NnzOffset first=k;
      while (first>record_begin && tape_val[val_starts[first]+8]==0.0) --first;
      for (NnzOffset t=first;t<=k;++t) {
        const auto ip=idx_starts[t],vp=val_starts[t];
        const int col=tape_idx[ip];const double incoming=z_org[col];
        const int side=incoming>0.0?0:1;
        const double coefficient=tape_val[vp+6+side];
        const bool supported=tape_val[vp+4+side]!=0.0 && coefficient!=0.0;
        bound_delta[col]=supported?incoming/coefficient:0.0;
        bound_side[col]=static_cast<std::uint8_t>(side);
        if (supported) z_org[col]=0.0;
      }
      for (NnzOffset t=first;t<=k;++t) {
        const auto ip=idx_starts[t],vp=val_starts[t];
        const int col=tape_idx[ip];const double delta=bound_delta[col];
        if(delta==0.0) continue;
        const bool lower=bound_side[col]==0;
        const int row=tape_idx[ip+(lower?1:2)];
        const int skip=lower?0:tape_idx[ip+3];
        const int count=tape_idx[ip+(lower?3:4)];
        const auto value_header=(val_starts[t+1]-vp)-(idx_starts[t+1]-ip-5);
        if(y_org) y_org[row]+=delta;
        for(int q=0;q<count;++q) {
          const int other=tape_idx[ip+5+skip+q];
          if(other!=col) z_org[other]-=tape_val[vp+value_header+skip+q]*delta;
        }
      }
      k=first;
      return true;
    }
  return false;
}

}  // namespace gpu_preprocessor::lp::presolve::postsolve
