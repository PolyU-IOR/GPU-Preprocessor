// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once
#include "gpu_preprocessor/lp/offset.hpp"
#include <cstdint>
#include <cstring>

namespace gpu_preprocessor::lp::presolve {
#if defined(__CUDACC__)
#define GPUP_PACKED_HD __host__ __device__
#else
#define GPUP_PACKED_HD
#endif
inline constexpr int packed_bound_words = sizeof(NnzOffset) / sizeof(int);
inline constexpr int packed_bound_header = 3 + packed_bound_words;
GPUP_PACKED_HD inline NnzOffset packed_bound_offset(const int* p) {
  if constexpr(sizeof(NnzOffset)==4) return p[0];
  const auto bits=std::uint64_t(std::uint32_t(p[0])) | (std::uint64_t(std::uint32_t(p[1]))<<32);
  return static_cast<NnzOffset>(bits);
}
GPUP_PACKED_HD inline void set_packed_bound_offset(int* p,NnzOffset value) {
  const auto lo=std::uint32_t(value);memcpy(p,&lo,sizeof(lo));
  if constexpr(sizeof(NnzOffset)==8) {const auto hi=std::uint32_t(std::uint64_t(value)>>32);memcpy(p+1,&hi,sizeof(hi));}
}
// A homogeneous batch shares its type/mode/offset metadata. The two optional
// reference planes retain original transfer order. A referenced coefficient
// identifies both the target column and its immutable source row.
struct PackedBoundView {
  const int* ids;
  GPUP_PACKED_HD int rows() const {return ids[0];}
  GPUP_PACKED_HD int transfers() const {return ids[1];}
  GPUP_PACKED_HD NnzOffset terms() const {return packed_bound_offset(ids+2);}
  GPUP_PACKED_HD int sides() const {return ids[2+packed_bound_words];}
  GPUP_PACKED_HD NnzOffset row_position(int r) const {return packed_bound_header+NnzOffset(r);}
  GPUP_PACKED_HD NnzOffset pointer_position(int r) const {return packed_bound_header+NnzOffset(rows())+NnzOffset(packed_bound_words)*r;}
  GPUP_PACKED_HD NnzOffset column_position(NnzOffset p) const {return packed_bound_header+NnzOffset(rows())+NnzOffset(packed_bound_words)*(NnzOffset(rows())+1)+p;}
  GPUP_PACKED_HD NnzOffset reference_position(int r,int side) const {
    const int plane=side==1 && (sides()&1)?1:0;
    return column_position(terms())+NnzOffset(packed_bound_words)*(NnzOffset(plane)*transfers()+r);
  }
  GPUP_PACKED_HD int row(int r) const {return ids[row_position(r)];}
  GPUP_PACKED_HD NnzOffset start(int r) const {return packed_bound_offset(ids+pointer_position(r));}
  GPUP_PACKED_HD int column(NnzOffset p) const {return ids[column_position(p)];}
  GPUP_PACKED_HD NnzOffset reference(int r,int side) const {return sides()&(1<<side)?packed_bound_offset(ids+reference_position(r,side)):NnzOffset(-1);}
  GPUP_PACKED_HD int owner(NnzOffset p) const {
    int lo=0,hi=rows();
    while(lo<hi) {const int mid=lo+(hi-lo)/2;if(start(mid+1)<=p)lo=mid+1;else hi=mid;}
    return lo;
  }
};
GPUP_PACKED_HD inline bool valid_packed_bound_header(const int* ids,NnzOffset indices,NnzOffset values) {
  if(indices<packed_bound_header || !ids) return false;
  const PackedBoundView b{ids};
  if(b.rows()<0 || b.transfers()<1 || b.terms()<0 || b.terms()!=values || b.sides()<0 || b.sides()>3) return false;
  const auto u=std::uint64_t(b.rows()),r=std::uint64_t(b.transfers()),s=std::uint64_t(b.terms());
  const auto planes=std::uint64_t(bool(b.sides()&1))+bool(b.sides()&2);
  return std::uint64_t(indices)==packed_bound_header+u+packed_bound_words*(u+1)+s+packed_bound_words*planes*r;
}
#undef GPUP_PACKED_HD
} // namespace gpu_preprocessor::lp::presolve
