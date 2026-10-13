#pragma once
#include "gpu_presolver/offset.hpp"
#include <cstdint>
#include <cub/block/block_scan.cuh>

namespace gpu_presolver::presolve {

// Visit the same lane's entries as stable compaction, without copying A/AT.
// Null masks retain the ordinary contiguous traversal.
struct LiveRowCursor {
  NnzOffset pos, end;
  const std::int32_t* indices;
  const std::uint8_t* keep;
  __device__ LiveRowCursor(NnzOffset first, NnzOffset last,
                          const std::int32_t* idx, const std::uint8_t* mask,
                          int lane = 0)
      : pos(first), end(last), indices(idx), keep(mask) {
    skip();
    advance(lane);
  }
  __device__ void skip() {
    while (keep && pos < end && !keep[indices[pos]]) ++pos;
  }
  __device__ void advance(int stride = 1) {
    if (!keep) { pos += stride; return; }
    for (int k = 0; k < stride && pos < end; ++k) { ++pos; skip(); }
  }
};

// Cooperative stable filtering: each lane receives the same sequence as it
// would after physical compaction. Scan each stored entry once, then distribute
// complete groups of live positions; carry a partial group to the next tile.
template<int Width> struct LiveRowStorage {
  NnzOffset positions[512]; // Two tiles per warp, or two tiles for one block.
  typename cub::BlockScan<int, 256>::TempStorage scan;
};
template<int Width> __device__ void live_row_sync() {
  if constexpr (Width == 32) __syncwarp(); else __syncthreads();
}
template<int Width, class Visit>
__device__ void for_each_live_entry(NnzOffset first, NnzOffset last,
    const int* indices, const std::uint8_t* keep,
    LiveRowStorage<Width>& storage, Visit visit) {
  const int lane = Width == 32 ? threadIdx.x & 31 : threadIdx.x;
  if (!keep) {
    for (NnzOffset p = first + lane; p < last; p += Width) visit(p);
    return;
  }
  const int base = Width == 32 ? (threadIdx.x / 32) * 64 : 0;
  int carry = 0;
  for (NnzOffset chunk = first; chunk < last; chunk += Width) {
    const NnzOffset p = chunk + lane;
    const int live = p < last && keep[indices[p]];
    int rank, count;
    if constexpr (Width == 32) {
      const unsigned mask = __ballot_sync(0xffffffffu, live);
      rank = __popc(mask & ((1u << lane) - 1u));
      count = __popc(mask);
    } else {
      cub::BlockScan<int, 256>(storage.scan).ExclusiveSum(live, rank, count);
    }
    if (live) storage.positions[base + carry + rank] = p;
    live_row_sync<Width>();
    const int total = carry + count;
    if (total >= Width) {
      visit(storage.positions[base + lane]);
      live_row_sync<Width>();
      if (lane < total - Width)
        storage.positions[base + lane] = storage.positions[base + Width + lane];
      live_row_sync<Width>();
      carry = total - Width;
    } else {
      carry = total;
    }
  }
  if (lane < carry) visit(storage.positions[base + lane]);
  live_row_sync<Width>();
}
} // namespace gpu_presolver::presolve
