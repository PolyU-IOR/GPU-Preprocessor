#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace gpu_presolver::presolve::detail {

// Tasks must not throw; callers preallocate all output and assign disjoint
// ranges. Join already-started workers if thread creation itself fails.
template<class Function>
void mapping_parallel_for(std::size_t n, Function fn) {
  constexpr unsigned workers_count=8;
  std::vector<std::thread> workers;
  workers.reserve(workers_count-1);
  auto run=[&](unsigned worker) { fn(worker,n*worker/workers_count,n*(worker+1)/workers_count); };
  try {
    for(unsigned worker=1;worker<workers_count;++worker)workers.emplace_back(run,worker);
    run(0);
  } catch(...) {
    for(auto& worker:workers)worker.join();
    throw;
  }
  for(auto& worker:workers)worker.join();
}

// Stable host compaction for large, non-identity recovery maps. Both kept and
// optional removed indices preserve source order across chunk boundaries.
inline void compact_mapping_stable(const std::vector<std::int32_t>& old_map,
                                  const std::vector<std::uint8_t>& keep,
                                  std::size_t expected,
                                  std::vector<std::int32_t>& next,
                                  std::vector<std::int32_t>* removed) {
  if(old_map.size()!=keep.size())throw std::runtime_error("mapping/mask size mismatch");
  std::array<std::size_t,9> prefix{};
  mapping_parallel_for(keep.size(),[&](unsigned worker,std::size_t begin,std::size_t end) {
    std::size_t count=0;
    for(auto i=begin;i<end;++i)count+=keep[i]!=0;
    prefix[worker+1]=count;
  });
  for(unsigned i=0;i<8;++i)prefix[i+1]+=prefix[i];
  if(prefix[8]!=expected)throw std::runtime_error("parallel mapping compaction count mismatch");
  next.resize(expected);
  const auto old_removed=removed?removed->size():0;
  if(removed)removed->resize(old_removed+keep.size()-expected);
  mapping_parallel_for(keep.size(),[&](unsigned worker,std::size_t begin,std::size_t end) {
    auto write=prefix[worker];auto drop=old_removed+begin-prefix[worker];
    for(auto i=begin;i<end;++i) {
      if(keep[i])next[write++]=old_map[i];
      else if(removed)(*removed)[drop++]=old_map[i];
    }
  });
}
} // namespace gpu_presolver::presolve::detail
