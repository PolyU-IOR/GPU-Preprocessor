// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "snapshot.hpp"
#include "gpu_preprocessor/lp/presolve/structural_dual.hpp"
#include "gpu_preprocessor/lp/presolve/device_recovery.hpp"
#include "gpu_preprocessor/lp/presolve/compact_bound_record.hpp"
#include "gpu_preprocessor/lp/presolve/packed_bound_record.hpp"
#include "gpu_preprocessor/lp/presolve/rules/rule_row_sparsification.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unistd.h>

namespace gpu_preprocessor::lp::diagnostics {
namespace {
namespace gp = presolve;
namespace gf = folding;

void checked(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void free_lp(gp::LPInfoGpu& lp) {
  for (void* p : {static_cast<void*>(lp.A.rowPtr), static_cast<void*>(lp.A.colVal),
                 static_cast<void*>(lp.A.nzVal), static_cast<void*>(lp.AT.rowPtr),
                 static_cast<void*>(lp.AT.colVal), static_cast<void*>(lp.AT.nzVal),
                 static_cast<void*>(lp.c), static_cast<void*>(lp.AL), static_cast<void*>(lp.AU),
                 static_cast<void*>(lp.l), static_cast<void*>(lp.u), static_cast<void*>(lp.AT_slack_after)})
    if (p) cudaFree(p);
  lp = {};
}

class Archive {
  FILE* file_ = nullptr;
  std::vector<unsigned char> buffer_{std::vector<unsigned char>(8 * 1024 * 1024)};
 public:
  const bool reading;
  std::uint32_t version = 7;
  Archive(const std::string& path, bool read) : reading(read) {
    if (read) file_ = std::fopen(path.c_str(), "rb");
    else {
      const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
      if (fd >= 0) {
        file_ = ::fdopen(fd, "wb");
        if (!file_) ::close(fd);
      }
    }
    if (!file_) throw std::runtime_error("Cannot open snapshot: " + path + ": " + std::strerror(errno));
  }
  ~Archive() { if (file_) std::fclose(file_); }
  void bytes(void* data, std::size_t size) {
    if (!size) return;
    const auto done = reading ? std::fread(data, 1, size, file_) : std::fwrite(data, 1, size, file_);
    if (done != size) throw std::runtime_error("Truncated snapshot or snapshot I/O failure");
  }
  template<class T> void scalar(T& value) {
    static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>);
    bytes(&value, sizeof(T));
  }
  void flag(bool& value) {
    std::uint8_t raw = value ? 1 : 0;
    scalar(raw);
    if (raw > 1) throw std::runtime_error("Invalid snapshot flag");
    if (reading) value = raw != 0;
  }
  std::size_t count(std::uint64_t n) {
    scalar(n);
    if (n > std::uint64_t(std::numeric_limits<std::int32_t>::max()))
      throw std::runtime_error("Snapshot collection count is out of range");
    return std::size_t(n);
  }
  template<class T> void vector(std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto n = count(values.size());
    if (reading) values.resize(n);
    bytes(values.data(), n * sizeof(T));
  }
  void string(std::string& value) {
    const auto n = count(value.size());
    if (reading) value.resize(n);
    bytes(value.data(), n);
  }
  template<class T> void device(T*& pointer, std::int64_t count) {
    if (count < 0 || std::uint64_t(count) > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::runtime_error("Invalid snapshot device-array length");
    bool present = pointer != nullptr;
    flag(present);
    const std::uint64_t expected = std::uint64_t(count) * sizeof(T);
    std::uint64_t size = present ? expected : 0;
    scalar(size);
    if (size != (present ? expected : 0)) throw std::runtime_error("Snapshot array-size mismatch");
    if (!present || !size) { if (reading) pointer = nullptr; return; }
    if (reading) checked(cudaMalloc(reinterpret_cast<void**>(&pointer), std::size_t(size)));
    auto* address = reinterpret_cast<unsigned char*>(pointer);
    for (std::uint64_t offset = 0; offset < size;) {
      const auto chunk = std::size_t(std::min<std::uint64_t>(size - offset, buffer_.size()));
      if (reading) {
        bytes(buffer_.data(), chunk);
        checked(cudaMemcpy(address + offset, buffer_.data(), chunk, cudaMemcpyHostToDevice));
      } else {
        checked(cudaMemcpy(buffer_.data(), address + offset, chunk, cudaMemcpyDeviceToHost));
        bytes(buffer_.data(), chunk);
      }
      offset += chunk;
    }
  }
  void header(const char* expected) {
    std::array<char,16> magic{};
    std::memcpy(magic.data(), expected, std::strlen(expected));
    const auto wanted = magic;
    bytes(magic.data(), magic.size());
    std::uint32_t endian = 0x01020304, offsets = sizeof(NnzOffset);
    scalar(version); scalar(endian); scalar(offsets);
    if (magic != wanted || (version < 1 || version > 7) || endian != 0x01020304 || offsets != sizeof(NnzOffset))
      throw std::runtime_error("Unsupported snapshot format, byte order or offset width");
  }
  void finish() {
    if (reading) {
      if (std::fgetc(file_) != EOF) throw std::runtime_error("Unexpected trailing snapshot bytes");
    } else if (std::fflush(file_) != 0) throw std::runtime_error("Snapshot flush failed");
  }
};

void matrix(Archive& a, gp::DeviceCsrMatrix& m) {
  a.scalar(m.rows); a.scalar(m.cols); a.scalar(m.nnz);
  if (m.rows < 0 || m.cols < 0 || m.nnz < 0) throw std::runtime_error("Invalid snapshot matrix dimensions");
  a.device(m.rowPtr, std::int64_t(m.rows) + 1);
  a.device(m.colVal, m.nnz); a.device(m.nzVal, m.nnz);
}
void model(Archive& a, gp::LPInfoGpu& lp) {
  matrix(a, lp.A); matrix(a, lp.AT);
  a.device(lp.c, lp.A.cols); a.device(lp.l, lp.A.cols); a.device(lp.u, lp.A.cols);
  a.device(lp.AL, lp.A.rows); a.device(lp.AU, lp.A.rows);
  a.scalar(lp.obj_constant); a.scalar(lp.AT_leading_slack);
  a.device(lp.AT_slack_after, lp.A.cols);
}
template<class T> void array(Archive& a, gp::RecoveryArray<T>& value) {
  a.scalar(value.count); a.device(value.data, value.count);
}
template<class T, class F> void list(Archive& a, std::vector<T>& items, F visitor) {
  const auto n = a.count(items.size());
  if (a.reading) items.resize(n);
  for (auto& item : items) visitor(a, item);
}
void group(Archive& a, gp::DeviceRecoveryGroup& g) {
  list(a, g.ids, array<std::int32_t>); list(a, g.values, array<double>);
}
void absolute(Archive& a, gp::DevicePiecewiseLinearReformulationAbsoluteValueStep& s) {
  a.string(s.pattern); group(a,s.splits); group(a,s.pairs); group(a,s.linked); list(a,s.max_slacks,group);
}
void structural(Archive& a, std::shared_ptr<gp::StructuralDualRecovery>& pointer) {
  if (a.reading) pointer = std::make_shared<gp::StructuralDualRecovery>();
  if (!pointer) throw std::runtime_error("Missing structural recovery checkpoint");
  auto& s = *pointer;
  model(a,s.model); array(a,s.rows); array(a,s.cols);
  array(a,s.keep_rows); array(a,s.keep_cols); array(a,s.row_scale);
  group(a,s.pairs); absolute(a,s.absolute);
  array(a,s.tree_parent); array(a,s.tree_row); array(a,s.tree_next);
  array(a,s.tree_head); array(a,s.tree_scale);
}
void tape(Archive& a, gp::PostsolveTapeGpu& t) {
  a.scalar(t.record_count); a.scalar(t.index_count); a.scalar(t.value_count);
  if (t.record_count < 0 || t.index_count < 0 || t.value_count < 0)
    throw std::runtime_error("Invalid snapshot tape dimensions");
  a.device(t.types, t.record_count);
  a.device(t.index_starts, std::int64_t(t.record_count) + 1);
  a.device(t.value_starts, std::int64_t(t.record_count) + 1);
  a.device(t.dual_modes, t.record_count);
  a.device(t.indices, t.index_count); a.device(t.vals, t.value_count);
  if (a.reading) {
    t.record_capacity = t.record_count; t.index_capacity = t.index_count; t.value_capacity = t.value_count;
  }
}
void record(Archive& a, gp::PresolveRecordGpu& r) {
  if (r.startup_poisson) throw std::runtime_error("Opaque startup inverse is not supported by snapshot version 1");
  a.scalar(r.m0); a.scalar(r.n0); a.scalar(r.m1); a.scalar(r.n1);
  a.scalar(r.obj_constant_old); a.scalar(r.obj_constant_new);
  a.flag(r.has_primal_only_piecewise_linear_reformulation_variable_pair_reduction);
  a.flag(r.has_primal_only_column_domination_reduction);
  a.flag(r.has_primal_only_projected_auxiliary_reduction);
  list(a, r.piecewise_linear_reformulation_absolute_value_recoveries, [](Archive& a, gp::PiecewiseLinearReformulationAbsoluteValuePrimalRecoveryStep& s) {
    a.string(s.pattern);
    list(a, s.splits, [](Archive& a, gp::PiecewiseLinearReformulationAbsoluteValueSplitRecovery& v) {
      a.scalar(v.t_col); a.scalar(v.e_col); a.scalar(v.rho);
    });
    list(a, s.outer_pairs, [](Archive& a, gp::AbsoluteValueOuterPairRecovery& v) {
      a.scalar(v.bound_col); a.scalar(v.free_col);
    });
    list(a, s.linked_slacks, [](Archive& a, gp::AbsoluteValueLinkedSlackRecovery& v) {
      a.scalar(v.slack_col); a.scalar(v.t_col); a.scalar(v.factor);
    });
    list(a, s.max_slacks, [](Archive& a, gp::AbsoluteValueMaxSlackRecovery& v) {
      a.scalar(v.slack_col); a.vector(v.t_cols); a.vector(v.factors);
    });
  });
  list(a, r.piecewise_linear_reformulation_variable_pair_recoveries, [](Archive& a, gp::PiecewiseLinearReformulationVariablePairRecoveryStep& s) {
    a.vector(s.plus_cols); a.vector(s.minus_cols); a.vector(s.root_plus_cols);
    a.vector(s.root_minus_cols); a.vector(s.plus_shift); a.vector(s.minus_shift);
  });
  list(a, r.primal_recovery_timeline, [](Archive& a, gp::PrimalRecoveryCheckpoint& c) {
    a.scalar(c.kind); a.scalar(c.payload_index); a.scalar(c.tape_position);
    if (a.version >= 3) a.scalar(c.dual_index);
  });
  if (a.version >= 3) list(a,r.structural_dual_recoveries,structural);
  bool device = bool(r.device); a.flag(device);
  if (!device) throw std::runtime_error("Snapshot version 1 requires device recovery records");
  if (a.reading) r.device = std::make_shared<gp::DeviceRecoveryState>();
  auto& d = *r.device;
  array(a, d.rows); array(a, d.cols);
  a.scalar(d.records); a.scalar(d.indices); a.scalar(d.values);
  list(a, d.segments, tape);
  list(a, d.fixed, [](Archive& a, gp::DeviceFixedSegment& f) { array(a,f.indices); array(a,f.values); });
  list(a, d.removed_rows, array<std::int32_t>); list(a, d.removed_cols, array<std::int32_t>);
  list(a, d.piecewise_linear_reformulation_variable_pair, group);
  list(a, d.piecewise_linear_reformulation_absolute_value, absolute);
  bool rows = bool(r.startup_rows); a.flag(rows);
  if (rows) {
    if (a.reading) r.startup_rows = std::make_shared<gp::StartupRowSparsification>();
    auto& s = *r.startup_rows;
    model(a, s.transformed);
    a.device(s.parents, s.transformed.A.rows); a.device(s.alpha, s.transformed.A.rows);
  }
  if (a.reading) gp::validate_device_recovery(r);
}
void summary(Archive& a, gp::GpuPresolveSummary& s, bool keep_model) {
  a.scalar(s.original_rows); a.scalar(s.original_cols);
  a.scalar(s.reduced_rows); a.scalar(s.reduced_cols); a.scalar(s.reduced_nnz);
  record(a, s.record);
  if (keep_model) {
    if (a.reading) s.owns_reduced_lp = true;
    model(a, s.reduced_lp);
  }
}
void pipeline(Archive& a, gf::FoldingPipelineSummary& p) {
  a.flag(p.presolve_enabled); a.flag(p.folding_applied); a.flag(p.startup_split);
  a.scalar(p.original_rows); a.scalar(p.original_cols);
  a.scalar(p.startup_rows); a.scalar(p.startup_cols); a.scalar(p.startup_nnz);
  a.scalar(p.folded_rows); a.scalar(p.folded_cols);
  if (!p.presolve_enabled) throw std::runtime_error("Snapshot version 1 requires presolve recovery");
  if (p.folding_applied) {
    auto& f = p.folding;
    if (a.reading) { f.applied = true; f.owns_folded_lp = true; f.map.owns_buffers = true; }
    a.scalar(f.original_rows); a.scalar(f.original_cols);
    a.scalar(f.folded_rows); a.scalar(f.folded_cols);
    auto& m = f.map;
    a.scalar(m.original_rows); a.scalar(m.original_cols);
    a.scalar(m.reduced_rows); a.scalar(m.reduced_cols);
    a.device(m.row_color_id, m.original_rows); a.device(m.col_color_id, m.original_cols);
    a.device(m.row_scale, m.original_rows); a.device(m.col_scale, m.original_cols);
    model(a, f.folded_lp);
  }
  bool startup = bool(p.startup); a.flag(startup);
  if (startup) {
    if (a.reading) p.startup = std::make_shared<gf::StartupPresolveState>();
    summary(a, p.startup->summary, true);
  }
  summary(a, p.presolve, a.version >= 2);
}
void release_lp_extra(gp::LPInfoGpu& lp) {
  if (lp.AT_slack_after) cudaFree(lp.AT_slack_after);
  lp.AT_slack_after = nullptr;
}
void release_row_extra(gp::PresolveRecordGpu& r) {
  if (r.startup_rows) release_lp_extra(r.startup_rows->transformed);
}
}  // namespace

PostsolveSnapshot::~PostsolveSnapshot() {
  cudaFree(x); cudaFree(y); cudaFree(z);
  free_lp(original);
  release_lp_extra(pipeline.folding.folded_lp);
  folding::free_folded_lp(pipeline.folding);
  release_row_extra(pipeline.presolve.record);
  release_lp_extra(pipeline.presolve.reduced_lp);
  presolve::free_gpu_presolve_reduced_lp(pipeline.presolve);
  if (pipeline.startup) {
    release_lp_extra(pipeline.startup->summary.reduced_lp);
    release_row_extra(pipeline.startup->summary.record);
  }
}

void save_snapshot(const std::string& path, const double* x, const double* y, const double* z,
                   const folding::FoldingPipelineSummary& p, const presolve::LPInfoGpu& original,
                   double tolerance) {
  checked(cudaDeviceSynchronize());
  Archive a(path, false); a.header("GPUP-PS-SNAPSHOT");
  a.scalar(tolerance);
  // The writing visitor reads descriptors and arrays without changing their contents.
  pipeline(a, const_cast<folding::FoldingPipelineSummary&>(p));
  model(a, const_cast<presolve::LPInfoGpu&>(original));
  auto* dx = const_cast<double*>(x); auto* dy = const_cast<double*>(y); auto* dz = const_cast<double*>(z);
  a.device(dx, p.presolve.reduced_cols); a.device(dy, p.presolve.reduced_rows); a.device(dz, p.presolve.reduced_cols);
  a.finish();
}

std::unique_ptr<PostsolveSnapshot> load_snapshot(const std::string& path) {
  auto out = std::make_unique<PostsolveSnapshot>();
  Archive a(path, true); a.header("GPUP-PS-SNAPSHOT");
  out->format_version = a.version;
  a.scalar(out->primal_tolerance); pipeline(a,out->pipeline); model(a,out->original);
  a.device(out->x, out->pipeline.presolve.reduced_cols);
  a.device(out->y, out->pipeline.presolve.reduced_rows);
  a.device(out->z, out->pipeline.presolve.reduced_cols);
  a.finish(); checked(cudaDeviceSynchronize());
  return out;
}

void save_reference(const std::string& path, const folding::UnfoldedSolutionHost& reference) {
  Archive a(path,false); a.version = 6; a.header("GPUP-PS-REF");
  auto& r = const_cast<folding::UnfoldedSolutionHost&>(reference);
  a.vector(r.x); a.vector(r.y); a.vector(r.z); a.finish();
}

folding::UnfoldedSolutionHost load_reference(const std::string& path) {
  Archive a(path,true); a.header("GPUP-PS-REF");
  folding::UnfoldedSolutionHost r;
  a.vector(r.x); a.vector(r.y); a.vector(r.z); a.finish();
  return r;
}

void print_snapshot_statistics(const PostsolveSnapshot& snapshot) {
  const auto print = [](const char* stage, const gp::PresolveRecordGpu& r) {
    std::array<std::uint64_t,20> counts{};
    std::size_t segment_index = 0;
    std::uint64_t exact=0,numerical=0,unavailable=0,legacy=0,compact=0,packed=0;
    double max_mismatch=0;
    for (const auto& segment : r.device->segments) {
      std::array<std::uint64_t,20> segment_counts{};
      std::vector<std::int32_t> types(segment.record_count);
      checked(cudaMemcpy(types.data(),segment.types,types.size()*sizeof(types[0]),cudaMemcpyDeviceToHost));
      for (auto type : types) {
        if (type < 0 || std::size_t(type) >= counts.size()) throw std::runtime_error("Unknown tape record type");
        ++counts[type];
        ++segment_counts[type];
      }
      compact+=segment_counts[int(presolve::PostsolveReductionType::BoundChangeCompact)];
      packed+=segment_counts[int(presolve::PostsolveReductionType::BoundPackedBatch)];
      if(segment_counts[int(presolve::PostsolveReductionType::BoundChangeSharedRow)] ||
         segment_counts[int(presolve::PostsolveReductionType::RowImpliedBoundChange)] ||
         segment_counts[int(presolve::PostsolveReductionType::VariableBoundChange)]) {
        std::vector<NnzOffset> is(segment.record_count+1),vs(segment.record_count+1);
        std::vector<double> values(segment.value_count);
        checked(cudaMemcpy(is.data(),segment.index_starts,is.size()*sizeof(NnzOffset),cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(vs.data(),segment.value_starts,vs.size()*sizeof(NnzOffset),cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(values.data(),segment.vals,values.size()*sizeof(double),cudaMemcpyDeviceToHost));
        for(std::size_t k=0;k<types.size();++k) {
          if(types[k]!=int(presolve::PostsolveReductionType::BoundChangeSharedRow) &&
             types[k]!=int(presolve::PostsolveReductionType::RowImpliedBoundChange) &&
             types[k]!=int(presolve::PostsolveReductionType::VariableBoundChange)) continue;
          if(is[k+1]-is[k]<5 || vs[k+1]-vs[k]<9) {++legacy;continue;}
          const auto v=vs[k];
          for(int side=0;side<2;++side) {
            if(values[v+side]==values[v+2+side]) continue;
            const int source=int(values[v+4+side]);
            if(source==1) ++exact;
            else if(source==2) {
              ++numerical;
              max_mismatch=std::max(max_mismatch,std::abs(values[v+2+side]-values[v+9+side]));
            } else ++unavailable;
          }
        }
      }
      std::printf("SNAPSHOT_SEGMENT {\"stage\":\"%s\",\"index\":%zu,\"records\":%lld,\"indices\":%lld,\"values\":%lld,\"type_counts\":[",
                  stage,segment_index++,static_cast<long long>(segment.record_count),
                  static_cast<long long>(segment.index_count),static_cast<long long>(segment.value_count));
      for (std::size_t i=0;i<segment_counts.size();++i)
        std::printf("%s%llu",i?",":"",static_cast<unsigned long long>(segment_counts[i]));
      std::printf("]}\n");
    }
    std::printf("BOUND_SOURCES {\"stage\":\"%s\",\"exact\":%llu,\"numerical\":%llu,\"unavailable\":%llu,\"legacy_records\":%llu,\"max_endpoint_mismatch\":%.17g,\"compact_records\":%llu,\"packed_batches\":%llu,\"endpoint_diagnostics_available\":%s}\n",
        stage,(unsigned long long)exact,(unsigned long long)numerical,(unsigned long long)unavailable,
        (unsigned long long)legacy,max_mismatch,(unsigned long long)compact,(unsigned long long)packed,(compact||packed)?"false":"true");
    std::printf("SNAPSHOT_STAGE {\"stage\":\"%s\",\"records\":%lld,\"segments\":%zu,\"type_counts\":[",
                stage,static_cast<long long>(r.device->records),r.device->segments.size());
    for (std::size_t i=0;i<counts.size();++i) std::printf("%s%llu",i?",":"",static_cast<unsigned long long>(counts[i]));
    std::printf("]}\n");
  };
  print("main",snapshot.pipeline.presolve.record);
  if (snapshot.pipeline.startup) print("startup",snapshot.pipeline.startup->summary.record);
}
}  // namespace gpu_preprocessor::lp::diagnostics
