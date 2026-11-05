// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

// Included by gpu_presolve_kernels.cu; not a standalone build source.
#include "gpu_preprocessor/lp/presolve/model_upload.hpp"
#include <cstring>
#include <limits>

namespace gpu_preprocessor::lp::presolve {
namespace model_upload_detail {
constexpr std::size_t min_count = (1024 * 1024) / sizeof(double);
struct Inputs {
  const double* data[6]; std::size_t count[6], offset[6], packed_offset[6];
};
struct Pattern { std::uint64_t values[4] = {}; unsigned count = 0, bad = 0; };
struct ScratchOom {};
void checked(cudaError_t status) {
  if (status == cudaErrorMemoryAllocation) throw ScratchOom{};
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void clear_oom() {
  if (cudaPeekAtLastError() == cudaErrorMemoryAllocation) (void)cudaGetLastError();
}
__device__ std::uint64_t bits_at(const double* input, std::size_t i) {
  std::uint64_t bits; memcpy(&bits, input+i, sizeof(bits)); return bits;
}
__device__ void insert(Pattern& p, std::uint64_t value) {
  for (unsigned i=0;i<p.count;++i) if (p.values[i] == value) return;
  if (p.count == 4) { p.bad = 1; return; }
  unsigned i = p.count++;
  while (i && p.values[i-1] > value) { p.values[i] = p.values[i-1]; --i; }
  p.values[i] = value;
}
struct Merge {
  __device__ Pattern operator()(Pattern a, const Pattern& b) const {
    if (a.bad || b.bad) { a.bad=1; a.count=0; return a; }
    for (unsigned i=0;i<b.count && !a.bad;++i) insert(a,b.values[i]);
    return a;
  }
};
__global__ void inspect(Inputs in, Pattern* out) {
  using Reduce = cub::BlockReduce<Pattern,256>;
  __shared__ typename Reduce::TempStorage temporary;
  const int field = blockIdx.y;
  const auto total = in.count[field];
  const auto chunks = (total+upload_chunk_elements-1)/upload_chunk_elements;
  for (std::size_t chunk=blockIdx.x;chunk<chunks;chunk+=gridDim.x) {
    const auto offset = chunk*upload_chunk_elements;
    const auto n = min(upload_chunk_elements,total-offset);
    Pattern local;
    for (std::size_t i=threadIdx.x;i<n && !local.bad;i+=blockDim.x)
      insert(local,bits_at(in.data[field]+offset,i));
    const auto result = Reduce(temporary).Reduce(local,Merge{});
    if (!threadIdx.x) out[in.offset[field]+chunk] = result;
    __syncthreads();
  }
}
__global__ void pack(Inputs in, const Pattern* patterns, unsigned char* packed, unsigned* changed) {
  const int field = blockIdx.y;
  if (in.packed_offset[field] == ~std::size_t(0)) return;
  const auto total = in.count[field];
  const auto chunks = (total+upload_chunk_elements-1)/upload_chunk_elements;
  for (std::size_t chunk=blockIdx.x;chunk<chunks;chunk+=gridDim.x) {
    const auto& p = patterns[in.offset[field]+chunk];
    if (p.bad || p.count < 2 || p.count > 4) continue;
    const auto offset = chunk*upload_chunk_elements;
    const auto n = min(upload_chunk_elements,total-offset);
    for (std::size_t i=threadIdx.x;i<(n+3)/4;i+=blockDim.x) {
      unsigned byte = 0;
      for (unsigned k=0;k<4 && i*4+k<n;++k) {
        const auto bits = bits_at(in.data[field],offset+i*4+k);
        unsigned code = 0;
        while (code<p.count && p.values[code]!=bits) ++code;
        if (code==p.count) { atomicExch(changed,1u); code=0; }
        byte |= code << (k*2);
      }
      packed[in.packed_offset[field]+offset/4+i] = byte;
    }
  }
}
__global__ void fill(double* output, std::size_t n, std::uint64_t bits) {
  for (std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
       i<n;i+=std::size_t(blockDim.x)*gridDim.x) memcpy(output+i,&bits,sizeof(bits));
}
__global__ void unpack(double* output, std::size_t offset, std::size_t n,
                       const unsigned char* packed, UploadChunkHint dictionary) {
  for (std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
       i<n;i+=std::size_t(blockDim.x)*gridDim.x) {
    const auto index = offset+i;
    const unsigned code = (packed[index/4] >> ((index%4)*2)) & 3u;
    const auto bits = dictionary.values[code];
    memcpy(output+index,&bits,sizeof(bits));
  }
}
int blocks(std::size_t count) { return int(std::min<std::size_t>(4096,(count+255)/256)); }
}  // namespace model_upload_detail

ModelUploadHints capture_model_upload_hints(const LPInfoGpu& model) {
  using namespace model_upload_detail;
  Inputs input{{model.A.nzVal,model.c,model.AL,model.AU,model.l,model.u},
               {std::size_t(std::max<NnzOffset>(0,model.A.nnz)),std::size_t(std::max(0,model.A.cols)),
                std::size_t(std::max(0,model.A.rows)),std::size_t(std::max(0,model.A.rows)),
                std::size_t(std::max(0,model.A.cols)),std::size_t(std::max(0,model.A.cols))},{},{}};
  std::size_t largest=0,total_chunks=0,downloaded=0;
  for (int i=0;i<6;++i) {
    if (!input.data[i] || input.count[i]<min_count) input.count[i]=0;
    input.offset[i]=total_chunks;
    const auto chunks=(input.count[i]+upload_chunk_elements-1)/upload_chunk_elements;
    total_chunks+=chunks; largest=std::max(largest,chunks);
    input.packed_offset[i]=std::numeric_limits<std::size_t>::max();
  }
  if (!largest) return {};
  auto cleanup=[](void* p) { cudaFree(p); };
  std::unique_ptr<void,decltype(cleanup)> owner(nullptr,cleanup),code_owner(nullptr,cleanup);
  try {
    void* allocation=nullptr;
    const auto metadata_bytes=sizeof(Pattern)*total_chunks;
    checked(cudaMalloc(&allocation,metadata_bytes+sizeof(unsigned))); owner.reset(allocation);
    auto* raw=static_cast<Pattern*>(allocation);
    auto* changed=reinterpret_cast<unsigned*>(static_cast<unsigned char*>(allocation)+metadata_bytes);
    checked(cudaMemset(changed,0,sizeof(unsigned)));
    const int grid=int(std::min<std::size_t>(4096,largest));
    inspect<<<dim3(grid,6),256>>>(input,raw); checked(cudaGetLastError());
    std::vector<Pattern> result(total_chunks);
    checked(cudaMemcpy(result.data(),raw,metadata_bytes,cudaMemcpyDeviceToHost));
    downloaded += metadata_bytes;
    ModelUploadHints hints;
    FieldUploadHint* fields[]={&hints.values,&hints.c,&hints.AL,&hints.AU,&hints.l,&hints.u};
    std::size_t total_packed=0;
    for (int i=0;i<6;++i) {
      auto& field=*fields[i]; field.count=input.count[i];
      const auto chunks=(field.count+upload_chunk_elements-1)/upload_chunk_elements;
      if (!chunks) continue;
      field.bits=result[input.offset[i]].values[0]; field.uniform=true;
      std::size_t encodable=0;
      field.chunks.reserve(chunks);
      for (std::size_t j=0;j<chunks;++j) {
        const auto& p=result[input.offset[i]+j]; UploadChunkHint chunk;
        chunk.count=p.bad ? 0 : p.count;
        std::copy(p.values,p.values+4,chunk.values); field.chunks.push_back(chunk);
        field.uniform &= chunk.count==1 && chunk.values[0]==field.bits;
        if (chunk.count>=2 && chunk.count<=4)
          encodable+=std::min(upload_chunk_elements,field.count-j*upload_chunk_elements);
      }
      if (field.uniform) field.chunks.clear();
      // The two extra packed transfers must save substantial raw traffic.
      else if (encodable >= (field.count+3)/4) {
        field.packed.resize((field.count+3)/4);
        input.packed_offset[i]=total_packed; total_packed+=field.packed.size();
      }
    }
    if (total_packed) {
      void* allocation=nullptr; checked(cudaMalloc(&allocation,total_packed)); code_owner.reset(allocation);
      auto* codes=static_cast<unsigned char*>(allocation); checked(cudaMemset(codes,0,total_packed));
      pack<<<dim3(grid,6),256>>>(input,raw,codes,changed); checked(cudaGetLastError());
      unsigned invalid=0; checked(cudaMemcpy(&invalid,changed,sizeof(invalid),cudaMemcpyDeviceToHost));
      downloaded += sizeof(invalid);
      if (invalid) throw std::runtime_error("Model changed during upload certification");
      for (int i=0;i<6;++i) if (!fields[i]->packed.empty()) {
        checked(cudaMemcpy(fields[i]->packed.data(),codes+input.packed_offset[i],fields[i]->packed.size(),cudaMemcpyDeviceToHost));
        downloaded += fields[i]->packed.size();
      }
    }
    hints.download_bytes=downloaded;
    return hints;
  } catch (const ScratchOom&) {
    code_owner.reset(); owner.reset(); clear_oom();
    ModelUploadHints omitted; omitted.download_bytes=downloaded; return omitted;
  } catch (const std::bad_alloc&) {
    ModelUploadHints omitted; omitted.download_bytes=downloaded; return omitted;
  }
}

static void fill_uniform_double_gpu(double* output, std::size_t count, std::uint64_t bits) {
  if (!count) return;
  model_upload_detail::fill<<<model_upload_detail::blocks(count),256>>>(output,count,bits);
  const auto status=cudaGetLastError();
  if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

bool upload_with_hint(double* output, const double* input, std::size_t count,
                      const FieldUploadHint& hint) {
  using namespace model_upload_detail;
  if (!count || hint.count!=count) return false;
  if (hint.uniform) { fill_uniform_double_gpu(output,count,hint.bits); return true; }
  const auto chunks=(count+upload_chunk_elements-1)/upload_chunk_elements;
  if (hint.chunks.size()!=chunks || (!hint.packed.empty() && hint.packed.size()!=(count+3)/4)) return false;
  auto cleanup=[](unsigned char* p) { cudaFree(p); };
  std::unique_ptr<unsigned char,decltype(cleanup)> owner(nullptr,cleanup);
  try {
  if (!hint.packed.empty()) {
    unsigned char* codes=nullptr;
    const auto status=cudaMalloc(&codes,hint.packed.size());
    if (status==cudaErrorMemoryAllocation) { clear_oom(); return false; }
    checked(status); owner.reset(codes);
    checked(cudaMemcpy(codes,hint.packed.data(),hint.packed.size(),cudaMemcpyHostToDevice));
  }
  auto kind=[&](std::size_t i) {
    const auto size=hint.chunks[i].count;
    return size==1 ? 1 : size>=2 && size<=4 && owner ? 2 : 0;
  };
  for (std::size_t begin=0;begin<chunks;) {
    const auto& pattern=hint.chunks[begin]; const int mode=kind(begin);
    std::size_t end=begin+1;
    while (end<chunks && kind(end)==mode) {
      if (mode && (hint.chunks[end].count!=pattern.count ||
          !std::equal(pattern.values,pattern.values+pattern.count,hint.chunks[end].values))) break;
      ++end;
    }
    const auto offset=begin*upload_chunk_elements;
    const auto length=std::min(count,end*upload_chunk_elements)-offset;
    if (mode==1) fill_uniform_double_gpu(output+offset,length,pattern.values[0]);
    else if (mode==2) {
      unpack<<<blocks(length),256>>>(output,offset,length,owner.get(),pattern); checked(cudaGetLastError());
    } else checked(cudaMemcpy(output+offset,input+offset,length*sizeof(double),cudaMemcpyHostToDevice));
    begin=end;
  }
    return true;
  } catch (const ScratchOom&) {
    owner.reset(); clear_oom(); return false;
  }
}

bool try_upload_uniform_double(double* output, const double* input, std::size_t count) {
  using namespace model_upload_detail;
  if (count<min_count) return false;
  std::uint64_t bits; std::memcpy(&bits,input,sizeof(bits));
  for (const auto i:{count/4,count/2,3*(count/4),count-1}) {
    std::uint64_t other; std::memcpy(&other,input+i,sizeof(other));
    if (bits!=other) return false;
  }
  std::array<std::uint64_t,512> expected; expected.fill(bits);
  const auto* bytes=reinterpret_cast<const char*>(input); const auto total=count*sizeof(double);
  for (std::size_t offset=0;offset<total;offset+=sizeof(expected))
    if (std::memcmp(bytes+offset,expected.data(),std::min(sizeof(expected),total-offset))) return false;
  fill_uniform_double_gpu(output,count,bits); return true;
}
}  // namespace gpu_preprocessor::lp::presolve
