// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once
#include "gpu_preprocessor/lp/presolve/presolve_structs.hpp"
#include <cstddef>
#include <cstdint>

namespace gpu_preprocessor::lp::presolve {
// Exact bit patterns only. A zero count means the field was not inspected.
inline constexpr std::size_t upload_chunk_elements = (256 * 1024) / sizeof(double);
struct UploadChunkHint {
  std::uint64_t values[4] = {};
  unsigned count = 0;  // zero: ordinary copy; one: fill; two to four: packed.
};
struct FieldUploadHint {
  std::uint64_t bits = 0;
  std::size_t count = 0;
  bool uniform = false;
  std::vector<UploadChunkHint> chunks;
  std::vector<unsigned char> packed;
};
struct ModelUploadHints {
  FieldUploadHint values, c, AL, AU, l, u;
  std::size_t download_bytes = 0;
};
// Read-only certificate for large fields of an immutable device model. Only
// compact bit dictionaries and lossless packed indices return to host.
// Scratch OOM returns uninspected hints, retaining the transfer byte count.
ModelUploadHints capture_model_upload_hints(const LPInfoGpu& model);
// Apply a matching certificate, coalescing adjacent ordinary-copy or equal-bit
// fill chunks. The certificate requires the same immutable index order.
// False requests an ordinary overwrite of the entire field before use.
bool upload_with_hint(double* output, const double* input, std::size_t count,
                      const FieldUploadHint& hint);
// Certify a large host field, then queue its exact fill; false means the caller
// must use its ordinary copy. No floating-point comparisons or arithmetic.
bool try_upload_uniform_double(double* output, const double* input, std::size_t count);
}  // namespace gpu_preprocessor::lp::presolve
