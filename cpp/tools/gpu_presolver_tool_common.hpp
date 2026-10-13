#include "gpu_presolver/offset.hpp"
#pragma once

#include "command_line.hpp"

#include "cpu_presolve/mpsreader.hpp"
#include "gpu_presolver/presolve/presolve_structs.hpp"
#include "gpu_presolver/folding/folding.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace gpu_presolver::tools {

struct FoldingPipelineOwner {
  folding::FoldingPipelineSummary summary;
  FoldingPipelineOwner() = default;
  FoldingPipelineOwner(const FoldingPipelineOwner&) = delete;
  FoldingPipelineOwner& operator=(const FoldingPipelineOwner&) = delete;
  ~FoldingPipelineOwner() {
    presolve::free_gpu_presolve_reduced_lp(summary.presolve);
    folding::free_folded_lp(summary.folding);
  }
};

inline void check_cuda(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

template <class T>
T* copy_to_device(const std::vector<T>& values) {
  if (values.empty()) {
    return nullptr;
  }
  T* device = nullptr;
  check_cuda(cudaMalloc(&device, sizeof(T) * values.size()), "cudaMalloc");
  const auto status = cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cudaFree(device);
    check_cuda(status, "cudaMemcpy H2D");
  }
  return device;
}

template <class T>
std::vector<T> copy_to_host(const T* device, std::size_t size) {
  std::vector<T> values(size);
  if (size == 0) {
    return values;
  }
  check_cuda(cudaMemcpy(values.data(), device, sizeof(T) * size, cudaMemcpyDeviceToHost),
             "cudaMemcpy D2H");
  return values;
}

inline std::vector<std::int32_t> to_i32(const std::vector<int>& input) {
  std::vector<std::int32_t> out(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    out[i] = static_cast<std::int32_t>(input[i]);
  }
  return out;
}

struct DeviceLpOwner {
  gpu_presolver::presolve::LPInfoGpu lp;

  DeviceLpOwner() = default;
  DeviceLpOwner(const DeviceLpOwner&) = delete;
  DeviceLpOwner& operator=(const DeviceLpOwner&) = delete;
  DeviceLpOwner(DeviceLpOwner&& other) noexcept : lp(std::exchange(other.lp, {})) {}
  DeviceLpOwner& operator=(DeviceLpOwner&& other) noexcept {
    if (this != &other) { reset(); lp = std::exchange(other.lp, {}); }
    return *this;
  }
  ~DeviceLpOwner() { reset(); }
  void reset() noexcept {
    cudaFree(lp.A.rowPtr);
    cudaFree(lp.A.colVal);
    cudaFree(lp.A.nzVal);
    cudaFree(lp.AT.rowPtr);
    cudaFree(lp.AT.colVal);
    cudaFree(lp.AT.nzVal);
    cudaFree(lp.c);
    cudaFree(lp.AL);
    cudaFree(lp.AU);
    cudaFree(lp.l);
    cudaFree(lp.u);
    lp = {};
  }
};

inline DeviceLpOwner upload_lp(const cpu_presolve::LpModel& model) {
  DeviceLpOwner owner;
  const cpu_presolve::CsrMatrix& csr = model.csr();
  const cpu_presolve::CscMatrix& csc = model.csc();

  checked_nnz_offset(csr.nnz());
  checked_nnz_offset(csc.nnz());
  const std::vector<NnzOffset> row_ptr = std::vector<NnzOffset>(csr.row_ptr().begin(), csr.row_ptr().end());
  const std::vector<std::int32_t> col_idx = to_i32(csr.col_idx());
  const std::vector<NnzOffset> at_row_ptr = std::vector<NnzOffset>(csc.col_ptr().begin(), csc.col_ptr().end());
  const std::vector<std::int32_t> at_col_idx = to_i32(csc.row_idx());

  owner.lp.A.rows = static_cast<std::int32_t>(csr.rows());
  owner.lp.A.cols = static_cast<std::int32_t>(csr.cols());
  owner.lp.A.nnz = static_cast<NnzOffset>(csr.nnz());
  owner.lp.A.rowPtr = copy_to_device(row_ptr);
  owner.lp.A.colVal = copy_to_device(col_idx);
  owner.lp.A.nzVal = copy_to_device(csr.values());

  owner.lp.AT.rows = static_cast<std::int32_t>(csc.cols());
  owner.lp.AT.cols = static_cast<std::int32_t>(csc.rows());
  owner.lp.AT.nnz = static_cast<NnzOffset>(csc.nnz());
  owner.lp.AT.rowPtr = copy_to_device(at_row_ptr);
  owner.lp.AT.colVal = copy_to_device(at_col_idx);
  owner.lp.AT.nzVal = copy_to_device(csc.values());

  owner.lp.c = copy_to_device(model.objective());
  owner.lp.AL = copy_to_device(model.row_lower());
  owner.lp.AU = copy_to_device(model.row_upper());
  owner.lp.l = copy_to_device(model.col_lower());
  owner.lp.u = copy_to_device(model.col_upper());
  owner.lp.obj_constant = model.obj_constant();
  return owner;
}

inline cpu_presolve::MpsModel read_mps_file(const std::string& path) {
  std::unique_ptr<std::ifstream> file;
  std::istream* input = &std::cin;
  if (path != "-") {
    file = std::make_unique<std::ifstream>(path);
    if (!file->is_open()) {
      throw std::runtime_error("failed to open " + path);
    }
    input = file.get();
  }
  return cpu_presolve::read_mps(*input);
}

template <class T>
void write_binary_vector(const std::filesystem::path& path, const std::vector<T>& values) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open " + path.string());
  }
  if (!values.empty()) {
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(sizeof(T) * values.size()));
  }
}

template <class T>
std::vector<T> read_binary_vector(const std::filesystem::path& path, std::size_t count) {
  std::vector<T> values(count);
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open " + path.string());
  }
  if (count > 0) {
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(sizeof(T) * count));
    if (!in) {
      throw std::runtime_error("failed to read " + path.string());
    }
  }
  return values;
}

}  // namespace gpu_presolver::tools
