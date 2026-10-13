#pragma once

// Explicit opt-in diagnostic output. Never enabled by the benchmark runner.
inline void write_diagnostic_snapshot(
    const char* label, const LPInfoGpu& lp, const PresolvePlanGpu* plan = nullptr) {
  const auto& root = runtime::options().snapshot_directory;
  if (root.empty()) return;
  const auto path = std::filesystem::path(root) / label;
  if (!std::filesystem::create_directories(path))
    throw std::runtime_error("diagnostic snapshot already exists: " + path.string());
  std::ofstream meta(path / "metadata.json");
  meta << "{\"rows\":" << lp.A.rows << ",\"cols\":" << lp.A.cols
       << ",\"nnz\":" << lp.A.nnz << ",\"offset_bytes\":" << sizeof(NnzOffset)
       << ",\"has_plan\":" << (plan ? "true" : "false") << "}\n";
  auto copy = [&](const std::string& name, const auto* device, std::size_t count) {
    using T = std::remove_cv_t<std::remove_pointer_t<decltype(device)>>;
    std::vector<T> host(count);
    if (count) throw_if_cuda_error(cudaMemcpy(host.data(), device, sizeof(T)*count,
                                             cudaMemcpyDeviceToHost), "diagnostic snapshot");
    std::ofstream out(path / (name + ".bin"), std::ios::binary);
    out.write(reinterpret_cast<const char*>(host.data()), sizeof(T)*count);
    if (!out) throw std::runtime_error("cannot write diagnostic snapshot " + name);
  };
  auto matrix = [&](const char* name, const DeviceCsrMatrix& a) {
    copy(std::string(name)+"_row_ptr", a.rowPtr, std::size_t(a.rows)+1);
    copy(std::string(name)+"_col_index", a.colVal, a.nnz);
    copy(std::string(name)+"_value", a.nzVal, a.nnz);
  };
  matrix("A", lp.A);
  matrix("AT", lp.AT);
  copy("row_lower", plan ? plan->new_AL : lp.AL, lp.A.rows);
  copy("row_upper", plan ? plan->new_AU : lp.AU, lp.A.rows);
  copy("col_lower", plan ? plan->new_l : lp.l, lp.A.cols);
  copy("col_upper", plan ? plan->new_u : lp.u, lp.A.cols);
  copy("objective", plan ? plan->new_c : lp.c, lp.A.cols);
  const double constant = lp.obj_constant;
  std::ofstream out(path / "objective_constant.bin", std::ios::binary);
  out.write(reinterpret_cast<const char*>(&constant), sizeof(constant));
  if (plan) {
    copy("keep_row", plan->keep_row_mask, lp.A.rows);
    copy("keep_col", plan->keep_col_mask, lp.A.cols);
  }
}
