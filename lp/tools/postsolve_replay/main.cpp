// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#include "snapshot.hpp"
#include "gpu_preprocessor/lp/presolve/structural_dual.hpp"
#include "kkt.hpp"
#include "gpu_preprocessor/lp/diagnostics/postsolve_observer.hpp"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;
bool measuring = false;
std::size_t row_bytes = 0, col_bytes = 0, copy_bytes = 0;
int copy_calls = 0;
double copy_seconds = 0;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
struct Comparison {
  std::size_t size = 0, different_bits = 0;
  double max_absolute = 0, max_relative = 0;
};
Comparison compare(const std::vector<double>& actual, const std::vector<double>& reference) {
  if (actual.size() != reference.size()) throw std::runtime_error("Reference dimension mismatch");
  Comparison c; c.size = actual.size();
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::memcmp(&actual[i], &reference[i], sizeof(double)) == 0) continue;
    ++c.different_bits;
    const double error = std::isfinite(actual[i]) && std::isfinite(reference[i])
        ? std::abs(actual[i] - reference[i]) : INFINITY;
    c.max_absolute = std::max(c.max_absolute, error);
    c.max_relative = std::max(c.max_relative, error / std::max(1.0, std::abs(reference[i])));
  }
  return c;
}
void print_comparison(const char* name, const Comparison& c) {
  std::printf("\"%s\":{\"size\":%zu,\"different_bits\":%zu,\"nonfinite_difference\":%s,\"max_absolute\":",
              name, c.size, c.different_bits, std::isfinite(c.max_absolute) ? "false" : "true");
  if (std::isfinite(c.max_absolute)) std::printf("%.17g",c.max_absolute); else std::printf("null");
  std::printf(",\"max_relative\":");
  if (std::isfinite(c.max_relative)) std::printf("%.17g",c.max_relative); else std::printf("null");
  std::printf("}");
}
}

extern "C" cudaError_t __real_cudaMemcpy(void*, const void*, std::size_t, cudaMemcpyKind);
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind) {
  const bool timed = measuring && kind == cudaMemcpyDeviceToHost && count > 0 &&
                     (count == row_bytes || count == col_bytes);
  const auto start = Clock::now();
  const auto status = __real_cudaMemcpy(dst, src, count, kind);
  if (timed) { copy_seconds += elapsed(start); copy_bytes += count; ++copy_calls; }
  return status;
}

int main(int argc, char** argv) {
  using namespace gpu_preprocessor::lp;
  try {
    std::string path, reference_path;
    int repeat = 3, warmup = 0;
    bool diagnose = false, diagnose_only = false, require_kkt = false;
    double kkt_tolerance = 1e-6;
    auto kkt_check = diagnostics::KktCheck::PrimalDualGap;
    std::string comparison = "strict";
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        std::puts("Usage: gpu_preprocessor_postsolve_replay --snapshot FILE [--reference FILE.reference] [--repeat 3] [--warmup 0] [--diagnose] [--diagnose-only] [--require-kkt] [--kkt-tolerance 1e-6] [--kkt-check pdg|full (default: pdg)] [--comparison strict|report|off]");
        return 0;
      }
      if (arg == "--diagnose") { diagnose = true; continue; }
      if (arg == "--diagnose-only") { diagnose = diagnose_only = true; continue; }
      if (arg == "--require-kkt") { diagnose = require_kkt = true; continue; }
      if (i + 1 == argc) throw std::runtime_error("Missing option value");
      const std::string value = argv[++i];
      if (arg == "--snapshot") path = value;
      else if (arg == "--reference") reference_path = value;
      else if (arg == "--repeat") repeat = std::stoi(value);
      else if (arg == "--warmup") warmup = std::stoi(value);
      else if (arg == "--comparison") comparison = value;
      else if (arg == "--kkt-tolerance") kkt_tolerance = std::stod(value);
      else if (arg == "--kkt-check") {
        if (value == "pdg") kkt_check = diagnostics::KktCheck::PrimalDualGap;
        else if (value == "full") kkt_check = diagnostics::KktCheck::Full;
        else throw std::runtime_error("Invalid KKT check; expected pdg or full");
      }
      else throw std::runtime_error("Unknown option: " + arg);
    }
    if (path.empty() || repeat < 1 || warmup < 0) throw std::runtime_error("Invalid snapshot path or repeat count");
    if (!std::isfinite(kkt_tolerance) || kkt_tolerance <= 0 ||
        (comparison != "strict" && comparison != "report" && comparison != "off"))
      throw std::runtime_error("Invalid KKT tolerance or comparison mode");
    if (reference_path.empty()) reference_path = path + ".reference";
    auto start = Clock::now();
    auto snapshot = diagnostics::load_snapshot(path);
    std::size_t structural_bytes=0,structural_count=0;
    auto count_stages=[&](const presolve::PresolveRecordGpu& record) {
      for(const auto& stage:record.structural_dual_recoveries) {
        ++structural_count; structural_bytes+=stage->device_bytes();
      }
    };
    count_stages(snapshot->pipeline.presolve.record);
    if(snapshot->pipeline.startup)count_stages(snapshot->pipeline.startup->summary.record);
    std::printf("STRUCTURAL_STORAGE {\"checkpoints\":%zu,\"device_bytes\":%zu}\n",structural_count,structural_bytes);

    std::printf("SNAPSHOT_LOAD {\"seconds\":%.9f}\n", elapsed(start));
    diagnostics::print_snapshot_statistics(*snapshot);
    bool final_kkt_pass = false;
    if (diagnose) {
      struct Context {
        double tolerance; diagnostics::KktCheck check;
        bool final_pass=false, final_available=false; std::string first_failure;
      } context{kkt_tolerance,kkt_check};
      diagnostics::postsolve_observer_context = &context;
      diagnostics::postsolve_observer = [](const char* label, const presolve::LPInfoGpu* model,
          const double* x, const double* y, const double* z, void* opaque) {
        auto& c = *static_cast<Context*>(opaque);
        std::string stage = label;
        const auto metrics = diagnostics::check_kkt(model,x,y,z);
        diagnostics::print_kkt(stage.c_str(),metrics,c.tolerance,c.check);
        if (metrics.available && !metrics.passed(c.tolerance,c.check) && c.first_failure.empty()) c.first_failure=stage;
        if (stage == "final") { c.final_pass=metrics.passed(c.tolerance,c.check); c.final_available=metrics.available; }
      };
      start = Clock::now();
      try {
        auto result = folding::postsolve_and_unfold_to_host(snapshot->x,snapshot->y,snapshot->z,
            snapshot->pipeline,&snapshot->original,snapshot->primal_tolerance);
      } catch (...) {
        diagnostics::postsolve_observer = nullptr; diagnostics::postsolve_observer_context = nullptr;
        throw;
      }
      diagnostics::postsolve_observer = nullptr; diagnostics::postsolve_observer_context = nullptr;
      final_kkt_pass=context.final_pass;
      std::printf("KKT_SUMMARY {\"seconds\":%.9f,\"format_version\":%u,\"legacy_incomplete_records\":%s,\"criterion\":\"%s\",\"first_failing_boundary\":\"%s\",\"final_available\":%s,\"final_status\":\"%s\",\"final_pass\":%s}\n",
          elapsed(start),snapshot->format_version,snapshot->format_version<2?"true":"null",
          diagnostics::check_name(kkt_check),context.first_failure.c_str(),context.final_available?"true":"false",
          context.final_available?(final_kkt_pass?"pass":"fail"):"unavailable",
          context.final_available?(final_kkt_pass?"true":"false"):"null");
      if (diagnose_only) return require_kkt && !final_kkt_pass ? 4 : 0;
    }
    start = Clock::now();
    const auto reference = comparison == "off" ? folding::UnfoldedSolutionHost{} : diagnostics::load_reference(reference_path);
    std::printf("REFERENCE_LOAD {\"seconds\":%.9f}\n", elapsed(start));
    row_bytes = std::size_t(snapshot->pipeline.original_rows) * sizeof(double);
    col_bytes = std::size_t(snapshot->pipeline.original_cols) * sizeof(double);
    if (comparison != "off" && (reference.x.size() * sizeof(double) != col_bytes || reference.z.size() != reference.x.size() ||
        reference.y.size() * sizeof(double) != row_bytes)) throw std::runtime_error("Invalid reference dimensions");
    std::vector<double> times, without_copy;
    bool all_exact = true;
    for (int run = -warmup; run < repeat; ++run) {
      copy_seconds = 0; copy_calls = 0; copy_bytes = 0;
      if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("CUDA synchronization failed");
      nvtxRangePushA("postsolve_replay");
      start = Clock::now(); measuring = true;
      auto result = folding::postsolve_and_unfold_to_host(snapshot->x, snapshot->y, snapshot->z,
          snapshot->pipeline, &snapshot->original, snapshot->primal_tolerance);
      measuring = false;
      const double seconds = elapsed(start);
      nvtxRangePop();
      const bool copy_valid = copy_calls == (col_bytes ? 2 : 0) + (row_bytes ? 1 : 0) &&
                              copy_bytes == 2 * col_bytes + row_bytes;
      start = Clock::now();
      const auto x = comparison == "off" ? Comparison{} : compare(result.x, reference.x);
      const auto y = comparison == "off" ? Comparison{} : compare(result.y, reference.y);
      const auto z = comparison == "off" ? Comparison{} : compare(result.z, reference.z);
      const bool exact = x.different_bits + y.different_bits + z.different_bits == 0;
      all_exact = all_exact && exact;
      std::printf("REPLAY {\"iteration\":%d,\"warmup\":%s,\"seconds\":%.9f,\"final_d2h_seconds\":%.9f,"
                  "\"excluding_final_d2h_seconds\":%.9f,\"final_d2h_calls\":%d,\"final_d2h_bytes\":%zu,"
                  "\"copy_accounting_valid\":%s,\"comparison_seconds\":%.9f,\"bitwise_equal\":%s,",
                  run, run < 0 ? "true" : "false", seconds, copy_seconds, seconds - copy_seconds,
                  copy_calls, copy_bytes, copy_valid ? "true" : "false", elapsed(start), comparison == "off" ? "null" : (exact ? "true" : "false"));
      std::printf("\"comparison_enabled\":%s,",comparison=="off"?"false":"true");
      if (comparison == "off") std::printf("\"x\":null,\"y\":null,\"z\":null");
      else { print_comparison("x", x); std::printf(","); print_comparison("y", y); std::printf(","); print_comparison("z", z); }
      std::puts("}"); std::fflush(stdout);
      if (!copy_valid) throw std::runtime_error("Final solution copy accounting failed");
      if (run >= 0) { times.push_back(seconds); without_copy.push_back(seconds - copy_seconds); }
    }
    std::sort(times.begin(), times.end()); std::sort(without_copy.begin(), without_copy.end());
    const auto median = [](const std::vector<double>& v) { return (v[(v.size()-1)/2] + v[v.size()/2]) / 2; };
    std::printf("REPLAY_SUMMARY {\"repeats\":%d,\"median_seconds\":%.9f,\"min_seconds\":%.9f,"
                "\"median_excluding_final_d2h_seconds\":%.9f,\"all_bitwise_equal\":%s}\n",
                repeat, median(times), times.front(), median(without_copy), comparison == "off" ? "null" : (all_exact ? "true" : "false"));
    if (require_kkt && !final_kkt_pass) return 4;
    return comparison == "strict" && !all_exact ? 3 : 0;
  } catch (const std::exception& error) {
    measuring = false; std::fprintf(stderr, "Postsolve replay failed: %s\n", error.what()); return 1;
  }
}
