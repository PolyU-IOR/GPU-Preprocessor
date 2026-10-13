#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::tools {

struct CommandLine {
  presolve::PresolveParams params;
  std::vector<std::string> positional;
  bool help = false;
  bool print_top_violations = false;
};

inline bool parse_bool(const std::string& value, const std::string& option) {
  if (value == "true" || value == "1") return true;
  if (value == "false" || value == "0") return false;
  throw std::invalid_argument(option + " expects true/false or 1/0");
}

inline double parse_nonnegative_double(const std::string& value, const std::string& option) {
  char* end = nullptr;
  errno = 0;
  const double result = std::strtod(value.c_str(), &end);
  if (errno || end == value.c_str() || *end || !std::isfinite(result) || result < 0)
    throw std::invalid_argument(option + " expects a finite nonnegative number");
  return result;
}

inline CommandLine parse_command_line(int argc, const char* const* argv,
                                     std::size_t positional_count,
                                     bool allow_top_violations = false) {
  CommandLine result;
  bool positional_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (!positional_only && option == "--help") { result.help = true; return result; }
    if (!positional_only && option == "--") { positional_only = true; continue; }
    if (positional_only || option == "-" || option.empty() || option[0] != '-') {
      result.positional.push_back(option);
      continue;
    }
    if (option == "--print-top-violations" && allow_top_violations) {
      result.print_top_violations = true;
      continue;
    }
    if (option != "--folding" && option != "--presolve" &&
        option != "--structure-specific" && option != "--max-iters" &&
        option != "--max-time" && option != "--folding-tolerance" && option != "--verbose")
      throw std::invalid_argument("unknown option: " + option);
    if (++i == argc) throw std::invalid_argument("missing value for " + option);
    const std::string value = argv[i];
    auto& p = result.params;
    if (option == "--folding") p.enable_folding = parse_bool(value, option);
    else if (option == "--presolve") p.enable_presolve = parse_bool(value, option);
    else if (option == "--structure-specific") p.enable_structure_specific_rules = parse_bool(value, option);
    else if (option == "--verbose") p.verbose = parse_bool(value, option);
    else if (option == "--max-time") p.max_time = parse_nonnegative_double(value, option);
    else if (option == "--folding-tolerance") p.folding_tolerance = parse_nonnegative_double(value, option);
    else {
      int parsed = 0;
      const auto end = value.data() + value.size();
      const auto status = std::from_chars(value.data(), end, parsed);
      if (status.ec != std::errc{} || status.ptr != end || parsed < 0)
        throw std::invalid_argument(option + " expects a nonnegative integer in int32 range");
      p.max_iters = parsed;
    }
  }
  if (result.positional.size() != positional_count)
    throw std::invalid_argument("expected " + std::to_string(positional_count) +
                                " path argument(s); use --help for usage");
  return result;
}

inline void print_usage(std::ostream& out, const char* tool, const char* paths,
                        bool allow_top_violations = false) {
  out << "usage: " << tool << ' ' << paths << " [options]\n"
      << "  --folding true|false          Fold before presolve (default: true)\n"
      << "  --presolve true|false         Enable presolve (default: true)\n"
      << "  --structure-specific true|false  Enable block reductions (default: true)\n"
      << "  --max-iters N                 Presolve iteration budget (default: 512)\n"
      << "  --max-time SECONDS            Soft presolve time limit (default: unlimited)\n"
      << "  --folding-tolerance VALUE     Folding tolerance (default: 1e-8)\n"
      << "  --verbose true|false          Print diagnostics (default: false)\n";
  if (allow_top_violations) out << "  --print-top-violations        Show the largest recovered primal violations\n";
  out << "  --help                       Show this help\n";
}

}  // namespace gpu_presolver::tools
