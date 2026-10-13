#include "../tools/command_line.hpp"

#include <cassert>
#include <iostream>
#include <sstream>

namespace tools = gpu_presolver::tools;

tools::CommandLine parse(std::initializer_list<const char*> arguments,
                         std::size_t paths = 1, bool checking = false) {
  const std::vector<const char*> argv(arguments);
  return tools::parse_command_line(static_cast<int>(argv.size()), argv.data(), paths, checking);
}

void rejects(std::initializer_list<const char*> arguments,
             std::size_t paths = 1, bool checking = false) {
  bool rejected = false;
  try { (void)parse(arguments, paths, checking); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}

int main() {
  const auto base = parse({"tool", "model.mps"});
  assert(base.params.enable_folding && base.params.enable_presolve);
  assert(base.params.max_iters == 512 && std::isinf(base.params.max_time));

  for (bool checking : {false, true}) {
    const auto defaults = parse({"tool", "model.mps", "output/"}, 2, checking);
    assert(defaults.params.enable_folding && defaults.params.enable_presolve);
    const auto presolve_only = parse({"tool", "model.mps", "output/", "--folding", "false"}, 2, checking);
    assert(!presolve_only.params.enable_folding && presolve_only.params.enable_presolve);
  }
  assert(!parse({"tool", "model.mps", "--folding", "false"}).params.enable_folding);

  // The same explicit settings must reach reporting, export, and recovery.
  for (std::size_t paths : {1U, 2U}) {
    std::vector<const char*> argv = {"tool", "--folding", "true", "model.mps",
      "--presolve", "0", "--structure-specific", "false", "--max-iters", "20",
      "--max-time", "1.25", "--folding-tolerance", "0", "--verbose", "1"};
    if (paths == 2) argv.push_back("results/");
    const auto options = tools::parse_command_line(static_cast<int>(argv.size()), argv.data(), paths);
    assert(options.params.enable_folding && !options.params.enable_presolve);
    assert(!options.params.enable_structure_specific_rules && options.params.verbose);
    assert(options.params.max_iters == 20 && options.params.max_time == 1.25);
    assert(options.params.folding_tolerance == 0 && options.positional.size() == paths);
  }
  assert(parse({"tool", "-"}).positional[0] == "-");
  assert(parse({"tool", "--", "-model.mps"}).positional[0] == "-model.mps");
  assert(parse({"tool", "--help"}, 2).help);
  assert(parse({"tool", "model.mps", "solution/", "--print-top-violations"}, 2, true).print_top_violations);
  assert(parse({"tool", "model.mps", "--max-iters", "0"}).params.max_iters == 0);
  assert(parse({"tool", "model.mps", "--max-iters", "2147483647"}).params.max_iters == 2147483647);

  rejects({"tool"});
  rejects({"tool", "model.mps", "extra"});
  rejects({"tool", "model.mps"}, 2);
  rejects({"tool", "model.mps", "--unknown", "1"});
  rejects({"tool", "model.mps", "--folding"});
  rejects({"tool", "model.mps", "--folding", "maybe"});
  rejects({"tool", "model.mps", "--print-top-violations"});
  for (const char* value : {"", "-1", "2147483648", "1.5", "2junk"})
    rejects({"tool", "model.mps", "--max-iters", value});
  for (const char* flag : {"--max-time", "--folding-tolerance"})
    for (const char* value : {"", "-1", "nan", "inf", "1e999", "1junk"})
      rejects({"tool", "model.mps", flag, value});

  std::ostringstream help;
  tools::print_usage(help, "tool", "<model> <solution>", true);
  assert(help.str().find("--max-iters") != std::string::npos);
  assert(help.str().find("--print-top-violations") != std::string::npos);
  std::cout << "Command-line parsing passed\n";
}
