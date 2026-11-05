// Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <cstring>

namespace gpu_preprocessor::lp::runtime {

// An unset option uses its code default. Explicit 0/1 values remain available
// for diagnostics; normal runs do not need an environment preset.
inline bool env_enabled(const char* name, bool default_value = false) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return default_value;
  return value[0] != '0' && std::strcmp(value, "false") != 0 &&
      std::strcmp(value, "FALSE") != 0 && std::strcmp(value, "off") != 0 &&
      std::strcmp(value, "OFF") != 0;
}

}  // namespace gpu_preprocessor::lp::runtime
