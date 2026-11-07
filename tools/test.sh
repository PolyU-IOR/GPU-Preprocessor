#!/usr/bin/env bash
set -euo pipefail
[[ $# -le 2 ]] || { echo "Usage: bash tools/test.sh [PHYSICAL_GPU_INDEX] [BUILD_DIR]" >&2; exit 2; }
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_dir/tools/gpu_idle_guard.sh"
select_idle_gpu "${1:-0}"
build_dir="${2:-build}"
[[ "$build_dir" = /* ]] || build_dir="$project_dir/$build_dir"
exec "${CTEST_BIN:-ctest}" --test-dir "$build_dir" --output-on-failure --no-tests=error
