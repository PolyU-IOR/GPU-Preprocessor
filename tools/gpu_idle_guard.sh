#!/usr/bin/env bash

select_idle_gpu() {
  local index="$1" uuid apps util memory reading
  [[ "$index" =~ ^[0-9]+$ ]] || { echo "Expected a physical GPU index." >&2; return 2; }
  uuid="$(nvidia-smi -i "$index" --query-gpu=uuid --format=csv,noheader)" || return 3
  [[ "$uuid" == GPU-* && "$uuid" != *$'\n'* ]] || { echo "Cannot identify GPU $index" >&2; return 3; }
  apps="$(nvidia-smi -i "$index" --query-compute-apps=pid --format=csv,noheader)" || return 3
  reading="$(nvidia-smi -i "$index" --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits)" || return 3
  IFS=, read -r util memory <<< "$reading"
  util="${util//[[:space:]]/}"
  memory="${memory//[[:space:]]/}"
  # Best-effort idle check, not a scheduler reservation. Fail closed on unknown data.
  if [[ -n "${apps//[[:space:]]/}" || ! "$util" =~ ^[0-9]+$ ||
        ! "$memory" =~ ^[0-9]+$ ]] || (( util != 0 || memory > 256 )); then
    echo "GPU $index is busy or cannot be verified idle; no CUDA program was started." >&2
    echo "PIDs: $apps; utilization: $util%; memory: $memory MiB" >&2
    return 3
  fi
  export CUDA_VISIBLE_DEVICES="$uuid"
}
