#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 ]] || exit 2
binary=$1
work=$(mktemp -d "$2/cli-XXXXXX")
mkdir "$work/data"
cp qp/examples/qp_fixed.mps "$work/data/model.mps"
gzip -c qp/examples/qp_fixed.mps > "$work/data/model.mps.gz"
"$binary" --help > "$work/help.txt"
grep -q 'gpu_preprocessor_qp' "$work/help.txt"
"$binary" qp/examples/qp_fixed.mps > "$work/default.log"
grep -q 'backend=CUDA status=OK' "$work/default.log"
"$binary" qp/examples/qp_fixed.mps --gpu 0 --folding false --max-time 1 --max-iters 2 > "$work/options.log"
grep -q 'folding_reductions=0' "$work/options.log"
for value in -1 2147483648 1.5; do
    if "$binary" qp/examples/qp_fixed.mps --gpu "$value"; then exit 1; fi
done
if "$binary" qp/examples/qp_fixed.mps --folding maybe; then exit 1; fi
if "$binary" qp/examples/qp_fixed.mps --max-time NaN; then exit 1; fi
if "$binary" qp/examples/qp_fixed.mps --gpu; then exit 1; fi
"$binary" --gpu 0 qp/examples/qp_fixed.mps "$work/reduced.mps"
"$binary" --gpu 0 "$work/reduced.mps"
cp "$work/reduced.mps" "$work/original-output.mps"
if "$binary" --gpu 0 qp/examples/qp_fixed.mps "$work/reduced.mps"; then
    echo 'Existing MPS output was incorrectly accepted.' >&2;exit 1
fi
cmp "$work/reduced.mps" "$work/original-output.mps"
"$binary" --gpu 0 --dataset "$work/data" "$work/valid.csv"
[[ $(wc -l < "$work/valid.csv") -eq 3 ]]
if "$binary" --gpu 0 --dataset "$work/data" "$work/valid.csv"; then
    echo 'Existing CSV output was incorrectly accepted.' >&2;exit 1
fi
printf 'NAME INVALID\nOBJSENSE\n MAX\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nENDATA\n' > "$work/data/invalid.mps"
if "$binary" --gpu 0 --dataset "$work/data" "$work/mixed.csv"; then
    echo 'Dataset parser error did not produce a failed exit status.' >&2;exit 1
fi
[[ $(wc -l < "$work/mixed.csv") -eq 4 ]]
[[ $(< "$work/mixed.csv") == *INVALID_ARGUMENT* ]]
if "$binary" --gpu 0 --max-iters -1 qp/examples/qp_fixed.mps; then exit 1;fi
if "$binary" --gpu 0 --max-seconds NaN qp/examples/qp_fixed.mps; then exit 1;fi
"$binary" --gpu 0 --schedule tiered --batch-limit 2 qp/examples/qp_fixed.mps > "$work/tiered.log"
if "$binary" --gpu 0 --schedule invalid qp/examples/qp_fixed.mps; then exit 1; fi
if "$binary" --gpu 0 --batch-limit 0 qp/examples/qp_fixed.mps; then exit 1; fi
echo 'CLI: presolve, export/reload, gzip dataset, failure reporting and output protection passed.'
