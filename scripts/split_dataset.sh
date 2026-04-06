#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:?"Usage: $0 <input.txt> [output-dir]"}
OUTDIR=${2:-data}
VAL_LINES=500

mkdir -p "$OUTDIR"

total=$(wc -l < "$INPUT")
train_lines=$((total - VAL_LINES))

head -n "$train_lines" "$INPUT" > "$OUTDIR/train.txt"
tail -n "$VAL_LINES" "$INPUT" > "$OUTDIR/val.txt"

echo "Split $total lines → train: $train_lines, val: $VAL_LINES"
echo "Output: $OUTDIR/train.txt, $OUTDIR/val.txt"
