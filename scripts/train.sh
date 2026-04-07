#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/train.sh                                      # train from scratch
#   ./scripts/train.sh --resume outputs/.../checkpoint-N   # resume training
#   ./scripts/train.sh --resume outputs/.../checkpoint-N --reset-steps  # finetune (load weights, reset steps)

RESUME=""
RESET_STEPS=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME="--resume $2"; shift 2 ;;
        --reset-steps) RESET_STEPS="--reset-steps"; shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

uv run accelerate launch --mixed_precision fp16 src/train.py \
  --train-dataset data/train.txt \
  --eval-dataset data/val.txt \
  --output-dir outputs/bert-char-he \
  --train-batch-size 64 \
  --epochs 3 \
  --encoder-lr 1e-4 \
  --warmup-steps 500 \
  $RESUME \
  $RESET_STEPS \
  "${EXTRA[@]+"${EXTRA[@]}"}"
