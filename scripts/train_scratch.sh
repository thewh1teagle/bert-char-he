#!/usr/bin/env bash
set -euo pipefail

uv run accelerate launch src/train.py \
  --train-dataset data/train.txt \
  --eval-dataset data/val.txt \
  --output-dir outputs/neobert-he \
  --train-batch-size 32 \
  --epochs 3 \
  --encoder-lr 1e-4 \
  --warmup-steps 500 \
  "$@"
