#!/usr/bin/env bash
# 6-hour runs for 355M and 124M, one after the other (the one-night comparison, part 2).
# Fastest configs that fit, from results/fastest-2.jsonl. Same optimizer (AdamW 8-bit) as the 577M run.
# Launch under systemd-inhibit so closing the lid does not suspend the laptop.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/python train.py --name night-355m --n-layer 24 --n-head 16 --n-embd 1024 \
    --batch-size 4 --accumulation 2 --checkpointing --optimizer adamw8bit --lr 3e-4 --hours 6 \
    > runs-night-355m.log 2>&1

.venv/bin/python train.py --name night-124m --n-layer 12 --n-head 12 --n-embd 768 \
    --batch-size 4 --accumulation 2 --optimizer adamw8bit --lr 6e-4 --hours 6 \
    > runs-night-124m.log 2>&1
