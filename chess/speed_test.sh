#!/usr/bin/env bash
# ~5 min each: 25M (8 layers) and 50M (16 layers), to measure chars/s, memory and epochs per 12 h.
set -eu
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for spec in "25m 8" "50m 16"; do
    set -- $spec
    $PY train_chess.py --name "speed-$1" --n-layer "$2" --n-head 8 --n-embd 512 \
        --hours 0.08 --eval-minutes 1 --checkpoint-minutes 600 > "runs-speed-$1.log" 2>&1 &
    pid=$!
    INTERVAL_S=20 ./temp_guard.sh "$pid" "results/temps-speed-$1.csv" &
    wait "$pid" || echo "speed-$1 exited with $?"
    wait
done
