#!/usr/bin/env bash
# Plays one checkpoint against Stockfish UCI_Elo levels with WORKERS parallel play.py processes,
# then merges the games with elo_report.py.
# Each worker plays GAMES/WORKERS games per level (colours alternate inside each worker; keep it even).
#
# Usage: run_elo_matches.sh <checkpoint> <games per level> <workers> <device cpu|cuda> <elo>...
#   e.g. run_elo_matches.sh runs/chess-50m-12h/ckpt_h12.pt 100 8 cuda 1320 1500 1700
# Output: results/elo/<run>-<ckpt>/worker-N.jsonl, worker-N.log, summary.json
set -eu
cd "$(dirname "$0")"
CKPT=${1:?checkpoint}
GAMES=${2:?games per level}
WORKERS=${3:?workers}
DEVICE=${4:?device}
shift 4
[ "$#" -ge 1 ] || { echo "give at least one Elo level"; exit 2; }
[ -e "$CKPT" ] || { echo "not found: $CKPT"; exit 2; }
PER_WORKER=$(( GAMES / WORKERS ))
if [ $(( PER_WORKER * WORKERS )) -ne "$GAMES" ] || [ $(( PER_WORKER % 2 )) -ne 0 ]; then
    echo "games ($GAMES) must split into an even number per worker ($WORKERS workers)"; exit 2
fi

PY=${PY:-python}
OUT="results/elo/$(basename "$(dirname "$CKPT")")-$(basename "$CKPT" .pt)"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1
[ "$DEVICE" = "cpu" ] && export CUDA_VISIBLE_DEVICES=

pids=()
for i in $(seq 0 $(( WORKERS - 1 ))); do
    $PY play.py "$CKPT" --elo "$@" --games "$PER_WORKER" --device "$DEVICE" --seed $(( 1337 + i )) \
        --out "$OUT/worker-$i.jsonl" > "$OUT/worker-$i.log" 2>&1 &
    pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
[ "$failed" -eq 0 ] || { echo "a worker failed, see $OUT/worker-*.log"; exit 1; }
$PY elo_report.py "$OUT"/worker-*.jsonl --out "$OUT/summary.json"
