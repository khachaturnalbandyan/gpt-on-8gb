#!/usr/bin/env bash
# Measures each new ckpt_h*.pt of a running run with eval_legal.py (CPU), until the run has ended
# and every checkpoint is in results/legal-<run>-p<positions>.csv.
# Usage: watch_legal.sh <run dir> <training pid> [positions, default 1000]
set -u
cd "$(dirname "$0")"
RUN=${1:?run dir}
PID=${2:?training pid}
POSITIONS=${3:-1000}
CSV="results/legal-$(basename "$RUN")-p${POSITIONS}.csv"
SETTLE_S=60   # a checkpoint must be this old, so it is fully written
INTERVAL_S=120
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4

while true; do
    running=0
    kill -0 "$PID" 2> /dev/null && running=1
    for ckpt in "$RUN"/ckpt_h*.pt; do
        [ -e "$ckpt" ] || continue
        grep -q "^$(basename "$ckpt")," "$CSV" 2> /dev/null && continue
        age=$(( $(date +%s) - $(stat -c %Y "$ckpt") ))
        [ "$age" -ge "$SETTLE_S" ] || continue
        echo "$(date +%F_%H:%M) measuring $ckpt"
        "${PY:-python}" eval_legal.py "$ckpt" --positions "$POSITIONS" || echo "eval_legal failed on $ckpt"
    done
    if [ "$running" -eq 0 ]; then
        pending=0
        for ckpt in "$RUN"/ckpt_h*.pt; do
            grep -q "^$(basename "$ckpt")," "$CSV" 2> /dev/null || pending=1
        done
        [ "$pending" -eq 0 ] && break
    fi
    sleep "$INTERVAL_S"
done
echo "$(date +%F_%H:%M) done"
