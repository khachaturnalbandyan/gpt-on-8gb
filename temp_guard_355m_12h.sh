#!/usr/bin/env bash
# Logs GPU temperature every minute while the night-355m-12h run is alive, to results/temps-355m-12h.csv.
# Safety: if the GPU stays at or above LIMIT_C for STRIKES checks in a row, stop the training
# (last.pt is kept, so the run can continue later with --resume).
set -u
cd "$(dirname "$0")"
LIMIT_C=90
STRIKES=3
INTERVAL_S=60
PATTERN="train.py --name night-355m-12h"
LOG=results/temps-355m-12h.csv

sleep 30  # let train.py start
echo "time,gpu_c,power_w,sm_mhz,throttle" >> "$LOG"
hot=0
while pgrep -f "$PATTERN" > /dev/null; do
    row=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks_throttle_reasons.active \
        --format=csv,noheader,nounits)
    echo "$(date +%F_%H:%M),${row// /}" >> "$LOG"
    temp=${row%%,*}
    if [ "${temp:-0}" -ge "$LIMIT_C" ]; then hot=$((hot + 1)); else hot=0; fi
    if [ "$hot" -ge "$STRIKES" ]; then
        echo "$(date +%F_%H:%M),GUARD: ${temp}C for ${STRIKES} checks, stopping training" >> "$LOG"
        pkill -f "$PATTERN"
        break
    fi
    sleep "$INTERVAL_S"
done
