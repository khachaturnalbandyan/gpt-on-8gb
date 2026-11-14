#!/usr/bin/env bash
# Logs GPU temperature every INTERVAL_S while a chess training run is alive.
# Safety: if the GPU stays at or above LIMIT_C for STRIKES checks in a row, stop the training
# (last.pt is kept, so the run can continue with --resume).
# Usage: temp_guard.sh <training pid> <csv log>
set -u
cd "$(dirname "$0")"
PID=${1:?pid}
LOG=${2:?log}
LIMIT_C=90
STRIKES=3
INTERVAL_S=${INTERVAL_S:-60}

echo "time,gpu_c,power_w,sm_mhz,throttle" >> "$LOG"
hot=0
while kill -0 "$PID" 2> /dev/null; do
    row=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks_throttle_reasons.active \
        --format=csv,noheader,nounits)
    echo "$(date +%F_%H:%M:%S),${row// /}" >> "$LOG"
    temp=${row%%,*}
    if [ "${temp:-0}" -ge "$LIMIT_C" ]; then hot=$((hot + 1)); else hot=0; fi
    if [ "$hot" -ge "$STRIKES" ]; then
        echo "$(date +%F_%H:%M:%S),GUARD: ${temp}C for ${STRIKES} checks, stopping training" >> "$LOG"
        kill "$PID"
        break
    fi
    sleep "$INTERVAL_S"
done
