#!/bin/bash
# Watch for 1.5B stage 3 LoRA, then launch autopilot_tier3 (T3+T4+T5 will all
# run; T1a/T1b/T2 skip because LoRAs already exist).
#
# Usage: GPU=3 nohup bash watch_and_finalize.sh > /tmp/watch_finalize.log 2>&1 &

set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-3}"

SIGNAL="$HERE/checkpoints/grpo_l40s_15b/stage3/lora_final/adapter_model.safetensors"

ts() { date "+%F %T"; }
log() { echo "[$(ts)] [watch] $*"; }

log "waiting for 1.5B stage 3 at $SIGNAL"
WAITED=0
while [ ! -f "$SIGNAL" ]; do
    sleep 60; WAITED=$((WAITED+60))
    [ $((WAITED % 600)) -eq 0 ] && log "  ... waited ${WAITED}s"
done
log "1.5B stage 3 detected (after ${WAITED}s). Launching autopilot_tier3."
sleep 10

GPU="$GPU" nohup bash "$HERE/autopilot_tier3.sh" > /tmp/autopilot_tier3_v2.log 2>&1 &
PID=$!
disown 2>/dev/null
log "autopilot_tier3 v2 launched as PID $PID — should skip T1a/T1b/T2 and run T3+T4+T5"
