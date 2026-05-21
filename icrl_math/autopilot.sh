#!/bin/bash
# Autopilot chain:
#   wait Stage 2 -> run Stage 3 -> 7-cell GSM8K eval -> 7-cell MATH500 eval -> summary
#
# Designed to be launched in the background. Each step writes to /tmp logs.
# Halts loudly on error; never auto-commits/pushes (left to user).
#
# Usage:
#   nohup bash autopilot.sh > /tmp/autopilot.log 2>&1 &

set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/home/jklee/ondevice/.venv-icrl-math-v2}"
GPU="${GPU:-0}"
PY="$VENV/bin/python"
LOG_DIR="${LOG_DIR:-/tmp/icrl_math_autopilot}"
mkdir -p "$LOG_DIR"

STAGE2_LORA="$HERE/checkpoints/grpo_l40s_phase_c_stage2/lora_final/adapter_model.safetensors"
STAGE3_DIR="$HERE/checkpoints/grpo_l40s_phase_c_stage3"
STAGE3_LORA_DIR="$STAGE3_DIR/lora_final"
STAGE3_LORA="$STAGE3_LORA_DIR/adapter_model.safetensors"

ts() { date "+%F %T"; }
log() { echo "[$(ts)] [autopilot] $*"; }

# ----- Step 1: wait for Stage 2 to finish -----
log "Step 1/4: wait for Stage 2 LoRA at $STAGE2_LORA"
WAITED=0
while [ ! -f "$STAGE2_LORA" ]; do
    sleep 30
    WAITED=$((WAITED + 30))
    if [ $((WAITED % 600)) -eq 0 ]; then
        log "  ... waited ${WAITED}s for Stage 2"
    fi
done
log "Stage 2 LoRA detected after ${WAITED}s wait"
ls -la "$STAGE2_LORA"

# Small grace period: Stage 2 trainer may still be flushing other files.
sleep 10

# ----- Step 2: Stage 3 (0-shot, 300 steps) -----
log "Step 2/4: launching Stage 3 (0-shot, 300 steps, ~45 min)"
if [ -f "$STAGE3_LORA" ]; then
    log "  Stage 3 LoRA already exists at $STAGE3_LORA — skipping training"
else
    GPU="$GPU" bash "$HERE/run_phase_c.sh" stage3 > "$LOG_DIR/stage3.log" 2>&1
fi
log "Stage 3 done. lora -> $STAGE3_LORA_DIR"

# ----- Step 3: 7-cell GSM8K eval (incl Phase C) -----
log "Step 3/4: GSM8K 7-cell eval (n=100)"
CUDA_VISIBLE_DEVICES="$GPU" UNSLOTH_VLLM_STANDBY=0 VLLM_USE_V1=0 \
HF_HOME="$HERE/.hf-cache" \
"$PY" "$HERE/scripts/eval_zero_shot_transfer.py" \
    --n 100 \
    --lora-phase-c "$STAGE3_LORA_DIR" \
    > "$LOG_DIR/eval_gsm8k.log" 2>&1
log "GSM8K eval done -> $HERE/eval_results/cross/summary.json"

# ----- Step 4: 7-cell MATH500 eval -----
log "Step 4/4: MATH500 7-cell eval (n=100, harder benchmark)"
CUDA_VISIBLE_DEVICES="$GPU" UNSLOTH_VLLM_STANDBY=0 VLLM_USE_V1=0 \
HF_HOME="$HERE/.hf-cache" \
"$PY" "$HERE/scripts/eval_math500.py" \
    --n 100 \
    --lora-phase-c "$STAGE3_LORA_DIR" \
    > "$LOG_DIR/eval_math500.log" 2>&1
log "MATH500 eval done -> $HERE/eval_results/math500/summary.json"

# ----- Summary -----
log "ALL DONE. Summaries:"
echo
echo "=== GSM8K (n=100) ==="
"$PY" -c "
import json
d = json.load(open('$HERE/eval_results/cross/summary.json'))
for k, v in d['results'].items():
    print(f'  {k:30s}  {v[\"em\"]:.3f}')
"
echo
echo "=== MATH500 (n=100) ==="
"$PY" -c "
import json
d = json.load(open('$HERE/eval_results/math500/summary.json'))
for k, v in d['results'].items():
    print(f'  {k:30s}  {v[\"em\"]:.3f}')
"
log "Logs saved under $LOG_DIR/. Final results in $HERE/eval_results/."
