#!/bin/bash
# Full autopilot chain (~6-7h):
#   1. wait Stage 2 -> 2. Stage 3 -> 3. GSM8K 7-cell -> 4. MATH500 7-cell
#   5. Phase D (2-shot no-curriculum, 500 step, 90min)
#   6. Phase D eval (GSM8K + MATH500)
#   7. Phase E (5-shot, 1500 step, 3h, depth check)
#   8. Phase E eval
#   9. Aggregate results -> ANALYSIS.md
#
# Halts loudly on any step error. Never auto-commits.
#
# Usage: GPU=2 nohup bash autopilot_full.sh > /tmp/autopilot_full.log 2>&1 &

set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/home/jklee/ondevice/.venv-icrl-math-v2}"
GPU="${GPU:-0}"
PY="$VENV/bin/python"
LOG_DIR="${LOG_DIR:-/tmp/icrl_math_autopilot}"
mkdir -p "$LOG_DIR"

# LoRA paths
PHASE_B_LORA="$HERE/checkpoints/grpo_l40s_phase_b/lora_final"
STAGE2_DIR="$HERE/checkpoints/grpo_l40s_phase_c_stage2"
STAGE2_LORA="$STAGE2_DIR/lora_final/adapter_model.safetensors"
STAGE3_DIR="$HERE/checkpoints/grpo_l40s_phase_c_stage3"
STAGE3_LORA_DIR="$STAGE3_DIR/lora_final"
STAGE3_LORA="$STAGE3_LORA_DIR/adapter_model.safetensors"
PHASE_D_DIR="$HERE/checkpoints/grpo_l40s_phase_d"
PHASE_D_LORA="$PHASE_D_DIR/lora_final"
PHASE_E_DIR="$HERE/checkpoints/grpo_l40s_phase_e"
PHASE_E_LORA="$PHASE_E_DIR/lora_final"

ts() { date "+%F %T"; }
log() { echo "[$(ts)] [autopilot_full] $*"; }

export CUDA_VISIBLE_DEVICES="$GPU"
export UNSLOTH_VLLM_STANDBY=0
export VLLM_USE_V1=0
export HF_HOME="$HERE/.hf-cache"

# ----- Step 1: wait Stage 2 -----
log "Step 1/9: wait Stage 2 LoRA at $STAGE2_LORA"
WAITED=0
while [ ! -f "$STAGE2_LORA" ]; do
    sleep 30; WAITED=$((WAITED+30))
    [ $((WAITED % 600)) -eq 0 ] && log "  ... waited ${WAITED}s"
done
log "Stage 2 detected after ${WAITED}s"
sleep 10

# ----- Step 2: Stage 3 -----
log "Step 2/9: Stage 3 (0-shot, 300 step, ~45min)"
if [ -f "$STAGE3_LORA" ]; then
    log "  Stage 3 already done, skipping"
else
    GPU="$GPU" bash "$HERE/run_phase_c.sh" stage3 > "$LOG_DIR/stage3.log" 2>&1
fi
log "Stage 3 done"

# ----- Step 3: GSM8K 7-cell (incl Phase C) -----
log "Step 3/9: GSM8K 7-cell eval"
"$PY" "$HERE/scripts/eval_zero_shot_transfer.py" --n 100 \
    --lora-phase-c "$STAGE3_LORA_DIR" \
    > "$LOG_DIR/eval_gsm8k_abc.log" 2>&1
log "GSM8K (abc) done -> $HERE/eval_results/cross/summary.json"

# ----- Step 4: MATH500 7-cell -----
log "Step 4/9: MATH500 7-cell eval"
"$PY" "$HERE/scripts/eval_math500.py" --n 100 \
    --lora-phase-c "$STAGE3_LORA_DIR" \
    > "$LOG_DIR/eval_math500_abc.log" 2>&1
log "MATH500 (abc) done"

# ----- Step 5: Phase D (2-shot no curriculum) -----
log "Step 5/9: Phase D training (2-shot only, fresh, 500 step, ~90 min)"
if [ -d "$PHASE_D_LORA" ]; then
    log "  Phase D already done, skipping"
else
    GPU="$GPU" bash "$HERE/run_phase_d.sh" > "$LOG_DIR/phase_d.log" 2>&1
fi
log "Phase D done -> $PHASE_D_LORA"

# ----- Step 6: Phase D eval (add D into the grid) -----
log "Step 6/9: Phase A/B/C/D eval on GSM8K + MATH500"
"$PY" "$HERE/scripts/eval_zero_shot_transfer.py" --n 100 \
    --lora-phase-c "$STAGE3_LORA_DIR" \
    --lora-phase-d "$PHASE_D_LORA" \
    --out-dir "$HERE/eval_results/cross_abcd" \
    > "$LOG_DIR/eval_gsm8k_abcd.log" 2>&1
"$PY" "$HERE/scripts/eval_math500.py" --n 100 \
    --lora-phase-c "$STAGE3_LORA_DIR" \
    --lora-phase-d "$PHASE_D_LORA" \
    --out-dir "$HERE/eval_results/math500_abcd" \
    > "$LOG_DIR/eval_math500_abcd.log" 2>&1
log "Phase D eval done"

# ----- Step 7: Phase E (5-shot longer training) -----
log "Step 7/9: Phase E training (5-shot, 1500 step, ~3h)"
if [ -d "$PHASE_E_LORA" ]; then
    log "  Phase E already done, skipping"
else
    GPU="$GPU" bash "$HERE/run_phase_e.sh" > "$LOG_DIR/phase_e.log" 2>&1
fi
log "Phase E done -> $PHASE_E_LORA"

# ----- Step 8: Phase E eval -----
log "Step 8/9: full grid eval (A/B/C/D/E)"
"$PY" "$HERE/scripts/eval_zero_shot_transfer.py" --n 100 \
    --lora-phase-c "$STAGE3_LORA_DIR" \
    --lora-phase-d "$PHASE_D_LORA" \
    --lora-phase-e "$PHASE_E_LORA" \
    --out-dir "$HERE/eval_results/cross_full" \
    > "$LOG_DIR/eval_gsm8k_full.log" 2>&1
"$PY" "$HERE/scripts/eval_math500.py" --n 100 \
    --lora-phase-c "$STAGE3_LORA_DIR" \
    --lora-phase-d "$PHASE_D_LORA" \
    --lora-phase-e "$PHASE_E_LORA" \
    --out-dir "$HERE/eval_results/math500_full" \
    > "$LOG_DIR/eval_math500_full.log" 2>&1
log "Full grid eval done"

# ----- Step 9: aggregate -----
log "Step 9/9: aggregating results into ANALYSIS.md"
"$PY" "$HERE/scripts/aggregate_results.py" > "$LOG_DIR/aggregate.log" 2>&1 || true
log "ALL DONE."

echo ""
echo "==================== FINAL SUMMARY ===================="
echo
echo "[GSM8K full grid]"
"$PY" -c "
import json, glob, os
for p in sorted(glob.glob('$HERE/eval_results/*/summary.json')):
    name = os.path.basename(os.path.dirname(p))
    d = json.load(open(p))
    print(f'\n--- {name} ---')
    for k, v in d['results'].items():
        print(f'  {k:30s}  {v[\"em\"]:.3f}')
"
echo ""
echo "Logs: $LOG_DIR/"
echo "Eval results: $HERE/eval_results/"
echo "ANALYSIS.md: $HERE/ANALYSIS.md (if aggregator ran)"
