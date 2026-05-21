#!/bin/bash
# Phase C with a different random seed — re-run the same 3-stage curriculum
# (5-shot stage 1 / 2-shot stage 2 / 0-shot stage 3) with a fresh seed for
# variance estimate.
#
# Usage:
#   GPU=3 SEED=1234 bash run_phase_c_seed.sh

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/home/jklee/ondevice/.venv-icrl-math-v2}"
GPU="${GPU:-0}"
SEED="${SEED:-1234}"

export CUDA_VISIBLE_DEVICES="$GPU"
export UNSLOTH_VLLM_STANDBY=0
export VLLM_USE_V1=0
export HF_HOME="$HERE/.hf-cache"
mkdir -p "$HF_HOME"
cd "$HERE"

ROOT="$HERE/checkpoints/grpo_l40s_seed_${SEED}"
PHASE_B_DIR="$ROOT/phase_b"
STAGE2_DIR="$ROOT/stage2"
STAGE3_DIR="$ROOT/stage3"

COMMON=(
    --model "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    --lora-rank 16 --lora-targets all
    --num-generations 2 --per-device-batch 1 --grad-accum 4
    --max-seq-length 2048 --max-prompt-length 1024 --max-completion-length 512
    --gpu-mem-util 0.55
    --correctness-weight 5.0 --int-weight 0.3 --soft-format-weight 0.3 --strict-format-weight 0.3
    --save-steps 100
    --seed "$SEED"
)

# 1. Phase B (5-shot, 500 steps, fresh)
if [ ! -d "$PHASE_B_DIR/lora_final" ]; then
    echo "[seed=$SEED] Phase B (5-shot, 500 steps)"
    "$VENV/bin/python" scripts/train_grpo_unsloth_l40s.py \
        --output-dir "$PHASE_B_DIR" \
        --demos-file "$HERE/example/math_demos_simple.txt" \
        --max-steps 500 "${COMMON[@]}"
fi

# 2. Stage 2 (2-shot, 200 steps, resume from Phase B')
if [ ! -d "$STAGE2_DIR/lora_final" ]; then
    echo "[seed=$SEED] Stage 2 (2-shot, 200 steps)"
    "$VENV/bin/python" scripts/train_grpo_unsloth_l40s.py \
        --output-dir "$STAGE2_DIR" \
        --demos-file "$HERE/example/math_demos_2shot.txt" \
        --resume-from-lora "$PHASE_B_DIR/lora_final" \
        --max-steps 200 "${COMMON[@]}"
fi

# 3. Stage 3 (0-shot, 300 steps, resume from Stage 2')
if [ ! -d "$STAGE3_DIR/lora_final" ]; then
    echo "[seed=$SEED] Stage 3 (0-shot, 300 steps)"
    "$VENV/bin/python" scripts/train_grpo_unsloth_l40s.py \
        --output-dir "$STAGE3_DIR" \
        --resume-from-lora "$STAGE2_DIR/lora_final" \
        --max-steps 300 "${COMMON[@]}"
fi

echo "[seed=$SEED] done. final LoRA: $STAGE3_DIR/lora_final"
