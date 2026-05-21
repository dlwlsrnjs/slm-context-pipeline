#!/bin/bash
# Phase C on Qwen2.5-1.5B-Instruct (smaller-scale validation).
# Same curriculum (5-shot / 2-shot / 0-shot) as Phase C on 3B.
# Smaller LoRA rank (r=8) because 1.5B has fewer params.

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/home/jklee/ondevice/.venv-icrl-math-v2}"
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
export UNSLOTH_VLLM_STANDBY=0
export VLLM_USE_V1=0
export HF_HOME="$HERE/.hf-cache"
mkdir -p "$HF_HOME"
cd "$HERE"

ROOT="$HERE/checkpoints/grpo_l40s_15b"
PHASE_B_DIR="$ROOT/phase_b"
STAGE2_DIR="$ROOT/stage2"
STAGE3_DIR="$ROOT/stage3"

MODEL="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"

COMMON=(
    --model "$MODEL"
    --lora-rank 8 --lora-targets all
    --num-generations 2 --per-device-batch 1 --grad-accum 4
    --max-seq-length 2048 --max-prompt-length 1024 --max-completion-length 512
    --gpu-mem-util 0.45
    --correctness-weight 5.0 --int-weight 0.3 --soft-format-weight 0.3 --strict-format-weight 0.3
    --save-steps 100
)

if [ ! -d "$PHASE_B_DIR/lora_final" ]; then
    echo "[1.5B] Phase B (5-shot, 500 steps)"
    "$VENV/bin/python" scripts/train_grpo_unsloth_l40s.py \
        --output-dir "$PHASE_B_DIR" \
        --demos-file "$HERE/example/math_demos_simple.txt" \
        --max-steps 500 "${COMMON[@]}"
fi

if [ ! -d "$STAGE2_DIR/lora_final" ]; then
    echo "[1.5B] Stage 2 (2-shot, 200 steps)"
    "$VENV/bin/python" scripts/train_grpo_unsloth_l40s.py \
        --output-dir "$STAGE2_DIR" \
        --demos-file "$HERE/example/math_demos_2shot.txt" \
        --resume-from-lora "$PHASE_B_DIR/lora_final" \
        --max-steps 200 "${COMMON[@]}"
fi

if [ ! -d "$STAGE3_DIR/lora_final" ]; then
    echo "[1.5B] Stage 3 (0-shot, 300 steps)"
    "$VENV/bin/python" scripts/train_grpo_unsloth_l40s.py \
        --output-dir "$STAGE3_DIR" \
        --resume-from-lora "$STAGE2_DIR/lora_final" \
        --max-steps 300 "${COMMON[@]}"
fi

echo "[1.5B] done. final LoRA: $STAGE3_DIR/lora_final"
