#!/bin/bash
# Phase D: 2-shot demos, NO curriculum, fresh LoRA (no resume).
# Ablation: isolates "curriculum effect" from "small demo count effect".
# If Phase D ~ Phase C, then curriculum was unnecessary (just 2-shot was enough).
# If Phase D << Phase C, the staged phase-out was essential.

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/home/jklee/ondevice/.venv-icrl-math-v2}"
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
export UNSLOTH_VLLM_STANDBY="${UNSLOTH_VLLM_STANDBY:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export HF_HOME="${HF_HOME:-$HERE/.hf-cache}"
mkdir -p "$HF_HOME"
cd "$HERE"

"$VENV/bin/python" scripts/train_grpo_unsloth_l40s.py \
    --output-dir "$HERE/checkpoints/grpo_l40s_phase_d" \
    --model "unsloth/Qwen2.5-3B-Instruct-bnb-4bit" \
    --demos-file "$HERE/example/math_demos_2shot.txt" \
    --lora-rank 16 --lora-targets all \
    --num-generations 2 --per-device-batch 1 --grad-accum 4 \
    --max-seq-length 2048 --max-prompt-length 1024 --max-completion-length 512 \
    --gpu-mem-util 0.55 \
    --correctness-weight 5.0 --int-weight 0.3 --soft-format-weight 0.3 --strict-format-weight 0.3 \
    --max-steps 500 --save-steps 100
