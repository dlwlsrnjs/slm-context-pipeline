#!/bin/bash
# Phase E: same config as Phase B (5-shot, no curriculum) but 3x longer (1500 steps).
# Tests whether Phase B's marginal GRPO-on-top-of-demos was saturation or just
# undertraining.

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
    --output-dir "$HERE/checkpoints/grpo_l40s_phase_e" \
    --model "unsloth/Qwen2.5-3B-Instruct-bnb-4bit" \
    --demos-file "$HERE/example/math_demos_simple.txt" \
    --lora-rank 16 --lora-targets all \
    --num-generations 2 --per-device-batch 1 --grad-accum 4 \
    --max-seq-length 2048 --max-prompt-length 1024 --max-completion-length 512 \
    --gpu-mem-util 0.55 \
    --correctness-weight 5.0 --int-weight 0.3 --soft-format-weight 0.3 --strict-format-weight 0.3 \
    --max-steps 1500 --save-steps 250
