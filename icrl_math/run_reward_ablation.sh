#!/bin/bash
# Reward ablation — train two LoRAs with single-component reward to show that
# both correctness AND format contribute (or that one dominates).
#
# Phase G: correctness-only  (5.0 / 0 / 0 / 0)
# Phase H: format-only        (0   / 0 / 1.0 / 1.0)
#
# Both use Phase B's exact training config (5-shot demos, r=16, 500 steps),
# so the only thing changed is the reward signal.
#
# Usage:
#   GPU=3 bash run_reward_ablation.sh g     # correctness-only, 500 steps
#   GPU=3 bash run_reward_ablation.sh h     # format-only, 500 steps

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/home/jklee/ondevice/.venv-icrl-math-v2}"
GPU="${GPU:-0}"
WHICH="${1:-g}"

export CUDA_VISIBLE_DEVICES="$GPU"
export UNSLOTH_VLLM_STANDBY=0
export VLLM_USE_V1=0
export HF_HOME="$HERE/.hf-cache"
mkdir -p "$HF_HOME"
cd "$HERE"

COMMON=(
    --model "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    --demos-file "$HERE/example/math_demos_simple.txt"
    --lora-rank 16 --lora-targets all
    --num-generations 2 --per-device-batch 1 --grad-accum 4
    --max-seq-length 2048 --max-prompt-length 1024 --max-completion-length 512
    --gpu-mem-util 0.55
    --max-steps 500 --save-steps 100
)

case "$WHICH" in
  g)
    OUT="$HERE/checkpoints/grpo_l40s_phase_g_correctness_only"
    echo "[ablation] Phase G: correctness-only (5.0 / 0 / 0 / 0)"
    "$VENV/bin/python" scripts/train_grpo_unsloth_l40s.py \
        --output-dir "$OUT" \
        --correctness-weight 5.0 --int-weight 0.0 --soft-format-weight 0.0 --strict-format-weight 0.0 \
        "${COMMON[@]}"
    ;;
  h)
    OUT="$HERE/checkpoints/grpo_l40s_phase_h_format_only"
    echo "[ablation] Phase H: format-only (0 / 0 / 1.0 / 1.0)"
    "$VENV/bin/python" scripts/train_grpo_unsloth_l40s.py \
        --output-dir "$OUT" \
        --correctness-weight 0.0 --int-weight 0.0 --soft-format-weight 1.0 --strict-format-weight 1.0 \
        "${COMMON[@]}"
    ;;
  *)
    echo "usage: $0 {g|h}"; exit 2 ;;
esac
