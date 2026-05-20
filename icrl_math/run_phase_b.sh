#!/bin/bash
# Phase B: Phase A baseline + ICRL spirit + methodology fixes.
#
# Changes vs Phase A:
#   + 5 math demos prepended to the system prompt (ICRL fewshot-in-rollout)
#   + LoRA r=16 with all linear targets (qkvo + gate/up/down)
#   + reward rebalanced: correctness 5.0 (up from 2.0), format/int 0.3 each (down from 0.5)
#   + 500 steps (up from 300)
#   + max_seq_length 2048 / max_prompt_length 1024 (demos fit)
#
# Usage:
#   GPU=0 bash run_phase_b.sh smoke   # 2-step OOM probe (~3 min)
#   GPU=0 bash run_phase_b.sh full    # 500 steps (~50 min)

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/home/jklee/ondevice/.venv-icrl-math-v2}"
GPU="${GPU:-0}"
MODE="${1:-full}"

if [ ! -x "$VENV/bin/python" ]; then
    echo "[error] venv not found at $VENV"; exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export UNSLOTH_VLLM_STANDBY="${UNSLOTH_VLLM_STANDBY:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export HF_HOME="${HF_HOME:-$HERE/.hf-cache}"
mkdir -p "$HF_HOME"

cd "$HERE"

DEMOS="$HERE/example/math_demos_simple.txt"
echo "[run_phase_b] mode=$MODE GPU=$GPU demos=$DEMOS"
nvidia-smi -i "$GPU" --query-gpu=name,memory.free,memory.used --format=csv,noheader

COMMON_ARGS=(
    --model "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    --demos-file "$DEMOS"
    --lora-rank 16
    --lora-targets all
    --num-generations 2
    --per-device-batch 1
    --grad-accum 4
    --max-seq-length 2048
    --max-prompt-length 1024
    --max-completion-length 512
    --gpu-mem-util 0.55
    --correctness-weight 5.0
    --int-weight 0.3
    --soft-format-weight 0.3
    --strict-format-weight 0.3
)

case "$MODE" in
  smoke)
    "$VENV/bin/python" scripts/train_grpo_unsloth_l40s.py --smoke \
        --output-dir "$HERE/checkpoints/grpo_l40s_phase_b_smoke" \
        "${COMMON_ARGS[@]}"
    ;;
  full)
    "$VENV/bin/python" scripts/train_grpo_unsloth_l40s.py \
        --output-dir "$HERE/checkpoints/grpo_l40s_phase_b" \
        --max-steps 500 --save-steps 100 \
        "${COMMON_ARGS[@]}"
    ;;
  *)
    echo "unknown mode: $MODE (use smoke | full)"; exit 2 ;;
esac
