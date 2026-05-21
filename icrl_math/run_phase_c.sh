#!/bin/bash
# Phase C: curriculum (Phase B = 3-shot stage 1) -> 2-shot stage 2 -> 0-shot stage 3.
#
# Each stage resumes LoRA weights from the previous stage so demo dependency is
# phased out gradually (ICRL Algorithm 1).
#
# Usage:
#   GPU=0 bash run_phase_c.sh stage2   # 2-shot, 200 steps from Phase B
#   GPU=0 bash run_phase_c.sh stage3   # 0-shot, 300 steps from stage 2

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/home/jklee/ondevice/.venv-icrl-math-v2}"
GPU="${GPU:-0}"
STAGE="${1:-stage2}"

if [ ! -x "$VENV/bin/python" ]; then
    echo "[error] venv not found at $VENV"; exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export UNSLOTH_VLLM_STANDBY="${UNSLOTH_VLLM_STANDBY:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export HF_HOME="${HF_HOME:-$HERE/.hf-cache}"
mkdir -p "$HF_HOME"

cd "$HERE"

PHASE_B_LORA="$HERE/checkpoints/grpo_l40s_phase_b/lora_final"
STAGE2_DIR="$HERE/checkpoints/grpo_l40s_phase_c_stage2"
STAGE3_DIR="$HERE/checkpoints/grpo_l40s_phase_c_stage3"

COMMON_ARGS=(
    --model "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
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
    --save-steps 100
)

nvidia-smi -i "$GPU" --query-gpu=name,memory.free,memory.used --format=csv,noheader

case "$STAGE" in
  stage2)
    DEMOS="$HERE/example/math_demos_2shot.txt"
    echo "[stage 2] demos=2 (resume from Phase B = 3-shot stage 1)"
    "$VENV/bin/python" scripts/train_grpo_unsloth_l40s.py \
        --output-dir "$STAGE2_DIR" \
        --demos-file "$DEMOS" \
        --resume-from-lora "$PHASE_B_LORA" \
        --max-steps 200 \
        "${COMMON_ARGS[@]}"
    ;;
  stage3)
    # 0-shot = no demos
    if [ ! -d "$STAGE2_DIR/lora_final" ]; then
        echo "[error] $STAGE2_DIR/lora_final not found; run stage2 first"
        exit 1
    fi
    echo "[stage 3] demos=0 (resume from stage 2 = 2-shot)"
    "$VENV/bin/python" scripts/train_grpo_unsloth_l40s.py \
        --output-dir "$STAGE3_DIR" \
        --resume-from-lora "$STAGE2_DIR/lora_final" \
        --max-steps 300 \
        "${COMMON_ARGS[@]}"
    ;;
  smoke)
    DEMOS="$HERE/example/math_demos_2shot.txt"
    "$VENV/bin/python" scripts/train_grpo_unsloth_l40s.py \
        --smoke \
        --output-dir "$HERE/checkpoints/grpo_l40s_phase_c_smoke" \
        --demos-file "$DEMOS" \
        --resume-from-lora "$PHASE_B_LORA" \
        "${COMMON_ARGS[@]}"
    ;;
  *)
    echo "usage: $0 {smoke|stage2|stage3}"; exit 2 ;;
esac
