#!/bin/bash
# Phase A: single-GPU GRPO smoke + full run on L40S 40GB.
# Uses the icrl_math venv created by setup_venv.sh.
#
# Usage:
#   bash run_phase_a.sh smoke     # 2-step OOM probe (~2-3 min)
#   bash run_phase_a.sh full      # 300 steps
#   GPU=2 bash run_phase_a.sh full  # use GPU 2 instead

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/home/jklee/ondevice/.venv-icrl-math-v2}"
GPU="${GPU:-0}"
MODE="${1:-smoke}"

if [ ! -x "$VENV/bin/python" ]; then
    echo "[error] venv not found at $VENV"
    echo "        run setup_venv.sh first"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
# Unsloth turns expandable_segments off automatically when STANDBY=1; we leave
# both fully off — STANDBY=1 with vLLM v1 graph capture hits a CUDA allocator
# internal assertion on L40S, and expandable_segments interacts badly with
# vLLM cudagraph in the same way.
export UNSLOTH_VLLM_STANDBY="${UNSLOTH_VLLM_STANDBY:-0}"
# vLLM 0.9+ v1 engine defaults to FlashAttention-3's AoT schedule for full
# cudagraph. Our box has no FA3 (FA2 is also broken, falls back to xformers),
# so v1 + Unsloth's full_cuda_graph=True raises:
#   "AoT scheduling is required for full cuda graph."
# Forcing the v0 engine restores the classic cudagraph path that xformers
# supports.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
# Use a dedicated HF cache to avoid permission clashes with root-owned
# ~/.cache/huggingface/hub/.locks (left over from prior Docker containers).
export HF_HOME="${HF_HOME:-$HERE/.hf-cache}"
mkdir -p "$HF_HOME"

cd "$HERE"

echo "[run_phase_a] mode=$MODE  GPU=$GPU  venv=$VENV"
nvidia-smi -i "$GPU" --query-gpu=name,memory.free,memory.used --format=csv,noheader

case "$MODE" in
  smoke)
    "$VENV/bin/python" scripts/train_grpo_unsloth_l40s.py \
        --smoke \
        --model "unsloth/Qwen2.5-3B-Instruct-bnb-4bit" \
        --output-dir "$HERE/checkpoints/grpo_l40s_smoke" \
        --num-generations 2 \
        --max-completion-length 256 \
        --max-seq-length 640 \
        --gpu-mem-util 0.45
    ;;
  full)
    "$VENV/bin/python" scripts/train_grpo_unsloth_l40s.py \
        --model "unsloth/Qwen2.5-3B-Instruct-bnb-4bit" \
        --output-dir "$HERE/checkpoints/grpo_l40s_full" \
        --num-generations 2 \
        --max-completion-length 512 \
        --max-seq-length 1024 \
        --max-steps 300 \
        --save-steps 100 \
        --gpu-mem-util 0.55
    ;;
  full-bigger)
    # only after `smoke` and `full` succeed and GPU has headroom
    "$VENV/bin/python" scripts/train_grpo_unsloth_l40s.py \
        --model "unsloth/Qwen2.5-3B-Instruct-bnb-4bit" \
        --output-dir "$HERE/checkpoints/grpo_l40s_n4" \
        --num-generations 4 \
        --max-completion-length 512 \
        --max-seq-length 1024 \
        --max-steps 500 \
        --gpu-mem-util 0.55 \
        --lora-targets all
    ;;
  *)
    echo "[run_phase_a] unknown mode: $MODE  (use smoke | full | full-bigger)"
    exit 2
    ;;
esac
