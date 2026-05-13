#!/bin/bash
# v4 full training chain — runs after smoke test passes
# Uses TRAIN-7B as policy + Qwen2.5-7B-Instruct as judge + 1.5B SLM as evaluator

cd /workspace
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$1 python scripts/rl_train_icl_grpo_ra_v4.py \
    --policy-path experiment_results/math_icl_sft/qwen7b_run2/final \
    --reward-model Qwen/Qwen2.5-1.5B-Instruct \
    --judge-model Qwen/Qwen2.5-7B-Instruct \
    --output-dir experiment_results/math_icl_rl/qwen7b_grpo_ra_v4 \
    --num-generations 4 --per-device-batch 1 --grad-accum 4 \
    --max-completion-tokens 900 --max-prompt-tokens 1024 \
    --max-steps 1500 \
    --learning-rate 8e-6 --beta 0.02 \
    --r-token 0.05 --r-structure 0.1 --r-format 0.5 --r-alignment 3.0 \
    --r-judge 1.0 --r-repetition 0.5 \
    --alignment-n 5 \
    --cluster-dir slm_context_pipeline/data/math_5k_clusters \
    --logging-steps 10
