#!/bin/bash
cd /workspace

# Re-run SmolLM2 RA random eval
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup bash -c "CUDA_VISIBLE_DEVICES=0 python scripts/eval_math_icl_baselines.py --n-per-dataset 2000 --gen-batch 8 --ans-batch 32 --gen-max-tokens 900 --output-dir eval_results/math_icl_baselines_smollm2_ra_random --train-1b HuggingFaceTB/SmolLM2-1.7B-Instruct --rl-1b experiment_results/math_icl_rl/smollm2_base_grpo_prl/final --use-prl-prompt --only-tags rl_1b --only-conditions rl_1b > eval_results/smollm2_ra_random_eval.log 2>&1" &

# Re-run Qwen1B-base-RA + SFT-RA PRL random eval
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup bash -c "CUDA_VISIBLE_DEVICES=2 python scripts/eval_math_icl_baselines.py --n-per-dataset 2000 --gen-batch 8 --ans-batch 32 --gen-max-tokens 900 --output-dir eval_results/math_icl_baselines_prl_random --train-1b experiment_results/math_icl_sft/qwen1b_run2/final --train-7b Qwen/Qwen2.5-1.5B-Instruct --rl-1b experiment_results/math_icl_rl/qwen1b_grpo_prl/final --rl-7b experiment_results/math_icl_rl/qwen1b_base_grpo_prl/final --use-prl-prompt --only-tags rl_1b,rl_7b --only-conditions rl_1b,rl_7b > eval_results/baselines_prl_random.log 2>&1" &
