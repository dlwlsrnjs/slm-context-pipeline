#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/experiment_results/grpo_icl"
mkdir -p "$OUT_DIR"

DATASETS=(BoolQ PIQA Hellaswag WinoGrande e2e_nlg viggo)
LOG_FILE="$OUT_DIR/run_grpo_6_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to $LOG_FILE"

run_one() {
  local ds="$1"
  echo "========================================" | tee -a "$LOG_FILE"
  echo "[RUN] Dataset: $ds" | tee -a "$LOG_FILE"
  echo "========================================" | tee -a "$LOG_FILE"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python3 "$ROOT_DIR/scripts/train_qwen_icl_grpo.py" \
    --dataset-path "$ROOT_DIR/local_datasets/$ds" \
    --dataset-name "$ds" \
    --steps "${GRPO_STEPS:-6}" \
    --group-size "${GRPO_GROUP_SIZE:-4}" \
    --shots "${GRPO_SHOTS:-4}" \
    --seed-pool-size "${GRPO_SEED_POOL:-12}" \
    --reward-batch-size "${GRPO_REWARD_BATCH:-16}" \
    --eval-episodes "${GRPO_EVAL_EPISODES:-6}" \
    --step-eval-episodes "${GRPO_STEP_EVAL_EPISODES:-2}" \
    --max-new-tokens "${GRPO_MAX_NEW_TOKENS:-96}" \
    --lr "${GRPO_LR:-5e-6}" \
    --temperature "${GRPO_TEMP:-0.7}" \
    --output-dir "$OUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"
}

for ds in "${DATASETS[@]}"; do
  run_one "$ds"
done

python3 "$ROOT_DIR/scripts/summarize_grpo_reports.py" --reports-dir "$OUT_DIR" 2>&1 | tee -a "$LOG_FILE"

echo "[DONE] 6-dataset GRPO run complete" | tee -a "$LOG_FILE"
