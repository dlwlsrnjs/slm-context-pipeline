#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY is not set"
  exit 1
fi

DATASETS=(
  "local_datasets/BoolQ"
  "local_datasets/PIQA"
  "local_datasets/WinoGrande"
  "local_datasets/Hellaswag"
  "local_datasets/e2e_nlg"
  "local_datasets/viggo"
)

python3 scripts/build_qwen_icl_distill_dataset.py \
  --dataset-paths "${DATASETS[@]}" \
  --samples-per-dataset 300 \
  --shots 4 \
  --seed-pool-size 12 \
  --teacher-model gpt-4o-mini \
  --max-tokens 320 \
  --temperature 0.4 \
  --retries 3 \
  --output-dir icl_distill_data

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="experiment_results/qwen_icl_sft/distill_sft_${STAMP}"

python3 scripts/finetune_qwen_icl_sft.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-dir icl_distill_data/hf_distilled_icl_dataset \
  --max-length 1024 \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 16 \
  --learning-rate 2e-5 \
  --tune-strategy norms_lm_head \
  --output-dir "$OUT_DIR"

echo "[DONE] distill+sft pipeline finished"
echo "Output dir: $OUT_DIR"
