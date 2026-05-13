#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

EMAIL="${EMAIL:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-64}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-32}"
LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/workspace/local_datasets}"

if [[ -z "$EMAIL" ]]; then
  echo "Usage: EMAIL=you@example.com $0"
  exit 1
fi

if [[ "${SLMBENCH_IN_DOCKER:-0}" != "1" ]]; then
  echo "[STEP] Launching Docker container for 18 individual runs"
  exec docker compose run --rm \
    -e SLMBENCH_IN_DOCKER=1 \
    -e SLMBENCH_AUTO_AGGREGATE="${SLMBENCH_AUTO_AGGREGATE:-1}" \
    -e EMAIL="$EMAIL" \
    -e MAX_TRAIN_SAMPLES="$MAX_TRAIN_SAMPLES" \
    -e MAX_EVAL_SAMPLES="$MAX_EVAL_SAMPLES" \
    -e LOCAL_DATASETS_DIR="$LOCAL_DATASETS_DIR" \
    slmbench bash /workspace/scripts/run_18_new_models.sh
fi

run_one() {
  local py_file="$1"
  echo "=================================================="
  echo "Running: $py_file"
  echo "=================================================="
  PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" /opt/venv/bin/python "$ROOT_DIR/scripts/$py_file" \
    --email "$EMAIL" \
    --local-datasets-dir "$LOCAL_DATASETS_DIR" \
    --max-train-samples "$MAX_TRAIN_SAMPLES" \
    --max-eval-samples "$MAX_EVAL_SAMPLES"
}

FAILED=()
for script in \
  run_qwen2_5_1_5b_instruct_boolq.py \
  run_qwen2_5_1_5b_instruct_piqa.py \
  run_qwen2_5_1_5b_instruct_hellaswag.py \
  run_qwen2_5_1_5b_instruct_winogrande.py \
  run_qwen2_5_1_5b_instruct_e2e_nlg.py \
  run_qwen2_5_1_5b_instruct_viggo.py \
  run_smollm2_1_7b_instruct_boolq.py \
  run_smollm2_1_7b_instruct_piqa.py \
  run_smollm2_1_7b_instruct_hellaswag.py \
  run_smollm2_1_7b_instruct_winogrande.py \
  run_smollm2_1_7b_instruct_e2e_nlg.py \
  run_smollm2_1_7b_instruct_viggo.py \
  run_llama32_1b_instruct_boolq.py \
  run_llama32_1b_instruct_piqa.py \
  run_llama32_1b_instruct_hellaswag.py \
  run_llama32_1b_instruct_winogrande.py \
  run_llama32_1b_instruct_e2e_nlg.py \
  run_llama32_1b_instruct_viggo.py
do
  if ! run_one "$script"; then
    FAILED+=("$script")
  fi
done

if (( ${#FAILED[@]} > 0 )); then
  echo "[WARN] Failed scripts: ${FAILED[*]}"
  exit 1
fi

echo "[OK] All 18 runs completed"
