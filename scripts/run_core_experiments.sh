#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DATASETS=(BoolQ PIQA Hellaswag WinoGrande e2e_nlg viggo)

EMAIL="${EMAIL:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-64}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-32}"
LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/workspace/local_datasets}"

backup_existing_results() {
  local backup_dir="$ROOT_DIR/experiment_runs/backup_before_core_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$backup_dir"

  shopt -s nullglob
  local files=(
    "$ROOT_DIR/experiment_results/GPT-Neo-1.3B"*_results.json
    "$ROOT_DIR/experiment_results/Phi-1.5"*_results.json
    "$ROOT_DIR/experiment_results/summary.json"
    "$ROOT_DIR/experiment_results/summary.csv"
  )

  if (( ${#files[@]} > 0 )); then
    mv "${files[@]}" "$backup_dir"/ 2>/dev/null || true
  fi

  echo "$backup_dir"
}

if [[ -z "$EMAIL" ]]; then
  echo "Usage: EMAIL=you@example.com $0"
  exit 1
fi

if [[ "${SLMBENCH_IN_DOCKER:-0}" == "1" ]]; then
  echo "[STEP] Backing up existing core results"
  backup_existing_results
fi

if [[ "${SLMBENCH_IN_DOCKER:-0}" != "1" ]]; then
  echo "[STEP] Launching Docker container for isolated execution"
  exec docker compose run --rm \
    -e SLMBENCH_IN_DOCKER=1 \
    -e SLMBENCH_AUTO_AGGREGATE="${SLMBENCH_AUTO_AGGREGATE:-1}" \
    -e EMAIL="$EMAIL" \
    -e MAX_TRAIN_SAMPLES="$MAX_TRAIN_SAMPLES" \
    -e MAX_EVAL_SAMPLES="$MAX_EVAL_SAMPLES" \
    -e LOCAL_DATASETS_DIR="$LOCAL_DATASETS_DIR" \
    slmbench bash /workspace/scripts/run_core_experiments.sh
fi

run_model () {
  local model_name="$1"
  local runner_py="$2"

  echo "=================================================="
  echo "Running model: $model_name"
  echo "Datasets: ${DATASETS[*]}"
  echo "=================================================="

  /opt/venv/bin/python "$runner_py" \
    --email "$EMAIL" \
    --local-datasets-dir "$LOCAL_DATASETS_DIR" \
    --datasets "${DATASETS[@]}" \
    --max-train-samples "$MAX_TRAIN_SAMPLES" \
    --max-eval-samples "$MAX_EVAL_SAMPLES"
}

run_model "GPT-Neo-1.3B" "$ROOT_DIR/scripts/run_gpt_neo_1_3b.py"
run_model "Phi-1.5" "$ROOT_DIR/scripts/run_phi_1_5.py"
