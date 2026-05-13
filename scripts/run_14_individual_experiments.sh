#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

EMAIL="${EMAIL:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-64}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-32}"
LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/workspace/local_datasets}"

backup_existing_results() {
  local backup_dir="$ROOT_DIR/experiment_runs/backup_before_14_$(date +%Y%m%d_%H%M%S)"
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
  echo "[STEP] Backing up existing GPT/Phi results"
  backup_existing_results
fi

if [[ "${SLMBENCH_IN_DOCKER:-0}" != "1" ]]; then
  echo "[STEP] Launching Docker container for 12 individual runs (CoQA excluded)"
  exec docker compose run --rm \
    -e SLMBENCH_IN_DOCKER=1 \
    -e SLMBENCH_AUTO_AGGREGATE="${SLMBENCH_AUTO_AGGREGATE:-1}" \
    -e EMAIL="$EMAIL" \
    -e MAX_TRAIN_SAMPLES="$MAX_TRAIN_SAMPLES" \
    -e MAX_EVAL_SAMPLES="$MAX_EVAL_SAMPLES" \
    -e LOCAL_DATASETS_DIR="$LOCAL_DATASETS_DIR" \
    slmbench bash /workspace/scripts/run_14_individual_experiments.sh
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

run_one "run_gpt_neo_1_3b_boolq.py"
run_one "run_gpt_neo_1_3b_piqa.py"
run_one "run_gpt_neo_1_3b_hellaswag.py"
run_one "run_gpt_neo_1_3b_winogrande.py"
run_one "run_gpt_neo_1_3b_e2e_nlg.py"
run_one "run_gpt_neo_1_3b_viggo.py"

run_one "run_phi_1_5_boolq.py"
run_one "run_phi_1_5_piqa.py"
run_one "run_phi_1_5_hellaswag.py"
run_one "run_phi_1_5_winogrande.py"
run_one "run_phi_1_5_e2e_nlg.py"
run_one "run_phi_1_5_viggo.py"
