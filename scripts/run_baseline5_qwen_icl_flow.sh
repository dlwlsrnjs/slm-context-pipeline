#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_FILE="$ROOT_DIR/model_configs/models_baseline_5.json"
DATASETS=(BoolQ PIQA Hellaswag WinoGrande e2e_nlg viggo)

EMAIL="${EMAIL:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-64}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-32}"
LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/workspace/local_datasets}"
ICL_GENERATOR_MODEL="${ICL_GENERATOR_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ICL_SHOTS="${ICL_SHOTS:-4}"
ICL_CANDIDATES="${ICL_CANDIDATES:-4}"
ICL_MAX_NEW_TOKENS="${ICL_MAX_NEW_TOKENS:-220}"
SKIP_BASELINE="${SKIP_BASELINE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

if [[ -z "$EMAIL" ]]; then
  echo "Usage: EMAIL=you@example.com $0"
  exit 1
fi

RUN_TAG="baseline5_vs_qwenicl_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/experiment_runs"
BASE_RESULTS_DIR="$ROOT_DIR/experiment_results/${RUN_TAG}_baseline"
ICL_RESULTS_DIR="$ROOT_DIR/experiment_results/${RUN_TAG}_qwen_icl"
mkdir -p "$LOG_DIR" "$BASE_RESULTS_DIR" "$ICL_RESULTS_DIR"

if [[ "${SLMBENCH_IN_DOCKER:-0}" != "1" ]]; then
  echo "[STEP] Launching Docker container for baseline-vs-ICL flow"
  exec docker compose run --rm \
    -e SLMBENCH_IN_DOCKER=1 \
    -e SLMBENCH_AUTO_AGGREGATE=0 \
    -e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    -e NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" \
    -e NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
    -e EMAIL="$EMAIL" \
    -e MAX_TRAIN_SAMPLES="$MAX_TRAIN_SAMPLES" \
    -e MAX_EVAL_SAMPLES="$MAX_EVAL_SAMPLES" \
    -e LOCAL_DATASETS_DIR="$LOCAL_DATASETS_DIR" \
    -e ICL_GENERATOR_MODEL="$ICL_GENERATOR_MODEL" \
    -e ICL_SHOTS="$ICL_SHOTS" \
    -e ICL_CANDIDATES="$ICL_CANDIDATES" \
    -e ICL_MAX_NEW_TOKENS="$ICL_MAX_NEW_TOKENS" \
    -e SKIP_BASELINE="$SKIP_BASELINE" \
    slmbench bash /workspace/scripts/run_baseline5_qwen_icl_flow.sh
fi

PYTHON_BIN="/opt/venv/bin/python"
AGG_BIN="$ROOT_DIR/scripts/aggregate_results.py"

export CUDA_VISIBLE_DEVICES
export NCCL_P2P_DISABLE
export NCCL_IB_DISABLE

run_phase() {
  local phase_name="$1"
  local results_dir="$2"
  shift 2
  local log_file="$LOG_DIR/${RUN_TAG}_${phase_name}.log"

  echo "=================================================="
  echo "[PHASE] $phase_name"
  echo "Models: baseline 5 (GPT-Neo, Phi, Qwen, SmolLM2, Llama-3.2-1B)"
  echo "Datasets: ${DATASETS[*]}"
  echo "Results: $results_dir"
  echo "Log: $log_file"
  echo "=================================================="

  PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" "$ROOT_DIR/train.py" \
    --email "$EMAIL" \
    --models-file "$MODEL_FILE" \
    --datasets "${DATASETS[@]}" \
    --local-datasets-dir "$LOCAL_DATASETS_DIR" \
    --max-train-samples "$MAX_TRAIN_SAMPLES" \
    --max-eval-samples "$MAX_EVAL_SAMPLES" \
    --results-dir "$results_dir" \
    "$@" \
    | tee "$log_file"

  "$PYTHON_BIN" "$AGG_BIN" --results-dir "$results_dir"
}

if [[ "$SKIP_BASELINE" != "1" ]]; then
  run_phase "baseline_no_icl" "$BASE_RESULTS_DIR"
else
  echo "[SKIP] baseline_no_icl skipped (using existing baseline results)"
fi
run_phase "qwen_icl" "$ICL_RESULTS_DIR" \
  --auto-icl \
  --icl-generator-model "$ICL_GENERATOR_MODEL" \
  --icl-shots "$ICL_SHOTS" \
  --icl-candidates "$ICL_CANDIDATES" \
  --icl-max-new-tokens "$ICL_MAX_NEW_TOKENS"

echo ""
echo "[DONE] Flow completed"
echo "Baseline results: $BASE_RESULTS_DIR"
echo "Qwen-ICL results: $ICL_RESULTS_DIR"
echo "Compare summaries:"
echo "  - $BASE_RESULTS_DIR/summary.csv"
echo "  - $ICL_RESULTS_DIR/summary.csv"
