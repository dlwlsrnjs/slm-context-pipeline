#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="${1:-}"
RESULTS_DIR="${2:-}"
EXPECTED_TOTAL="${EXPECTED_TOTAL:-30}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-10}"

MODELS=(
  "GPT-Neo-1.3B"
  "Phi-1.5"
  "Qwen2.5-1.5B-Instruct"
  "SmolLM2-1.7B-Instruct"
  "Llama-3.2-1B-Instruct"
)

latest_qwen_icl_log() {
  ls -1t "$ROOT_DIR"/experiment_runs/run_qwen_icl_only_*.log 2>/dev/null | head -n 1 || true
}

latest_qwen_icl_results_dir() {
  ls -1dt "$ROOT_DIR"/experiment_results/baseline5_vs_qwenicl_*_qwen_icl 2>/dev/null | head -n 1 || true
}

count_model_results() {
  local model_name="$1"
  find "$RESULTS_DIR" -maxdepth 1 -type f -name "${model_name}_*_results.json" 2>/dev/null | wc -l | tr -d ' '
}

if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE="$(latest_qwen_icl_log)"
fi

if [[ -z "$RESULTS_DIR" ]]; then
  RESULTS_DIR="$(latest_qwen_icl_results_dir)"
fi

if [[ -z "$LOG_FILE" ]]; then
  echo "No qwen_icl log found under experiment_runs/."
  exit 1
fi

if [[ -z "$RESULTS_DIR" ]]; then
  echo "No qwen_icl results directory found under experiment_results/."
  exit 1
fi

if [[ "$LOG_FILE" != /* ]]; then
  LOG_FILE="$ROOT_DIR/$LOG_FILE"
fi

if [[ "$RESULTS_DIR" != /* ]]; then
  RESULTS_DIR="$ROOT_DIR/$RESULTS_DIR"
fi

if [[ ! -f "$LOG_FILE" ]]; then
  echo "Log file not found: $LOG_FILE"
  exit 1
fi

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Results dir not found: $RESULTS_DIR"
  exit 1
fi

echo "Watching log: $LOG_FILE"
echo "Watching results: $RESULTS_DIR"
echo "Target total: $EXPECTED_TOTAL"
echo "Refresh interval: ${INTERVAL_SECONDS}s"
echo

while true; do
  total_count="$(find "$RESULTS_DIR" -maxdepth 1 -type f -name '*_results.json' 2>/dev/null | wc -l | tr -d ' ')"
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

  clear
  echo "[$timestamp] Qwen-ICL progress"
  echo "Completed: $total_count / $EXPECTED_TOTAL"
  echo "Results dir: $RESULTS_DIR"
  echo "Log file:    $LOG_FILE"
  echo

  for model in "${MODELS[@]}"; do
    model_count="$(count_model_results "$model")"
    printf "%-24s %s\n" "$model:" "$model_count"
  done

  echo
  echo "--- recent log ---"
  tail -n 25 "$LOG_FILE" || true

  if [[ "$total_count" -ge "$EXPECTED_TOTAL" ]]; then
    echo
    echo "Target reached."
    break
  fi

  sleep "$INTERVAL_SECONDS"
done
