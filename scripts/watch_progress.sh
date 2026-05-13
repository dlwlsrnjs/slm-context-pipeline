#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="${1:-}"
EXPECTED_TOTAL="${2:-46}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-10}"

count_results() {
  local pattern="$1"
  find "$ROOT_DIR/experiment_results" -maxdepth 1 -type f -name "$pattern" 2>/dev/null | wc -l | tr -d ' '
}

latest_log() {
  ls -1t "$ROOT_DIR"/experiment_runs/run_*.log 2>/dev/null | head -n 1 || true
}

if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE="$(latest_log)"
fi

if [[ -z "$LOG_FILE" ]]; then
  echo "No experiment log found in experiment_runs/."
  exit 1
fi

if [[ "$LOG_FILE" != /* ]]; then
  LOG_FILE="$ROOT_DIR/$LOG_FILE"
fi

if [[ ! -f "$LOG_FILE" ]]; then
  echo "Log file not found: $LOG_FILE"
  exit 1
fi

echo "Watching: $LOG_FILE"
echo "Target total: $EXPECTED_TOTAL"
echo "Refresh interval: ${INTERVAL_SECONDS}s"
echo

while true; do
  gpt_count="$(count_results 'GPT-Neo-1.3B*_results.json')"
  phi_count="$(count_results 'Phi-1.5*_results.json')"
  total_count=$((gpt_count + phi_count))
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

  clear
  echo "[$timestamp] SLM-Bench progress"
  echo "Completed: $total_count / $EXPECTED_TOTAL"
  echo "GPT-Neo-1.3B: $gpt_count"
  echo "Phi-1.5:      $phi_count"
  echo "Log: $LOG_FILE"
  echo
  echo "--- recent log ---"
  tail -n 20 "$LOG_FILE" || true

  if [[ "$total_count" -ge "$EXPECTED_TOTAL" ]]; then
    echo
    echo "Target reached."
    break
  fi

  sleep "$INTERVAL_SECONDS"
done
