#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "${SLMBENCH_IN_DOCKER:-0}" != "1" ]]; then
	exec docker compose run --rm \
		-e SLMBENCH_IN_DOCKER=1 \
		slmbench bash /workspace/scripts/run_gpt_neo_1_3b.sh "$@"
fi

/opt/venv/bin/python "$ROOT_DIR/scripts/run_gpt_neo_1_3b.py" "$@"
