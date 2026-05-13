#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

create_env () {
  local env_name="$1"
  local env_dir="$ROOT_DIR/.venv_${env_name}"

  if [[ -d "$env_dir" ]]; then
    echo "[SKIP] exists: $env_dir"
    return 0
  fi

  python3 -m venv "$env_dir"
  "$env_dir/bin/pip" install --upgrade pip setuptools wheel
  "$env_dir/bin/pip" install -r "$ROOT_DIR/requirements.txt"

  echo "[OK] created: $env_dir"
}

create_env "llama32_1b"
create_env "gpt_neo_1_3b"
create_env "phi_1_5"

echo ""
echo "Activate examples:"
echo "source $ROOT_DIR/.venv_llama32_1b/bin/activate"
echo "source $ROOT_DIR/.venv_gpt_neo_1_3b/bin/activate"
echo "source $ROOT_DIR/.venv_phi_1_5/bin/activate"
