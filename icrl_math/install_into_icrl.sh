#!/bin/bash
# Install icrl_math patches into a local ICRL checkout.
#
# What this does (idempotent):
#   1. Copy verl_patches/math_fewshot.py into ICRL/verl/utils/reward_score/
#   2. Patch ICRL/verl/trainer/main_ppo_fewshot.py: add 'math' branch in
#      _select_rm_score_fn so data_source='math' routes to math_fewshot.compute_score_fewshot
#   3. (Optional) print sandbox + curriculum next-step instructions.
#
# Usage:
#   ICRL_DIR=/home/jklee/ondevice/ICRL bash install_into_icrl.sh
#
# Safe to re-run; existing patches are detected and skipped.

set -e

ICRL_DIR="${ICRL_DIR:-/home/jklee/ondevice/ICRL}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$ICRL_DIR" ]; then
    echo "[install] ICRL_DIR not found: $ICRL_DIR"
    echo "          Clone first:  git clone https://github.com/applese233/ICRL.git $ICRL_DIR"
    exit 1
fi

REWARD_DIR="$ICRL_DIR/verl/utils/reward_score"
TRAINER_FILE="$ICRL_DIR/verl/trainer/main_ppo_fewshot.py"

# --- 1. copy reward function ---
echo "[install] copying math_fewshot.py -> $REWARD_DIR/"
cp "$HERE/verl_patches/math_fewshot.py" "$REWARD_DIR/math_fewshot.py"

# --- 2. patch main_ppo_fewshot.py to route data_source='math' ---
if grep -q "math_fewshot.compute_score_fewshot" "$TRAINER_FILE"; then
    echo "[install] main_ppo_fewshot.py already patched (skip)"
else
    echo "[install] patching $TRAINER_FILE"
    python3 - "$TRAINER_FILE" <<'PY'
import sys, re, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()

# Add import for math_fewshot below the existing qa_em_fewshot import.
if "from verl.utils.reward_score import math_fewshot" not in src:
    src = src.replace(
        "from verl.utils.reward_score import qa_em_fewshot",
        "from verl.utils.reward_score import qa_em_fewshot\nfrom verl.utils.reward_score import math_fewshot",
        1,
    )

# Inject a 'math' branch at the top of _select_rm_score_fn.
pattern = r"(def _select_rm_score_fn\(data_source[^\n]*\n(?:[^\n]*\n)*?    )(?=if data_source == 'nq')"
def repl(m):
    return m.group(1) + (
        "if data_source == 'math':\n"
        "        if reward_type == 'fewshot':\n"
        "            return math_fewshot.compute_score_fewshot\n"
        "        return math_fewshot.compute_score_em\n"
        "    "
    )
new, n = re.subn(pattern, repl, src, count=1)
if n == 0:
    print("[install] WARN: could not locate _select_rm_score_fn pattern; manual edit required.")
    sys.exit(2)
p.write_text(new)
print("[install] patched _select_rm_score_fn (1 site)")
PY
fi

# --- 3. sanity check ---
echo
echo "[install] verifying patch..."
python3 - <<PY
import importlib.util, sys
spec = importlib.util.spec_from_file_location("math_fewshot", "$REWARD_DIR/math_fewshot.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assert hasattr(m, "compute_score_fewshot")
assert hasattr(m, "compute_score_em")
print("[install]   math_fewshot.compute_score_fewshot OK")

src = open("$TRAINER_FILE").read()
assert "math_fewshot.compute_score_fewshot" in src
assert "if data_source == 'math':" in src
print("[install]   main_ppo_fewshot.py routing OK")
PY

cat <<NOTE

[install] Done.

Next steps:
  1. Start the Python sandbox (acts as ICRL's retriever):
       python $HERE/sandbox/python_sandbox_server.py --port 8000 --timeout 5

  2. Prepare 3 / 2 / 0-shot math parquet:
       python $HERE/scripts/data_process/math_fewshot.py \\
           --dataset math --num_examples 3 \\
           --local_dir $HERE/data/math_3shot
       python $HERE/scripts/data_process/math_fewshot.py \\
           --dataset math --num_examples 2 \\
           --local_dir $HERE/data/math_2shot
       python $HERE/scripts/data_process/math_fewshot.py \\
           --dataset math --template_type zeroshot \\
           --local_dir $HERE/data/math_0shot

  3. Curriculum train (3 -> 2 -> 0, skipping 1-shot per ICRL ablation):
       bash $HERE/train_curriculum_math.sh

NOTE
