#!/bin/bash
# Set up the icrl_math virtualenv (torch 2.6 + cu124 + Unsloth + TRL GRPO + vLLM).
# Idempotent: re-running upgrades pins without breaking torch.
#
# Usage:
#   bash setup_venv.sh
#   VENV=~/.venvs/icrl-math bash setup_venv.sh

set -e

VENV="${VENV:-/home/jklee/ondevice/.venv-icrl-math}"

if [ ! -d "$VENV" ]; then
    echo "[setup] creating venv at $VENV"
    python3 -m venv "$VENV"
fi

PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

"$PIP" install --upgrade --quiet pip wheel setuptools

# 1) torch 2.6 + cu124 (matches host CUDA)
if ! "$PY" -c "import torch; assert torch.__version__.startswith('2.6')" 2>/dev/null; then
    echo "[setup] installing torch 2.6.0+cu124"
    "$PIP" install --quiet torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
fi

# 2) core RL stack. peft<0.17 because newer peft requires torch>=2.11.
#    datasets unconstrained because trl>=0.16 needs datasets>=3.0.
echo "[setup] installing peft/transformers/accelerate/bitsandbytes/datasets/trl"
"$PIP" install --quiet \
    'peft<0.17' \
    'transformers>=4.50,<4.55' \
    'accelerate>=1.0,<1.5' \
    'bitsandbytes>=0.45,<0.50' \
    'datasets>=3.0' \
    'trl>=0.16,<0.20'

# 3) vLLM (cu124-compatible 0.7.x line)
echo "[setup] installing vllm 0.7.x"
"$PIP" install --quiet 'vllm>=0.7.3,<0.8'

# 4) Unsloth (depends on the stack above)
echo "[setup] installing unsloth + unsloth_zoo"
"$PIP" install --quiet unsloth unsloth_zoo

# 5) sandbox HTTP server deps
echo "[setup] installing fastapi/uvicorn/requests"
"$PIP" install --quiet fastapi uvicorn requests

echo
echo "=== installed versions ==="
"$PY" - <<'PY'
import torch, unsloth, trl, peft, vllm, bitsandbytes, datasets, transformers
print(f"torch         = {torch.__version__}   cuda={torch.cuda.is_available()}")
print(f"unsloth       = {unsloth.__version__}")
print(f"trl           = {trl.__version__}")
print(f"peft          = {peft.__version__}")
print(f"vllm          = {vllm.__version__}")
print(f"bitsandbytes  = {bitsandbytes.__version__}")
print(f"datasets      = {datasets.__version__}")
print(f"transformers  = {transformers.__version__}")
PY

echo
echo "[setup] done. activate with:  source $VENV/bin/activate"
echo "[setup] then run a smoke:     bash run_phase_a.sh smoke"
