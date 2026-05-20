#!/bin/bash
# Set up the icrl_math virtualenv on torch 2.7.1 + cu124.
#
# Why this version of the stack:
#   - torch 2.7.1 adds the torch.int{1..7} / uint{1..7} dtypes that
#     torchao >= 0.10 references unconditionally at import time. On older
#     torch (2.5/2.6) Unsloth's torchao>=0.16 dependency deadlocks.
#   - vllm 0.9.x is the first line that fully supports torch 2.7.
#   - cu124 matches host CUDA driver (CUDA 12.4 on L40S).
#
# Idempotent: re-running upgrades pins without breaking torch.
#
# Usage:
#   bash setup_venv.sh
#   VENV=~/.venvs/icrl-math bash setup_venv.sh

set -e

VENV="${VENV:-/home/jklee/ondevice/.venv-icrl-math-v2}"

if [ ! -d "$VENV" ]; then
    echo "[setup] creating venv at $VENV"
    python3 -m venv "$VENV"
fi

PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

"$PIP" install --upgrade --quiet pip wheel setuptools

# 1) torch 2.7.1 + cu124 (must precede anything that pulls in torch as a dep)
if ! "$PY" -c "import torch; assert torch.__version__.startswith('2.7')" 2>/dev/null; then
    echo "[setup] installing torch 2.7.1+cu126 (cu126 wheel works on driver 12.4 via minor forward compat)"
    "$PIP" install --quiet \
        torch==2.7.1 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126
fi

# 2) vLLM 0.9.x (compatible with torch 2.7)
echo "[setup] installing vllm 0.9.x"
"$PIP" install --quiet 'vllm>=0.9.0,<0.10'

# 3) transformers / peft / trl / accelerate / bitsandbytes / datasets
#    transformers 5.x has a dataclass ordering bug (non-default 'vision_config'
#    follows default arg) that breaks trl 0.24's GRPOTrainer import; pin to 4.x.
echo "[setup] installing transformers/peft/trl/accelerate/bnb/datasets"
"$PIP" install --quiet \
    'transformers>=4.55,<5.0' \
    'peft' \
    'trl' \
    'accelerate' \
    'bitsandbytes' \
    'datasets'

# 4) Unsloth (depends on stack above + torchao>=0.16 which now works on torch 2.7)
echo "[setup] installing unsloth + unsloth_zoo"
"$PIP" install --quiet unsloth unsloth_zoo

# 5) sandbox HTTP server deps
echo "[setup] installing fastapi/uvicorn/requests"
"$PIP" install --quiet fastapi uvicorn requests

echo
echo "=== installed versions ==="
"$PY" - <<'PY'
import torch, torchao, unsloth, trl, peft, vllm, bitsandbytes, datasets, transformers
print(f"torch         = {torch.__version__}   cuda={torch.cuda.is_available()} devs={torch.cuda.device_count()}")
print(f"torchao       = {torchao.__version__}")
print(f"vllm          = {vllm.__version__}")
print(f"unsloth       = {unsloth.__version__}")
print(f"trl           = {trl.__version__}")
print(f"peft          = {peft.__version__}")
print(f"bitsandbytes  = {bitsandbytes.__version__}")
print(f"datasets      = {datasets.__version__}")
print(f"transformers  = {transformers.__version__}")
print()
print("torch.int1?", hasattr(torch, "int1"))
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
print("FastLanguageModel + GRPOTrainer OK")
PY

echo
echo "[setup] done."
echo "[setup] next:  GPU=0 bash run_phase_a.sh smoke"
