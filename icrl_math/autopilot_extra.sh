#!/bin/bash
# Tier 2 extras (runs after autopilot_full.sh finishes):
#   E1. n=500 cross eval (full grid with A/B/C/D/E if present)
#   E2. AIME2024 + AIME2025 evaluation (3rd benchmark)
#   E3. Phase G — correctness-only reward ablation training (500 steps)
#   E4. Phase H — format-only reward ablation training (500 steps)
#   E5. Eval Phases G + H on GSM8K + MATH500
#   E6. Re-aggregate all results into ANALYSIS.md
#
# Triggered by waiting on autopilot_full.sh's ANALYSIS.md signal file.
#
# Usage:  GPU=3 nohup bash autopilot_extra.sh > /tmp/autopilot_extra.log 2>&1 &

set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/home/jklee/ondevice/.venv-icrl-math-v2}"
GPU="${GPU:-0}"
PY="$VENV/bin/python"
LOG_DIR="${LOG_DIR:-/tmp/icrl_math_autopilot}"
mkdir -p "$LOG_DIR"

STAGE3_LORA_DIR="$HERE/checkpoints/grpo_l40s_phase_c_stage3/lora_final"
PHASE_D_LORA="$HERE/checkpoints/grpo_l40s_phase_d/lora_final"
PHASE_E_LORA="$HERE/checkpoints/grpo_l40s_phase_e/lora_final"
PHASE_G_LORA="$HERE/checkpoints/grpo_l40s_phase_g_correctness_only/lora_final"
PHASE_H_LORA="$HERE/checkpoints/grpo_l40s_phase_h_format_only/lora_final"

ts() { date "+%F %T"; }
log() { echo "[$(ts)] [autopilot_extra] $*"; }

export CUDA_VISIBLE_DEVICES="$GPU"
export UNSLOTH_VLLM_STANDBY=0
export VLLM_USE_V1=0
export HF_HOME="$HERE/.hf-cache"

# ----- Wait for autopilot_full to finish -----
SIGNAL="$HERE/ANALYSIS.md"
log "Waiting for autopilot_full signal at $SIGNAL"
WAITED=0
while [ ! -f "$SIGNAL" ]; do
    sleep 60; WAITED=$((WAITED+60))
    [ $((WAITED % 600)) -eq 0 ] && log "  ... waited ${WAITED}s"
done
log "autopilot_full done. Tier 2 starting."

# Optional Phase E LoRA arg (may exist)
PHASE_E_ARG=""
[ -d "$PHASE_E_LORA" ] && PHASE_E_ARG="--lora-phase-e $PHASE_E_LORA"

# ----- E1: n=500 cross eval (full grid) -----
log "E1/6: n=500 cross eval (GSM8K full grid)"
"$PY" "$HERE/scripts/eval_zero_shot_transfer.py" --n 500 \
    --lora-phase-c "$STAGE3_LORA_DIR" \
    --lora-phase-d "$PHASE_D_LORA" \
    $PHASE_E_ARG \
    --out-dir "$HERE/eval_results/cross_n500" \
    > "$LOG_DIR/extra_gsm8k_n500.log" 2>&1
"$PY" "$HERE/scripts/eval_math500.py" --n 500 \
    --lora-phase-c "$STAGE3_LORA_DIR" \
    --lora-phase-d "$PHASE_D_LORA" \
    --out-dir "$HERE/eval_results/math500_n500" \
    > "$LOG_DIR/extra_math500_n500.log" 2>&1
log "E1 done"

# ----- E2: AIME 2024 + 2025 -----
log "E2/6: AIME 2024 + 2025 evaluation"
"$PY" "$HERE/scripts/eval_aime.py" --n 60 \
    --lora-phase-c "$STAGE3_LORA_DIR" \
    --lora-phase-d "$PHASE_D_LORA" \
    $PHASE_E_ARG \
    > "$LOG_DIR/extra_aime.log" 2>&1
log "E2 done"

# ----- E3: Phase G (correctness-only) -----
log "E3/6: Phase G — correctness-only reward training (500 steps)"
if [ -d "$PHASE_G_LORA" ]; then
    log "  Phase G already done, skipping"
else
    GPU="$GPU" bash "$HERE/run_reward_ablation.sh" g > "$LOG_DIR/phase_g.log" 2>&1
fi
log "E3 done"

# ----- E4: Phase H (format-only) -----
log "E4/6: Phase H — format-only reward training (500 steps)"
if [ -d "$PHASE_H_LORA" ]; then
    log "  Phase H already done, skipping"
else
    GPU="$GPU" bash "$HERE/run_reward_ablation.sh" h > "$LOG_DIR/phase_h.log" 2>&1
fi
log "E4 done"

# ----- E5: eval Phases G + H on GSM8K + MATH500 (separate from main grid) -----
log "E5/6: eval Phase G + H on GSM8K + MATH500"
"$PY" - <<PYEOF > "$LOG_DIR/extra_gh_eval.log" 2>&1
import os, sys, json, re, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES","$GPU")
os.environ.setdefault("UNSLOTH_VLLM_STANDBY","0")
os.environ.setdefault("VLLM_USE_V1","0")
os.environ.setdefault("HF_HOME","$HERE/.hf-cache")
from datasets import load_dataset
from unsloth import FastLanguageModel
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from pathlib import Path

SYSTEM_PROMPT_BASE = """You are a careful step-by-step math reasoner.

Respond in EXACTLY this format and nothing else:

<reasoning>
your step-by-step reasoning here
</reasoning>
<answer>
the final numeric answer here, e.g. 42
</answer>
"""
HASH = re.compile(r"####\s*(\-?[0-9\.,]+)")
ANS  = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
def gold(t):
    m = HASH.search(t or ""); return m.group(1).replace(",","").strip() if m else None
def pred(t):
    m = ANS.search(t or "")
    if not m:
        nums = re.findall(r"\-?\d+(?:\.\d+)?", t or ""); return nums[-1] if nums else ""
    raw = m.group(1).strip()
    nums = re.findall(r"\-?\d+(?:\.\d+)?", raw); return nums[-1] if nums else raw
def ok(p, g):
    if not p or not g: return False
    if p.replace(",","").strip() == g.replace(",","").strip(): return True
    try: return abs(float(p) - float(g)) < 1e-6
    except: return False

print("loading model")
model, tok = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit", max_seq_length=2048,
    load_in_4bit=True, fast_inference=True, max_lora_rank=16,
    gpu_memory_utilization=0.30)
ds = load_dataset("openai/gsm8k","main",split="test").select(range(100))
exs = [{"q": e["question"], "g": gold(e["answer"])} for e in ds]
sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=512,
                    stop=["</answer>"], include_stop_str_in_output=True)
prompts = [tok.apply_chat_template(
    [{"role":"system","content":SYSTEM_PROMPT_BASE},{"role":"user","content":e["q"]}],
    tokenize=False, add_generation_prompt=True) for e in exs]
out = {}
for label, path in [("phase_g","$PHASE_G_LORA"),("phase_h","$PHASE_H_LORA")]:
    if not Path(path).exists():
        print(f"skip {label}, no lora at {path}"); continue
    lr = LoRARequest(label, hash(label)%10000, path)
    res = model.fast_generate(prompts, sampling_params=sp, lora_request=lr)
    n_ok = 0
    for e, r in zip(exs, res):
        n_ok += int(ok(pred(r.outputs[0].text), e["g"]))
    em = n_ok / len(exs)
    print(f"[{label}] EM = {em:.3f} ({n_ok}/{len(exs)})")
    out[label] = {"em": em, "correct": n_ok}
Path("$HERE/eval_results/ablation_gh").mkdir(parents=True, exist_ok=True)
with open("$HERE/eval_results/ablation_gh/summary.json","w") as f:
    json.dump({"n": 100, "results": out}, f, indent=2)
PYEOF
log "E5 done"

# ----- E6: re-aggregate -----
log "E6/6: re-aggregate into ANALYSIS.md"
"$PY" "$HERE/scripts/aggregate_results.py" > "$LOG_DIR/extra_aggregate.log" 2>&1 || true

log "TIER 2 ALL DONE."
echo ""
echo "========= TIER 2 FINAL SUMMARY ========="
for d in cross_n500 math500_n500 aime ablation_gh; do
    p="$HERE/eval_results/$d/summary.json"
    if [ -f "$p" ]; then
        echo
        echo "--- $d ---"
        "$PY" -c "
import json
d = json.load(open('$p'))
for k, v in d['results'].items():
    print(f'  {k:30s}  {v[\"em\"]:.3f}')
"
    fi
done
