#!/bin/bash
# Tier 3 chain — main conf long-paper-grade extras.
# Waits for Tier 2 done signal (cross_n500/summary.json), then runs:
#   T1. Multi-seed Phase C × 2 (seeds 1234, 5678) — variance estimate
#   T2. Qwen 1.5B Phase C  — smaller-scale validation
#   T3. CoT baseline (no LoRA, no demos) — comparison baseline
#   T4. Eval seed_1234 / seed_5678 / 1.5B Phase C on GSM8K + MATH500
#   T5. Final aggregate
#
# Usage:  GPU=3 nohup bash autopilot_tier3.sh > /tmp/autopilot_tier3.log 2>&1 &

set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/home/jklee/ondevice/.venv-icrl-math-v2}"
GPU="${GPU:-0}"
PY="$VENV/bin/python"
LOG_DIR="${LOG_DIR:-/tmp/icrl_math_autopilot}"
mkdir -p "$LOG_DIR"

ts() { date "+%F %T"; }
log() { echo "[$(ts)] [autopilot_tier3] $*"; }

export CUDA_VISIBLE_DEVICES="$GPU"
export UNSLOTH_VLLM_STANDBY=0
export VLLM_USE_V1=0
export HF_HOME="$HERE/.hf-cache"

# ----- Wait for Tier 2 done -----
SIGNAL="$HERE/eval_results/cross_n500/summary.json"
log "Waiting for Tier 2 signal at $SIGNAL"
WAITED=0
while [ ! -f "$SIGNAL" ]; do
    sleep 60; WAITED=$((WAITED+60))
    [ $((WAITED % 600)) -eq 0 ] && log "  ... waited ${WAITED}s"
done
log "Tier 2 done. Tier 3 starting."

# ----- T1a: Multi-seed Phase C (seed 1234) -----
log "T1a/5: Multi-seed Phase C — seed=1234"
GPU="$GPU" SEED=1234 bash "$HERE/run_phase_c_seed.sh" > "$LOG_DIR/seed_1234.log" 2>&1
log "T1a done"

# ----- T1b: Multi-seed Phase C (seed 5678) -----
log "T1b/5: Multi-seed Phase C — seed=5678"
GPU="$GPU" SEED=5678 bash "$HERE/run_phase_c_seed.sh" > "$LOG_DIR/seed_5678.log" 2>&1
log "T1b done"

# ----- T2: Qwen 1.5B Phase C -----
log "T2/5: Qwen 1.5B Phase C (full 3-stage curriculum)"
GPU="$GPU" bash "$HERE/run_phase_c_15b.sh" > "$LOG_DIR/phase_c_15b.log" 2>&1
log "T2 done"

# ----- T3: CoT baseline -----
log "T3/5: CoT baseline (no LoRA, no demos)"
"$PY" "$HERE/scripts/eval_cot_baseline.py" --n 100 > "$LOG_DIR/cot_baseline.log" 2>&1
log "T3 done"

# ----- T4: eval all new LoRAs (multi-seed + 1.5B) on GSM8K -----
# Each (model, LoRA) gets its own Python subprocess so vLLM frees memory cleanly.
log "T4/5: Eval seed_1234 / seed_5678 / 1.5B Phase C on GSM8K (separate subprocesses)"

SEED_1234_LORA="$HERE/checkpoints/grpo_l40s_seed_1234/stage3/lora_final"
SEED_5678_LORA="$HERE/checkpoints/grpo_l40s_seed_5678/stage3/lora_final"
LORA_15B="$HERE/checkpoints/grpo_l40s_15b/stage3/lora_final"
OUT_T3="$HERE/eval_results/tier3"

# Init empty summary so subprocess writes accumulate
mkdir -p "$OUT_T3"
[ ! -f "$OUT_T3/summary.json" ] && echo '{"n": 100, "results": {}}' > "$OUT_T3/summary.json"

for cfg in \
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit|$SEED_1234_LORA|phase_c_seed1234|16" \
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit|$SEED_5678_LORA|phase_c_seed5678|16" \
    "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit|$LORA_15B|phase_c_15b|8"
do
    IFS='|' read -r model_id lora_path label rank <<<"$cfg"
    if [ ! -d "$lora_path" ]; then
        log "  skip $label: LoRA not found"
        continue
    fi
    log "  evaluating $label"
    "$PY" "$HERE/scripts/eval_single_lora.py" \
        --model "$model_id" --lora "$lora_path" --label "$label" \
        --lora-rank "$rank" --n 100 --out-dir "$OUT_T3" \
        >> "$LOG_DIR/tier3_eval.log" 2>&1 || log "  $label eval failed (continuing)"
done

# (legacy inline path retained but skipped)
if false; then
"$PY" - <<PYEOF > /dev/null 2>&1
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
    m=HASH.search(t or ""); return m.group(1).replace(",","").strip() if m else None
def pred(t):
    m=ANS.search(t or "")
    if not m:
        nums=re.findall(r"\-?\d+(?:\.\d+)?", t or ""); return nums[-1] if nums else ""
    raw=m.group(1).strip()
    nums=re.findall(r"\-?\d+(?:\.\d+)?", raw); return nums[-1] if nums else raw
def ok(p,g):
    if not p or not g: return False
    if p.replace(",","").strip()==g.replace(",","").strip(): return True
    try: return abs(float(p)-float(g))<1e-6
    except: return False

# Each new LoRA pair (model_repo, lora_path, label)
configs = [
    ("unsloth/Qwen2.5-3B-Instruct-bnb-4bit", "$SEED_1234_LORA", "phase_c_seed1234"),
    ("unsloth/Qwen2.5-3B-Instruct-bnb-4bit", "$SEED_5678_LORA", "phase_c_seed5678"),
    ("unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit", "$LORA_15B", "phase_c_15b"),
]

OUT = Path("$HERE/eval_results/tier3"); OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = {"n": 100, "results": {}}

for model_id, lora_path, label in configs:
    if not Path(lora_path).exists():
        print(f"SKIP {label}: lora not found {lora_path}"); continue
    print(f"\n=== {label} ({model_id}) ===")
    rank = 8 if "1.5B" in model_id else 16
    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_id, max_seq_length=2048, load_in_4bit=True,
        fast_inference=True, max_lora_rank=rank, gpu_memory_utilization=0.30)
    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=512,
                        stop=["</answer>"], include_stop_str_in_output=True)
    ds = load_dataset("openai/gsm8k","main",split="test").select(range(100))
    exs = [{"q":e["question"],"g":gold(e["answer"])} for e in ds]
    prompts = [tok.apply_chat_template(
        [{"role":"system","content":SYSTEM_PROMPT_BASE},{"role":"user","content":e["q"]}],
        tokenize=False, add_generation_prompt=True) for e in exs]
    lr = LoRARequest(label, hash(label)%10000, lora_path)
    res = model.fast_generate(prompts, sampling_params=sp, lora_request=lr)
    n_ok = sum(int(ok(pred(r.outputs[0].text), e["g"])) for e, r in zip(exs, res))
    em = n_ok / len(exs)
    print(f"  GSM8K {label} EM = {em:.3f} ({n_ok}/{len(exs)})")
    SUMMARY["results"][f"{label}__base"] = {"em": em, "correct": n_ok}
    # release vLLM memory between configs
    del model, tok; import gc; gc.collect()
    import torch; torch.cuda.empty_cache()

with open(OUT/"summary.json","w") as f:
    json.dump(SUMMARY, f, indent=2)
print(f"\nsaved -> {OUT}/summary.json")
PYEOF
fi  # close `if false` — legacy inline block kept dormant
log "T4 done"

# ----- T5: final aggregate -----
log "T5/5: re-aggregate all results"
"$PY" "$HERE/scripts/aggregate_results.py" > "$LOG_DIR/tier3_aggregate.log" 2>&1 || true

log "TIER 3 ALL DONE."
echo ""
echo "========= TIER 3 FINAL ========="
for d in tier3 cot_baseline; do
    p="$HERE/eval_results/$d/summary.json"
    [ -f "$p" ] && {
        echo
        echo "--- $d ---"
        "$PY" -c "
import json
d = json.load(open('$p'))
for k, v in d['results'].items():
    print(f'  {k:30s}  {v[\"em\"]:.3f}')
"
    }
done
