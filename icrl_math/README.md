# ICRL-Math: In-Context RL for Math Reasoning at SLM Scale

> Apply ICRL (Ye, Zhao et al., 2026) **almost verbatim** to mathematical reasoning,
> targeting **Qwen2.5-3B-Instruct** — the smallest scale ICRL hasn't been
> validated on — and replacing the search retriever with a sandboxed
> **Python interpreter**.

## Why this exists

ICRL showed that you can train tool-using LLMs with **no SFT**, just a few-shot
prompt that you gradually phase out during GRPO rollouts. The original paper
mainly validated this on web search (NQ → multi-hop QA); the math + code section
was a brief generalization (AIME only, 8B model).

This codebase extends ICRL along three axes:

| Axis | ICRL original | This work |
|---|---|---|
| Domain | search (NQ → TriviaQA, HotpotQA, ...) | **math + Python tool** (GSM8K / MATH → AIME, MATH500, ...) |
| Model | 3B / 7B / 8B / 14B | **3B SLM focus** (does ICRL hold at this scale?) |
| Tool | live web search (Serper / BM25) | **sandboxed Python interpreter** |
| Demo pool | fixed 3 few-shot examples | same (fixed 3-5), with hooks for cluster-adaptive selection later |

We **reuse ICRL's entire training infrastructure**:
- `verl.trainer.main_ppo_fewshot` (GRPO + few-shot rollouts)
- `verl/trainer/ppo/ray_trainer._create_loss_mask` (mask tool outputs)
- `search_r1/llm_agent/generation.py` (multi-turn rollout loop)

What we add is small and contained:
1. `example/math_examples.txt` — 5 math few-shot demos in ICRL's tag format
2. `scripts/data_process/math_fewshot.py` — GSM8K/MATH → parquet, like ICRL's NQ script
3. `sandbox/python_sandbox_server.py` — code execution HTTP server, drop-in for ICRL's retriever
4. `verl_patches/math_fewshot.py` — boxed-answer EM + format-penalty reward
5. `install_into_icrl.sh` — copies (4) into ICRL and patches `main_ppo_fewshot._select_rm_score_fn` to route `data_source='math'`
6. `train_curriculum_math.sh` — Stage 1→2→3 (3-shot → 2-shot → 0-shot, skipping 1-shot per ICRL ablation)
7. `scripts/eval_math.py` — vLLM-based standalone eval on GSM8K/MATH500/AIME2024/AIME2025

## Tag convention

The training infrastructure remains untouched, so we **keep ICRL's tag names**
(`<search>`, `<information>`) but redefine their semantics in the prompt:

| Tag | ICRL meaning | Our meaning |
|---|---|---|
| `<think>...</think>` | reasoning | same |
| `<search>...</search>` | search query | **Python source code** (executed in our sandbox) |
| `<information>...</information>` | retrieved docs (loss-masked) | **stdout from the sandbox** (loss-masked) |
| `<answer>\boxed{...}</answer>` | final answer | same |

The system prompt explicitly tells the model that `<search>` invokes a Python
interpreter. Loss masking on `<information>` content is already implemented in
ICRL's trainer — we get it for free.

## Setup

1. **Clone ICRL** (the upstream framework, untouched except for our small patch):
   ```bash
   cd /home/jklee/ondevice
   git clone https://github.com/applese233/ICRL.git
   conda env create -f ICRL/environment.yml
   conda activate icrl
   pip install -e ICRL
   pip install flash-attn --no-build-isolation
   pip install fastapi uvicorn   # for our sandbox
   ```

2. **Install our patches into ICRL** (idempotent):
   ```bash
   ICRL_DIR=/home/jklee/ondevice/ICRL \
     bash icrl_math/install_into_icrl.sh
   ```
   This copies `verl_patches/math_fewshot.py` into ICRL's `verl/utils/reward_score/`
   and adds a `data_source == 'math'` branch in `_select_rm_score_fn`.

3. **Start the Python sandbox** (replaces ICRL's retrieval server):
   ```bash
   python icrl_math/sandbox/python_sandbox_server.py \
       --host 127.0.0.1 --port 8000 \
       --timeout 5 --memory-mb 1024 --cpu-sec 5
   ```
   The endpoint `POST /retrieve` accepts ICRL's normal `{"queries": [...], "topk": 1}`
   payload; each query is interpreted as Python source.

4. **Prepare 3 / 2 / 0-shot parquet datasets** (MATH train + test):
   ```bash
   python icrl_math/scripts/data_process/math_fewshot.py \
       --dataset math --num_examples 3 --local_dir icrl_math/data/math_3shot
   python icrl_math/scripts/data_process/math_fewshot.py \
       --dataset math --num_examples 2 --local_dir icrl_math/data/math_2shot
   python icrl_math/scripts/data_process/math_fewshot.py \
       --dataset math --template_type zeroshot --local_dir icrl_math/data/math_0shot
   ```
   For GSM8K-only or mixed training, swap `--dataset` to `gsm8k` or `mixed`.

5. **Curriculum train** (Qwen2.5-3B-Instruct, 4×A100):
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 ICRL_DIR=/home/jklee/ondevice/ICRL \
     bash icrl_math/train_curriculum_math.sh
   ```
   Stages: 3-shot → 2-shot → 0-shot, 100 steps each (configurable via
   `STAGE{1,2,3}_STEPS` env vars). Skips the 1-shot stage per ICRL Section 4.1.

6. **Evaluate** on standard math benchmarks:
   ```bash
   python icrl_math/scripts/eval_math.py \
       --model-path icrl_math/checkpoints/icrl-math-stage3-0shot-qwen2.5-3b/.../hf \
       --datasets gsm8k math500 aime2024 aime2025 \
       --sandbox-url http://127.0.0.1:8000/retrieve \
       --num-shots 0 \
       --out-dir icrl_math/eval_results/
   ```

## Reward design

`compute_score_fewshot(solution, ground_truth) = 0.8·acc + 0.2·format`

- **Accuracy** (binary): extract last `<answer>\\boxed{...}</answer>`, then
  compare to gold via three fallbacks: Hendrycks-MATH string normalization → numeric
  tolerance → sympy `simplify(a-b)==0` (handles e.g. `100-25*pi` vs `-25\\pi+100`).
- **Format** (penalty-based, in [0,1]): missing/unbalanced `<think>`, `<search>`,
  `<answer>` tags or empty extracted answer incur graduated penalties.

This is intentionally identical in structure to ICRL's `qa_em_fewshot.compute_score_fewshot`
— we only swap the answer extractor (NQ EM → boxed extraction) and the equivalence
checker (string EM → MATH string-equiv + numeric + sympy).

## Hyperparameters

`train_curriculum_math.sh` defaults (3B; tighter than ICRL's 7B run):

| Knob | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Max prompt length | 4096 |
| Max response length | 2048 |
| Max obs (sandbox stdout) | 500 chars |
| Train batch | 64 |
| PPO mini / micro batch | 32 / 8 |
| Rollouts per query (`n_agent`) | 8 |
| Temperature | 1.0 (rollout) / 0.0 (eval) |
| LR / KL coef | 1e-6 / 0.001 (ICRL same) |
| Steps per stage | 100 |
| Max turns | 6 |
| Accuracy / format weight | 0.8 / 0.2 |

Override any via env var, e.g. `STAGE1_STEPS=200 N_AGENT_ROLLOUTS=4 bash train_curriculum_math.sh`.

## Files

```
icrl_math/
├── README.md                        (this file)
├── example/
│   └── math_examples.txt            5 demos: algebra, gcd, geometry, combinatorics, calculus
├── scripts/
│   ├── data_process/math_fewshot.py GSM8K/MATH/mixed -> parquet
│   └── eval_math.py                 vLLM eval with sandbox tool loop
├── sandbox/
│   └── python_sandbox_server.py     FastAPI; /retrieve endpoint = code exec
├── verl_patches/
│   └── math_fewshot.py              boxed-EM + format reward (drop into ICRL)
├── install_into_icrl.sh             one-shot patcher
├── train_curriculum_math.sh         Qwen2.5-3B, 3->2->0
└── configs/                         (reserved for future hyperparameter sweeps)
```

## Roadmap / ablations

Things deliberately left out of v1, in priority order:

1. **Cluster-adaptive demos** — sample 3 demos from the Auto-CoT cluster nearest
   to the query (re-using `scripts/build_autocot_clusters.py` from the parent
   `slm-context-pipeline`). Ablation against fixed-3 demos.
2. **Reward decomposition** — split per-step rewards (code success vs answer
   correctness) for credit assignment; current reward is trajectory-level.
3. **GSM8K-only vs MATH-only vs mixed** training-set ablation.
4. **Curriculum schedule** — replicate ICRL's 4-stage (3→2→1→0) vs 3-stage
   ablation in the math setting; ICRL found 3-stage strictly better for QA.
5. **Tool sandbox isolation** — wrap sandbox in firejail / docker for stronger
   guarantees if running untrusted RL workloads.

## Citation / inspiration

- Ye, Zhao, et al. *In-Context Reinforcement Learning for Tool Use in Large Language Models.* (Mar 2026)
- Shao et al. *DeepSeekMath: GRPO.* (2024)
- Sheng et al. *HybridFlow / VeRL.* (2024)
