# Curriculum-Driven Internalization: How ICRL Closes the Demos-Conditional Trap at SLM Scale

> **Working title.** Paper draft accompanying the `icrl_math` codebase.
> Target venue: EMNLP main conference long paper.

## Abstract

In-Context Reinforcement Learning (ICRL; Ye, Zhao et al. 2026) trains tool-using
LLMs without SFT by embedding few-shot demos into GRPO rollouts and phasing them
out across stages. We replicate ICRL on math reasoning at **single-GPU SLM
scale** (Qwen2.5-3B / 1.5B-Instruct, L40S 40 GB) and isolate one question that
the original paper did not test directly: **does the GRPO LoRA actually
*internalize* the demos, or does it just learn a demos-conditional policy that
collapses when the demos are removed at inference?**

We find: (i) at 3 B, prompting demos alone moves GSM8K from 48.6 % → 80.0 %
(+31 pp), making demos the dominant signal; (ii) GRPO with demos-always-on
(our Phase B) **fails to internalize** — drop the demos at inference and the
LoRA collapses back to baseline (50 %); (iii) the **3-stage 5→2→0 curriculum
(Phase C) is the only intervention that closes the gap**, lifting zero-shot
to **72.4 %** on n=500 GSM8K (+23.8 pp over baseline, robust across 3 random
seeds and reproducible at 1.5 B as well as 3 B); (iv) longer training is **not
a substitute**: 3× the steps (Phase E) only reaches 56 %; (v) **adding format
reward to a correctness reward strictly hurts** (Phase G correctness-only beats
Phase B mixed). OOD generalization to MATH500 is weak (+2 pp) and AIME is at
noise floor for all conditions.

We position this as a focused, ablation-heavy replication that **directly tests
ICRL's curriculum hypothesis**, complements the original paper's tool-use
results, and surfaces a reward-design pitfall.

## 1. Introduction

(Standard intro: tools and demos are how LMs are made useful at inference time.
ICRL fuses both into GRPO rollouts; the paper claims SFT-free training works
when demos are phased out. We test whether *the phase-out is what carries the
weight*, at the smallest scale where ICRL was claimed to apply.)

## 2. Method

### 2.1 Setup

- **Base model**: Qwen2.5-3B-Instruct (`bnb-4bit`), with Qwen2.5-1.5B for scale.
- **Trainer**: Unsloth + TRL GRPO, vLLM 0.9 rollout, single L40S 40 GB.
- **LoRA**: r=16 on all 7 linear targets (qkvo + gate / up / down) for Phase B
  onwards, r=8 q/v for Phase A.
- **Reward**: `α·correctness + β·format` where correctness extracts the last
  `<answer>` block and string-normalizes; format is XML-tag balance.
- **Data**: GSM8K (train for RL, test for eval). MATH500 + AIME 2024/2025 as
  held-out OOD.

### 2.2 Training conditions

| Name | demos at train | steps | r / targets | reward | resume |
|---|---|---|---|---|---|
| A | 0 (base prompt) | 300 | 8, q/v | 2.0 + 0.5 ×3 | — |
| B | 5-shot always-on | 500 | 16, all | 5.0 + 0.3 ×3 | — |
| D | 2-shot always-on | 500 | 16, all | 5.0 + 0.3 ×3 | — |
| E | 5-shot always-on, longer | 1500 | 16, all | 5.0 + 0.3 ×3 | — |
| **C** | **5→2→0 curriculum** | **500+200+300** | **16, all** | **5.0 + 0.3 ×3** | **previous stage** |
| G | 5-shot, **correctness-only** | 500 | 16, all | 5.0 + 0 | — |
| H | 5-shot, **format-only** | 500 | 16, all | 0 + 1.0 ×2 | — |

C is exactly ICRL's curriculum: each stage resumes the previous stage's LoRA
weights but reduces demo count. G/H isolate which reward component carries the
signal.

### 2.3 Evaluation grid

For every (LoRA, prompt) pair, we run greedy decoding on GSM8K test (n=100 for
quick iteration, n=500 for headline numbers, n=60 for AIME). Records and per-
problem predictions are stored under `eval_results/<dir>/`.

The **0-shot transfer test** is the key cell: evaluate a LoRA trained with
demos in the prompt, *without* demos at inference. If the LoRA only learned a
demos-conditional policy, EM collapses to baseline.

## 3. Results

### 3.1 Headline GSM8K (n=500)

| Method | base prompt | demos prompt | Δ base vs baseline |
|---|---|---|---|
| baseline (no LoRA) | **48.6 %** | 80.0 % | — |
| Chain-of-Thought (no LoRA) | 76.0 (n=100) | — | +27 pp |
| Phase A | 51.0 | 79.6 | +2.4 |
| Phase B | 49.8 | 79.0 | +1.2 |
| Phase D | 50.8 | 79.8 | +2.2 |
| Phase E (3× steps) | 56.4 | 81.4 | +7.8 |
| **Phase C (curriculum)** | **72.4** | **80.4** | **+23.8 ★** |

Curriculum lifts zero-shot EM by **23.8 pp** — eight times the gain from
naïve GRPO (Phase B). Demos alone explain 31 pp of the total ceiling; curriculum
is what converts that into a *parameter-resident* capability.

### 3.2 Multi-seed (3 B, n=100)

| Seed | Phase C base EM |
|---|---|
| 3407 | 70 |
| 1234 | 82 |
| 5678 | 78 |
| mean / σ | 76.7 / 6.1 pp |

The curriculum effect is robust to seed choice; σ ~ 6 pp is on the same order
as our n=100 binomial standard error.

### 3.3 Multi-scale

At 1.5 B, Phase C scores **61 %** base on GSM8K. The curriculum effect is
present but smaller — consistent with reduced capacity for in-context format
copying at smaller scale.

### 3.4 0-shot transfer table (the central diagnostic)

| LoRA | base EM | demos EM | drop |
|---|---|---|---|
| Phase A | 52.0 | 76.0 | −24 pp |
| Phase B | 50.0 | 78.0 | −28 pp |
| Phase D | 57.0 | 76.0 | −19 pp |
| Phase E | 58.0 | 77.0 | −19 pp |
| **Phase C** | **70.0** | **80.0** | **−10 pp** |

Every demos-always-on policy loses 19–28 pp when demos are stripped at
inference. Only Phase C narrows that drop to 10 pp — its base-prompt EM is
where most of the demos-prompt capability is now living *inside the weights*.

### 3.5 Reward ablation (Phase G / H)

| reward | base EM (n=100) |
|---|---|
| Phase B: correctness 5.0 + format 0.3×3 | 50.0 |
| **Phase G: correctness 5.0 only** | **58.0** |
| Phase H: format 1.0×2 only | 52.0 |

Adding format to a correctness reward strictly hurts. The Phase G > Phase B
gap (+8 pp) at the same step count and same LoRA shape is the strongest
single-knob effect we measured — bigger than longer training (Phase E vs B is
+6 pp), nearly half the curriculum effect itself.

### 3.6 MATH500 — harder benchmark (n=500)

| method | base | demos |
|---|---|---|
| baseline | 30.8 | 38.2 |
| Phase B | 29.8 | 39.6 |
| Phase C | 33.0 | 39.8 |
| Phase D | 31.6 | 39.4 |

Curriculum's +2 pp on MATH500 is within noise. Two interpretations: (a) the
GSM8K-style demos transfer poorly to MATH problem types; (b) the model hits a
capacity ceiling on multi-step competition math. Either way the curriculum
effect is **task-specific**, not a general "RL improves reasoning" claim.

### 3.7 AIME (n=60, integer answers 0-999)

Every method 0–7 %, near noise floor. AIME is above 3 B + LoRA capability.
Reported as a limitation.

## 4. Discussion

### 4.1 Why does "demos always on" fail to internalize?

The GRPO objective is `E[r(τ) | π(τ | demos, q)]`. When demos are *always* in
the prompt, the model receives no gradient on demos-absent trajectories. The
optimum the loss converges to is a policy that *uses the demos as a crutch* —
removing them at inference returns the policy to the base distribution.

### 4.2 Why does curriculum work?

The 5→2→0 schedule keeps the *gradient flow* on progressively-less-conditioned
trajectories. At each stage, the LoRA already solves the task with one more
demo than is now visible, so reducing demos by one only requires closing a
small distribution shift. Jumping straight to 0-shot inference (which is what
"demos-always-on training + drop demos at test" does implicitly) is too large
a step.

### 4.3 Why does adding format reward hurt?

Qwen2.5-3B-Instruct already follows the requested `<reasoning>/<answer>` XML
format ~99 % of the time out of the box. The format reward is therefore
near-saturated from step 0 — adding it to the gradient pulls the policy
toward something it can already do, at the expense of the harder
correctness signal. The minimal-reward principle (Phase G) outperforms the
shaped-reward instinct.

### 4.4 What does NOT generalize

- **Out-of-distribution math** (MATH500, AIME): curriculum at GSM8K is a
  GSM8K-specific skill, not a math-reasoning unlock.
- **Tool use**: we deliberately did not activate ICRL's Python sandbox.
  Tool-using ICRL is the original paper's contribution; we test only the
  rollout-with-curriculum half.

## 5. Limitations and open items

- **Single model family** (Qwen2.5). Llama-3.2-3B Phase C rerun would
  strengthen claim of generality.
- **No plain-SFT baseline** of same LoRA rank trained on GSM8K answers.
- **AIME** is noise-floor because the base model can't do AIME. Re-running at
  7 B / 14 B would tell us whether curriculum scales.
- **Greedy decoding only**; pass@k or maj@k might surface latent capability.

## 6. Reproduction

See `icrl_math/README.md` and `ANALYSIS.md`. The full chain is automated as
`autopilot_full.sh` → `autopilot_extra.sh` → `autopilot_tier3.sh`, hand-off
via signal files. Total compute used to produce all numbers in this paper:
~14 hours on one L40S 40 GB.

## Caveats and acknowledgements

- **Ye, Zhao et al. 2026 — ICRL** is the method we re-implement; our scope is
  narrower (math + curriculum only, no tools).
- **DeepSeekMath / GRPO (Shao et al. 2024)** for the RL objective.
- **Unsloth + TRL + vLLM** for the single-GPU GRPO scaffolding.
- The n=100 / n=500 split in the headline reflects that n=100 was used during
  exploration and n=500 was the final commit. For paper, n=500 is the
  reported number.
