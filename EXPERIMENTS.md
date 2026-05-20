# 실험 전체 흐름 (Experiments Overview)

> SLM에게 "Q → A"가 아니라 "Q → C*"를 가르치는 파이프라인 실험 기록.
> 데이터 생성 → SFT → DPO → GRPO(RL) → 평가 5단계로 구성됩니다.

---

## 0. 디렉토리 한눈에 보기

```
slm-context-pipeline/
├── pipeline/            # Stage 1~3: Planner / Evidence Builder / Judge
├── data_processing/     # 후보 컨텍스트 생성
├── training/            # SFT, DPO 학습 + 학습 데이터 가공
├── scripts/             # ICL distill 빌드, GRPO(RL) 학습, 평가 baseline
├── evaluation/          # 다운스트림 평가
├── model_configs/       # SLM-Bench용 모델 설정(json)
├── config/              # settings.yaml (teacher/answer/student 모델, retriever 설정)
├── data/                # 작은 샘플 데이터 (real_tiny 등). 대용량 데이터는 .gitignore 처리
├── v4_full_chain.sh     # RL v4 학습 실행 스크립트
└── restart_evals_after_smoke.sh
```

핵심 모델 구성 (`config/settings.yaml`):

| 역할 | 모델 |
|---|---|
| Teacher (데이터 생성) | `gpt-4o-mini` |
| Answer (다운스트림 평가) | `gpt-4o-mini` 또는 Qwen2.5-1.5B-Instruct |
| Student (학습 대상) | Qwen2.5-1.5B-Instruct / Qwen2.5-7B-Instruct |
| Judge (RL reward) | Qwen2.5-7B-Instruct |
| Reward model (RL) | Qwen2.5-1.5B-Instruct |
| Encoder (clustering) | `sentence-transformers/all-MiniLM-L6-v2` |

---

## 1. Stage 1~3 — Teacher 파이프라인으로 C* 데이터 만들기

`pipeline/`의 3단계로 (질문 → 최소 충분 컨텍스트) 라벨을 자동 생성합니다.

| 단계 | 파일 | 입력 | 출력 |
|---|---|---|---|
| Planner | `pipeline/planner.py` | `data/questions*.jsonl` | 질문 유형/엔티티/제약/서브질문 |
| Evidence Builder | `pipeline/evidence_builder.py` | Planner 출력 | retrieved + generated evidence (하이브리드) |
| Judge | `pipeline/judge.py` | 후보 컨텍스트 | utility 기준 필터링 (정보밀도/distractor/leakage) |
| Teacher 실행 | `pipeline/run_teacher.py` | 위 3단계 통합 | `teacher_output*.jsonl`, `labeled*.jsonl` |

후보 컨텍스트 풀은 `data_processing/generate_candidates.py`로 만들고, 결과는 `data/candidates*.jsonl` 입니다.

샘플 실행:
```bash
python -m pipeline.run_teacher \
    --questions data/questions_real_tiny.jsonl \
    --out data/teacher_output_real_tiny.jsonl
```

---

## 2. 학습 데이터 가공 — 4가지 Task로 쪼개기

`training/prepare_training_data.py`는 Teacher 출력을 4개의 학습 task로 분해합니다.

| Task | 출력 파일 | 학습 목적 |
|---|---|---|
| A. Necessity | `task_a_necessity.jsonl` | 각 evidence가 정답에 필요한지 0/1 판단 |
| B. Extraction | `task_b_extraction.jsonl` | 긴 문서에서 필요한 문장만 추출 |
| C. Compression | `task_c_compression.jsonl` | helpful evidence를 정보밀도 높여 압축 |
| D. Full generation | `task_d_full_generation.jsonl` | Q → C* 직접 생성 (end-to-end) |

추가로 `dpo_preferences.jsonl`이 만들어져 DPO 단계에서 사용됩니다.

샘플 실행:
```bash
python -m training.prepare_training_data \
    --teacher-output data/teacher_output_real_tiny.jsonl \
    --output-dir data/train_real_tiny
```

---

## 3. SFT — Q → C* 능력 1차 학습

`training/train_sft.py`로 Student 모델(1.5B 또는 7B)을 4개 task의 혼합 데이터(`sft_combined.jsonl`)로 지도 학습합니다.

기본:
- Base: `Qwen/Qwen2.5-1.5B-Instruct` (또는 7B)
- Loss: causal LM
- LR / batch / steps는 `train_sft.py` 인자 참고

```bash
python -m training.train_sft \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --data data/train_real_tiny/sft_combined.jsonl \
    --output-dir checkpoints/sft_1.5b_run1
```

---

## 4. DPO — 선호도 기반 정렬

`training/train_dpo.py`로 `dpo_preferences.jsonl`(chosen=짧고 깔끔한 C*, rejected=장황/노이지 C*)을 사용해 정렬합니다. SFT 체크포인트를 초기 정책으로 받습니다.

```bash
python -m training.train_dpo \
    --policy-path checkpoints/sft_1.5b_run1/final \
    --pref-data data/train_real_tiny/dpo_preferences.jsonl \
    --output-dir checkpoints/dpo_1.5b_run1
```

---

## 5. GRPO (RL) — 최종 보상 신호로 미세조정

본 작업의 핵심 강화학습 단계입니다. `scripts/rl_train_icl_grpo*.py` 시리즈로 진행했고, 버전별로 보상 설계가 다릅니다.

| 버전 | 파일 | 핵심 변경점 |
|---|---|---|
| v0 (base) | `rl_train_icl_grpo.py` | SLM answer correctness를 reward로 GRPO |
| PRL 변형 | `rl_train_icl_grpo_prl.py` | PRL 스타일 reasoning 단계 reward 추가 |
| v2 | `rl_train_icl_grpo_ra_v2.py` | reward aggregation 개편 (가중 합) |
| v3 | `rl_train_icl_grpo_ra_v3.py` | structure / format reward 분리 |
| **v4** | `rl_train_icl_grpo_ra_v4.py` | **alignment reward(top-N) + cluster 기반 샘플링.** 메인 실험 |
| v5 | `rl_train_icl_grpo_ra_v5.py` | judge 비중 조정, repetition penalty 강화 |
| v5-direct | `rl_train_icl_grpo_ra_v5_direct.py` | judge 우회 (direct scoring) |

`v4_full_chain.sh`가 메인 학습 스크립트입니다:

```bash
bash v4_full_chain.sh 0   # GPU 0
```

내부 설정 요약:
- Policy: SFT된 Qwen2.5-7B (`experiment_results/math_icl_sft/qwen7b_run2/final`)
- Reward model: `Qwen2.5-1.5B-Instruct`
- Judge model: `Qwen2.5-7B-Instruct`
- Generations / step: 4, batch 1, grad-accum 4
- Max steps: 1500, LR 8e-6, β 0.02
- 보상 가중치: token 0.05 / structure 0.1 / format 0.5 / **alignment 3.0** / judge 1.0 / repetition 0.5
- Cluster dir: `slm_context_pipeline/data/math_5k_clusters` (Auto-CoT 군집 → 다양한 시드)

`scripts/debug_grpo_diversity.py`는 generation 다양성을 추적하기 위한 점검 도구입니다.

---

## 6. ICL Distill 데이터셋 빌드 (수학 도메인)

수학 task에 특화된 별도 데이터 빌드 트랙입니다.

| 스크립트 | 역할 |
|---|---|
| `scripts/build_autocot_clusters.py` | Auto-CoT 방식으로 질문을 임베딩→KMeans 군집화 |
| `scripts/select_grpo_datasets.py` | GRPO 학습용 데이터셋 선별 |
| `scripts/build_math_icl_distill_dataset.py` | 1차 (4-shot) distill 데이터 빌드 |
| `scripts/build_math_icl_distill_prl_dataset.py` | PRL 추론 단계 포함 빌드 |
| `scripts/build_math_icl_distill_prl_v2_dataset.py` | reasoning 품질 필터 추가 |
| `scripts/build_math_icl_distill_prl_v3_dataset.py` | 최종. teacher=gpt-4o-mini, samples/dataset=5000, shots=4, seed_pool=12 |
| `scripts/convert_distill_to_prl.py` | 기존 포맷 → PRL 포맷 변환 |
| `scripts/run_grpo_6_datasets.sh` | 6개 데이터셋에 대해 GRPO 일괄 실행 |

---

## 7. 평가 (Evaluation)

다운스트림 정확도와 컨텍스트 품질을 다각도로 측정합니다.

| 스크립트 | 평가 대상 |
|---|---|
| `evaluation/evaluate_downstream.py` | 학습된 student로 Q+C* → A 답변 정확도 |
| `scripts/eval_math_icl_baselines.py` | 4가지 baseline 비교 (NO-ICL / BASE-1B / BASE-7B / OUR-RL) |
| `scripts/eval_1b_from_context_pack.py` | 미리 만들어둔 context pack으로 1B 평가 |
| `scripts/eval_1b_streaming_from_two_contexts.py` | 두 컨텍스트 비교 (streaming) |
| `scripts/eval_cli_structured_outputs.py` | CLI 기반 정형 출력 검증 |
| `scripts/rejudge_100q_and_overlap_audit.py` | judge 모델로 100문항 재채점 + leakage 감사 |
| `scripts/manual_probe_10.py` | 10문항 수동 점검용 |
| `scripts/aggregate_results.py` | 결과 JSON 집계 |
| `scripts/summarize_grpo_reports.py` | GRPO 학습 로그 요약 |

baseline 정의:
- `NO-ICL`: ICL 없이 Q → A
- `BASE-1B`: vanilla Qwen2.5-1.5B-Instruct가 4개 ICL demo 생성
- `BASE-7B`: vanilla Qwen2.5-7B-Instruct가 4개 ICL demo 생성
- `OUR-RL`: GRPO로 학습된 모델이 ICL demo 생성

---

## 8. 환경 설정

| 문서 | 내용 |
|---|---|
| `MODEL_ENV_SETUP.md` | 모델/HF 캐시/토큰 환경 변수 설정 |
| `DOCKER_SETUP.md` | Docker / GPU 컨테이너 가이드 |
| `KOREAN_DATASET_GENERATION_GUIDE.md` | 한국어 데이터셋 생성 절차 |
| `PIPELINE_STAGE_CODEFLOW_DETAILED.md` | 파이프라인 단계별 코드 흐름 상세 |
| `requirements.txt` | Python 의존성 |
| `setup.py` | 패키지 설치 |

기본 환경:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...        # teacher용
export HF_TOKEN=...              # gated 모델용
```

---

## 9. End-to-end 재현 시퀀스 (요약)

```bash
# (1) 후보 + Teacher 라벨 생성
python -m data_processing.generate_candidates --questions data/questions_real_tiny.jsonl
python -m pipeline.run_teacher --questions data/questions_real_tiny.jsonl --out data/teacher_output_real_tiny.jsonl

# (2) 4-task 학습 데이터 가공
python -m training.prepare_training_data \
    --teacher-output data/teacher_output_real_tiny.jsonl \
    --output-dir data/train_real_tiny

# (3) SFT
python -m training.train_sft --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --data data/train_real_tiny/sft_combined.jsonl \
    --output-dir checkpoints/sft_1.5b

# (4) DPO
python -m training.train_dpo --policy-path checkpoints/sft_1.5b/final \
    --pref-data data/train_real_tiny/dpo_preferences.jsonl \
    --output-dir checkpoints/dpo_1.5b

# (5) GRPO (RL) — 수학 도메인 메인 실험
bash v4_full_chain.sh 0

# (6) 평가
python scripts/eval_math_icl_baselines.py --judge-model Qwen/Qwen2.5-7B-Instruct
```

> 대용량 결과/체크포인트는 git에서 제외되어 있습니다. (`experiment_results/`, `checkpoints/`, `data/run_en_v2/` 등)

---

## 10. 새 연구 라인 — ICRL-Math (`icrl_math/`)

위 1~9의 SFT+DPO+GRPO 파이프라인은 teacher distillation 비용/복잡도가 크고, 데이터 손실에 취약합니다 (실제로 한 번 잃었습니다). 이를 대체하는 **새 연구 라인** `icrl_math/` 가 추가됐습니다.

**핵심 아이디어:** ICRL (Ye, Zhao et al., 2026)의 RL-only + few-shot curriculum 방식을 거의 그대로 차용하되, 도메인을 **수학 추론 + Python 도구**, 타깃을 **Qwen2.5-3B-Instruct SLM**, demo 풀을 (향후) Auto-CoT 클러스터로 다양화.

### 흐름

```
HF raw (GSM8K / MATH)  ──>  3-shot / 2-shot / 0-shot parquet
                                       │
                                       ▼
                       GRPO 학습 (verl, ICRL infra)
                       │  rollout: <think>·<search Python>·<information stdout>·<answer>
                       │  reward: 0.8·boxed-EM + 0.2·format
                       │  loss mask: <information> 토큰 제외 (ICRL 그대로)
                       ▼
              Stage 1 (3-shot) → Stage 2 (2-shot) → Stage 3 (0-shot)
                                       │   (1-shot은 ICRL ablation 따라 건너뜀)
                                       ▼
                eval: GSM8K · MATH500 · AIME2024 · AIME2025
```

### 우리가 새로 작성한 부분 (작음)

| 파일 | 역할 |
|---|---|
| `icrl_math/example/math_examples.txt` | 5개 demo (algebra, gcd, 기하, 조합, 미분) |
| `icrl_math/scripts/data_process/math_fewshot.py` | HF GSM8K/MATH → parquet, N-shot 선택 |
| `icrl_math/sandbox/python_sandbox_server.py` | FastAPI 코드 실행기, ICRL retriever URL 자리에 plug |
| `icrl_math/verl_patches/math_fewshot.py` | boxed-EM + 포맷 페널티 reward (sympy/string-equiv 다중 fallback) |
| `icrl_math/install_into_icrl.sh` | ICRL repo에 reward 복사 + `main_ppo_fewshot._select_rm_score_fn` 1줄 패치 |
| `icrl_math/train_curriculum_math.sh` | Stage 1→2→3 (3B용 메모리 튜닝) |
| `icrl_math/scripts/eval_math.py` | vLLM 기반 standalone eval (tool loop 포함) |

### 이전 작업(섹션 1~7)과의 관계

| 구분 | 이전 (slm_context_pipeline) | 새 (icrl_math) |
|---|---|---|
| 도메인 | Q → C* context 압축 | 수학 추론 + Python tool |
| 학습 | SFT → DPO → GRPO (3단계) | **GRPO 단일 단계** (curriculum 안에서) |
| Teacher API | gpt-4o-mini 수천 콜 | **없음** (사람이 작성한 demo 5개) |
| 학습 데이터 | teacher labeled jsonl (잃어버림) | **HF raw GSM8K/MATH** (재다운로드 free) |
| 자산 재활용 | — | `scripts/build_autocot_clusters.py` (향후 cluster-adaptive demos) |

### 실행 시퀀스 (요약)

```bash
# 1. ICRL clone + 환경
git clone https://github.com/applese233/ICRL.git /home/jklee/ondevice/ICRL
conda env create -f /home/jklee/ondevice/ICRL/environment.yml && conda activate icrl

# 2. 패치 설치 (idempotent)
ICRL_DIR=/home/jklee/ondevice/ICRL bash icrl_math/install_into_icrl.sh

# 3. sandbox 띄우기
python icrl_math/sandbox/python_sandbox_server.py --port 8000 --timeout 5 &

# 4. parquet 준비
python icrl_math/scripts/data_process/math_fewshot.py --dataset math --num_examples 3 --local_dir icrl_math/data/math_3shot
python icrl_math/scripts/data_process/math_fewshot.py --dataset math --num_examples 2 --local_dir icrl_math/data/math_2shot
python icrl_math/scripts/data_process/math_fewshot.py --dataset math --template_type zeroshot --local_dir icrl_math/data/math_0shot

# 5. curriculum 학습
CUDA_VISIBLE_DEVICES=0,1,2,3 ICRL_DIR=/home/jklee/ondevice/ICRL bash icrl_math/train_curriculum_math.sh

# 6. 평가
python icrl_math/scripts/eval_math.py \
    --model-path icrl_math/checkpoints/icrl-math-stage3-0shot-qwen2.5-3b/actor/global_step_100/hf \
    --datasets gsm8k math500 aime2024 aime2025 \
    --num-shots 0
```

자세한 설명은 [`icrl_math/README.md`](icrl_math/README.md).
