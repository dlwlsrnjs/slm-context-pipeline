# SLM Minimal Sufficient Context Pipeline

> SLM이 배워야 하는 것은 지식을 길게 말하는 능력이 아니라,
> 질문에 맞는 정보만 짧게 남기는 능력이다.

## 핵심 철학

학습 목표를 `Q → A`가 아니라 `Q → C*`로 재정의합니다.

- **C*** = 최소 충분 컨텍스트 (Minimal Sufficient Context)
- 길고 똑똑한 배경설명이 아니라, **짧고 구조화되어 있고 노이즈가 적은** 작업용 입력

> 실험을 어떻게 진행했는지 한 문서로 보고 싶다면 → **[EXPERIMENTS.md](EXPERIMENTS.md)** (데이터 빌드 → SFT → DPO → GRPO(RL) → 평가 5단계 + RL v1~v5 비교 + end-to-end 재현 시퀀스)
>
> 새로 시작한 RL-only 라인은 → **[icrl_math/README.md](icrl_math/README.md)** (ICRL을 Qwen2.5-3B 수학 추론에 그대로 적용, teacher distill 비용 0)

## 아키텍처

```
Question
    │
    ▼
┌─────────────────────────────────┐
│  Stage 1: Planner               │  질문 해부 (유형, 엔티티, 제약, 서브질문)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Stage 2: Evidence Builder      │  retrieved + generated evidence 하이브리드
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Stage 3: Judge                 │  utility 기반 필터링 (정보밀도, distractor, leakage)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Context Candidates Generator   │  C_pos / C_long / C_noisy / C_null / C_leaky
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Downstream Evaluator           │  answer model로 성능 측정 → preference label
└──────────────┬──────────────────┘
               │
               ▼
        Training Data
   (SFT → DPO → Utility Reranking)
```

## 프로젝트 구조

```
slm_context_pipeline/
├── config/
│   ├── settings.example.yaml    # 설정 예제
│   └── settings.yaml            # 실제 설정 (직접 생성)
├── data/
│   └── questions.jsonl          # 입력 질문 데이터
├── pipeline/
│   ├── planner.py               # Stage 1: 질문 분석
│   ├── evidence_builder.py      # Stage 2: 증거 수집/생성
│   ├── judge.py                 # Stage 3: 품질 판정
│   └── run_teacher.py           # 전체 파이프라인 실행
├── data_processing/
│   └── generate_candidates.py   # 5종 컨텍스트 후보 생성
├── evaluation/
│   └── evaluate_downstream.py   # 다운스트림 성능 평가
├── training/
│   ├── prepare_training_data.py # 학습 데이터 준비
│   ├── train_sft.py             # SFT 학습
│   └── train_dpo.py             # DPO 학습
├── models/
│   └── schemas.py               # 데이터 스키마 정의
├── utils/
│   ├── llm_client.py            # LLM API 클라이언트
│   └── file_utils.py            # 파일 유틸리티
└── requirements.txt
```

## 4개 하위 태스크

| Task | 설명 | 목적 |
|------|------|------|
| **A. Context Necessity** | 외부 컨텍스트 필요 여부 분류 | 불필요한 context 생성 방지 |
| **B. Extraction** | 핵심 엔티티/제약조건 추출 | 질문 이해력 학습 |
| **C. Compression** | 증거 → 작업용 메모 압축 | 정보 밀도 최적화 |
| **D. Full Generation** | 전체 구조화 컨텍스트 생성 | 통합 능력 학습 |

## 빠른 시작

### 1. 설치

```bash
cd slm_context_pipeline
pip install -r requirements.txt

# 설정 파일 복사
cp config/settings.example.yaml config/settings.yaml
# settings.yaml에서 API 키 환경변수 설정
```

### 2. 환경변수 설정

```bash
export OPENAI_API_KEY="your-openai-api-key"
# 또는
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 3. 파이프라인 실행

```bash
# 1. Teacher 파이프라인으로 데이터 생성
python -m pipeline.run_teacher \
    --input data/questions.jsonl \
    --output data/teacher_output.jsonl \
    --config config/settings.yaml

# 2. 5종 컨텍스트 후보 생성
python -m data_processing.generate_candidates \
    --input data/teacher_output.jsonl \
    --output data/candidates.jsonl

# 3. 다운스트림 평가 → preference label
python -m evaluation.evaluate_downstream \
    --input data/candidates.jsonl \
    --output data/labeled.jsonl

# 4. 학습 데이터 구성
python -m training.prepare_training_data \
    --teacher-output data/teacher_output.jsonl \
    --labeled data/labeled_preferences.jsonl \
    --output-dir data/train/

# 5. SLM 학습
python -m training.train_sft --config config/settings.yaml
python -m training.train_dpo --config config/settings.yaml
```

## 구조화된 컨텍스트 출력 형식

```json
{
  "need_context": true,
  "question_type": "comparison",
  "entities": ["Tesla", "BYD"],
  "constraints": ["time=2023", "metric=전기차 판매량"],
  "subquestions": [],
  "useful_facts": [
    "BYD 2023년 판매량: 약 300만대",
    "Tesla 2023년 판매량: 약 180만대"
  ],
  "missing_info": [],
  "answer_hint": "두 회사의 2023년 판매량 수치를 비교하라"
}
```

## 5종 컨텍스트 후보

| Type | 설명 | 용도 |
|------|------|------|
| **C_pos** | 압축된 핵심 컨텍스트 | Positive sample |
| **C_long** | 장황한 설명형 | 비효율성 학습 |
| **C_noisy** | 관련·비관련 혼합 | Distractor 인식 학습 |
| **C_null** | 컨텍스트 없음 | 필요성 판단 학습 |
| **C_leaky** | 답 누설 컨텍스트 | Leakage 탐지 학습 |

## 관련 연구

이 파이프라인은 다음 연구들의 아이디어를 통합합니다:

- **Self-RAG**: 적응형 retrieval 판단
- **RECOMP / LongLLMLingua**: 질문 의존적 압축
- **Query2doc / GenRead**: LLM 기반 pseudo-document 생성
- **COMBO**: Retrieved + Generated knowledge 결합
- **RAFT / DRAG**: Distractor 무시 능력의 distillation

## 핵심 원칙

1. **Teacher가 써야 하는 것은 essay가 아니라 working memory**
2. **"LLM이 좋아하는 컨텍스트"가 아니라 "다운스트림 성능 차이"로 라벨링**
3. **분해학습 후 통합** (4개 태스크 → Full generation)
4. **Generated context만 쓰지 말고, retrieval-aware synthetic data로 구성**

## License

MIT
