# SLM Context Pipeline 상세 코드 플로우 (Planner / Evidence Builder / Judge 중심)

## 문서 목적
이 문서는 현재 운영 중인 파이프라인에서,
- 단계별로 코드가 어떤 순서로 실행되는지,
- 각 단계가 어떤 입력을 받아 어떤 객체/필드를 출력하는지,
- `config/settings.yaml`의 주요 옵션이 실제 어디에 연결되는지
를 **코드 레벨로 상세히 설명**한다.

범위:
- `pipeline/run_teacher.py`
- `pipeline/planner.py`
- `pipeline/evidence_builder.py`
- `pipeline/judge.py`
- `config/settings.yaml`
- (연결 포인트) `data_processing/generate_candidates.py`, `evaluation/evaluate_downstream.py`

---

## 0) 왜 이 구조로 설계했는가 (설계 의도)

이 파이프라인은 단순히 “정답 생성”이 아니라, **SLM이 잘 풀 수 있도록 최소 충분 컨텍스트(C*)를 안정적으로 생산**하는 것이 목표다.

핵심 설계 의도:
1. **역할 분리(Planner → Evidence → Judge)**
  - 한 번에 모든 걸 생성하면 품질 오류 원인 추적이 어렵다.
  - 단계별 책임을 분리해 디버깅/개선 포인트를 명확히 한다.
2. **구조화 출력(JSON) 우선**
  - 자유 텍스트보다 후처리·검증·재학습 데이터화가 쉽다.
  - 질문 유형/제약/근거/최종 컨텍스트를 필드 단위로 관리할 수 있다.
3. **운영 안정성 우선(JSONL append + resume)**
  - 대규모 장시간 실행 중 장애/중단이 발생해도 부분 결과를 보존한다.
  - 재시작 시 이미 처리한 질문을 건너뛰어 비용을 줄인다.
4. **품질 평가를 생성과 분리**
  - teacher가 만든 후보를 downstream utility로 다시 평가해 “실제로 SLM에 도움되는지”를 검증한다.
  - LLM 선호만으로 라벨링하지 않고, 학생 모델 성능 기준으로 필터링한다.

대안 대비 선택 이유:
- 단일 거대 프롬프트 방식은 구현은 쉽지만, 실패 원인 분석과 부분 개선이 어렵다.
- 반면 현재 3단계 방식은 호출 수가 늘어 비용이 증가할 수 있으나, 품질 통제와 재현성이 높다.

---

## 1) 실행 진입점과 전체 호출 체인

## 1.1 CLI 진입점
파일: `pipeline/run_teacher.py`

`main()`이 수행하는 순서:
1. argparse로 `--input`, `--output`, `--config`, `--no-resume` 파싱
2. `load_yaml(args.config)`로 설정 로드
3. `create_llm_client(config.get("teacher", {}))`로 teacher LLM 클라이언트 생성
4. `TeacherPipeline(llm_client, config)` 생성
5. `pipeline.process_batch(...)` 실행

핵심 포인트:
- teacher 모델/temperature/max_tokens는 여기서 만든 `llm_client`를 통해 3단계 모두에 공통 사용된다.
- 질문 단위 결과는 JSONL로 append 저장되어 장시간 실행/중단/재시작(resume)에 유리하다.

왜 이렇게 하는가:
- 실험/운영 환경에서 가장 빈번한 실패는 네트워크/API/프로세스 중단이다.
- 질문 단위 append는 전체 배치를 원자적으로 재시작하지 않아도 되어 비용과 시간을 줄인다.

## 1.2 배치 처리 체인
`TeacherPipeline.process_batch()` 내부 흐름:
1. 출력 디렉토리 생성 (`ensure_dir(output_path.parent)`)
2. resume 모드면 기존 출력 JSONL을 읽어 `processed_ids` 집합 구성
3. 입력 JSONL을 `iter_jsonl`로 순회
4. 각 레코드마다 `process(question_id, question, answer)` 호출
5. 결과를 `append_jsonl`로 즉시 저장
6. stats(`success/failed/skipped`) 누적

질문 ID 결정 규칙:
- `id` 우선
- 없으면 `question_id`
- 없으면 현재 인덱스 문자열

질문/정답 필드 fallback:
- question: `question` → `text`
- answer: `answer` → `ground_truth`

왜 fallback이 필요한가:
- 데이터셋 소스(BoolQ/PIQA/GSM8K 등)마다 필드명이 다를 수 있다.
- ingestion 단계에서 스키마 차이를 흡수해 파이프라인 코어를 단순하게 유지한다.

---

## 2) 데이터 스키마 관점의 단계 연결

파일: `models/schemas.py`

단계별 주 객체:
- Planner 출력: `PlannerOutput`
  - `question_type`, `need_external_context`, `entities`, `constraints`, `subquestions`, `retrieval_queries`
- Evidence 단위: `Evidence`
  - `source`, `content`, `relevance_score`, `is_distractor`
- Judge 출력: `JudgeOutput`
  - `selected_facts`, `rejected_facts`, `information_density_score`, `has_answer_leakage`, `distractor_ratio`, `final_context`
- 최종 구조화 컨텍스트: `StructuredContext`
  - `need_context`, `question_type`, `entities`, `constraints`, `subquestions`, `useful_facts`, `missing_info`, `answer_hint`

핵심 타입 변환:
- 문자열 question_type은 `QuestionType` enum으로 변환 (`from_dict`, parse 단계)
- 변환 실패 시 기본값(`factoid`)로 fallback

---

## 2.5) 전체 구조 순서도 (중간 요약)

아래 다이어그램은 실제 코드 실행 경로와 산출물 흐름을 한 번에 보여준다.

```text
┌──────────────────────────────────────────────────────────────┐
│ CLI: run_teacher.py main                                     │
│ - load_yaml(settings)                                        │
│ - create_llm_client(teacher)                                 │
│ - TeacherPipeline.process_batch()                            │
└───────────────┬──────────────────────────────────────────────┘
                │
                │ (입력 레코드에서 question/answer/id 정규화)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 1: Planner                                             │
│ 질문 타입 분류 + entities/constraints/subquestions 추출         │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (산출물: planner_output)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 2: Evidence Builder                                    │
│ retrieved + generated evidence 하이브리드 (현재 retrieval off) │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (산출물: evidences + pseudo_document)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 3: Judge                                               │
│ 정보밀도 / distractor / leakage 기준 최소 충분 컨텍스트 선별      │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (산출물: final_context + judge_scores)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ Teacher Output JSONL                                         │
│ question_id, planner_output, evidence_output, final_context  │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (후처리 단계로 전달)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ Context Candidates Generator                                 │
│ C_pos / C_long / C_noisy / C_null / C_leaky 생성              │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (후보별 답변 성능 차이를 비교하기 위한 입력 준비)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ Downstream Evaluator                                         │
│ answer model(or student_hf)로 utility 측정 + preference 생성   │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (최종 학습/분석 아티팩트 저장)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ Outputs                                                      │
│ evaluated.jsonl / *_preferences.jsonl / *_helpful_contexts   │
│ *_summary.json                                               │
└──────────────────────────────────────────────────────────────┘
```

읽는 방법:
- 위쪽은 teacher 3-stage 생성 파트, 아래쪽은 후보 생성/유틸리티 평가 파트다.
- `settings.yaml`은 클라이언트 생성, 분기(`use_retrieval`, `use_llm_variants`, `evaluation.backend`) 지점에서 동작을 바꾼다.

---

## 3) Stage 1: Planner 상세

파일: `pipeline/planner.py`

## 3.1 초기화
`Planner.__init__(llm_client, config)`:
- `max_entities`, `max_constraints`, `max_subquestions`를 config에서 로드
- 미설정 시 기본값 사용

현재 설정 연결:
- `pipeline.planner.max_entities`
- `pipeline.planner.max_constraints`
- `pipeline.planner.max_subquestions`

## 3.2 핵심 실행 메서드
`analyze(question)`:
1. few-shot + 현재 질문을 합쳐 prompt 구성
2. `self.llm.generate_json(prompt, PLANNER_SYSTEM_PROMPT)` 호출
3. 결과를 `_parse_result`로 정규화
4. 예외 발생 시 `_default_output(question)` 반환

왜 Planner를 분리하는가:
- 하위 단계에서 필요한 정보(엔티티/제약/서브질문)를 미리 구조화하면,
  Evidence/Judge 단계 프롬프트가 짧아지고 초점이 선명해진다.
- 특히 multi-hop에서 서브질문 분해가 없으면 근거 생성이 장황해지거나 누락될 가능성이 커진다.

## 3.3 `_parse_result` 동작
- `question_type` 문자열을 enum으로 캐스팅
- `entities/constraints/subquestions`는 길이 제한(max_*) 적용
- 결과를 `PlannerOutput`으로 반환

## 3.4 `_default_output` 동작
실패 시 안전 기본값:
- `question_type=factoid`
- `need_external_context=True`
- `retrieval_queries=[원문 질문]`

왜 보수적 기본값인가:
- 실패 시 `need_external_context=False`로 두면 필요한 정보가 누락될 위험이 크다.
- 기본값을 보수적으로 잡아 “모르는 경우라도 조사 가능한 상태”를 유지한다.

## 3.5 언어 제약
`PLANNER_SYSTEM_PROMPT`에 영어-only 제약이 포함되어,
`entities/constraints/subquestions/retrieval_queries`를 영어로만 생성하도록 요구한다.

## 3.6 Planner 내부 순서도 (중간 설명 포함)
```text
┌──────────────────────────────────────────────────────────────┐
│ Planner.analyze(question) 호출                                │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (few-shot + 현재 question으로 prompt 구성)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ llm.generate_json(prompt, PLANNER_SYSTEM_PROMPT)             │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (원시 JSON 결과를 안전 파싱 단계로 전달)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ _parse_result(result)                                        │
│ - question_type enum 캐스팅                                   │
│ - entities/constraints/subquestions 길이 제한                 │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (파싱 실패 시 보수적 fallback 적용)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ _default_output(question)                                    │
│ - factoid / need_external_context=True                       │
│ - retrieval_queries=[question]                               │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (최종 PlannerOutput 반환)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ PlannerOutput                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 4) Stage 2: Evidence Builder 상세

파일: `pipeline/evidence_builder.py`

## 4.1 초기화
`EvidenceBuilder.__init__(llm_client, retriever, config)`:
- `use_retrieval`, `use_generation`, `max_evidences`, `pseudo_doc_length` 로드
- retriever 미지정 시 `DummyRetriever` 사용

현재 설정 연결:
- `pipeline.evidence_builder.use_retrieval: false`
- `pipeline.evidence_builder.use_generation: true`
- `pipeline.evidence_builder.max_evidences`
- `pipeline.evidence_builder.pseudo_doc_length`

## 4.2 `build(question, planner_output)` 실행 순서
1. `retrieved_evidence` 초기화
2. 조건: `use_retrieval=True` 그리고 `planner_output.need_external_context=True`
   - 쿼리(`planner_output.retrieval_queries` 또는 원문 질문)로 retriever 호출
3. 조건: `use_generation=True`
   - `_generate_evidence(...)` 호출
   - `generated_evidence`, `pseudo_document` 생성
4. `_combine_evidence(retrieved, generated)`로 병합/중복제거/정렬
5. `EvidenceBuilderOutput` 반환

현재 설정상 실제 동작:
- retrieval이 꺼져 있으므로 2번 단계는 스킵되고,
- 생성 근거 중심으로 동작한다.

왜 retrieval을 끈 상태로 운영하는가:
- 현재 파이프라인의 목표가 retrieval 시스템 품질 평가가 아니라, context distillation 데이터 생산에 있기 때문이다.
- 검색 인프라/인덱스 품질 변수를 제거해 teacher 출력 품질과 평가 로직 검증에 집중한다.

## 4.3 `_generate_evidence` 동작
1. few-shot + question + planner_output dict를 prompt로 구성
2. `self.llm.generate_json(prompt, EVIDENCE_GENERATION_SYSTEM_PROMPT)` 호출
3. `generated_evidence` 배열 각 항목을 `Evidence(source="generated", ...)`로 변환
4. `max_evidences`로 컷
5. `pseudo_document`는 길이 제한 적용(`pseudo_doc_length * 2`)
6. 실패 시 빈 결과 반환

## 4.4 `_combine_evidence` 동작
중복 제거 키:
- `content.lower().strip()[:100]`

병합 규칙:
1. retrieved 먼저 삽입(우선순위)
2. generated 추가
3. `relevance_score` 내림차순 정렬
4. `max_evidences`로 최종 컷

왜 이 단순 결합 전략을 쓰는가:
- 초기 운영 단계에서는 고정 규칙이 디버깅/재현성 측면에서 유리하다.
- 복잡한 semantic dedup은 정확도가 올라갈 수 있으나, 비용/지연/실패 모드가 증가한다.

## 4.5 언어 제약
`EVIDENCE_GENERATION_SYSTEM_PROMPT`에 영어-only 제약 포함.
`generated_evidence[].content`와 `pseudo_document` 영어 출력을 강제한다.

## 4.6 Evidence Builder 내부 순서도 (중간 설명 포함)
```text
┌──────────────────────────────────────────────────────────────┐
│ EvidenceBuilder.build(question, planner_output)              │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (분기 1: use_retrieval && need_external_context)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ Retrieval branch                                             │
│ - retrieval_queries 또는 question으로 retriever 호출           │
│ - retrieved_evidence 생성                                     │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (분기 2: use_generation=True)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ _generate_evidence(question, planner_output)                 │
│ - generated_evidence / pseudo_document 생성                   │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (retrieved + generated 병합)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ _combine_evidence(...)                                       │
│ - dedup / 정렬 / max_evidences 컷                             │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (EvidenceBuilderOutput 포맷으로 반환)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ EvidenceBuilderOutput                                        │
│ retrieved_evidence, generated_evidence, combined_evidence,   │
│ pseudo_document                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 5) Stage 3: Judge 상세

파일: `pipeline/judge.py`

## 5.1 초기화
`Judge.__init__(llm_client, config)`:
- `min_information_density`
- `max_distractor_ratio`
- `leakage_detection`

현재 설정 연결:
- `pipeline.judge.min_information_density: 0.3`
- `pipeline.judge.max_distractor_ratio: 0.7`
- `pipeline.judge.leakage_detection: true`

## 5.2 `judge(question, planner_output, evidence_output)` 실행 순서
1. `_format_evidence(evidence_output)`로 evidence를 문자열화
   - `combined_evidence`를 `- "..."` 리스트로 생성
   - `pseudo_document`가 있으면 별도 블록 추가
2. few-shot + question + planner_output + evidence를 합쳐 prompt 구성
3. `self.llm.generate_json(prompt, JUDGE_SYSTEM_PROMPT)` 호출
4. `_parse_result(result, planner_output)`로 정규화
5. 실패 시 `_default_output(planner_output, evidence_output)` 반환

왜 Judge를 마지막에 두는가:
- Evidence 단계는 “가능한 근거를 넓게” 수집하는 역할이고,
- Judge 단계는 “정답에 실제로 기여하는 최소 집합”으로 압축하는 역할이기 때문이다.
- 이 분리를 통해 recall(근거 포착)과 precision(불필요 정보 제거)을 단계적으로 달성한다.

## 5.3 `_parse_result` 동작
- `final_context.question_type`를 enum으로 캐스팅
- 실패 시 planner의 question_type 사용
- `StructuredContext` 구성 시 planner 값을 fallback으로 사용
- `JudgeOutput` 객체로 반환

## 5.4 `_default_output` 동작
- `combined_evidence` 상위 5개를 `useful_facts`로 사용
- leakage/distractor를 보수적 기본값으로 설정

## 5.5 `passes_quality_threshold` 규칙
아래 중 하나라도 위배되면 False:
- `information_density_score < min_information_density`
- `distractor_ratio > max_distractor_ratio`
- `leakage_detection=True`이고 `has_answer_leakage=True`

참고:
- 현재 `run_teacher.py`에서는 이 메서드로 필터링하지 않고, judge 결과를 그대로 저장한다.
- 즉, threshold는 “판정 함수”로 제공되지만 저장 단계에서 하드 필터는 적용되지 않는다.

왜 하드 필터를 바로 적용하지 않는가:
- 운영 중에는 borderline 케이스를 포함해 데이터를 보존해야, 이후 규칙 조정/재평가가 가능하다.
- 초기에 과도한 필터링을 하면 유용한 학습 신호까지 손실될 수 있다.

## 5.6 언어 제약
`JUDGE_SYSTEM_PROMPT`에 영어-only 제약 포함.
`selected_facts/rejected_facts/rejection_reasons/final_context` 전반을 영어로 요구한다.

## 5.7 Judge 내부 순서도 (중간 설명 포함)
```text
┌──────────────────────────────────────────────────────────────┐
│ Judge.judge(question, planner_output, evidence_output)       │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (evidence를 prompt 주입 가능한 문자열로 변환)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ _format_evidence(evidence_output)                            │
│ - combined_evidence bullet화                                 │
│ - pseudo_document 블록 결합                                   │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (few-shot + question/planner/evidence 합성)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ llm.generate_json(prompt, JUDGE_SYSTEM_PROMPT)               │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (원시 JSON을 스키마로 정규화)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ _parse_result(result, planner_output)                        │
│ - StructuredContext 구성                                      │
│ - JudgeOutput 생성                                            │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (실패 시 fallback 경로)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ _default_output(planner_output, evidence_output)             │
└───────────────────────────────┬──────────────────────────────┘
                │
                │ (최종 JudgeOutput 반환)
                ▼
┌──────────────────────────────────────────────────────────────┐
│ JudgeOutput                                                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 6) settings.yaml 주요 항목과 실제 코드 연결

## 6.1 Teacher 모델 설정
```yaml
teacher:
  model: gpt-4o-mini
  api_base: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
  temperature: 0.2
  max_tokens: 1536
```

실제 반영 지점:
- `run_teacher.py` → `create_llm_client(config.get("teacher", {}))`
- `utils/llm_client.py`의 `create_llm_client`
  - model 문자열에 `gpt` 포함 시 `OpenAIClient` 선택
  - `temperature`, `max_tokens`, `api_base`, `api_key` 전달

결과:
- Planner/Evidence/Judge의 `generate_json` 호출이 모두 해당 teacher 설정을 공유한다.

## 6.2 Retrieval 비활성
```yaml
pipeline:
  evidence_builder:
    use_retrieval: false
```

실제 반영 지점:
- `EvidenceBuilder.build()`에서 retrieval 분기 미실행
- retrieved_evidence는 사실상 빈 리스트 유지

## 6.3 Candidate generation 변동성/비용 제어
```yaml
candidate_generation:
  use_llm_variants: false
```

실제 반영 지점:
- `data_processing/generate_candidates.py`
- `_generate_verbose/_generate_noisy/_generate_leaky`에서
  - true: LLM 호출
  - false: 템플릿/규칙 기반 deterministic 텍스트 생성

결과:
- 후보 컨텍스트 생성 비용 절감
- 실행 간 변동성 감소

왜 `use_llm_variants=false`가 중요한가:
- 후보 생성 단계까지 LLM 호출을 확장하면 비용이 급증하고 실험 재현성이 낮아진다.
- 현재는 teacher 산출물 검증과 평가 루프 안정화가 우선이므로 deterministic 변형이 실무적으로 유리하다.

## 6.4 Evaluation backend
```yaml
evaluation:
  backend: student_hf
student:
  model_name: Qwen/Qwen2.5-1.5B
```

실제 반영 지점:
- `evaluation/evaluate_downstream.py`의 `create_answer_function`
- backend가 `student_hf`면 `LocalSLMResponder`를 통해 HF 모델 로컬 추론 사용
- 아니면 `answer_model` API 클라이언트 사용

결과:
- 현재는 API 답변 모델이 아니라 로컬 student SLM 추론으로 utility를 산출한다.

왜 `student_hf` backend를 쓰는가:
- 최종 목표가 SLM 학습 데이터 품질 향상이므로, utility도 학생 모델 기준으로 측정하는 것이 정합적이다.
- teacher/answer API 기준 점수만 보면 실제 타깃 모델 개선과 괴리가 생길 수 있다.

---

## 7) Stage 결과가 다음 단계로 전달되는 필드 체인

질문 1건 기준 데이터 전달:
1. Planner
   - 출력: `question_type`, `entities`, `constraints`, `subquestions`, `retrieval_queries`
2. Evidence Builder
   - 입력: Planner 출력
   - 출력: `generated_evidence`, `pseudo_document`, `combined_evidence`
3. Judge
   - 입력: Planner + Evidence 출력
   - 출력: `selected_facts`, `rejected_facts`, `final_context`
4. Teacher Output JSONL
   - 위 3단계 결과를 한 레코드로 저장

후속 단계 연결:
- `generate_candidates.py`는 teacher의 `final_context`를 기반으로 5종 컨텍스트 생성
- `evaluate_downstream.py`는 후보별 utility를 계산하고 preference/helpful-context를 추출

---

## 8) 한글 데이터셋 관점에서의 해석 포인트

이미 생성된 한글 레코드에서는 보통 다음과 같은 전파가 보인다.
1. planner의 문자열 필드에 한글이 섞임
2. evidence 생성 시 해당 표현이 재사용/확장됨
3. judge의 `answer_hint/useful_facts`까지 한글이 이어짐

즉, 다단계 구조에서는 상위 단계 언어가 하위 단계 입력 프롬프트에 포함되므로,
언어 드리프트가 누적될 수 있다.

현재 코드는 각 단계 시스템 프롬프트에 영어-only 제약을 넣어 이 누적 전파를 차단하는 방식으로 운영 중이다.

왜 단계별로 모두 언어 제약을 넣는가:
- 한 단계만 강제하면 상/하위 단계에서 다시 언어 혼입이 발생할 수 있다.
- 다단계 파이프라인에서는 제약을 중복 적용해야 언어 일관성이 안정화된다.

---

## 9) 운영 체크리스트 (실행 전/후)

실행 전:
1. `config/settings.yaml`의 teacher/temperature 확인
2. `pipeline.evidence_builder.use_retrieval` 의도 확인
3. `candidate_generation.use_llm_variants` 비용 정책 확인
4. `evaluation.backend`가 기대값(`student_hf`)인지 확인

실행 중:
1. teacher output JSONL이 append로 누적되는지 확인
2. 실패 레코드(`success=false`) 비율 확인

실행 후:
1. 샘플 레코드에서 `planner/evidence/judge/final_context` 필드 완전성 점검
2. 언어 혼입(한글/영문) 스캔
3. `evaluate_downstream` 결과의 helpful-context 및 summary 확인

---

## 10) 요약
현재 파이프라인은 `run_teacher.py` 기준으로
- Planner(질문 구조화) → Evidence Builder(근거 생성/결합) → Judge(최소 충분 컨텍스트 선별)
을 순차 수행하고,
- 설정 파일의 옵션은 각 클래스 초기화와 분기 로직에서 직접 반영된다.

특히 현재 설정에서는
- teacher: `gpt-4o-mini`, temperature `0.2`
- retrieval 비활성
- 후보 변형 LLM 비활성(`use_llm_variants=false`)
- utility 평가는 `student_hf` (`Qwen/Qwen2.5-1.5B`)
로 동작한다.

설계 관점 한 줄 요약:
- 이 구조는 "생성"보다 "검증 가능한 컨텍스트 생산"을 우선하는 아키텍처이며,
  단계 분리/구조화 출력/재시작 가능성/학생모델 기준 평가를 통해 운영 안정성과 품질 개선 루프를 동시에 확보한다.
