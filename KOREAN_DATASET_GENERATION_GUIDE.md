# 한글 데이터셋 기반 생성 파이프라인 설명서

## 1) 문서 목적
이 문서는 **이미 생성된 한글 포함 데이터셋**을 기준으로,
- 어떤 구조의 데이터가 만들어졌는지,
- 실제 파이프라인이 어떤 순서/설정으로 생성하는지,
- 왜 한글이 섞였는지와 현재 운영 방식이 무엇인지
를 설명하기 위한 기술 문서다.

---

## 2) 한글 데이터셋 현황 (감사 결과)
기준 파일: `data/korean_audit/summary.json`

- `teacher_output_real_max`
  - total: **628**
  - Korean 포함: **387**
  - Non-Korean: **241**
- `teacher_output_real_tiny`
  - total: **6**
  - Korean 포함: **0**
  - Non-Korean: **6**

분리 저장 파일:
- `data/korean_audit/teacher_output_real_max_korean.jsonl`
- `data/korean_audit/teacher_output_real_max_non_korean.jsonl`
- `data/korean_audit/teacher_output_real_tiny_korean.jsonl`
- `data/korean_audit/teacher_output_real_tiny_non_korean.jsonl`

해석 포인트:
- 대규모 실행(`real_max`)에서는 한글 혼입이 높았고,
- 소규모 검증(`real_tiny`)에서는 혼입이 관찰되지 않았다.
- 즉, 동일 코드라도 프롬프트/샘플/런 조건 영향으로 출력 언어 안정성이 달라질 수 있다.

---

## 3) 한 레코드가 만들어지는 방식 (실제 산출 구조)
각 레코드는 질문 하나를 입력으로 받아 아래 필드를 생성한다.

- 입력/메타
  - `question_id`, `question`, `ground_truth_answer`
- Stage 1 출력
  - `planner_output`
- Stage 2 출력
  - `evidence_output`
- Stage 3 출력
  - `judge_output`, `final_context`
- 실행 상태
  - `success`, `error_message`

핵심은 **질문 → 구조화 분석 → 근거 생성/결합 → 최소 충분 컨텍스트 선별** 흐름이 JSON 단위로 누적된다는 점이다.

---

## 4) 현재 데이터 생성 파이프라인 (운영 기준)
엔트리포인트: `pipeline/run_teacher.py`

### 단계별 처리
1. **Planner (`pipeline/planner.py`)**
   - 질문 타입 분류(`factoid/comparison/multi-hop` 등)
   - 엔티티/제약/서브질문/검색쿼리 추출

2. **Evidence Builder (`pipeline/evidence_builder.py`)**
   - 현재 설정상 retrieval 비활성(`use_retrieval: false`)
   - 생성 근거 중심(`use_generation: true`)
   - `generated_evidence`, `pseudo_document`, `combined_evidence` 생성

3. **Judge (`pipeline/judge.py`)**
   - 정보밀도/누설/잡음 기준으로 선별
   - `selected_facts`, `rejected_facts`, `final_context` 생성

### 현재 주요 설정 (`config/settings.yaml`)
- Teacher 모델: `gpt-4o-mini`
- temperature: `0.2`
- retrieval: `disabled`
- candidate generation: `use_llm_variants: false` (비용/변동성 완화)
- evaluation backend: `student_hf` (`Qwen/Qwen2.5-1.5B`)

즉, 현재 운영은
- teacher가 컨텍스트를 만들고,
- downstream utility 평가를 student(HF)로 수행하는 **teacher-student 분리 구조**다.

---

## 5) 왜 한글이 생성되었는가 (한글 데이터셋 기준 설명 포인트)
한글 혼입 레코드를 보면 다음 패턴이 반복된다.

- `planner_output.subquestions` 또는 `entities`가 한국어
- `evidence_output.generated_evidence[].content`에 한국어 문장 다수
- `judge_output.final_context.answer_hint/useful_facts`에 한국어 포함

실무적으로 설명하면,
- 모델 자체는 다국어 생성이 가능하고,
- 프롬프트/예시/문맥 중 한국어 신호가 있으면 해당 언어로 수렴할 수 있다.
- 특히 다단계 파이프라인에서는 상위 단계 출력 언어가 하위 단계 입력으로 전달되어,
  **언어 드리프트가 연쇄적으로 증폭**된다.

---

## 6) 현재는 어떻게 생성하고 있는가 (재발 방지 기준)
현재 코드 기준으로 planner/evidence/judge 시스템 프롬프트에 공통적으로
**"영어만 출력" 규칙**이 명시되어 있다.

적용 위치:
- `pipeline/planner.py` → `PLANNER_SYSTEM_PROMPT`
- `pipeline/evidence_builder.py` → `EVIDENCE_GENERATION_SYSTEM_PROMPT`
- `pipeline/judge.py` → `JUDGE_SYSTEM_PROMPT`

운영 해석:
- 언어 제약을 각 단계에 중복 적용해 상위/하위 단계 모두에서 영어를 강제한다.
- 기존 오염 데이터는 `data/korean_audit/`로 분리 보관하고,
- 신규 생성은 별도 clean 경로에서 다시 수행한다.

---

## 7) 설명 시 바로 쓰기 좋은 요약 문장
- "이 데이터셋은 질문별로 Planner→Evidence→Judge 3단계를 거쳐 최소 충분 컨텍스트를 만든다."
- "한글 데이터셋은 대규모 run에서 언어 제약이 약했을 때 생긴 산출물이며, 현재는 단계별 영어 강제 프롬프트로 통제한다."
- "기존 산출물은 Korean/Non-Korean으로 분리 보관했고, clean run은 별도 폴더에서 재생성 중이다."
- "최종적으로는 student utility 기반 평가를 붙여, 단순 선호가 아니라 실제 도움이 되는 컨텍스트만 남긴다."

---

## 8) 참고 파일
- `data/korean_audit/summary.json`
- `data/korean_audit/teacher_output_real_max_korean.jsonl`
- `pipeline/run_teacher.py`
- `pipeline/planner.py`
- `pipeline/evidence_builder.py`
- `pipeline/judge.py`
- `config/settings.yaml`
