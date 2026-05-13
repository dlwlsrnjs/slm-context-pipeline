# 모델별 분리 가상환경 실행 가이드 (3개 모델 전용)

이 설정은 아래 3개 모델만 대상으로, 모델별 독립 가상환경과 모델별 실행 파이썬 파일을 분리합니다.

- Llama-3.2-1B
- GPT-Neo-1.3B
- Phi-1.5

## 추가된 파일

### 모델 설정
- `model_configs/models_llama32_1b.json`
- `model_configs/models_gpt_neo_1_3b.json`
- `model_configs/models_phi_1_5.json`
- `model_configs/models_edge_trio.json`

### 모델별 실행 Python
- `scripts/run_llama32_1b.py`
- `scripts/run_gpt_neo_1_3b.py`
- `scripts/run_phi_1_5.py`

### 모델별 venv/실행 스크립트
- `scripts/setup_model_envs.sh`
- `scripts/run_llama32_1b.sh`
- `scripts/run_gpt_neo_1_3b.sh`
- `scripts/run_phi_1_5.sh`

## 1) 모델별 가상환경 생성 (한 번만)

```bash
cd /home/jklee/ondevice/SLM-Bench
chmod +x scripts/setup_model_envs.sh
./scripts/setup_model_envs.sh
```

생성되는 환경:
- `.venv_llama32_1b`
- `.venv_gpt_neo_1_3b`
- `.venv_phi_1_5`

## 2) 모델별 개별 실행

### Llama-3.2-1B
```bash
cd /home/jklee/ondevice/SLM-Bench
chmod +x scripts/run_llama32_1b.sh
./scripts/run_llama32_1b.sh --email you@example.com --max-datasets 2
```

### GPT-Neo-1.3B
```bash
cd /home/jklee/ondevice/SLM-Bench
chmod +x scripts/run_gpt_neo_1_3b.sh
./scripts/run_gpt_neo_1_3b.sh --email you@example.com --max-datasets 2
```

### Phi-1.5
```bash
cd /home/jklee/ondevice/SLM-Bench
chmod +x scripts/run_phi_1_5.sh
./scripts/run_phi_1_5.sh --email you@example.com --max-datasets 2
```

## 3) 직접 가상환경 활성화 후 실행 (원하면)

### Llama-3.2-1B
```bash
source /home/jklee/ondevice/SLM-Bench/.venv_llama32_1b/bin/activate
python /home/jklee/ondevice/SLM-Bench/scripts/run_llama32_1b.py --email you@example.com
```

### GPT-Neo-1.3B
```bash
source /home/jklee/ondevice/SLM-Bench/.venv_gpt_neo_1_3b/bin/activate
python /home/jklee/ondevice/SLM-Bench/scripts/run_gpt_neo_1_3b.py --email you@example.com
```

### Phi-1.5
```bash
source /home/jklee/ondevice/SLM-Bench/.venv_phi_1_5/bin/activate
python /home/jklee/ondevice/SLM-Bench/scripts/run_phi_1_5.py --email you@example.com
```

## 참고

- 세 실행 파일은 각각 하나의 모델만 로드하도록 고정되어 있습니다.
- 한 번에 하나의 모델만 실행하면 환경/캐시 충돌 가능성을 크게 줄일 수 있습니다.
- Llama-3.2-1B는 Hugging Face 권한이 필요할 수 있습니다.

## 4) 69개 실험을 한 번에 순차 실행

3개 모델 × 23개 데이터셋 = 총 69개 실험을 순차 실행하려면:

```bash
cd /home/jklee/ondevice/SLM-Bench
chmod +x scripts/run_all_69_experiments.sh
EMAIL=you@example.com ./scripts/run_all_69_experiments.sh
```

이 스크립트는 각 모델별 venv를 분리해서 활성화한 뒤,
모델별 Python 진입점으로 23개 데이터셋을 순차 학습/테스트합니다.
실행 전 `scripts/setup_model_envs.sh`가 자동으로 호출되어,
존재하지 않는 venv만 생성하고 기존 venv는 재사용합니다.

## 5) 로컬 데이터셋 우선 사용

`train.py`는 이제 `local_datasets/` 폴더가 있으면 Hugging Face에서 다시 받지 않고,
먼저 로컬 저장본을 읽습니다. 따라서 23개 데이터셋은 오프라인/반오프라인 환경에서도 재사용할 수 있습니다.
