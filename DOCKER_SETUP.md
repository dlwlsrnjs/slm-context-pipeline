# SLM-Bench Docker 가상환경 가이드

이 문서는 현재 `ondevice/SLM-Bench`에 적용된 Docker 기반 가상환경 설정과, 이후 작업 시작 방법을 정리합니다.

## 1) 현재 Docker 설정 상태

### 구성 파일
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

### 핵심 설정 요약
- 베이스 이미지: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- 컨테이너 내부 Python 가상환경(venv): `/opt/venv`
- 작업 디렉토리: `/workspace`
- 프로젝트 바인드 마운트: `./ -> /workspace`
- Hugging Face 캐시 마운트: `${HOME}/.cache/huggingface -> /workspace/.cache/huggingface`
- GPU 사용: Compose에서 NVIDIA GPU 예약(`gpus`)
- 기본 실행: `bash` (필요 시 `python train.py` 직접 실행)

## 2) 왜 이 방식이 안전한가

- 패키지 설치/업데이트는 컨테이너 내부 환경(`/opt/venv`)에만 반영됩니다.
- 호스트(로컬) 기본 Python, 다른 가상환경(`.venv`, conda 등)은 건드리지 않습니다.
- 코드/결과는 바인드 마운트로 파일만 공유되고, 실행 환경은 격리됩니다.

## 3) 처음 1회 세팅

프로젝트 루트로 이동:

```bash
cd /home/jklee/ondevice/SLM-Bench
```

이미지 빌드:

```bash
docker compose build
```

구성 검증:

```bash
docker compose config
```

## 4) 매번 작업 시작할 때

### A. 컨테이너 쉘로 들어가서 작업

```bash
cd /home/jklee/ondevice/SLM-Bench
docker compose run --rm slmbench bash
```

컨테이너 안에서:

```bash
python -V
python train.py
```

### B. 쉘 진입 없이 단일 명령 실행

```bash
cd /home/jklee/ondevice/SLM-Bench
docker compose run --rm slmbench python train.py
```

## 5) 자주 쓰는 명령어

이미지 다시 빌드(의존성/도커파일 변경 후):

```bash
docker compose build --no-cache
```

컨테이너에서 패키지 목록 확인:

```bash
docker compose run --rm slmbench pip list
```

GPU 인식 확인:

```bash
docker compose run --rm slmbench nvidia-smi
```

## 6) 주의사항

- 호스트에서 직접 `pip install` 하지 말고, 반드시 `docker compose run ...` 안에서 실행하세요.
- 의존성 변경 시 `requirements.txt` 수정 후 `docker compose build`를 다시 실행하세요.
- GPU 없는 환경에서는 `docker-compose.yml`의 `gpus` 블록을 잠시 제거/주석 처리 후 실행하면 됩니다.

## 로컬 데이터셋 우선 사용

`train.py`는 `local_datasets/`가 존재하면 Hugging Face 원격보다 이 로컬 저장본을 먼저 사용합니다.
따라서 한 번 내려받은 23개 데이터셋을 그대로 재사용할 수 있습니다.

69개 실험을 돌릴 때도 이 폴더를 공유하면 됩니다.

## 자동 집계

결과가 쌓일 때마다 집계를 갱신하려면 `SLMBENCH_AUTO_AGGREGATE=1`을 사용합니다.

집계 파일:
- `experiment_results/summary.json`
- `experiment_results/summary.csv`

예시:

```bash
cd /home/jklee/ondevice/SLM-Bench
SLMBENCH_AUTO_AGGREGATE=1 EMAIL=you@example.com ./scripts/run_all_69_experiments.sh
```

## 대표 데이터셋만 돌리기

현재처럼 전체 23개를 한 번에 돌리기보다, 안정적인 대표 데이터셋만 먼저 돌릴 수 있습니다.

선택한 데이터셋:
- BoolQ
- PIQA
- Hellaswag
- WinoGrande
- e2e_nlg
- viggo

실행 예시:

```bash
cd /home/jklee/ondevice/SLM-Bench
SLMBENCH_AUTO_AGGREGATE=1 EMAIL=you@example.com ./scripts/run_core_experiments.sh
```

기본 샘플 수는 학습 64개, 평가 32개로 설정되어 있어 GPU 메모리 부담이 훨씬 적습니다.

## 12개 개별 Python 코드로 실행 (2모델 x 6데이터셋, CoQA 제외)

모델/데이터셋 단위로 완전히 분리된 Python 실행 파일 12개를 사용합니다.

- GPT-Neo-1.3B: BoolQ, PIQA, Hellaswag, WinoGrande, e2e_nlg, viggo
- Phi-1.5: BoolQ, PIQA, Hellaswag, WinoGrande, e2e_nlg, viggo

한 번에 12개를 순차 실행:

```bash
cd /home/jklee/ondevice/SLM-Bench
SLMBENCH_AUTO_AGGREGATE=1 EMAIL=you@example.com ./scripts/run_14_individual_experiments.sh
```

개별 1개만 실행 예시:

```bash
cd /home/jklee/ondevice/SLM-Bench
docker compose run --rm slmbench /opt/venv/bin/python /workspace/scripts/run_phi_1_5_boolq.py --email you@example.com --local-datasets-dir /workspace/local_datasets --max-train-samples 64 --max-eval-samples 32
```

## 진행 상황 바로 보기

실행 중인 로그와 결과 파일 개수를 함께 보고 싶으면 아래 스크립트를 사용하세요.

```bash
cd /home/jklee/ondevice/SLM-Bench
bash ./scripts/watch_progress.sh experiment_runs/run_46_gpu_strict_20260412_101604.log 46
```

인자를 생략하면 가장 최근 로그를 자동으로 찾습니다.

```bash
cd /home/jklee/ondevice/SLM-Bench
bash ./scripts/watch_progress.sh
```

화면에는 다음이 계속 갱신됩니다.
- 완료 개수: GPT-Neo-1.3B / Phi-1.5 결과 파일 수
- 최근 로그 20줄
- 현재 실행 중인 로그 파일 경로

## 7) 빠른 시작(복붙용)

```bash
cd /home/jklee/ondevice/SLM-Bench
docker compose build
docker compose run --rm slmbench python -V
docker compose run --rm slmbench python train.py
```
