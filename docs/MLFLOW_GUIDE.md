# MLflow 통합 가이드

이 가이드는 PilotScope에 통합된 MLflow 실험 추적 시스템 사용법을 설명합니다.

## 개요

MLflow는 머신러닝 실험을 추적하고 관리하는 오픈소스 플랫폼입니다. PilotScope는 MLflow를 사용하여:

- **자동 실험 추적**: 하이퍼파라미터, 메트릭, 모델 파일 자동 저장
- **학습 과정 모니터링**: Epoch별 loss 변화 추적
- **모델 비교**: 여러 실험 결과를 쉽게 비교
- **재현성 보장**: 코드 버전, 환경 정보 자동 기록

## 빠른 시작

### 1. MLflow 통합 테스트 실행

```bash
# Docker 컨테이너 내부에서
cd test_example_algorithms
python unified_test.py --algo mscn --db stats_tiny

# MLflow가 자동으로:
# 1. 학습 파라미터 기록
# 2. Epoch별 loss 추적
# 3. 테스트 결과 저장
# 4. 모델 파일 백업
```

### 2. MLflow UI 실행

**방법 1: 스크립트 사용 (권장)**
```bash
# Docker 내부에서
./scripts/mlflow_ui.sh

# 브라우저에서 http://localhost:5000 접속
```

**방법 2: 직접 실행**
```bash
mlflow ui --backend-store-uri file://$(pwd)/mlruns --host 0.0.0.0 --port 5000
```

### 3. MLflow UI 사용법

MLflow UI에서 다음을 확인할 수 있습니다:

- **Experiments 탭**: 모든 실험 목록 (MSCN, Lero, Baseline 등)
- **Runs 탭**: 각 실험의 run 목록과 메트릭
- **Compare Runs**: 여러 run을 선택하여 비교
- **Artifacts**: 저장된 모델 파일 다운로드

## 주요 기능

### 1. 자동 실험 추적

MSCN 알고리즘을 예로 들면:

```python
# unified_test.py에서 자동으로:

# 학습 시작
scheduler, tracker = get_mscn_preset_scheduler(
    config,
    enable_training=True,
    num_epoch=100,
    use_mlflow=True  # MLflow 활성화
)

# 파라미터 자동 로깅:
# - algorithm: mscn
# - dataset: stats_tiny
# - num_epoch: 100
# - num_training: -1
# - num_collection: -1

# Epoch별 loss 자동 로깅 (100번)
# - train_loss at step 0
# - train_loss at step 1
# ...
# - train_loss at step 99

# 테스트 결과 자동 로깅:
# - test_total_time: 13.497s
# - test_average_time: 0.092s
# - test_query_count: 146
# - test_db_execution_time: 0.122s
# - test_overhead_time: 13.375s

# 모델 파일 자동 백업
# - ExampleData/Mscn/Model/mscn_20251019_151808
# - ExampleData/Mscn/Model/mscn_20251019_151808.json
```

### 2. CLI로 실험 조회

**모든 실험 목록**
```bash
python scripts/mlflow_query.py list

# 출력 예시:
# ╔════╦═══════════════════╦═════════╦════════╗
# ║ ID ║ Name              ║ Stage   ║ # Runs ║
# ╠════╬═══════════════════╬═════════╬════════╣
# ║ 1  ║ mscn_stats_tiny   ║ active  ║ 5      ║
# ║ 2  ║ lero_stats_tiny   ║ active  ║ 3      ║
# ║ 3  ║ baseline_stats... ║ active  ║ 2      ║
# ╚════╩═══════════════════╩═════════╩════════╝
```

**특정 실험의 run 목록**
```bash
python scripts/mlflow_query.py runs mscn_stats_tiny --limit 5

# 출력 예시:
# ╔══════════╦════════════════════════════════╦══════════╦═════════════╦═══════════╦═════════════╦═══════════╗
# ║ Run ID   ║ Name                           ║ Status   ║ Total Time  ║ Avg Time  ║ Algorithm   ║ Dataset   ║
# ╠══════════╬════════════════════════════════╬══════════╬═════════════╬═══════════╬═════════════╬═══════════╣
# ║ 3a2f1... ║ mscn_stats_tiny_20251019_15... ║ FINISHED ║ 13.497s     ║ 0.0925s   ║ mscn        ║ stats_... ║
# ║ 4b3c2... ║ mscn_stats_tiny_20251019_14... ║ FINISHED ║ 14.123s     ║ 0.0968s   ║ mscn        ║ stats_... ║
# ╚══════════╩════════════════════════════════╩══════════╩═════════════╩═══════════╩═════════════╩═══════════╝
```

**최고 성능 run 찾기**
```bash
python scripts/mlflow_query.py best mscn_stats_tiny --metric test_total_time

# 출력 예시:
# ===============================================================================
# Best run in mscn_stats_tiny (by test_total_time)
# ===============================================================================
# Run ID: 3a2f1d4b5c6e7f8a
# Run Name: mscn_stats_tiny_20251019_152112
# test_total_time: 13.497000
#
# Parameters:
#   algorithm: mscn
#   dataset: stats_tiny
#   num_epoch: 100
#   num_training: -1
#   num_collection: -1
#
# Artifact URI: file:///home/pilotscope/workspace/mlruns/1/3a2f1d4b5c6e7f8a/artifacts
```

**두 run 비교**
```bash
python scripts/mlflow_query.py compare 3a2f1d4b 4b3c2a5d

# 출력 예시:
# ╔═══════════╦═════════════════════════════════════╦═════════════════════════════════════╗
# ║           ║ Run 1                               ║ Run 2                               ║
# ╠═══════════╬═════════════════════════════════════╬═════════════════════════════════════╣
# ║ Run ID    ║ 3a2f1d4b                            ║ 4b3c2a5d                            ║
# ║ Run Name  ║ mscn_stats_tiny_20251019_152112     ║ mscn_stats_tiny_20251019_141023     ║
# ║ Status    ║ FINISHED                            ║ FINISHED                            ║
# ╚═══════════╩═════════════════════════════════════╩═════════════════════════════════════╝
#
# Metrics:
# ╔═══════════════════════════╦══════════╦══════════╦═════════════╗
# ║ Metric                    ║ Run 1    ║ Run 2    ║ Diff (2-1)  ║
# ╠═══════════════════════════╬══════════╬══════════╬═════════════╣
# ║ test_total_time           ║ 13.4970  ║ 14.1230  ║ +0.6260 ⬆️   ║
# ║ test_average_time         ║ 0.0925   ║ 0.0968   ║ +0.0043 ⬆️   ║
# ║ test_db_execution_time    ║ 0.1227   ║ 0.1301   ║ +0.0074 ⬆️   ║
# ╚═══════════════════════════╩══════════╩══════════╩═════════════╝
```

### 3. Python API로 최고 모델 로드

```python
from pilotscope.Common.MLflowTracker import MLflowTracker

# 최고 성능 모델 찾기
best_run = MLflowTracker.get_best_run(
    experiment_name="mscn_stats_tiny",
    metric="test_total_time",
    ascending=True  # 시간은 작을수록 좋음
)

print(f"Best model: {best_run['run_name']}")
print(f"Total time: {best_run['metric_value']:.3f}s")
print(f"Parameters: {best_run['params']}")

# 모델 파일 경로 가져오기
model_id = best_run['params'].get('model_id')
if model_id:
    from algorithm_examples.Mscn.MscnPilotModel import MscnPilotModel
    model = MscnPilotModel.load_model(model_id, "mscn")
```

### 4. 학습 없이 최고 모델만 사용

```python
from pilotscope.PilotConfig import PostgreSQLConfig
from algorithm_examples.Mscn.MscnPresetScheduler import get_mscn_preset_scheduler

config = PostgreSQLConfig(db="stats_tiny")

# MLflow에서 자동으로 최고 모델 로드
scheduler, _ = get_mscn_preset_scheduler(
    config,
    enable_training=False,  # 학습 안 함
    enable_collection=False,
    use_mlflow=True  # MLflow에서 최고 모델 자동 로드
)

# 바로 테스트 실행
test_sqls = load_test_sql("stats_tiny")
for sql in test_sqls:
    scheduler.execute(sql)
```

## 디렉토리 구조

```
pilotscope/
├── mlruns/                          # MLflow 실험 데이터 (Git 무시됨)
│   ├── 0/                          # Default experiment
│   ├── 1/                          # mscn_stats_tiny experiment
│   │   ├── 3a2f1d4b5c6e7f8a/       # Run 1
│   │   │   ├── artifacts/          # 모델 파일, 메타데이터
│   │   │   ├── metrics/            # 메트릭 데이터
│   │   │   ├── params/             # 파라미터 데이터
│   │   │   └── tags/               # 태그 데이터
│   │   └── 4b3c2a5d6e7f8a9b/       # Run 2
│   ├── 2/                          # lero_stats_tiny experiment
│   └── .trash/                     # 삭제된 실험
├── algorithm_examples/ExampleData/  # 기존 모델 저장 (여전히 사용됨)
│   └── Mscn/Model/
│       ├── mscn_20251019_151808
│       └── mscn_20251019_151808.json
└── test_example_algorithms/results/ # JSON 결과 파일 (호환성 유지)
    └── mscn_stats_tiny_20251019_152112.json
```

**중요**:
- `mlruns/`: MLflow가 관리하는 실험 데이터 (`.gitignore`에 포함)
- `ExampleData/`: 기존 모델 파일 (여전히 사용됨, MLflow에도 백업)
- `results/`: 이전 버전 호환성을 위한 JSON 파일

## 고급 기능

### 1. MLflow 비활성화 (디버깅용)

```bash
# MLflow 없이 테스트
python unified_test.py --algo mscn --db stats_tiny --no-mlflow
```

또는 코드에서:

```python
scheduler, _ = get_mscn_preset_scheduler(
    config,
    enable_training=True,
    use_mlflow=False  # MLflow 비활성화
)
```

### 2. 특정 metric 최적화

```python
# 평균 시간이 가장 짧은 모델 찾기
best_run = MLflowTracker.get_best_run(
    experiment_name="mscn_stats_tiny",
    metric="test_average_time",
    ascending=True
)

# 정확도가 가장 높은 모델 찾기 (미래 기능)
best_run = MLflowTracker.get_best_run(
    experiment_name="mscn_stats_tiny",
    metric="test_accuracy",
    ascending=False  # 높을수록 좋음
)
```

### 3. 실험 태그 필터링

```python
import mlflow

# 특정 태그로 run 검색
runs = mlflow.search_runs(
    experiment_names=["mscn_stats_tiny"],
    filter_string="tags.stage = 'tested' AND params.num_epoch = '100'"
)

print(runs[['run_id', 'metrics.test_total_time', 'params.num_training']])
```

### 4. 모델 아티팩트 다운로드

```bash
# CLI로 다운로드
mlflow artifacts download --run-id 3a2f1d4b5c6e7f8a --dst-path ./downloaded_model
```

Python에서:

```python
import mlflow

# 특정 run의 artifact 다운로드
artifact_uri = "runs:/3a2f1d4b5c6e7f8a/model"
local_path = mlflow.artifacts.download_artifacts(artifact_uri)
print(f"Downloaded to: {local_path}")
```

## 문제 해결

### MLflow UI가 실행되지 않음

```bash
# 1. MLflow 설치 확인
pip show mlflow

# 2. 재설치
pip install --upgrade mlflow>=2.8.0

# 3. 포트 변경 (5000 포트가 사용 중인 경우)
./scripts/mlflow_ui.sh 5001
```

### 브라우저에서 접속 불가

```bash
# Docker 컨테이너 내부에서 MLflow UI 실행 시
# docker-compose.yml에서 포트 매핑 확인:
ports:
  - "5000:5000"   # MLflow UI

# 호스트에서 http://localhost:5000 접속
```

### mlruns 디렉토리가 없음

```bash
# 정상입니다. 첫 실험 실행 시 자동 생성됩니다.
python unified_test.py --algo mscn --db stats_tiny

# 수동 생성 (선택사항)
mkdir mlruns
```

### 이전 JSON 결과 파일 호환성

MLflow 통합 후에도 기존 JSON 파일은 여전히 생성됩니다:

```python
# test_example_algorithms/results/mscn_stats_tiny_20251019_152112.json
{
  "algorithm": "mscn",
  "database": "stats_tiny",
  "timestamp": "20251019_152112",
  "metrics": { ... },
  "extra_info": { ... }
}
```

기존 `compare_results.py`도 계속 사용 가능합니다:

```bash
python algorithm_examples/compare_results.py --latest mscn lero baseline
```

## 다음 단계

### Lero와 Baseline에도 MLflow 적용

현재 MSCN만 MLflow 통합이 완료되었습니다. Lero와 Baseline도 동일한 패턴으로 적용 가능:

```python
# algorithm_examples/Lero/LeroPresetScheduler.py
def get_lero_preset_scheduler(config, ..., use_mlflow=True) -> tuple:
    mlflow_tracker = None
    if use_mlflow:
        mlflow_tracker = MLflowTracker(experiment_name=f"lero_{config.db}")
        mlflow_tracker.start_training(...)

    # ... (기존 코드)

    return scheduler, mlflow_tracker
```

### 원격 MLflow 서버 설정 (선택사항)

팀 협업을 위해 원격 MLflow 서버 설정 가능:

```bash
# 서버 실행
mlflow server \
    --backend-store-uri postgresql://user:pass@host/mlflow \
    --default-artifact-root s3://my-bucket/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000

# 클라이언트 설정
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

## 참고 자료

- [MLflow 공식 문서](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking API](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Python API Reference](https://mlflow.org/docs/latest/python_api/index.html)
