# MLflow 통합 완료 요약

## 통합된 알고리즘

### ✅ MSCN (Cardinality Estimation)
- **PresetScheduler**: [MscnPresetScheduler.py](../algorithm_examples/Mscn/MscnPresetScheduler.py)
- **Event**: [EventImplement.py](../algorithm_examples/Mscn/EventImplement.py)
- **Model**: [mscn_model.py](../algorithm_examples/Mscn/source/mscn_model.py)
- **추적 메트릭**:
  - `train_loss` (epoch별, 100 steps)
  - `training_time_seconds`
  - `batch_size`
  - `num_training_samples`
  - 테스트 메트릭 (total_time, average_time, db_execution_time 등)

### ✅ Lero (Learned Query Optimizer)
- **PresetScheduler**: [LeroPresetScheduler.py](../algorithm_examples/Lero/LeroPresetScheduler.py)
- **Event**: [EventImplement.py](../algorithm_examples/Lero/EventImplement.py)
- **Model**: [model.py](../algorithm_examples/Lero/source/model.py)
- **추적 메트릭**:
  - `train_loss` (epoch별, 100 steps)
  - `training_time_seconds`
  - `batch_size`
  - `num_training_pairs`
  - 테스트 메트릭 (total_time, average_time, db_execution_time 등)

### ✅ Baseline (No AI)
- **PresetScheduler**: [BaselinePresetScheduler.py](../algorithm_examples/Baseline/BaselinePresetScheduler.py)
- **추적 메트릭**:
  - 학습 메트릭 없음 (AI 미사용)
  - 테스트 메트릭만 기록 (total_time, average_time, db_execution_time 등)

### ✅ Index Selection (Extend Algorithm)
- **PresetScheduler**: [IndexPresetScheduler.py](../algorithm_examples/Index/IndexPresetScheduler.py)
- **Event**: [EventImplement.py](../algorithm_examples/Index/EventImplement.py)
- **추적 메트릭**:
  - `index_optimization_time_seconds` (periodic update마다)
  - `num_indexes_selected`
  - `num_queries`
  - 테스트 메트릭 (total_time, average_time, db_execution_time 등)

### ✅ KnobTuning (Configuration Optimization)
- **PresetScheduler**: [KnobPresetScheduler.py](../algorithm_examples/KnobTuning/KnobPresetScheduler.py)
- **Event**: [EventImplement.py](../algorithm_examples/KnobTuning/EventImplement.py)
- **추적 메트릭**:
  - `knob_optimization_time_seconds` (periodic update마다)
  - `best_performance`
  - `num_knobs_tuned`
  - `target_metric`
  - 테스트 메트릭 (total_time, average_time, db_execution_time 등)

## 테스트 방법

### 1. 단일 알고리즘 테스트

```bash
# MSCN 테스트
cd test_example_algorithms
python unified_test.py --algo mscn --db stats_tiny

# Lero 테스트
python unified_test.py --algo lero --db stats_tiny

# Baseline 테스트
python unified_test.py --algo baseline --db stats_tiny

# Index Selection 테스트
python unified_test.py --algo index --db stats_tiny

# KnobTuning 테스트
python unified_test.py --algo knob --db stats_tiny
```

### 2. 여러 알고리즘 비교

```bash
# 모든 알고리즘 한 번에 테스트
python unified_test.py --algo baseline mscn lero index knob --db stats_tiny --compare

# Query optimization 알고리즘만 비교
python unified_test.py --algo baseline mscn lero --db stats_tiny --compare
```

### 3. MLflow UI로 결과 확인

```bash
# MLflow UI 시작
../scripts/mlflow_ui.sh

# 브라우저에서 http://localhost:5000 접속
```

### 4. CLI로 결과 조회

```bash
# 모든 실험 목록
python ../scripts/mlflow_query.py list

# MSCN 실험의 run 목록
python ../scripts/mlflow_query.py runs mscn_stats_tiny

# Lero 실험의 최고 성능 run
python ../scripts/mlflow_query.py best lero_stats_tiny

# Baseline과 MSCN 비교
python ../scripts/mlflow_query.py runs baseline_stats_tiny --limit 1
python ../scripts/mlflow_query.py runs mscn_stats_tiny --limit 1
# (run_id를 확인 후)
python ../scripts/mlflow_query.py compare <baseline_run_id> <mscn_run_id>
```

## 추적되는 정보

### 공통 (모든 알고리즘)

**Parameters**:
- `algorithm`: 알고리즘 이름 (mscn, lero, baseline)
- `dataset`: 데이터셋 이름 (stats_tiny, imdb, etc.)
- `enable_collection`: 데이터 수집 여부
- `enable_training`: 학습 실행 여부
- `num_epoch`: Epoch 수
- `num_training`: 학습 쿼리 수
- `num_collection`: 수집 쿼리 수

**Test Metrics**:
- `test_total_time`: 전체 테스트 실행 시간 (초)
- `test_average_time`: 쿼리당 평균 시간 (초)
- `test_query_count`: 테스트 쿼리 수
- `test_db_execution_time`: 순수 DB 실행 시간 (초)
- `test_overhead_time`: AI 추론 오버헤드 (초)

**Tags**:
- `stage`: "training" 또는 "tested"
- `started_at`: 학습 시작 시간
- `tested_at`: 테스트 완료 시간
- `test_dataset`: 테스트 데이터셋 이름

### MSCN 추가 메트릭

**Training Metrics** (100 steps):
- `train_loss`: Epoch별 Q-error loss
- `training_time_seconds`: 총 학습 시간
- `batch_size`: 배치 크기 (2048)
- `num_training_samples`: 학습 샘플 수

**Model Artifacts**:
- `model/mscn_YYYYMMDD_HHMMSS`: 모델 파일
- `metadata/mscn_YYYYMMDD_HHMMSS.json`: 메타데이터

### Lero 추가 메트릭

**Training Metrics** (100 steps):
- `train_loss`: Epoch별 BCE loss (pairwise learning)
- `training_time_seconds`: 총 학습 시간
- `batch_size`: 배치 크기 (64 or 64*GPU_COUNT)
- `num_training_pairs`: 학습 plan pair 수

**Model Artifacts**:
- `model/lero_pair_YYYYMMDD_HHMMSS`: 모델 파일
- `metadata/lero_pair_YYYYMMDD_HHMMSS.json`: 메타데이터

### Baseline

**Parameters**:
- `no_ai`: True
- `direct_execution`: True

**Test Metrics만 기록** (학습 없음)

## 사용 예시

### Python API로 최고 모델 찾기

```python
from pilotscope.Common.MLflowTracker import MLflowTracker

# MSCN 최고 모델
best_mscn = MLflowTracker.get_best_run(
    experiment_name="mscn_stats_tiny",
    metric="test_total_time",
    ascending=True
)
print(f"Best MSCN: {best_mscn['run_name']}, Time: {best_mscn['metric_value']:.3f}s")

# Lero 최고 모델
best_lero = MLflowTracker.get_best_run(
    experiment_name="lero_stats_tiny",
    metric="test_total_time",
    ascending=True
)
print(f"Best Lero: {best_lero['run_name']}, Time: {best_lero['metric_value']:.3f}s")

# Baseline (최근 run)
baseline_runs = MLflowTracker.list_runs("baseline_stats_tiny", limit=1)
if baseline_runs:
    baseline_time = baseline_runs[0].get('metrics.test_total_time', 0)
    print(f"Baseline: {baseline_time:.3f}s")

# 개선도 계산
if best_mscn and baseline_runs:
    improvement = (baseline_time - best_mscn['metric_value']) / baseline_time * 100
    print(f"MSCN improvement over baseline: {improvement:.1f}%")
```

### 학습 없이 최고 모델만 사용

```python
from pilotscope.PilotConfig import PostgreSQLConfig
from algorithm_examples.Mscn.MscnPresetScheduler import get_mscn_preset_scheduler
from algorithm_examples.utils import load_test_sql

config = PostgreSQLConfig(db="stats_tiny")

# MLflow에서 자동으로 최고 모델 로드
scheduler, _ = get_mscn_preset_scheduler(
    config,
    enable_training=False,
    enable_collection=False,
    use_mlflow=True  # MLflow에서 최고 모델 자동 검색
)

# 테스트 실행
test_sqls = load_test_sql("stats_tiny")
for sql in test_sqls:
    scheduler.execute(sql)
```

## 디렉토리 구조

```
pilotscope/
├── mlruns/                                # MLflow 실험 데이터
│   ├── 0/                                # Default experiment
│   ├── 1/                                # mscn_stats_tiny
│   │   └── <run_id>/
│   │       ├── artifacts/model/          # 모델 백업
│   │       ├── metrics/                  # train_loss, test_*
│   │       └── params/                   # 하이퍼파라미터
│   ├── 2/                                # lero_stats_tiny
│   ├── 3/                                # baseline_stats_tiny
│   └── .trash/                           # 삭제된 실험
├── algorithm_examples/ExampleData/        # 기존 모델 저장 (여전히 사용)
│   ├── Mscn/Model/
│   └── Lero/Model/
└── test_example_algorithms/results/      # JSON 결과 (호환성 유지)
```

## 호환성

### 기존 워크플로우 유지

✅ **JSON 결과 파일**: `results/*.json` 여전히 생성됨
✅ **모델 파일**: `ExampleData/*/Model/*` 여전히 사용됨
✅ **compare_results.py**: 기존 비교 스크립트 계속 사용 가능
✅ **ModelRegistry**: 기존 레지스트리 API 유지

### MLflow 비활성화

```python
# MLflow 없이 실행
scheduler, _ = get_mscn_preset_scheduler(
    config,
    enable_training=True,
    use_mlflow=False  # MLflow 비활성화
)
```

또는 CLI:

```bash
python unified_test.py --algo mscn --db stats_tiny --no-mlflow
```

## 문제 해결

### MLflow 경로 오류

```bash
# 증상: "file://mlruns is not a valid remote uri"
# 해결: 최신 코드 pull (절대 경로 사용으로 수정됨)
git pull origin master
```

### mlruns 디렉토리 없음

```bash
# 정상입니다. 첫 실험 실행 시 자동 생성됩니다.
python unified_test.py --algo mscn --db stats_tiny
```

### MLflow UI 접속 불가

```bash
# Docker 포트 매핑 확인
docker-compose ps
# 5000:5000 매핑 확인

# 재시작
docker-compose restart pilotscope-dev

# 컨테이너 내부에서 UI 실행
docker-compose exec pilotscope-dev bash
./scripts/mlflow_ui.sh
```

## 참고 문서

- **빠른 시작**: [MLFLOW_QUICKSTART.md](./MLFLOW_QUICKSTART.md)
- **상세 가이드**: [MLFLOW_GUIDE.md](./MLFLOW_GUIDE.md)
- **Docker 환경**: [DOCKER_GUIDE.md](./DOCKER_GUIDE.md)
- **프로젝트 개요**: [CLAUDE.md](../CLAUDE.md)

## 다음 단계

1. **KnobTuning 통합** (선택사항): 동일한 패턴 적용
2. **원격 MLflow 서버**: 팀 협업 환경 구축
3. **커스텀 메트릭**: Q-error, 정확도 등 추가
4. **모델 배포**: MLflow Models로 프로덕션 배포
