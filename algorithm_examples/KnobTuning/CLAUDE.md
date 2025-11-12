# KnobTuning (Database Configuration Optimization)

KnobTuning은 **데이터베이스 설정 최적화(Configuration Optimization)**를 위한 알고리즘입니다.

## 알고리즘 개요

**목적**: 데이터베이스 성능 파라미터(Knob)를 워크로드에 맞게 자동 튜닝

**작동 방식**:
1. 다양한 Knob 설정으로 워크로드 실행
2. 각 설정의 성능 측정
3. 최적 Knob 조합 탐색 (Bayesian Optimization 등)
4. 최적 설정 적용

**적합한 워크로드**:
- 워크로드가 명확하게 정의된 경우
- 성능 개선을 위해 DB 설정을 조정하고 싶은 경우
- OLTP, OLAP 모두 가능

**튜닝 대상 Knob 예시**:
- `shared_buffers`: 공유 버퍼 크기
- `work_mem`: 작업 메모리
- `effective_cache_size`: 캐시 크기
- `random_page_cost`: 랜덤 I/O 비용
- 등 (PostgreSQL 기준 100+ Knobs)

## 파일 구조

```
KnobTuning/
├── KnobPresetScheduler.py      # PostgreSQL용 팩토리 함수
├── SparkKnobPresetScheduler.py # Spark용 팩토리 함수
├── EventImplement.py           # 학습/탐색 로직
└── llamatune/                  # LlamaTune 알고리즘 구현
    └── ...
```

## 사용 방법

### PostgreSQL Knob Tuning

```python
from pilotscope.DBInteractor.PilotDataInteractor import PostgreSQLConfig
from algorithm_examples.KnobTuning.KnobPresetScheduler import get_knob_preset_scheduler

# 설정
config = PostgreSQLConfig(db="your_db")

# Scheduler 생성
scheduler, tracker = get_knob_preset_scheduler(
    config,
    enable_collection=True,   # 워크로드 데이터 수집
    enable_training=True,     # Knob 탐색
    dataset_name="your_db"
)

# 최적 Knob 설정 자동 탐색 및 적용
# (scheduler.init() 시 자동 실행)
```

### Spark Knob Tuning

```python
from pilotscope.DBInteractor.PilotDataInteractor import SparkConfig
from algorithm_examples.KnobTuning.SparkKnobPresetScheduler import get_spark_knob_preset_scheduler

config = SparkConfig(app_name="your_app")

scheduler, tracker = get_spark_knob_preset_scheduler(
    config,
    enable_collection=True,
    enable_training=True,
    dataset_name="your_workload"
)
```

## 주요 컴포넌트

### 1. KnobPresetScheduler

**역할**: Knob 튜닝 알고리즘 설정

**MSCN/Lero와의 차이**:
- 모델 학습이 아닌 **탐색(Search)** 수행
- 워크로드 전체를 반복 실행하며 최적 설정 찾기

**주요 파라미터**:
- `num_epoch`: 탐색 반복 횟수 (기본: 100)
- `num_training`: 워크로드 쿼리 수
- `num_collection`: 초기 데이터 수집 쿼리 수

### 2. Knob 탐색 알고리즘

**위치**: `llamatune/` 폴더

**지원 알고리즘**:
- **LlamaTune**: Bayesian Optimization 기반 Knob 탐색
- 기타 알고리즘 추가 가능

**탐색 과정**:
```python
# 1. 초기 Knob 설정
knob_config = {
    'shared_buffers': '128MB',
    'work_mem': '4MB',
    # ...
}

# 2. 워크로드 실행 및 성능 측정
total_time = 0
for sql in workload_queries:
    result = execute_with_knobs(sql, knob_config)
    total_time += result.execution_time

# 3. Bayesian Optimization으로 다음 Knob 제안
next_knob_config = optimizer.suggest(total_time)

# 4. 반복 (num_epoch 횟수만큼)
```

### 3. EventImplement

**역할**: 워크로드 실행 및 Knob 탐색

**데이터 수집**:
```python
def iterative_data_collection(self):
    # 워크로드 쿼리 로드
    workload_queries = dataset.read_train_sql()

    # 각 Knob 설정에 대해 워크로드 실행
    for iteration in range(self.num_epoch):
        # Knob 설정 적용
        self._apply_knobs(current_knob_config)

        # 워크로드 실행
        total_time = 0
        for sql in workload_queries:
            exec_time = self.data_interactor.pull_execution_time(sql)
            total_time += exec_time

        # 결과 저장
        self.data_manager.save(self.data_save_table, {
            'iteration': iteration,
            'knob_config': current_knob_config,
            'total_time': total_time
        })

        # 다음 Knob 제안
        current_knob_config = self.optimizer.suggest(total_time)
```

**최적 Knob 찾기**:
```python
def custom_model_training(self):
    # 모든 시도 중 최적 Knob 선택
    all_results = self.data_manager.read(self.data_save_table)
    best_result = min(all_results, key=lambda x: x['total_time'])

    # 최적 Knob 적용
    self._apply_knobs(best_result['knob_config'])

    # MLflow에 기록
    self.mlflow_tracker.log_params(best_result['knob_config'])
    self.mlflow_tracker.log_metric("best_total_time", best_result['total_time'])
```

## 수정 시 주의사항

### 1. 탐색 Knob 범위 변경

**위치**: `llamatune/` 또는 `EventImplement.py`

**주의**:
- Knob 범위가 너무 넓으면 탐색 시간 증가
- 너무 좁으면 최적 설정을 찾지 못할 수 있음

```python
# Knob 탐색 범위 정의
knob_search_space = {
    'shared_buffers': ['128MB', '256MB', '512MB', '1GB'],
    'work_mem': ['4MB', '8MB', '16MB', '32MB'],
    # ...
}
```

### 2. 워크로드 정의

**위치**: Dataset 클래스

**주의**:
- Knob 튜닝은 **대표 워크로드**가 중요
- 학습 쿼리와 실제 워크로드가 유사해야 함

```python
# 대표 워크로드 선정
# - OLTP: 짧은 트랜잭션 쿼리
# - OLAP: 복잡한 분석 쿼리
# - Mixed: 혼합

# train.txt에 대표 쿼리 배치
```

### 3. 탐색 알고리즘 변경

**위치**: `EventImplement.py`

**현재**: LlamaTune (Bayesian Optimization)

**다른 알고리즘 추가**:
```python
# Grid Search
for knob_config in all_possible_configs:
    performance = evaluate_workload(knob_config)

# Random Search
for i in range(num_trials):
    knob_config = sample_random_config()
    performance = evaluate_workload(knob_config)

# Genetic Algorithm
# ...
```

## 성능 튜닝

### 탐색 속도 개선

```python
# 1. 탐색 반복 횟수 감소
num_epoch=50  # 기본 100

# 2. 워크로드 크기 축소
num_training=50  # 대표 쿼리 50개만 사용

# 3. 탐색 공간 축소
# 핵심 Knob만 튜닝 (shared_buffers, work_mem, effective_cache_size)
```

### 더 나은 Knob 찾기

```python
# 1. 더 많은 반복
num_epoch=200

# 2. 더 넓은 탐색 공간
# 더 많은 Knob 포함

# 3. 워크로드를 실제와 유사하게 구성
```

## KnobTuning vs MSCN/Lero 비교

| 항목 | KnobTuning | MSCN/Lero |
|------|------------|-----------|
| **목적** | DB 설정 최적화 | 쿼리 최적화 |
| **적용 시점** | DB 시작 시 (전역 설정) | 쿼리 실행 시 (쿼리별) |
| **변경 빈도** | 낮음 (한 번 설정) | 높음 (쿼리마다) |
| **학습 방식** | 탐색 (Search) | 학습 (Learning) |
| **워크로드 의존성** | 높음 (워크로드별 최적 설정) | 중간 (쿼리 패턴) |

**조합 사용**:
1. KnobTuning으로 DB 전역 설정 최적화
2. MSCN/Lero로 개별 쿼리 최적화

## 문제 해결

### Q1. 탐색이 너무 느림
- `num_epoch` 감소
- 워크로드 크기 축소 (`num_training`)
- 탐색 Knob 수 감소

### Q2. 최적 Knob이 baseline보다 나쁨
- 탐색 반복 부족: `num_epoch` 증가
- 탐색 공간이 최적 설정을 포함하지 않음: 범위 확대
- 워크로드가 실제와 다름: 대표 쿼리 재선정

### Q3. Knob 적용이 안 됨
- PostgreSQL 재시작 필요한 Knob 확인 (`pg_settings` 테이블)
- 권한 문제: `ALTER SYSTEM` 권한 확인

### Q4. 워크로드마다 최적 설정이 다름
- 의도된 동작: Knob은 워크로드별로 튜닝해야 함
- 여러 워크로드에 대해 절충안(trade-off) 필요

## 관련 문서

- **공통 패턴**: [algorithm_examples/CLAUDE.md](../CLAUDE.md)
- **상세 가이드**: [docs/PRODUCTION_OPTIMIZATION.md](../../docs/PRODUCTION_OPTIMIZATION.md)

## 참고 논문

- Van Aken et al., "Automatic Database Management System Tuning Through Large-scale Machine Learning"
- PilotScope 논문: `paper/PilotScope.pdf`
