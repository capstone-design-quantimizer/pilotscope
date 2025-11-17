# KnobTuning

DB 설정 파라미터(Knob) 자동 튜닝. Bayesian Optimization으로 최적 설정 탐색.

## 작동 방식

1. 다양한 Knob 설정으로 워크로드 실행
2. 각 설정의 성능 측정
3. 최적 Knob 조합 탐색
4. 최적 설정 적용

---

## 전체 파이프라인 상세 분석

### 아키텍처 개요

```
사용자 코드
    ↓
get_knob_preset_scheduler() → PilotScheduler 생성
    ↓
KnobPeriodicModelUpdateEvent 등록
    ↓
scheduler.init() 호출
    ↓
llamatune() 실행 (Bayesian Optimization)
    ↓
SMAC Optimizer (Random Forest Surrogate)
    ↓  ↓  ↓ (반복 50회)
    ↓  ↓  ↓ 1. Knob 설정 제안
    ↓  ↓  ↓ 2. SysmlExecutor로 평가
    ↓  ↓  ↓ 3. 성능 피드백
    ↓  ↓  ↓ 4. Surrogate 모델 업데이트
    ↓  ↓  ↓
    ↓  ↓  └→ 워크로드 쿼리 실행 → 성능 메트릭 수집
    ↓  ↓
    ↓  └→ 최적 Knob 발견
    ↓
최적 Knob 적용 → DB 재시작
```

### 1단계: 초기화 및 쿼리 로딩

#### 파일: `KnobPresetScheduler.py:11-70`

```python
def get_knob_preset_scheduler(config, dataset_name=None, **kwargs):
    # 1. MLflow 트래커 초기화 (실험 추적용)
    mlflow_tracker = MLflowTracker(experiment_name=f"knob_{config.db}")

    # 2. PilotScheduler 생성
    scheduler = SchedulerFactory.create_scheduler(config)

    # 3. Event 등록: KnobPeriodicModelUpdateEvent
    periodic_model_update_event = KnobPeriodicModelUpdateEvent(
        config, 200,  # 200쿼리마다 업데이트 (실제론 init 시 1회만)
        execute_on_init=True,
        llamatune_config_file="../algorithm_examples/KnobTuning/llamatune/configs/llama_config.ini",
        optimizer_type="smac",
        dataset_name=dataset_name
    )
    scheduler.register_events([periodic_model_update_event])

    # 4. scheduler.init() 호출 시 Event 실행됨
    scheduler.init()
    return scheduler, mlflow_tracker
```

**쿼리 로딩 과정** (`EventImplement.py:116-122`):

```python
# load_training_sql()로 데이터셋의 train 쿼리 로드
train_sqls = load_training_sql(self.dataset_name)
# 예: "stats_tiny" → StatsTinyDataset().read_train_sql()

# 임시 파일에 쿼리 저장
temp_sql_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
for sql in train_sqls:
    temp_sql_file.write(sql + '\n')
temp_sql_file.close()
```

**데이터셋별 쿼리 로딩** (`utils.py:30-51`):

- `StatsTinyDataset`: Stack Overflow 데이터셋 (작은 버전)
- `StatsDataset`: Stack Overflow 데이터셋 (전체)
- `ImdbDataset`: IMDb 영화 데이터셋
- `TpcdsDataset`: TPC-DS 벤치마크
- 각 Dataset 클래스는 `read_train_sql()` 메서드로 쿼리 리스트 반환

---

### 2단계: Knob 설정 공간 정의

#### 파일: `space.py:11-199`

**ConfigSpaceGenerator**: Knob 탐색 공간 생성

```python
# 설정 파일에서 정의 읽기
spaces = ConfigSpaceGenerator.from_config(config)
```

**설정 예시** (`llama_config.ini:24-30`):

```ini
[spaces]
definition=postgres-13           # Knob 정의 파일
ignore=postgres-none             # 무시할 Knob 목록
adapter_alias=hesbo              # 차원 축소 방법 (HesBO)
le_low_dim=16                    # 저차원 공간 크기
bias_prob_sv=0.2                 # 기본값 편향 확률
quantization_factor=10000        # 양자화 계수
target_metric=throughput         # 목표 메트릭
```

**Knob 정의** (`spaces/definitions/postgres-13.json`):

```json
[
  {
    "id": 8,
    "name": "autovacuum",
    "type": "enum",
    "default": "on",
    "choices": ["on", "off"]
  },
  {
    "id": 9,
    "name": "autovacuum_analyze_scale_factor",
    "type": "real",
    "default": 0.1,
    "min": 0.0,
    "max": 100.0
  },
  {
    "id": 10,
    "name": "autovacuum_analyze_threshold",
    "type": "integer",
    "default": 50,
    "min": 0,
    "max": 1000
  },
  // ... 100+ knobs 정의
]
```

**Knob 타입**:
- `enum`: 카테고리형 (예: `on`/`off`)
- `integer`: 정수형 (예: `50` ~ `1000`)
- `real`: 실수형 (예: `0.0` ~ `100.0`)

**탐색 공간 생성** (`space.py:61-96`):

```python
def generate_input_space(self, seed: int):
    input_dimensions = []
    for info in self.knobs:
        if knob_type == 'enum':
            dim = CSH.CategoricalHyperparameter(
                name=name, choices=info['choices'], default_value=info['default'])
        elif knob_type == 'integer':
            dim = CSH.UniformIntegerHyperparameter(
                name=name, lower=info['min'], upper=info['max'],
                default_value=info['default'])
        elif knob_type == 'real':
            dim = CSH.UniformFloatHyperparameter(
                name=name, lower=info['min'], upper=info['max'],
                default_value=info['default'])
        input_dimensions.append(dim)

    input_space = CS.ConfigurationSpace(name="input", seed=seed)
    input_space.add_hyperparameters(input_dimensions)
    return input_space
```

**차원 축소** (고급):

- **HesBO (Heterogeneous Bayesian Optimization)**: 고차원 공간 → 저차원 공간 (16차원)
- **Bias Sampling**: 기본값 근처를 20% 확률로 샘플링
- **Quantization**: 연속 공간을 이산화 (10000 레벨)

---

### 3단계: Bayesian Optimization 루프

#### 파일: `EventImplement.py:30-92`, `run_smac.py:72-136`

**최적화 알고리즘**: SMAC (Sequential Model-based Algorithm Configuration)

```python
def llamatune(conf):
    # 1. 설정 로드
    config.update_from_file(conf["conf_filepath"])

    # 2. 탐색 공간 생성
    spaces = ConfigSpaceGenerator.from_config(config)

    # 3. Executor 생성 (Knob 평가 담당)
    executor = ExecutorFactory.from_config(config, spaces, storage)

    # 4. SMAC Optimizer 생성
    optimizer = get_smac_optimizer(
        config, spaces,
        evaluate_func=evaluate_dbms_conf,  # 평가 함수
        exp_state=exp_state
    )

    # 5. 기본 설정 평가
    default_config = spaces.get_default_configuration()
    perf = evaluate_dbms_conf(spaces, executor, storage, columns,
                              default_config, state=exp_state)

    # 6. 최적화 루프 시작 (50회 반복)
    optimizer.optimize()

    return exp_state  # 최적 설정 및 성능 반환
```

**SMAC Optimizer 설정** (`optimizer.py:82-178`):

```python
def get_smac_optimizer(config, spaces, tae_runner, state):
    # 입력 공간 생성
    input_space = spaces.generate_input_space(config.seed)

    # Scenario 설정
    scenario = Scenario({
        "run_obj": "quality",              # 품질 최적화
        "runcount-limit": config.iters,    # 50회 반복
        "cs": input_space,                 # 탐색 공간
        "deterministic": "true",
        "always_race_default": "false",
        "limit_resources": "false",
        "output_dir": state.results_path,
    })

    # Random Forest Surrogate 모델 설정
    model_kwargs = {
        'num_trees': 100,                  # Random Forest 트리 개수
        'log_y': False,                    # y 로그 스케일 사용 안 함
        'ratio_features': 1,               # 전체 feature 사용
        'min_samples_split': 2,            # 분할 최소 샘플 수
        'min_samples_leaf': 3,             # 리프 최소 샘플 수
        'max_depth': 2**20,                # 최대 깊이
    }

    # SMAC4HPO: Hyperparameter Optimization
    optimizer = SMAC4HPO(
        scenario=scenario,
        tae_runner=tae_runner,             # 평가 함수
        rng=config.seed,                   # 랜덤 시드
        model_kwargs=model_kwargs,         # RF 설정
        initial_design=LHDesignWithBiasedSampling,  # 초기 샘플링: Latin Hypercube + Bias
        initial_design_kwargs={
            "init_budget": 10,              # 초기 10개 랜덤 샘플
            "max_config_fracs": 1,
        },
        random_configuration_chooser_kwargs={
            'prob': 0.1,                    # 10% 확률로 랜덤 샘플링
        },
    )

    return optimizer
```

**Bayesian Optimization 작동 원리**:

1. **Surrogate Model**: Random Forest
   - 입력: Knob 설정 (예: `{shared_buffers: 128MB, work_mem: 4MB, ...}`)
   - 출력: 성능 예측 (예: `throughput: 1500 ops/sec`)
   - 이전 평가 결과로 학습됨

2. **Acquisition Function**: Expected Improvement (EI)
   - 다음 탐색할 Knob 설정 선택
   - Exploration (탐험) vs Exploitation (활용) 균형

3. **반복 과정**:
   ```
   for i in range(50):
       1. Surrogate Model로 유망한 Knob 설정 제안
       2. 제안된 설정으로 워크로드 실행 → 실제 성능 측정
       3. (설정, 성능) 데이터 추가
       4. Surrogate Model 업데이트 (Random Forest 재학습)
       5. 최적 설정 업데이트
   ```

---

### 4단계: Knob 평가 (Executor)

#### 파일: `executors/executor.py:213-372`, `run_smac.py:72-136`

**SysmlExecutor**: PostgreSQL에 Knob 적용 후 성능 측정

```python
def evaluate_dbms_conf(spaces, executor, storage, columns, sample, state):
    # 1. Knob 설정 준비
    conf = spaces.finalize_conf(sample)  # 설정 포맷 변환
    # 예: {'shared_buffers': 128, 'work_mem': 4} → {'shared_buffers': '128MB', 'work_mem': '4MB'}

    dbms_info = {
        'name': 'postgres',
        'config': conf,
        'version': '13.1'
    }

    # 2. Executor로 평가
    perf_stats = executor.evaluate_configuration(dbms_info, benchmark_info)

    # 3. 성능 메트릭 추출
    if state.target_metric == 'throughput':
        perf = perf_stats['throughput']  # ops/sec
    else:
        perf = perf_stats['latency']     # 95-th percentile latency (ms)

    # 4. 최적 설정 업데이트
    if state.best_perf is None or state.is_better_perf(perf, state.best_perf):
        state.best_conf = sample
        state.best_perf = perf

    # 5. 결과 저장
    storage.store_result_summary({
        'Iteration': state.iter,
        'Performance': perf,
        'Optimum': state.best_perf,
        'Runtime': perf_stats['runtime']
    })
    state.iter += 1

    # SMAC은 항상 최소화하므로, throughput은 음수로 반환
    return perf if state.minimize else -perf
```

**SysmlExecutor 상세** (`executors/executor.py:322-372`):

```python
class SysmlExecutor:
    def evaluate_configuration(self, dbms_info, benchmark_info):
        # 1. SQL 파일 읽기
        with open(self.sqls_file_path, "r") as f:
            sqls = f.readlines()  # 워크로드 쿼리들

        # 2. Knob 적용 (PostgreSQL에 설정 전송)
        self.data_interactor.push_knob(dbms_info["config"])
        # → ALTER SYSTEM SET shared_buffers = '128MB';
        # → ALTER SYSTEM SET work_mem = '4MB';
        # → ... (모든 knob 설정)

        # 3. 실행 시간 수집 시작
        self.data_interactor.pull_execution_time()

        # 4. 워크로드 쿼리 실행
        execution_times = []
        accu_execution_time = 0

        for sql in sqls:
            data = self.data_interactor.execute(sql)
            if data.execution_time is None:
                raise TimeoutError  # 타임아웃 시 실패 처리
            execution_times.append(data.execution_time)
            accu_execution_time += data.execution_time

        # 5. 성능 메트릭 계산
        perf = {
            "latency": sorted(execution_times)[int(0.95 * len(sqls))],  # 95th percentile
            "runtime": accu_execution_time,                              # 총 실행 시간
            "throughput": len(sqls) / accu_execution_time                # 쿼리/초
        }

        # 6. (선택적) DB 내부 메트릭 수집
        if self.parse_metrics:
            metrics = self.get_internal_metrics()  # pg_stat_* 뷰에서 수집
            return perf, metrics

        return perf
```

**DB 내부 메트릭 수집** (`executors/executor.py:258-320`):

```python
def get_internal_metrics(self):
    # PostgreSQL 통계 뷰에서 메트릭 수집
    PG_STAT_VIEWS = [
        "pg_stat_archiver",           # 아카이버 통계
        "pg_stat_bgwriter",           # Background Writer 통계
        "pg_stat_database",           # 데이터베이스 통계
        "pg_stat_database_conflicts", # 충돌 통계
        "pg_stat_user_tables",        # 테이블 통계
        "pg_statio_user_tables",      # 테이블 I/O 통계
        "pg_stat_user_indexes",       # 인덱스 통계
        "pg_statio_user_indexes"      # 인덱스 I/O 통계
    ]

    metrics_dict = {'global': {}, 'local': {}}

    for view in PG_STAT_VIEWS:
        results = self.db_controller.execute(f"SELECT * FROM {view}")
        # 예: pg_stat_bgwriter → buffers_alloc, buffers_backend, ...
        # 예: pg_stat_database → blks_hit, blks_read, tup_fetched, ...
        metrics_dict[...][view] = results

    # 메트릭 집계 (합산)
    valid_metrics = {}
    for name, values in metrics.items():
        if name in NUMERIC_METRICS:
            valid_metrics[name] = sum(values)

    return valid_metrics
    # 반환 예: {'buffers_alloc': 12345, 'blks_hit': 98765, ...}
```

**메트릭 예시**:

- `buffers_alloc`: 할당된 버퍼 수
- `blks_hit`: 캐시 히트 블록 수
- `blks_read`: 디스크에서 읽은 블록 수
- `tup_fetched`: 가져온 튜플 수
- `idx_scan`: 인덱스 스캔 횟수
- 총 60개 메트릭 (선택적 수집)

---

### 5단계: 최적 Knob 적용

#### 파일: `EventImplement.py:176-179`

```python
# 최적 설정을 파일에 저장
db_controller.write_knob_to_file(dict(exp_state.best_conf))
# → postgresql.auto.conf 파일에 기록
# 예:
# shared_buffers = '256MB'
# work_mem = '8MB'
# effective_cache_size = '2GB'
# ...

# DB 재시작하여 적용
db_controller.restart()
```

**적용 과정**:

1. `write_knob_to_file()`: `postgresql.auto.conf`에 최적 설정 기록
2. `restart()`: PostgreSQL 재시작
3. PostgreSQL은 시작 시 `postgresql.auto.conf` 읽어서 설정 적용
4. 이후 모든 쿼리는 최적화된 Knob으로 실행됨

---

## 데이터 흐름 요약

### 입력 데이터

1. **워크로드 쿼리** (Training SQLs):
   - 형식: SQL 문자열 리스트
   - 출처: `Dataset.read_train_sql()`
   - 예시: `["SELECT * FROM posts WHERE id = 1", "SELECT COUNT(*) FROM users", ...]`
   - 용도: 각 Knob 설정의 성능 평가

2. **Knob 정의** (Configuration Space):
   - 형식: JSON (postgres-13.json)
   - 내용: Knob 이름, 타입, 범위, 기본값
   - 예시: `{"name": "shared_buffers", "type": "integer", "min": 16, "max": 8192, "default": 128}`
   - 용도: 탐색 공간 정의

### 중간 데이터

1. **Knob 설정 샘플** (Configuration):
   - 형식: Dict[str, Union[int, float, str]]
   - 예시: `{"shared_buffers": 256, "work_mem": 8, "random_page_cost": 1.5}`
   - 생성: SMAC Optimizer가 제안
   - 처리: `finalize_conf()` → PostgreSQL 형식으로 변환 → `push_knob()`

2. **성능 메트릭** (Performance Stats):
   - 형식: Dict[str, float]
   - 내용:
     - `throughput`: 쿼리 처리량 (ops/sec)
     - `latency`: 95번째 백분위 지연시간 (ms)
     - `runtime`: 총 실행 시간 (sec)
   - 예시: `{"throughput": 1500.5, "latency": 25.3, "runtime": 120.5}`
   - 수집: `SysmlExecutor.evaluate_configuration()`

3. **DB 내부 메트릭** (Optional):
   - 형식: Dict[str, float] (60개 메트릭)
   - 내용: `pg_stat_*` 뷰의 통계
   - 예시: `{"buffers_alloc": 12345, "blks_hit": 98765, ...}`
   - 용도: 고급 분석 (현재는 미사용)

### 출력 데이터

1. **최적 Knob 설정** (Best Configuration):
   - 형식: Dict[str, Union[int, float, str]]
   - 저장 위치: `postgresql.auto.conf`
   - 예시:
     ```
     shared_buffers = '256MB'
     work_mem = '8MB'
     effective_cache_size = '2GB'
     random_page_cost = 1.5
     ```

2. **최적 성능**:
   - 형식: float
   - 내용: 최적 설정에서의 성능 메트릭
   - 예시: `throughput: 2500 ops/sec`

3. **최적화 히스토리** (Storage):
   - 형식: CSV 또는 JSON
   - 내용: 각 반복의 설정, 성능, 최적값
   - 저장 위치: `results/{benchmark}.{workload}/seed{seed}/`

---

## Surrogate Model 상세 분석

### Random Forest란?

**Random Forest**: 여러 개의 결정 트리(Decision Tree)를 앙상블한 모델

```
입력: Knob 설정 (고차원 벡터)
  ↓
100개의 Decision Tree
  ↓ ↓ ↓ ↓ ↓
  각 트리가 성능 예측
  ↓ ↓ ↓ ↓ ↓
평균 계산 → 최종 성능 예측
```

### 입력 (Features)

- **Feature**: Knob 설정값
- **차원**: 탐색하는 Knob 개수 (예: 100개)
- **타입**: 혼합 (정수, 실수, 카테고리)
- **예시**: `[shared_buffers=256, work_mem=8, autovacuum=on, ...]`

**HesBO 사용 시**: 100차원 → 16차원으로 축소 (Linear Embedding)

### 출력 (Target)

- **Target**: 성능 메트릭 (throughput 또는 latency)
- **타입**: 실수 (float)
- **예시**: `throughput = 1500.5 ops/sec`

### 학습 과정

```python
# 초기: 10개 랜덤 샘플로 Random Forest 학습
samples = [(config_1, perf_1), (config_2, perf_2), ...]
rf_model.fit(samples)

# 반복 (50회):
for i in range(10, 50):
    # 1. Acquisition Function으로 다음 탐색 지점 선택
    next_config = acquisition_function(rf_model)
    # Expected Improvement (EI): 개선이 기대되는 설정 선택

    # 2. 실제 평가
    perf = evaluate_configuration(next_config)

    # 3. 데이터 추가
    samples.append((next_config, perf))

    # 4. Random Forest 재학습
    rf_model.fit(samples)
```

### Random Forest 설정

- **트리 개수**: 100개
- **최대 깊이**: 무제한 (2^20)
- **최소 분할 샘플**: 2개
- **최소 리프 샘플**: 3개
- **Feature 비율**: 100% (모든 feature 사용)

### 예측 및 불확실성

Random Forest는 **예측값**과 **불확실성**을 모두 제공:

```python
# 예측
mean, std = rf_model.predict(config, return_std=True)
# mean: 평균 성능 예측 (100개 트리의 평균)
# std: 표준편차 (100개 트리의 분산)

# Acquisition Function (Expected Improvement)
EI = (mean - current_best) * Φ((mean - current_best) / std) + std * φ((mean - current_best) / std)
# Φ: 누적 분포 함수
# φ: 확률 밀도 함수
# → mean이 크고, std가 클수록 탐색 우선순위 높음
```

**Exploration vs Exploitation**:

- **Exploitation**: `mean`이 큰 영역 → 성능이 좋을 것으로 예상되는 곳
- **Exploration**: `std`가 큰 영역 → 불확실한 곳 (탐색되지 않은 곳)

---

## 학습하는 것 vs 추정하는 것

### KnobTuning은 ML 모델 학습이 아닌 "탐색"

**MSCN/Lero와의 차이**:

| 항목 | MSCN/Lero | KnobTuning |
|------|-----------|------------|
| **목적** | 카디널리티 예측 모델 학습 | 최적 Knob 설정 찾기 |
| **입력** | 쿼리 특징 (조건, 조인 등) | Knob 설정 |
| **출력** | 카디널리티 예측값 | 성능 메트릭 (throughput/latency) |
| **학습 데이터** | (쿼리, 실제 카디널리티) 쌍 | (Knob 설정, 성능) 쌍 |
| **모델** | 신경망 (MSCN) 또는 강화학습 (Lero) | Random Forest (Surrogate) |
| **학습 방식** | Offline 학습 (사전 학습) | Online 학습 (탐색 중 학습) |
| **적용** | 쿼리마다 카디널리티 힌트 주입 | DB 시작 시 Knob 설정 적용 (1회) |

### Random Forest Surrogate Model

**학습하는 것**:

```
입력: Knob 설정 (예: {shared_buffers: 256, work_mem: 8, ...})
출력: 성능 예측 (예: throughput: 1500 ops/sec)
```

- **학습 데이터**: 이전에 평가한 (Knob 설정, 성능) 쌍들
- **학습 목표**: Knob 설정 → 성능의 관계를 모델링
- **활용**: 다음 탐색할 Knob 설정 제안 (Acquisition Function)

**추정하는 것**:

```
새로운 Knob 설정 → Random Forest → 성능 예측 (평균 ± 표준편차)
```

- 아직 평가하지 않은 Knob 설정의 성능을 예측
- 불확실성(표준편차)도 함께 제공 → Exploration 가이드

### 최종 목표: 최적 Knob 찾기

```
50회 반복 후:
    → 최적 Knob 설정: {shared_buffers: 256, work_mem: 8, ...}
    → 최적 성능: throughput: 2500 ops/sec
```

- Random Forest 모델 자체는 **버려짐** (저장 안 함)
- **최적 Knob 설정**만 저장되어 DB에 적용됨
- 다음 실행 시 처음부터 다시 탐색 (또는 이전 결과 로드 가능)

---

## 적합 워크로드

- 워크로드가 명확하게 정의된 경우
- DB 설정 조정으로 성능 개선 원함
- OLTP, OLAP 모두 가능

## 튜닝 대상 Knob

PostgreSQL: `shared_buffers`, `work_mem`, `effective_cache_size`, `random_page_cost` 등 100+ knobs

## 파일

```
KnobTuning/
├── KnobPresetScheduler.py      # PostgreSQL용
├── SparkKnobPresetScheduler.py # Spark용
├── EventImplement.py           # 탐색 로직
└── llamatune/                  # LlamaTune (Bayesian Optimization)
```

## 사용

```python
from algorithm_examples.KnobTuning.KnobPresetScheduler import get_knob_preset_scheduler

scheduler, tracker = get_knob_preset_scheduler(
    config,
    enable_collection=True,
    enable_training=True,
    dataset_name="your_db"
)
# scheduler.init() 시 최적 Knob 자동 탐색 및 적용
```

## 주요 컴포넌트

**EventImplement**: 워크로드 실행 및 Knob 탐색 (Bayesian Optimization)

**LlamaTune**: Bayesian Optimization 기반 탐색 알고리즘

**특이사항**:
- 모델 학습 아닌 **탐색(Search)** 수행
- 워크로드 전체를 반복 실행 (num_epoch 횟수)

## KnobTuning vs MSCN/Lero

| | KnobTuning | MSCN/Lero |
|---|---|---|
| 목적 | DB 설정 최적화 | 쿼리 최적화 |
| 적용 시점 | DB 시작 시 (전역) | 쿼리 실행 시 (쿼리별) |
| 변경 빈도 | 낮음 (한 번) | 높음 (쿼리마다) |
| 방식 | 탐색 | 학습 |

**조합 사용**: KnobTuning (DB 전역) + MSCN/Lero (개별 쿼리)

## 수정 시 주의

**탐색 Knob 범위**: 너무 넓으면 탐색 시간 증가, 너무 좁으면 최적 못 찾음

**워크로드 정의**: 대표 워크로드가 중요, 학습 쿼리와 실제 유사해야

**탐색 알고리즘 변경**: LlamaTune 외 Grid Search, Random Search, Genetic Algorithm 등 가능

## 성능 튜닝

**탐색 속도**: `num_epoch` 감소, 워크로드 크기 축소 (`num_training`), 탐색 공간 축소

**더 나은 Knob**: 더 많은 반복 (`num_epoch=200`), 더 넓은 탐색 공간, 실제와 유사한 워크로드

## 문제 해결

- 탐색 느림 → `num_epoch`, `num_training` 감소, 탐색 Knob 수 감소
- Baseline보다 나쁨 → 탐색 부족 (epoch 증가), 탐색 공간 확대, 워크로드 재선정
- Knob 적용 안 됨 → PostgreSQL 재시작 필요한 Knob 확인, 권한 확인 (`ALTER SYSTEM`)
- 워크로드마다 다름 → 정상 (Knob은 워크로드별 튜닝 필요)
