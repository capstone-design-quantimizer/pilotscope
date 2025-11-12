# Algorithm Examples - 공통 구현 패턴

이 문서는 PilotScope의 모든 AI4DB 알고리즘이 따르는 **공통 구현 패턴**을 설명합니다. 새로운 알고리즘 개발 또는 기존 알고리즘 수정 시 이 패턴을 따라야 합니다.

## 알고리즘 폴더 구조

모든 알고리즘은 다음 4개의 핵심 파일을 포함합니다:

```
algorithm_examples/YourAlgorithm/
├── YourAlgorithmPresetScheduler.py    # 팩토리 함수 (진입점)
├── YourAlgorithmPilotModel.py         # 모델 래퍼
├── YourAlgorithmHandler.py            # DB 인터셉션 로직
└── EventImplement.py                  # 학습/데이터 수집 로직
```

**예시**:
- `algorithm_examples/Mscn/`: MSCN 카디널리티 추정
- `algorithm_examples/Lero/`: Lero 쿼리 플랜 최적화
- `algorithm_examples/KnobTuning/`: DB 노브 튜닝
- `algorithm_examples/Index/`: 인덱스 선택

## 핵심 컴포넌트

### 1. PresetScheduler (진입점)

**역할**: 알고리즘을 쉽게 사용할 수 있는 팩토리 함수 제공

**파일**: `*PresetScheduler.py`

**함수 시그니처** (모든 알고리즘 공통):

```python
def get_*_preset_scheduler(
    config,                 # PostgreSQLConfig 또는 SparkConfig
    enable_collection,      # 학습 데이터 수집 여부
    enable_training,        # 모델 학습 여부
    num_collection=-1,      # 수집할 쿼리 수 (-1: 전체)
    num_training=-1,        # 학습에 사용할 쿼리 수 (-1: 전체)
    num_epoch=100,          # 학습 에포크
    load_model_id=None,     # 기존 모델 ID (None: 새 모델)
    use_mlflow=True,        # MLflow 사용 여부
    experiment_name=None,   # MLflow 실험 이름
    dataset_name=None       # 데이터셋/워크로드 이름
) -> tuple:                 # (scheduler, mlflow_tracker)
```

**필수 구현 로직**:

```python
def get_mscn_preset_scheduler(config, enable_collection, enable_training, ...):
    # 1. MLflow 초기화
    mlflow_tracker = None
    if use_mlflow and enable_training:
        exp_name = experiment_name if experiment_name else f"mscn_{config.db}"
        mlflow_tracker = MLflowTracker(experiment_name=exp_name)

        hyperparams = {
            "num_epoch": num_epoch,
            "num_training": num_training,
            # ...
        }
        mlflow_tracker.start_training(
            algo_name="mscn",
            dataset=dataset_name if dataset_name else config.db,
            params=hyperparams
        )

    # 2. 모델 로딩 로직 (3가지 경로)
    if load_model_id:
        # 경로 1: 특정 모델 ID로 로드
        model = MscnPilotModel.load_model(load_model_id, "mscn")
    elif not enable_training:
        # 경로 2: MLflow에서 최적 모델 자동 로드
        best_run = MLflowTracker.get_best_run(
            experiment_name=exp_name,
            metric="test_total_time",
            ascending=True
        )
        if best_run:
            model_id = best_run['params'].get('model_id')
            model = MscnPilotModel.load_model(model_id, "mscn")
        else:
            model = MscnPilotModel(model_name, mlflow_tracker=mlflow_tracker)
    else:
        # 경로 3: 새 모델 생성 (학습용)
        model = MscnPilotModel(model_name, mlflow_tracker=mlflow_tracker)
        model.set_training_info(dataset_name, hyperparams)

    # 3. Scheduler 생성 및 구성
    scheduler = SchedulerFactory.create_scheduler(config)

    # 4. Event 등록 (학습/데이터 수집)
    if enable_collection or enable_training:
        data_table = f"mscn_pretraining_{dataset_name if dataset_name else config.db}"
        event = MscnPretrainingModelEvent(
            config, model, data_table,
            enable_collection=enable_collection,
            enable_training=enable_training,
            num_collection=num_collection,
            num_training=num_training,
            num_epoch=num_epoch,
            mlflow_tracker=mlflow_tracker,
            dataset_name=dataset_name
        )
        scheduler.register_events([event])

    # 5. Handler 등록 (AI 로직 주입)
    scheduler.register_custom_handlers([MscnCardPushHandler(model, config)])

    # 6. Required data 등록 (테스트 메트릭 수집)
    test_data_table = f"{model_name}_test_data_table"
    scheduler.register_required_data(test_data_table, pull_execution_time=True)

    # 7. Scheduler에 모델/트래커 첨부
    scheduler.pilot_model = model
    scheduler.mlflow_tracker = mlflow_tracker

    # 8. 초기화 및 반환
    scheduler.init()
    return scheduler, mlflow_tracker
```

**중요**: 모든 알고리즘은 이 구조를 따라야 통일된 인터페이스 제공 가능.

### 2. PilotModel (모델 래퍼)

**역할**: AI 모델을 PilotScope와 연동

**파일**: `*PilotModel.py`

**핵심 메서드**:

```python
class MscnPilotModel(EnhancedPilotModel):
    def __init__(self, model_name, mlflow_tracker=None, save_to_local=False):
        super().__init__(model_name, mlflow_tracker, save_to_local)
        self.model = None  # 실제 AI 모델

    def _load_model_impl(self):
        """모델 초기화 로직"""
        # 예: self.model = MSCN(...)
        pass

    def _save_model_impl(self, model_path):
        """모델 저장 로직"""
        # 예: torch.save(self.model.state_dict(), model_path)
        pass

    def _train_model_impl(self, training_data):
        """모델 학습 로직"""
        # 예: self.model.train(training_data)
        pass

    def _predict_impl(self, input_data):
        """추론 로직"""
        # 예: return self.model.predict(input_data)
        pass
```

**EnhancedPilotModel이 제공하는 기능**:
- 타임스탬프 기반 모델 저장
- MLflow 통합
- 메타데이터 관리
- 모델 로딩/저장 자동화

**주의**: `_impl` 메서드만 구현하면 나머지는 부모 클래스가 처리.

### 3. Handler (DB 인터셉션)

**역할**: DB 쿼리 실행 중 AI 로직 주입

**파일**: `*Handler.py`

**두 가지 타입**:
- **PushHandler**: DB에 힌트 주입 (카디널리티, 플랜 등)
- **PullHandler**: DB에서 데이터 수집 (실행 시간, 플랜 등)

**PushHandler 예시**:

```python
class MscnCardPushHandler(BasePushHandler):
    def __init__(self, pilot_model, config):
        super().__init__(config)
        self.pilot_model = pilot_model

    def acquire_injected_data(self, sql):
        """DB에 주입할 데이터 생성"""
        # 1. SQL 파싱 및 피처 추출
        features = self._extract_features(sql)

        # 2. AI 모델 추론
        predicted_cards = self.pilot_model.predict(features)

        # 3. 카디널리티 힌트 생성
        card_hints = self._format_card_hints(predicted_cards)

        return card_hints

    def inject_data(self, sql, injected_data):
        """DB에 데이터 주입"""
        # PilotDataInteractor를 통해 DB에 힌트 전달
        self.data_interactor.push_card(injected_data, sql)
```

**PullHandler 예시**:

```python
class MetricsCollector(BasePullHandler):
    def pull_data(self, sql, execution_data):
        """DB에서 데이터 수집"""
        # 실행 시간, 플랜 등 수집
        execution_time = self.data_interactor.pull_execution_time(sql)
        physical_plan = self.data_interactor.pull_physical_plan(sql)

        return {
            'sql': sql,
            'execution_time': execution_time,
            'plan': physical_plan
        }
```

**실행 순서**:
1. `scheduler.execute(sql)` 호출
2. `acquire_injected_data(sql)` 실행 (AI 추론)
3. `inject_data(sql, data)` 실행 (DB 힌트 주입)
4. DB가 힌트를 사용하여 쿼리 실행
5. `pull_data(sql, ...)` 실행 (결과 수집)

### 4. Event (학습/데이터 수집)

**역할**: 학습 데이터 수집 및 모델 학습 로직

**파일**: `EventImplement.py`

**PretrainingModelEvent 구조**:

```python
class MscnPretrainingModelEvent(PretrainingModelEvent):
    def __init__(self, config, pilot_model, data_save_table,
                 enable_collection, enable_training,
                 num_collection, num_training, num_epoch,
                 mlflow_tracker, dataset_name):
        super().__init__(config, pilot_model, data_save_table)
        self.enable_collection = enable_collection
        self.enable_training = enable_training
        self.num_collection = num_collection
        self.num_training = num_training
        self.num_epoch = num_epoch
        self.mlflow_tracker = mlflow_tracker
        self.dataset_name = dataset_name

    def iterative_data_collection(self):
        """학습 데이터 수집"""
        if not self.enable_collection:
            return

        # 1. Dataset에서 학습 쿼리 로드
        dataset = load_dataset(self.dataset_name)
        training_queries = dataset.read_train_sql()

        # 2. 쿼리 실행하며 데이터 수집
        for i, sql in enumerate(training_queries):
            if self.num_collection > 0 and i >= self.num_collection:
                break

            # DB 실행하여 ground truth 수집
            execution_time = self.data_interactor.pull_execution_time(sql)
            cardinality = self.data_interactor.pull_subquery_card(sql)

            # 데이터 저장
            self.data_manager.save(self.data_save_table, {
                'sql': sql,
                'execution_time': execution_time,
                'cardinality': cardinality
            })

    def custom_model_training(self):
        """모델 학습"""
        if not self.enable_training:
            return

        # 1. 학습 데이터 로드
        training_data = self.data_manager.read(
            self.data_save_table,
            limit=self.num_training
        )

        # 2. 모델 학습
        self.pilot_model.train(training_data, num_epoch=self.num_epoch)

        # 3. 모델 저장
        model_id = self.pilot_model.save_model()

        # 4. MLflow에 메트릭 기록
        if self.mlflow_tracker:
            self.mlflow_tracker.log_metric("train_samples", len(training_data))
            self.mlflow_tracker.log_param("model_id", model_id)
```

**실행 시점**: `scheduler.init()` 호출 시 자동 실행

## 공통 파라미터 규칙

모든 알고리즘의 `get_*_preset_scheduler()` 함수는 동일한 파라미터 시그니처를 따라야 합니다:

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `config` | Config | 필수 | PostgreSQLConfig 또는 SparkConfig |
| `enable_collection` | bool | 필수 | 학습 데이터 수집 여부 |
| `enable_training` | bool | 필수 | 모델 학습 여부 |
| `num_collection` | int | -1 | 수집할 쿼리 수 (-1: 전체) |
| `num_training` | int | -1 | 학습 쿼리 수 (-1: 전체) |
| `num_epoch` | int | 100 | 학습 에포크 |
| `load_model_id` | str | None | 로드할 모델 ID |
| `use_mlflow` | bool | True | MLflow 사용 여부 |
| `experiment_name` | str | None | MLflow 실험 이름 |
| `dataset_name` | str | None | 데이터셋/워크로드 이름 |

**중요**: 이 규칙을 따라야 `unified_test.py`에서 모든 알고리즘을 통일된 방식으로 실행할 수 있습니다.

## 데이터 테이블 네이밍 규칙

학습 데이터와 테스트 데이터를 분리하여 관리:

```python
# 학습 데이터 테이블
pretraining_data_table = f"{algo_name}_pretraining_{dataset_name}"
# 예: "mscn_pretraining_stats_tiny"
# 예: "mscn_pretraining_production_oltp"

# 테스트 데이터 테이블
test_data_table = f"{algo_name}_test_data_table"
# 예: "mscn_test_data_table"
```

**이유**: `dataset_name`을 포함하여 여러 워크로드의 데이터를 분리 저장.

## MLflow 통합 규칙

모든 알고리즘은 MLflow를 통해 실험을 추적해야 합니다:

### 1. Experiment 네이밍

```python
# 기본: "{algo_name}_{db_name}"
experiment_name = f"mscn_{config.db}"  # 예: "mscn_stats_tiny"

# 커스텀 (사용자 지정 시)
experiment_name = user_provided_name  # 예: "production_tuning"
```

### 2. Run 정보 기록

```python
# 학습 시작 시
mlflow_tracker.start_training(
    algo_name="mscn",
    dataset=dataset_name,
    params={
        "num_epoch": num_epoch,
        "num_training": num_training,
        # ...
    },
    db_name=config.db,
    workload=workload,  # OLTP, OLAP, Mixed 등
    num_queries=num_training
)

# 테스트 시작 시
mlflow_tracker.start_testing(
    algo_name="mscn",
    dataset=dataset_name,
    db_name=config.db,
    workload=workload
)

# 메트릭 기록
mlflow_tracker.log_metric("test_total_time", total_time)
mlflow_tracker.log_metric("test_queries", num_queries)

# 종료
mlflow_tracker.end_run()
```

## 새로운 알고리즘 추가 체크리스트

새 알고리즘을 추가할 때 다음을 확인하세요:

- [ ] 폴더 생성: `algorithm_examples/YourAlgorithm/`
- [ ] 4개 파일 작성:
  - [ ] `YourAlgorithmPresetScheduler.py`
  - [ ] `YourAlgorithmPilotModel.py`
  - [ ] `YourAlgorithmHandler.py`
  - [ ] `EventImplement.py`
- [ ] `get_*_preset_scheduler()` 함수 시그니처 준수
- [ ] MLflow 통합
- [ ] `unified_test.py`의 `ALGORITHM_REGISTRY`에 등록:
  ```python
  ALGORITHM_REGISTRY = {
      # ...
      'youralgo': {
          'module': 'algorithm_examples.YourAlgorithm.YourAlgorithmPresetScheduler',
          'function': 'get_youralgo_preset_scheduler',
          'description': 'Your algorithm description'
      }
  }
  ```
- [ ] 테스트 작성: `test_example_algorithms/test_youralgo_example.py`
- [ ] 문서 작성: `algorithm_examples/YourAlgorithm/CLAUDE.md`

## 기존 알고리즘 수정 시 주의사항

### 파라미터 시그니처 변경 금지

모든 알고리즘은 동일한 파라미터를 받아야 합니다. 알고리즘별 특수 파라미터가 필요한 경우:

```python
# 나쁜 예: 시그니처 변경
def get_mscn_preset_scheduler(config, enable_collection, enable_training,
                               special_mscn_param):  # ❌
    pass

# 좋은 예: kwargs 사용
def get_mscn_preset_scheduler(config, enable_collection, enable_training,
                               num_collection=-1, num_training=-1, num_epoch=100,
                               load_model_id=None, use_mlflow=True,
                               experiment_name=None, dataset_name=None,
                               **kwargs):  # ✅
    special_mscn_param = kwargs.get('special_mscn_param', default_value)
    # ...
```

### 데이터 테이블 변경 시

데이터 테이블 이름을 변경할 경우, 기존 데이터와 충돌하지 않도록 주의:

```python
# 이전 버전
old_table = "mscn_pretraining_data"

# 새 버전 (dataset_name 추가)
new_table = f"mscn_pretraining_{dataset_name}"

# 마이그레이션 필요 시
if data_manager.table_exists(old_table) and not data_manager.table_exists(new_table):
    data_manager.copy_table(old_table, new_table)
```

## 알고리즘별 상세 문서

각 알고리즘의 구체적인 구현 세부사항은 개별 CLAUDE.md 참조:

- **[Mscn/CLAUDE.md](Mscn/CLAUDE.md)** - MSCN 카디널리티 추정
- **[Lero/CLAUDE.md](Lero/CLAUDE.md)** - Lero 쿼리 플랜 최적화
- **[KnobTuning/CLAUDE.md](KnobTuning/CLAUDE.md)** - DB 노브 튜닝
- **[Index/CLAUDE.md](Index/CLAUDE.md)** - 인덱스 선택

## 문제 해결

### Q1. 특정 알고리즘만 문제가 발생
- 다른 알고리즘의 구현과 비교하여 공통 패턴 준수 확인
- PresetScheduler의 파라미터 시그니처 확인
- MLflow 통합이 올바르게 구현되었는지 확인

### Q2. unified_test.py에서 인식 안 됨
- `ALGORITHM_REGISTRY`에 등록했는지 확인
- 모듈 경로가 올바른지 확인
- 함수 이름이 일치하는지 확인

### Q3. MLflow에서 결과를 찾을 수 없음
- `mlflow_tracker.start_training()` 호출 확인
- `experiment_name` 일관성 확인
- `mlflow_tracker.end_run()` 호출 확인

---

**핵심**: 모든 알고리즘은 동일한 구조를 따라야 통일된 인터페이스 제공 및 유지보수가 가능합니다.
