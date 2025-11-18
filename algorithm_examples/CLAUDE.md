# Algorithm Implementation Pattern

모든 알고리즘은 동일한 구조를 따라야 함.

## 필수 파일

```
algorithm_examples/YourAlgorithm/
├── YourAlgorithmPresetScheduler.py    # 팩토리 (진입점)
├── YourAlgorithmPilotModel.py         # 모델 래퍼
├── YourAlgorithmHandler.py            # DB 인터셉션
└── EventImplement.py                  # 학습/수집
```

## PresetScheduler (팩토리)

**함수 시그니처** (모든 알고리즘 통일):

```python
def get_*_preset_scheduler(
    config,                 # PostgreSQLConfig or SparkConfig
    enable_collection,      # bool
    enable_training,        # bool
    num_collection=-1,      # int (-1: all)
    num_training=-1,        # int (-1: all)
    num_epoch=100,          # int
    load_model_id=None,     # str or None
    use_mlflow=True,        # bool
    experiment_name=None,   # str or None
    dataset_name=None       # str or None
) -> tuple:                 # (scheduler, mlflow_tracker)
```

**필수 로직**:

```python
# 1. MLflow 초기화
mlflow_tracker = MLflowTracker(experiment_name) if use_mlflow else None

# 2. 모델 로딩 (3가지 경로)
if load_model_id:
    model = YourModel.load_model(load_model_id, "algo_name")
elif not enable_training:
    # MLflow에서 최적 모델 자동 로드
    best_run = MLflowTracker.get_best_run(...)
    model = YourModel.load_model(best_run['params']['model_id'], "algo_name")
else:
    model = YourModel(model_name, mlflow_tracker)

# 3. Scheduler 구성
scheduler = SchedulerFactory.create_scheduler(config)

# 4. Event 등록
if enable_collection or enable_training:
    data_table = f"{algo_name}_pretraining_{dataset_name}"
    event = YourPretrainingEvent(config, model, data_table, ...)
    scheduler.register_events([event])

# 5. Handler 등록
scheduler.register_custom_handlers([YourHandler(model, config)])

# 6. Required data 등록
scheduler.register_required_data(test_table, pull_execution_time=True)

# 7. 첨부 및 초기화
scheduler.pilot_model = model
scheduler.mlflow_tracker = mlflow_tracker
scheduler.init()
return scheduler, mlflow_tracker
```

## PilotModel (모델 래퍼)

```python
class YourPilotModel(EnhancedPilotModel):
    def _load_model_impl(self): pass  # 모델 초기화
    def _save_model_impl(self, path): pass  # 모델 저장
    def _train_model_impl(self, data): pass  # 학습
    def _predict_impl(self, input): pass  # 추론
```

## Handler (DB 인터셉션)

**PushHandler**: DB 힌트 주입

```python
class YourPushHandler(BasePushHandler):
    def acquire_injected_data(self, sql):
        # AI 추론
        return hints

    def inject_data(self, sql, hints):
        # DB에 주입
        self.data_interactor.push_card(hints, sql)
```

**PullHandler**: 데이터 수집

```python
class YourPullHandler(BasePullHandler):
    def pull_data(self, sql, execution_data):
        return self.data_interactor.pull_execution_time(sql)
```

## Event (학습/수집)

```python
class YourPretrainingModelEvent(PretrainingModelEvent):
    def iterative_data_collection(self):
        # 학습 데이터 수집
        for sql in training_queries:
            data = execute_and_collect(sql)
            self.data_manager.save(self.data_save_table, data)

    def custom_model_training(self):
        # 모델 학습
        data = self.data_manager.read(self.data_save_table)
        self.pilot_model.train(data, num_epoch=self.num_epoch)
        model_id = self.pilot_model.save_model()
        self.mlflow_tracker.log_param("model_id", model_id)
```

## 공통 규칙

**파라미터 시그니처**: 모든 알고리즘 통일 필수 (unified_test.py 호환성)

**데이터 테이블 네이밍**:
- 학습: `{algo_name}_pretraining_{dataset_name}`
- 테스트: `{algo_name}_test_data_table`

**MLflow Experiment 네이밍**:
- 기본: `{algo_name}_{db_name}`
- 커스텀: `experiment_name` 파라미터 사용

**시그니처 변경 금지**: 알고리즘별 특수 파라미터는 `**kwargs` 사용

## 새 알고리즘 추가

1. 폴더 생성: `algorithm_examples/YourAlgorithm/`
2. 4개 파일 작성 (위 구조 준수)
3. `unified_test.py` ALGORITHM_REGISTRY 등록:
   ```python
   'youralgo': {
       'module': 'algorithm_examples.YourAlgorithm.YourAlgorithmPresetScheduler',
       'function': 'get_youralgo_preset_scheduler',
       'description': 'Your algorithm'
   }
   ```

## 수정 시 주의

- 파라미터 시그니처 변경 금지
- 데이터 테이블 이름 변경 시 마이그레이션 필요
- 피처 형식 변경 시 기존 모델과 비호환
