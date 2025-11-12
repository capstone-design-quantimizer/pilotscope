# MSCN (Multi-Set Convolutional Network)

MSCN은 **카디널리티 추정(Cardinality Estimation)**을 위한 딥러닝 알고리즘입니다.

## 알고리즘 개요

**목적**: PostgreSQL 옵티마이저의 카디널리티 추정을 AI 모델로 대체하여 쿼리 플랜 품질 향상

**작동 방식**:
1. SQL 쿼리를 피처 벡터로 변환
2. MSCN 모델로 각 서브쿼리의 카디널리티 예측
3. 예측된 카디널리티를 PostgreSQL에 힌트로 주입
4. PostgreSQL이 AI 예측을 사용하여 실행 플랜 생성

**적합한 워크로드**:
- 복잡한 JOIN 쿼리가 많은 경우
- 옵티마이저의 카디널리티 추정이 부정확한 경우
- OLAP 워크로드 (복잡한 분석 쿼리)

## 파일 구조

```
Mscn/
├── MscnPresetScheduler.py          # 팩토리 함수 (진입점)
├── MscnPilotModel.py               # 모델 래퍼
├── MscnParadigmCardAnchorHandler.py # 카디널리티 힌트 주입
├── EventImplement.py               # 학습/데이터 수집
└── source/                         # MSCN 모델 구현
    └── mscn.py
```

## 사용 방법

### 기본 사용

```python
from pilotscope.DBInteractor.PilotDataInteractor import PostgreSQLConfig
from algorithm_examples.Mscn.MscnPresetScheduler import get_mscn_preset_scheduler

# 설정
config = PostgreSQLConfig(db="your_db")

# Scheduler 생성
scheduler, tracker = get_mscn_preset_scheduler(
    config,
    enable_collection=True,   # 학습 데이터 수집
    enable_training=True,     # 모델 학습
    num_epoch=100,
    dataset_name="your_db"
)

# 테스트 쿼리 실행
test_queries = load_test_queries()
for sql in test_queries:
    result = scheduler.execute(sql)
    print(f"Execution time: {result.execution_time}ms")
```

### 기존 모델 사용

```python
# MLflow에서 최적 모델 자동 로드
scheduler, tracker = get_mscn_preset_scheduler(
    config,
    enable_collection=False,
    enable_training=False,
    dataset_name="your_db"
)

# 또는 특정 모델 ID 지정
scheduler, tracker = get_mscn_preset_scheduler(
    config,
    enable_collection=False,
    enable_training=False,
    load_model_id="mscn_20241112_120000"
)
```

## 주요 컴포넌트

### 1. MscnPresetScheduler

**역할**: MSCN 알고리즘 설정 및 초기화

**주요 파라미터**:
- `num_epoch`: 학습 에포크 수 (기본: 100)
- `num_training`: 학습 쿼리 수 (기본: -1, 전체)
- `num_collection`: 수집 쿼리 수 (기본: -1, 전체)

**특이사항**:
- Lero와 달리 데이터 수집 시 기존 데이터를 자동으로 삭제하지 않음
- 동일한 `dataset_name`으로 여러 번 실행 시 데이터 누적

### 2. MscnPilotModel

**역할**: MSCN 모델 래퍼

**핵심 메서드**:
- `_load_model_impl()`: MSCN 모델 초기화
- `_train_model_impl(training_data)`: 모델 학습
- `_predict_impl(query_features)`: 카디널리티 예측
- `_save_model_impl(model_path)`: 모델 저장

**모델 저장 위치**: `ExampleData/Mscn/Model/mscn_{timestamp}`

### 3. MscnCardPushHandler

**역할**: AI 예측 카디널리티를 PostgreSQL에 주입

**핵심 메서드**:
- `acquire_injected_data(sql)`: 쿼리의 카디널리티 예측
- `inject_data(sql, card_hints)`: PostgreSQL에 힌트 주입

**작동 원리**:
```python
# 1. 쿼리 피처 추출
features = self._extract_features(sql)

# 2. MSCN 모델 추론
predicted_cards = self.pilot_model.predict(features)
# 예: {"users": 1000, "orders": 5000, "join_users_orders": 3000}

# 3. PostgreSQL에 힌트 주입
self.data_interactor.push_card(predicted_cards, sql)
# PostgreSQL은 이 카디널리티를 사용하여 실행 플랜 생성
```

### 4. MscnPretrainingModelEvent

**역할**: 학습 데이터 수집 및 모델 학습

**데이터 수집**:
```python
def iterative_data_collection(self):
    # 1. 학습 쿼리 로드
    training_queries = dataset.read_train_sql()

    # 2. 각 쿼리 실행하여 ground truth 수집
    for sql in training_queries:
        # PostgreSQL 실행하여 실제 카디널리티 획득
        true_cards = self.data_interactor.pull_subquery_card(sql)
        execution_time = self.data_interactor.pull_execution_time(sql)

        # 데이터 저장
        self.data_manager.save(self.data_save_table, {
            'sql': sql,
            'true_cards': true_cards,
            'execution_time': execution_time
        })
```

**모델 학습**:
```python
def custom_model_training(self):
    # 1. 데이터 로드
    training_data = self.data_manager.read(self.data_save_table)

    # 2. 피처 추출
    features, labels = self._prepare_training_data(training_data)

    # 3. MSCN 모델 학습
    self.pilot_model.train(features, labels, num_epoch=self.num_epoch)

    # 4. 모델 저장 및 MLflow 기록
    model_id = self.pilot_model.save_model()
    self.mlflow_tracker.log_param("model_id", model_id)
```

## 수정 시 주의사항

### 1. 피처 추출 로직 변경

**위치**: `MscnPilotModel.py` 또는 `source/mscn.py`

**주의**:
- 피처 형식이 변경되면 기존 모델과 호환되지 않음
- 새로운 피처 버전은 별도의 `model_name`으로 관리 권장

```python
# 나쁜 예: 기존 모델과 호환 불가
def _extract_features(self, sql):
    return new_feature_format(sql)  # ❌ 기존 모델 사용 불가

# 좋은 예: 버전 분리
class MscnPilotModelV2(MscnPilotModel):
    def __init__(self, model_name="mscn_v2", ...):  # ✅
        super().__init__(model_name, ...)
```

### 2. 하이퍼파라미터 변경

**위치**: `MscnPresetScheduler.py` 또는 `source/mscn.py`

**권장 방법**:
- `get_mscn_preset_scheduler()` 파라미터로 제공
- 하드코딩하지 말고 외부에서 주입

```python
# 나쁜 예: 하드코딩
def get_mscn_preset_scheduler(...):
    learning_rate = 0.001  # ❌ 하드코딩

# 좋은 예: 파라미터로 받기
def get_mscn_preset_scheduler(..., **kwargs):
    learning_rate = kwargs.get('learning_rate', 0.001)  # ✅
```

### 3. 데이터 수집 로직 변경

**위치**: `EventImplement.py`

**주의**:
- `data_save_table` 이름 변경 시 기존 데이터 접근 불가
- 데이터 스키마 변경 시 기존 데이터와 호환 불가

```python
# 기존 데이터 활용
old_table = "mscn_pretraining_old"
new_table = f"mscn_pretraining_{dataset_name}"

if data_manager.table_exists(old_table):
    data_manager.copy_table(old_table, new_table)
```

### 4. Handler 로직 변경

**위치**: `MscnParadigmCardAnchorHandler.py`

**주의**:
- `push_card()` 인터페이스는 PilotScope 코어에 정의
- 임의로 변경하면 다른 알고리즘에 영향

```python
# 올바른 방법: 표준 인터페이스 사용
self.data_interactor.push_card(card_dict, sql)

# 잘못된 방법: 커스텀 메서드 추가
self.data_interactor.push_card_custom(...)  # ❌ 코어 수정 필요
```

## 성능 튜닝

### 학습 속도 개선

```python
# 1. 학습 데이터 크기 제한
scheduler, tracker = get_mscn_preset_scheduler(
    config,
    enable_collection=True,
    enable_training=True,
    num_collection=500,   # 500개만 수집
    num_training=500,     # 500개로 학습
    num_epoch=50          # 에포크 감소
)

# 2. 빠른 쿼리만 필터링
# train.txt에서 실행 시간 짧은 쿼리만 선별
```

### 예측 정확도 향상

```python
# 1. 더 많은 학습 데이터
num_collection=2000
num_training=2000

# 2. 더 많은 에포크
num_epoch=200

# 3. 학습 쿼리와 테스트 쿼리의 분포 유사하게 조정
```

## 문제 해결

### Q1. 학습이 너무 느림
- `num_collection`, `num_training` 감소
- 복잡한 쿼리 제외
- GPU 사용 (MSCN은 CPU 기반이지만, PyTorch 버전으로 변경 시 GPU 가능)

### Q2. 예측이 부정확함
- 학습 데이터 부족: 최소 500개 이상 권장
- 학습 쿼리와 테스트 쿼리 분포 차이: 비슷한 패턴으로 조정
- 통계 정보 업데이트: `ANALYZE` 실행

### Q3. 모델이 저장되지 않음
- `ExampleData/Mscn/Model/` 폴더 권한 확인
- 디스크 공간 확인
- `save_to_local=False`인 경우 저장 안 됨 (의도된 동작)

### Q4. Baseline보다 성능이 나쁨
- MSCN은 복잡한 JOIN 쿼리에 효과적
- 단순 쿼리에는 오버헤드가 더 클 수 있음
- 워크로드에 맞는 알고리즘 선택 (Lero 등 다른 알고리즘 시도)

## 관련 문서

- **공통 패턴**: [algorithm_examples/CLAUDE.md](../CLAUDE.md)
- **Lero 비교**: [algorithm_examples/Lero/CLAUDE.md](../Lero/CLAUDE.md)
- **상세 가이드**: [docs/PRODUCTION_OPTIMIZATION.md](../../docs/PRODUCTION_OPTIMIZATION.md)

## 참고 논문

- Kipf et al., "Learned Cardinalities: Estimating Cardinality with Deep Learning"
- PilotScope 논문: `paper/PilotScope.pdf`
