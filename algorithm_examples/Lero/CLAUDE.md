# Lero (Learning to Rewrite Optimizer)

Lero는 **쿼리 플랜 최적화(Query Plan Optimization)**를 위한 강화학습 기반 알고리즘입니다.

## 알고리즘 개요

**목적**: PostgreSQL 옵티마이저가 생성한 여러 쿼리 플랜 중 최적의 플랜을 선택

**작동 방식**:
1. PostgreSQL에서 여러 가능한 실행 플랜 생성 (hint 변경)
2. Lero 모델로 각 플랜의 비용 예측
3. 예측 비용이 가장 낮은 플랜 선택
4. 선택된 플랜으로 쿼리 실행

**적합한 워크로드**:
- 쿼리 플랜이 다양하게 나올 수 있는 경우
- JOIN 순서가 성능에 큰 영향을 미치는 경우
- OLAP 워크로드 (복잡한 분석 쿼리)

**MSCN과의 차이**:
- MSCN: 카디널리티 추정 (중간 결과 크기 예측)
- Lero: 플랜 선택 (전체 실행 비용 예측)

## 파일 구조

```
Lero/
├── LeroPresetScheduler.py          # 팩토리 함수 (진입점)
├── LeroPilotModel.py               # 모델 래퍼
├── LeroParadigmCardAnchorHandler.py # 플랜 힌트 주입
├── LeroPilotAdapter.py             # Lero 원본 코드 어댑터
├── EventImplement.py               # 학습/데이터 수집
└── source/                         # Lero 모델 구현 (PyTorch)
    ├── balsa.py
    └── ...
```

## 사용 방법

### 기본 사용

```python
from pilotscope.DBInteractor.PilotDataInteractor import PostgreSQLConfig
from algorithm_examples.Lero.LeroPresetScheduler import get_lero_preset_scheduler

# 설정
config = PostgreSQLConfig(db="your_db")

# Scheduler 생성
scheduler, tracker = get_lero_preset_scheduler(
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
scheduler, tracker = get_lero_preset_scheduler(
    config,
    enable_collection=False,
    enable_training=False,
    dataset_name="your_db"
)
```

## 주요 컴포넌트

### 1. LeroPresetScheduler

**역할**: Lero 알고리즘 설정 및 초기화

**MSCN과의 차이**:
- `enable_collection=True` 시 기존 학습 데이터를 **자동 삭제**
- Physical plan을 함께 수집 (`pull_physical_plan=True`)

```python
# 데이터 자동 삭제 로직
if enable_collection:
    data_manager.remove_table_and_tracker(pretraining_data_table)
```

**주요 파라미터**:
- `num_epoch`: 학습 에포크 수 (기본: 100)
- `num_training`: 학습 쿼리 수 (기본: -1, 전체)
- `num_collection`: 수집 쿼리 수 (기본: -1, 전체)

### 2. LeroPilotModel

**역할**: Lero 모델 래퍼

**핵심 메서드**:
- `_load_model_impl()`: Lero 모델 초기화 (PyTorch 기반)
- `_train_model_impl(training_data)`: 강화학습으로 모델 학습
- `_predict_impl(plan_features)`: 플랜 비용 예측
- `_save_model_impl(model_path)`: 모델 저장 (PyTorch checkpoint)

**모델 저장 위치**: `ExampleData/Lero/Model/lero_{timestamp}`

**GPU 지원**: PyTorch 기반이므로 CUDA 사용 가능

```python
import torch
if torch.cuda.is_available():
    print("Using GPU")
    # Lero 모델은 자동으로 GPU 사용
```

### 3. LeroCardPushHandler

**역할**: AI가 선택한 최적 플랜을 PostgreSQL에 적용

**핵심 메서드**:
- `acquire_injected_data(sql)`: 여러 플랜 생성 및 최적 플랜 선택
- `inject_data(sql, plan_hints)`: PostgreSQL에 플랜 힌트 주입

**작동 원리**:
```python
def acquire_injected_data(self, sql):
    # 1. 여러 가능한 플랜 생성 (hint 변경)
    candidate_plans = []
    for hint_config in hint_space:
        plan = self._generate_plan_with_hint(sql, hint_config)
        candidate_plans.append(plan)
    # 예: [plan1 (nested loop join), plan2 (hash join), plan3 (merge join)]

    # 2. Lero 모델로 각 플랜 비용 예측
    predicted_costs = []
    for plan in candidate_plans:
        features = self._extract_plan_features(plan)
        cost = self.pilot_model.predict(features)
        predicted_costs.append(cost)
    # 예: [100ms, 50ms, 80ms]

    # 3. 최소 비용 플랜 선택
    best_plan_idx = np.argmin(predicted_costs)
    best_plan = candidate_plans[best_plan_idx]

    # 4. 플랜 힌트 생성
    plan_hints = self._plan_to_hints(best_plan)
    return plan_hints
```

### 4. LeroPretrainingModelEvent

**역할**: 학습 데이터 수집 및 모델 학습

**데이터 수집** (MSCN과 다른 점):
```python
def iterative_data_collection(self):
    # 1. 학습 쿼리 로드
    training_queries = dataset.read_train_sql()

    # 2. 각 쿼리에 대해 여러 플랜 실행 및 비용 측정
    for sql in training_queries:
        for hint_config in hint_space:
            # 힌트 적용하여 플랜 생성
            plan = self._generate_plan_with_hint(sql, hint_config)

            # 실제 실행하여 비용 측정
            execution_time = self.data_interactor.pull_execution_time(sql)

            # 데이터 저장
            self.data_manager.save(self.data_save_table, {
                'sql': sql,
                'plan': plan,
                'execution_time': execution_time,  # Ground truth 비용
                'hint_config': hint_config
            })
```

**모델 학습** (강화학습):
```python
def custom_model_training(self):
    # 1. 데이터 로드
    training_data = self.data_manager.read(self.data_save_table)

    # 2. 강화학습 환경 구성
    # State: 쿼리 + 현재 플랜
    # Action: 다음 플랜 선택
    # Reward: -(execution_time)

    # 3. Lero 강화학습
    self.pilot_model.train(training_data, num_epoch=self.num_epoch)

    # 4. 모델 저장
    model_id = self.pilot_model.save_model()
```

### 5. 동적 학습 (Dynamic Learning)

Lero는 **주기적 모델 업데이트**를 지원합니다 (MSCN은 미지원):

```python
# 동적 학습 활성화
scheduler = get_lero_dynamic_preset_scheduler(config, dataset_name="your_db")

# 100개 쿼리마다 자동으로 재학습
# - LeroPeriodicCollectEvent: 데이터 수집
# - LeroPeriodicModelUpdateEvent: 모델 업데이트
```

## 수정 시 주의사항

### 1. 플랜 피처 추출 로직 변경

**위치**: `LeroPilotAdapter.py` 또는 `source/balsa.py`

**주의**:
- Lero는 플랜 트리 구조를 피처로 사용
- 피처 형식 변경 시 기존 모델과 호환 불가

```python
# 플랜 피처 예시
plan_features = {
    'operators': ['SeqScan', 'HashJoin', 'Aggregate'],
    'costs': [100, 500, 200],
    'cardinalities': [1000, 5000, 100],
    # ...
}
```

### 2. Hint Space 변경

**위치**: `LeroParadigmCardAnchorHandler.py`

**주의**:
- Hint space가 크면 데이터 수집 시간 증가
- 너무 작으면 최적 플랜을 찾지 못할 수 있음

```python
# Hint space 예시
hint_space = [
    {'join_method': 'nested_loop'},
    {'join_method': 'hash_join'},
    {'join_method': 'merge_join'},
    # ...
]

# Hint space 크기 조정 (상황에 따라)
hint_space = generate_hint_space(size='small')  # 'small', 'medium', 'large'
```

### 3. 강화학습 하이퍼파라미터 변경

**위치**: `LeroPilotModel.py` 또는 `source/balsa.py`

**권장 방법**:
- `get_lero_preset_scheduler()` 파라미터로 제공
- MLflow에 기록하여 추적

```python
def get_lero_preset_scheduler(..., **kwargs):
    learning_rate = kwargs.get('learning_rate', 0.001)
    discount_factor = kwargs.get('discount_factor', 0.99)
    # ...
```

### 4. 데이터 자동 삭제 로직

**위치**: `LeroPresetScheduler.py`

**현재 동작**:
```python
if enable_collection:
    data_manager.remove_table_and_tracker(pretraining_data_table)
```

**주의**:
- MSCN과 달리 Lero는 기존 데이터를 자동 삭제
- 데이터 누적을 원한다면 이 로직 제거 (비권장)

**이유**: Lero는 플랜 공간 탐색이 중요하므로, 이전 데이터가 새로운 탐색에 방해될 수 있음

## 성능 튜닝

### 학습 속도 개선

```python
# 1. GPU 사용 (자동 감지)
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# 2. Hint space 축소
# LeroParadigmCardAnchorHandler.py에서 hint_space 크기 조정

# 3. 학습 데이터 크기 제한
scheduler, tracker = get_lero_preset_scheduler(
    config,
    enable_collection=True,
    enable_training=True,
    num_collection=200,   # 200개만 수집
    num_training=200,     # 200개로 학습
    num_epoch=50          # 에포크 감소
)
```

### 예측 정확도 향상

```python
# 1. 더 많은 학습 데이터 (다양한 플랜 탐색)
num_collection=1000
num_training=1000

# 2. 더 많은 에포크
num_epoch=200

# 3. Hint space 확대 (더 많은 플랜 후보)
```

## MSCN vs Lero 비교

| 항목 | MSCN | Lero |
|------|------|------|
| **목적** | 카디널리티 추정 | 플랜 선택 |
| **AI 기술** | 딥러닝 (CNN) | 강화학습 (RL) |
| **학습 방식** | 지도학습 | 강화학습 |
| **데이터 수집** | 카디널리티 ground truth | 플랜 + 실행 시간 |
| **추론 속도** | 빠름 | 느림 (여러 플랜 탐색) |
| **GPU 지원** | 선택사항 | 권장 (PyTorch) |
| **데이터 관리** | 누적 가능 | 자동 삭제 |
| **적용 시점** | 카디널리티 추정 단계 | 플랜 선택 단계 |

**선택 가이드**:
- 옵티마이저의 카디널리티 추정이 문제 → MSCN
- 옵티마이저가 잘못된 플랜 선택 → Lero
- 복잡한 JOIN 쿼리가 많음 → 둘 다 시도 후 비교

## 문제 해결

### Q1. 학습이 매우 느림
- GPU 사용 확인 (`torch.cuda.is_available()`)
- Hint space 크기 감소
- `num_collection`, `num_training` 감소

### Q2. 메모리 부족
- GPU 메모리 부족 시 배치 크기 감소
- 또는 CPU 사용 (`CUDA_VISIBLE_DEVICES=""`)

### Q3. Baseline보다 성능이 나쁨
- Lero는 학습 초기에 성능이 나쁠 수 있음 (탐색 단계)
- 더 많은 에포크 학습 (200+)
- Hint space가 최적 플랜을 포함하는지 확인

### Q4. 기존 데이터가 삭제됨
- 의도된 동작 (강화학습 특성상 새로운 탐색 필요)
- 누적을 원한다면 `LeroPresetScheduler.py`의 삭제 로직 제거 (비권장)

### Q5. 추론이 너무 느림
- Hint space 크기 감소
- 또는 MSCN 사용 고려 (추론 속도가 중요한 경우)

## 관련 문서

- **공통 패턴**: [algorithm_examples/CLAUDE.md](../CLAUDE.md)
- **MSCN 비교**: [algorithm_examples/Mscn/CLAUDE.md](../Mscn/CLAUDE.md)
- **상세 가이드**: [docs/PRODUCTION_OPTIMIZATION.md](../../docs/PRODUCTION_OPTIMIZATION.md)

## 참고 논문

- Marcus et al., "Bao: Making Learned Query Optimization Practical"
- PilotScope 논문: `paper/PilotScope.pdf`
