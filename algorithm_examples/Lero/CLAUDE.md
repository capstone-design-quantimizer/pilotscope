# Lero (Learning to Rewrite Optimizer)

쿼리 플랜 최적화. Tree-CNN 기반 pairwise ranking으로 최적 플랜 선택.

## 핵심 개념

**Lero는 카디널리티를 조작하여 다양한 플랜을 생성하고, Tree-CNN으로 각 플랜의 실행 시간을 예측하여 최적 플랜을 선택합니다.**

- 학습: 같은 쿼리의 여러 플랜을 pairwise로 비교 학습
- 추론: 카디널리티 조작으로 여러 플랜 생성 → 스코어 예측 → 최저 스코어 선택
- 특징: 플랜 트리 구조를 Tree Convolution으로 직접 학습

## 전체 파이프라인 개요

```
[학습 Phase]
1. 트레이닝 SQL 로드 (load_training_sql)
   ↓
2. 각 SQL마다 카디널리티 조작으로 여러 플랜 생성 (CardsPickerModel)
   ↓
3. 각 플랜 실행 및 실행 시간 측정 (PilotDataInteractor)
   ↓
4. (sql, plan, time) 데이터 저장 (DataManager)
   ↓
5. 같은 SQL의 플랜들을 pairwise 조합 (extract_plan_pairs)
   ↓
6. Feature 추출: JSON 플랜 → SampleEntity 트리 (FeatureGenerator)
   ↓
7. Tree-CNN 학습: pairwise ranking (LeroModelPairWise)
   ↓
8. 모델 저장 (LeroPilotModel)

[추론 Phase]
1. 사용자 쿼리 입력
   ↓
2. 서브쿼리 카디널리티 추출 (pull_subquery_card)
   ↓
3. CardsPickerModel로 여러 카디널리티 조합 생성
   ↓
4. 각 조합으로 플랜 생성 (push_card + pull_physical_plan)
   ↓
5. 각 플랜의 스코어 예측 (LeroModelPairWise.predict)
   ↓
6. 최저 스코어 플랜의 카디널리티 선택
   ↓
7. PostgreSQL에 주입하여 실행
```

## 적합 워크로드

- 플랜이 다양하게 나올 수 있는 경우
- JOIN 순서가 성능에 큰 영향
- OLAP (복잡한 분석 쿼리)

## MSCN vs Lero

- MSCN: 카디널리티 추정 (중간 결과 크기)
- Lero: 플랜 선택 (전체 실행 비용)

| | MSCN | Lero |
|---|---|---|
| AI 기술 | 딥러닝 (CNN) | 강화학습 (RL) |
| 학습 방식 | 지도학습 | 강화학습 |
| 추론 속도 | 빠름 | 느림 (여러 플랜 탐색) |
| GPU | 선택 | 권장 (PyTorch) |
| 데이터 관리 | 누적 가능 | 자동 삭제 |

---

## 상세 파이프라인 분석

### 1. 트레이닝 쿼리 로딩

**파일**: `algorithm_examples/utils.py:30-51`

```python
def load_training_sql(dataset_name):
    # dataset_name에 따라 적절한 Dataset 클래스 선택
    # 예: "stats_tiny" → StatsTinyDataset
    return StatsTinyDataset(DatabaseEnum.POSTGRESQL).read_train_sql()
```

**흐름**:
- `LeroPretrainingModelEvent.load_sql()` (EventImplement.py:56)
- → `load_training_sql(self.dataset_name)` 호출
- → Dataset 클래스가 SQL 파일에서 쿼리 리스트 반환
- → `self.sqls`에 저장

**데이터 형식**: `List[str]` (SQL 문자열 리스트)

### 2. 데이터 수집 파이프라인

**파일**: `EventImplement.py:59-95` (`iterative_data_collection`)

**핵심 로직** (각 SQL마다):

```python
# 1. 원본 카디널리티 추출
pull_subquery_card()  # PostgreSQL의 EXPLAIN에서 서브쿼리별 카디널리티 수집
data = execute(sql)
subquery_2_card = data.subquery_2_card  # Dict[str, float]
# 예: {"SELECT * FROM users": 1000, "SELECT * FROM orders WHERE ...": 500}

# 2. CardsPickerModel 초기화
cards_picker = CardsPickerModel(subquery_2_card.keys(), subquery_2_card.values())
# → 서브쿼리를 JOIN 테이블 수로 분류
# → swing_factor (0.01, 0.1, 1, 10, 100) 준비

# 3. 다양한 카디널리티 조합으로 플랜 생성 루프
while not finish:
    # a. 조작된 카디널리티 주입
    push_card(scale_subquery_2_card)  # PostgreSQL에 힌트 주입

    # b. 플랜 및 실행 시간 수집
    pull_physical_plan()
    pull_execution_time()
    data = execute(sql)

    # c. 수집 데이터 저장
    column_2_value = {
        "sql": sql,
        "plan": data.physical_plan,  # JSON 형식 플랜 트리
        "time": data.execution_time   # 실제 실행 시간 (ms)
    }

    # d. 플랜 카디널리티를 원본으로 복원 (중요!)
    cards_picker.replace(plan)  # 조작된 Plan Rows를 원래 값으로 되돌림

    # e. 다음 카디널리티 조합 생성
    finish, new_cards = cards_picker.get_cards()
    scale_subquery_2_card = {sq: new_card for sq, new_card in zip(...)}
```

**CardsPickerModel 상세** (`LeroPilotAdapter.py:5-43`):

```
입력: subqueries, rows
처리:
  1. 서브쿼리를 테이블 수로 그룹화
     - 1-table: base relation (조작 안함)
     - 2-table JOIN: swing_factor 적용
     - 3-table JOIN: swing_factor 적용
     ...

  2. CardPicker가 체계적으로 탐색 (card_picker.py)
     - 테이블 수가 많은 JOIN부터 조작 (복잡한 쿼리일수록 추정 오류 크므로)
     - swing_factors = [0.01, 0.1, 1, 10, 100]
     - 각 swing_factor로 카디널리티 스케일링

  3. 예시:
     원본: JOIN(A,B) = 1000, JOIN(A,B,C) = 5000
     조합1: JOIN(A,B) = 1000, JOIN(A,B,C) = 50 (x0.01)
     조합2: JOIN(A,B) = 1000, JOIN(A,B,C) = 500 (x0.1)
     조합3: JOIN(A,B) = 1000, JOIN(A,B,C) = 5000 (x1)
     ...
     → 각 조합마다 다른 플랜 생성 가능
```

**중요**: `PlanCardReplacer` (utils.py:27-99)
- 조작된 카디널리티로 플랜을 생성한 후
- 모델 입력 전에 **원래 카디널리티로 복원** 필요
- 이유: 실제 데이터 분포 기반 feature로 학습해야 일반화

**수집 결과**:
```python
[
  {"sql": "SELECT ...", "plan": "{JSON plan tree}", "time": 123.45},
  {"sql": "SELECT ...", "plan": "{JSON plan tree}", "time": 98.76},
  ...
]
```

테이블에 저장: `lero_pretraining_{dataset_name}`

### 3. Pairwise 데이터 생성

**파일**: `EventImplement.py:17-39` (`extract_plan_pairs`)

```python
# 1. SQL별로 플랜 그룹화
sql_2_plans = {}
for sql in unique_sqls:
    plans = data[data["sql"] == sql]
    for plan in plans:
        plan_json = json.loads(plan)
        plan_json["Execution Time"] = time  # 실행 시간 추가
        sql_2_plans[sql].append(json.dumps(plan_json))

# 2. 같은 SQL의 플랜들을 pairwise 조합
for sql, plans in sql_2_plans.items():
    if len(plans) < 2:
        continue  # 최소 2개 필요

    # 모든 쌍 생성 (train.py:27-41)
    for i in range(len(plans)):
        for j in range(i+1, len(plans)):
            plans1.append(plans[i])
            plans2.append(plans[j])

# 결과: (plan1, plan2) 쌍들
# 예: SQL1의 플랜 3개 → 3쌍 생성
# (plan_a, plan_b), (plan_a, plan_c), (plan_b, plan_c)
```

### 4. Feature 추출

**파일**: `source/feature.py`

#### 4.1. FeatureGenerator 초기화 (`fit`)

```python
feature_generator = FeatureGenerator()
feature_generator.fit(plans1 + plans2)  # 모든 플랜으로 정규화 범위 계산

# fit 과정:
for plan in plans:
    json_obj = json.loads(plan)
    recurse(json_obj["Plan"]):  # 트리 전체 순회
        startup_costs.append(node["Startup Cost"])
        total_costs.append(node["Total Cost"])
        rows.append(node["Plan Rows"])
        input_relations.add(node.get("Relation Name"))

# 정규화 범위 계산
startup_costs = log(startup_costs + 1)
startup_costs_min = min(startup_costs)
startup_costs_max = max(startup_costs)
# ... total_costs, rows도 동일

# Normalizer 생성
normalizer = Normalizer(
    mins={"Startup Cost": min, "Total Cost": min, "Plan Rows": min},
    maxs={"Startup Cost": max, "Total Cost": max, "Plan Rows": max}
)
```

#### 4.2. Feature 변환 (`transform`)

```python
local_features, y = feature_generator.transform(plans)

# 각 플랜마다:
json_obj = json.loads(plan)
plan_tree = json_obj["Plan"]
execution_time = json_obj["Execution Time"]

# AnalyzeJsonParser가 재귀적으로 트리 파싱 (feature.py:199-237)
sample_entity = extract_feature(plan_tree)

# SampleEntity 구조 (feature.py:119-161):
SampleEntity {
    node_type: np.ndarray,           # one-hot encoding [0,0,1,0,...] (23개 op types)
    startup_cost: float,             # 정규화된 값 (사용 안함: None)
    total_cost: float,               # 정규화된 값 (사용 안함: None)
    rows: float,                     # 정규화된 log(Plan Rows + 1)
    width: int,                      # Plan Width (바이트)
    left: SampleEntity,              # 왼쪽 자식 (트리 구조)
    right: SampleEntity,             # 오른쪽 자식
    startup_time: float,             # Actual Startup Time (label용)
    total_time: float,               # Actual Total Time (label용)
    input_tables: List[str],         # 참조 테이블명
    encoded_input_tables: np.ndarray # 테이블 one-hot encoding
}

# get_feature() 메소드 (feature.py:144-146):
# 실제 모델 입력 벡터 생성
return np.hstack([
    node_type,              # 23차원 (operator type)
    encoded_input_tables,   # N차원 (테이블 수 + 1)
    [width, rows]           # 2차원
])
# → 총 (23 + N + 2)차원 벡터

# 예시:
# 테이블 3개 (users, orders, products) 쿼리
# → input_feature_dim = 23 + 4 + 2 = 29
```

**Feature 요약**:
- **Node Type**: Hash Join, Seq Scan 등 23가지 operator one-hot
- **Input Tables**: 쿼리에서 사용하는 테이블들의 one-hot encoding
- **Plan Rows**: 정규화된 카디널리티 (log scale)
- **Plan Width**: 튜플 너비 (bytes)
- **트리 구조**: left/right로 연결된 재귀 트리

### 5. 모델 학습

**파일**: `source/model.py:235-331` (`LeroModelPairWise.fit`)

#### 5.1. 모델 아키텍처 (`LeroNet`)

```python
LeroNet(input_feature_dim):
    tree_conv = Sequential(
        # Layer 1: Tree Convolution
        ConvTree(input_dim, 256),       # 자식 노드 정보 집계
        LayerNormTree(),                # 트리 각 노드 정규화
        LeakyReLU(),

        # Layer 2
        ConvTree(256, 128),
        LayerNormTree(),
        LeakyReLU(),

        # Layer 3
        ConvTree(128, 64),
        LayerNormTree(),

        # Pooling: 트리 전체를 단일 벡터로
        DynamicPoolingTree(),           # 64차원 벡터 출력

        # Fully Connected
        Linear(64, 32),
        LeakyReLU(),
        Linear(32, 1)                   # 스코어 (scalar)
    )
```

**Tree Convolution** (`tcnn/module.py`):
- 각 노드의 feature와 자식 노드들의 feature를 결합
- `output_node = linear([node_feature, sum(child_features)])`
- 트리 구조를 유지하며 상향식(bottom-up)으로 전파

**DynamicPoolingTree**:
- 모든 노드의 feature를 sum-pooling
- 가변 크기 트리 → 고정 크기 벡터

#### 5.2. Pairwise 학습

```python
# 학습 데이터 준비
pairs = []
for i in range(len(X1)):
    label = 1.0 if Y1[i] >= Y2[i] else 0.0  # Y1이 더 느리면 1
    pairs.append((X1[i], X2[i], label))

# 학습 루프 (각 epoch마다)
for x1, x2, label in dataset:
    # 1. 트리 빌드
    tree_x1 = build_trees(x1)  # SampleEntity → PyTorch tensor tree
    tree_x2 = build_trees(x2)

    # 2. Forward pass
    score1 = net(tree_x1)  # scalar
    score2 = net(tree_x2)  # scalar

    # 3. Pairwise ranking loss
    diff = score1 - score2
    prob = sigmoid(diff)  # score1이 더 크면 확률 높음

    # 4. Binary Cross Entropy
    loss = BCE(prob, label)
    # label=1: score1이 더 커야 함 (time1이 더 느림)
    # label=0: score2가 더 커야 함 (time2가 더 느림)

    # 5. Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**학습 목표**:
- **느린 플랜에 높은 스코어 부여**
- 추론 시 최소 스코어 선택 → 가장 빠른 플랜 선택

**하이퍼파라미터**:
- Batch size: 64 (GPU 사용 시 GPU 수 × 64)
- Optimizer: Adam
- Loss: Binary Cross Entropy
- Epochs: 100 (기본값)

### 6. 추론 파이프라인

**파일**: `LeroParadigmCardAnchorHandler.py:39-72` (`acquire_injected_data`)

```python
# 1. 원본 카디널리티 추출
pull_subquery_card()
data = execute(sql)
subquery_2_card = data.subquery_2_card

# 2. CardsPickerModel로 여러 카디널리티 조합 생성
cards_picker = CardsPickerModel(subquery_2_card.keys(), subquery_2_card.values())
plans = []
cardss = []

# 3. 모든 카디널리티 조합으로 플랜 생성
while not finish:
    # a. 카디널리티 주입
    push_card(scale_subquery_2_card)

    # b. 플랜 수집 (실행 안함!)
    pull_physical_plan()
    data = execute(sql)  # EXPLAIN만 실행 (실제 쿼리 실행 X)

    # c. 카디널리티 복원
    cards_picker.replace(data.physical_plan)

    plans.append(data.physical_plan)
    cardss.append(scale_subquery_2_card)

    # d. 다음 조합
    finish, new_cards = cards_picker.get_cards()

# 4. 각 플랜의 스코어 예측 (LeroParadigmCardAnchorHandler.py:20-37)
feature_generator = model._feature_generator
x, _ = feature_generator.transform(plans)  # List[SampleEntity]
scores = model.predict(x)  # np.ndarray (N,)

# 5. 최소 스코어 플랜 선택
best_idx = np.argmin(scores)
selected_card = cardss[best_idx]

# 6. 선택된 카디널리티 반환
return selected_card  # PostgreSQL에 주입됨
```

**추론 과정**:
1. 여러 플랜 후보 생성 (EXPLAIN만, 실행 안함)
2. 각 플랜의 예상 실행 시간 스코어 계산
3. 최소 스코어 플랜의 카디널리티 선택
4. 실제 쿼리 실행은 선택된 카디널리티로 1번만

---

## 파일 구조

```
Lero/
├── LeroPresetScheduler.py          # 팩토리 (스케줄러 생성)
├── LeroPilotModel.py               # 모델 래퍼 (저장/로드)
├── LeroParadigmCardAnchorHandler.py # 추론: 플랜 선택 및 힌트 주입
├── LeroPilotAdapter.py             # CardsPickerModel (카디널리티 조작)
├── EventImplement.py               # 학습: 데이터 수집 및 학습
└── source/
    ├── feature.py                  # FeatureGenerator, SampleEntity
    ├── model.py                    # LeroNet, LeroModelPairWise
    ├── train.py                    # 학습 헬퍼 함수
    ├── card_picker.py              # CardPicker (swing factor 탐색)
    ├── utils.py                    # PlanCardReplacer
    └── tcnn/                       # Tree CNN 레이어
        ├── module.py               # ConvTree, DynamicPoolingTree
        └── util.py                 # prepare_trees
```

## 사용

```python
from algorithm_examples.Lero.LeroPresetScheduler import get_lero_preset_scheduler

scheduler, tracker = get_lero_preset_scheduler(
    config,
    enable_collection=True,
    enable_training=True,
    num_epoch=100,
    dataset_name="your_db"
)
```

## 주요 컴포넌트

**LeroCardPushHandler**: 여러 플랜 생성 → 비용 예측 → 최적 플랜 선택 → PostgreSQL 주입

**LeroPretrainingModelEvent**: 여러 플랜 실행 및 비용 측정 (강화학습 데이터)

**특이사항**:
- `enable_collection=True` 시 기존 데이터 **자동 삭제** (새로운 탐색 필요)
- Physical plan 함께 수집 (`pull_physical_plan=True`)
- GPU 지원 (PyTorch)
- 모델 저장: `ExampleData/Lero/Model/lero_{timestamp}`

## 동적 학습

주기적 모델 업데이트 지원 (MSCN 미지원):

```python
scheduler = get_lero_dynamic_preset_scheduler(config, dataset_name="your_db")
# 100개 쿼리마다 자동 재학습
```

---

## 데이터 흐름 요약

### 학습 Phase

```
1. SQL 문자열 리스트 (load_training_sql)
   ↓
2. 각 SQL → 서브쿼리 카디널리티 Dict[str, float] (pull_subquery_card)
   ↓
3. CardsPickerModel → 다양한 카디널리티 조합 생성
   ↓
4. 각 조합 → PostgreSQL EXPLAIN → JSON 플랜 트리
   ↓
5. 실제 실행 → 실행 시간 측정
   ↓
6. PlanCardReplacer → 플랜의 카디널리티를 원본으로 복원
   ↓
7. 저장: [(sql, plan_json, execution_time), ...]
   ↓
8. extract_plan_pairs → 같은 SQL의 플랜들을 pairwise 조합
   결과: [(plan1, plan2), ...], [(time1, time2), ...]
   ↓
9. FeatureGenerator.fit → 모든 플랜에서 정규화 범위 계산
   ↓
10. FeatureGenerator.transform → JSON 플랜 → SampleEntity 트리
    각 노드: [node_type(23) + input_tables(N) + width(1) + rows(1)]
   ↓
11. LeroModelPairWise.fit → Pairwise 학습
    입력: (SampleEntity_tree1, SampleEntity_tree2)
    라벨: 1 if time1 >= time2 else 0
    출력: (score1, score2)
    손실: BCE(sigmoid(score1 - score2), label)
   ↓
12. 모델 저장 (LeroPilotModel._save_model_impl)
```

### 추론 Phase

```
1. 사용자 쿼리 (SQL 문자열)
   ↓
2. pull_subquery_card → 서브쿼리 카디널리티 Dict[str, float]
   ↓
3. CardsPickerModel → 여러 카디널리티 조합 생성
   [card_combo1, card_combo2, ..., card_comboN]
   ↓
4. 각 조합 → PostgreSQL EXPLAIN (실행 안함!)
   [plan1, plan2, ..., planN]
   ↓
5. PlanCardReplacer → 각 플랜의 카디널리티 복원
   ↓
6. FeatureGenerator.transform → [SampleEntity_tree1, ..., SampleEntity_treeN]
   ↓
7. LeroModelPairWise.predict → [score1, score2, ..., scoreN]
   ↓
8. np.argmin(scores) → best_idx
   ↓
9. selected_card = cardss[best_idx]
   ↓
10. PostgreSQL에 selected_card 주입 → 실제 쿼리 실행
```

### 핵심 데이터 구조

**서브쿼리 카디널리티**:
```python
Dict[str, float] = {
    "SELECT * FROM users": 1000,
    "SELECT * FROM orders WHERE user_id = users.id": 5000,
    ...
}
```

**JSON 플랜 트리** (PostgreSQL EXPLAIN 결과):
```json
{
  "Plan": {
    "Node Type": "Hash Join",
    "Plan Rows": 5000,
    "Plan Width": 128,
    "Startup Cost": 10.5,
    "Total Cost": 250.3,
    "Execution Time": 123.45,
    "Plans": [
      {
        "Node Type": "Seq Scan",
        "Relation Name": "users",
        "Plan Rows": 1000,
        ...
      },
      {
        "Node Type": "Hash",
        "Plans": [...]
      }
    ]
  }
}
```

**SampleEntity 트리** (모델 입력):
```python
SampleEntity {
    get_feature() → np.ndarray([
        0,0,0,1,0,...,0,  # node_type (Hash Join = 1)
        1,0,0,0,          # input_tables (users table)
        128,              # width
        8.517             # normalized log(rows+1)
    ]),
    left: SampleEntity { ... },
    right: SampleEntity { ... }
}
```

**모델 출력**:
- 학습: `score = scalar` (느린 플랜일수록 높은 값)
- 추론: `scores = [score1, score2, ..., scoreN]` → argmin 선택

---

## 수정 시 주의

**플랜 피처 변경**: 플랜 트리 구조 피처 사용, 형식 변경 시 기존 모델 비호환

**Hint Space 변경**: 크면 수집 시간 증가, 작으면 최적 플랜 못 찾음
- 기본값: swing_factor = [0.01, 0.1, 1, 10, 100] (5단계)
- JOIN 2개 쿼리: 최대 5 × 5 = 25개 플랜
- JOIN 3개 쿼리: 최대 5 × 5 × 5 = 125개 플랜

**데이터 자동 삭제**: 강화학습 특성상 이전 데이터가 탐색에 방해 (삭제 로직 제거 비권장)
- `enable_collection=True`일 때 기존 테이블 삭제 (LeroPresetScheduler.py:36-37)
- 새로운 데이터셋/워크로드마다 재수집 필요

**PlanCardReplacer의 중요성**:
- 조작된 카디널리티로 플랜을 유도하지만
- 모델 입력 feature는 원본 카디널리티 사용
- 이유: 실제 데이터 분포에서 일반화해야 함

## 성능 튜닝

**학습 속도**: GPU 사용, Hint space 축소, `num_collection`/`num_training` 감소

**정확도**: 더 많은 데이터/epoch, Hint space 확대

## 문제 해결

**학습 매우 느림**:
- GPU 확인: `torch.cuda.is_available()` (model.py:17-26)
- Hint space 감소: swing_factor 단계 줄이기 (LeroPilotAdapter.py:7-9)
- 데이터 감소: `num_collection`, `num_training` 조정
- 예: 100개 쿼리 × 25개 플랜/쿼리 = 2500개 플랜 수집 (수 시간 소요)

**메모리 부족**:
- GPU 메모리: 배치 크기 감소 (model.py:264, 기본 64)
- CPU 사용: GPU 비활성화 시 자동 CPU 모드
- DataParallel 오버헤드: Single GPU는 최소화됨 (model.py:32)

**Baseline보다 성능 나쁨**:
- 학습 초기: 모델이 아직 탐색 중 (pairwise ranking 수렴 필요)
- Epoch 증가: 100 → 200+ (LeroPresetScheduler.py:52)
- 데이터 부족: 더 많은 쿼리/플랜 조합 필요
- 모델 확인: `hasattr(model._net, '_net')` 및 `model._net is not None`

**추론 느림**:
- Hint space 감소: swing_factor 단계 줄이기 (탐색 플랜 수 감소)
- MSCN 고려: 카디널리티만 개선하면 되는 경우 더 빠름
- 예: swing_factor 5단계 × JOIN 3개 = 125개 플랜 생성 및 예측

**모델 로드 실패**:
- `FileNotFoundError`: 모델 학습 안됨 → `enable_training=True` 필요
- `AttributeError: _feature_generator is None`: 학습 미완료
- MLflow 사용 시: 실험명 확인 (`f"lero_{config.db}"`)

**수집 타임아웃**:
- Config의 `timeout` 값 증가 (EventImplement.py:85)
- 복잡한 쿼리: 실행 시간이 timeout 초과 시 스킵

---

## 참고 자료

**원본 논문**:
- Marcus, Ryan, et al. "Towards a Hands-Free Query Optimizer through Deep Learning." CIDR 2019.

**핵심 알고리즘**:
1. **Cardinality Hint Injection**: PostgreSQL의 `pg_hint_plan`을 통한 카디널리티 조작
2. **Tree Convolution**: 플랜 트리 구조를 직접 학습하는 CNN 변형
3. **Pairwise Ranking**: 절대 시간 예측보다 상대 순위 학습

**PilotScope 내 관련 알고리즘**:
- **MSCN**: 카디널리티 추정 (Lero는 플랜 선택)
- **Index Selection**: 인덱스 선택 (Lero는 플랜 선택)
- **Knob Tuning**: 설정 최적화 (Lero는 플랜 선택)
