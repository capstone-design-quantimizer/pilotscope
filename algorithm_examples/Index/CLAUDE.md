# Index Selection

인덱스 자동 추천. 워크로드 분석하여 최적 인덱스 집합 선택.

**중요**: Index는 ML 모델이 아닌 **휴리스틱 기반 탐색 알고리즘**입니다. MSCN/Lero는 신경망 학습이지만, Index는 **비용 기반 탐색**으로 인덱스를 선택합니다.

## 작동 방식

1. 워크로드 분석 → 인덱스 후보 생성
2. 각 인덱스 조합의 비용-효과 분석 (PostgreSQL 쿼리 옵티마이저 사용)
3. 탐욕적(greedy) 탐색으로 최적 인덱스 집합 선택
4. 추천된 인덱스를 실제 DB에 생성

## 적합 워크로드

- 읽기 성능 중요 (OLAP, 분석)
- 특정 쿼리 패턴 반복
- 디스크 I/O 병목

## 고려사항

- 쓰기 성능 감소 (INSERT, UPDATE, DELETE)
- 스토리지 공간 필요
- 유지보수 비용 (VACUUM, ANALYZE)

## 파일

```
Index/
├── IndexPresetScheduler.py     # 팩토리
├── EventImplement.py           # 인덱스 선택
└── index_selection_evaluation/ # 벤치마크 도구
```

## 사용

```python
from algorithm_examples.Index.IndexPresetScheduler import get_index_preset_scheduler

scheduler, tracker = get_index_preset_scheduler(
    config,
    enable_collection=True,
    enable_training=True,
    dataset_name="your_db"
)
# scheduler.init() 시 추천 인덱스 자동 생성
```

## 주요 컴포넌트

**EventImplement**: 워크로드 분석 및 인덱스 생성

**Hypothetical Index (PostgreSQL)**: 실제 생성 없이 효과 추정 (`hypopg` 확장)

**특이사항**:
- 쿼리 실행 최적화 아닌 **스키마 변경**
- 한 번 실행하면 인덱스 생성 (지속적 효과)
- 최대 인덱스 수 제한 (기본: 10, 테이블당 5~10개 권장)

## 인덱스 후보 추출

- WHERE 절 컬럼
- JOIN 키
- ORDER BY 컬럼
- (선택적) 복합 인덱스

## Index vs MSCN/Lero/Knob

| | Index | MSCN/Lero | KnobTuning |
|---|---|---|---|
| 목적 | 스키마 최적화 | 쿼리 최적화 | DB 설정 최적화 |
| 적용 시점 | 사전 (스키마 변경) | 실시간 | 사전 (전역 설정) |
| 변경 빈도 | 낮음 (주기적) | 높음 | 낮음 (한 번) |
| 효과 지속성 | 지속적 | 일시적 | 지속적 |
| 부작용 | 쓰기 성능 감소 | 없음 | 없음 |

**조합 사용**: Index (스키마) + KnobTuning (설정) + MSCN/Lero (쿼리)

## 수정 시 주의

**인덱스 후보 추출**: 너무 많으면 탐색 증가, 너무 적으면 최적 못 찾음

**최대 인덱스 수**: 너무 많으면 쓰기 성능 저하 (테이블당 5~10개 권장)

**비용-효과 계산**: `score = benefit / cost` (다른 방식 가능)

**복합 인덱스**: 컬럼 순서 중요 (선택도 높은 컬럼 먼저)

## 성능 튜닝

**탐색 속도**: 인덱스 후보 수 제한 (`max_candidates=50`), 워크로드 크기 축소, Hypothetical Index 사용

**더 나은 인덱스**: 더 많은 후보, 복합 인덱스 고려 (조합 폭발 주의), 실제 워크로드와 유사하게

## 문제 해결

- 추천 인덱스 효과 없음 → 워크로드 재선정, `ANALYZE` 실행, `random_page_cost` 조정
- 쓰기 성능 저하 → `max_indexes` 감소, 불필요한 인덱스 제거
- Hypothetical Index 안 됨 → `CREATE EXTENSION hypopg` 확인
- 디스크 공간 부족 → 불필요한 인덱스 제거

## 인덱스 모니터링

```sql
-- 사용 통계
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0;  -- 사용 안 되는 인덱스

-- 크기
SELECT indexname, pg_size_pretty(pg_relation_size(indexrelid))
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC;
```

## 유지보수

```sql
-- 재구축 (비대해지면)
REINDEX INDEX idx_name;

-- 제거 (불필요하면)
DROP INDEX idx_name;
```

---

# 상세 파이프라인 문서

## 전체 실행 흐름

```
사용자 코드
    ↓
get_index_preset_scheduler(config, dataset_name)
    ↓
SchedulerFactory.create_scheduler()
    ↓
IndexPeriodicModelUpdateEvent 등록
    ↓
scheduler.init() → Event 실행
    ↓
[1단계] 워크로드 로딩
    ↓
[2단계] SQL 파싱 및 컬럼 추출
    ↓
[3단계] 인덱스 후보 생성
    ↓
[4단계] 탐욕적 탐색 (Extend Algorithm)
    ↓
[5단계] 선택된 인덱스 생성
    ↓
[6단계] MLflow 로깅
```

## 1단계: 워크로드 로딩

**파일**: `algorithm_examples/Index/EventImplement.py`

**함수**: `IndexPeriodicModelUpdateEvent._load_sql()`

```python
def _load_sql(self):
    sqls: list = load_test_sql(self.dataset_name)  # utils.py 호출
    random.shuffle(sqls)  # 랜덤 셔플
    if "imdb" in self.config.db:
        sqls = sqls[0:len(sqls) // 2]  # IMDB는 절반만
    return sqls
```

**데이터 흐름**:
1. `load_test_sql(dataset_name)` → `algorithm_examples/utils.py:54`
2. Dataset 클래스 호출 (예: `StatsTinyDataset.read_test_sql()`)
3. SQL 파일 읽기 (`pilotscope/Dataset/BaseDataset.py`)
4. SQL 문자열 리스트 반환

**입력**: `dataset_name` (예: "stats_tiny", "imdb")
**출력**: SQL 문자열 리스트 (예: `["SELECT * FROM users WHERE id=1", ...]`)

## 2단계: SQL 파싱 및 컬럼 추출

**파일**: `algorithm_examples/Index/index_selection_evaluation/selection/index_selection_evaluation.py`

**함수**: `to_workload(sqls)`

```python
def to_workload(sqls):
    from sqlglot import parse_one
    query_list = []
    table_dict = dict()

    for i, sql in enumerate(sqls):
        sql_ast = parse_one(sql)  # SQL → AST
        unalias = dict()

        # 1. 테이블 추출 및 alias 처리
        for table in sql_ast.find_all(exp.Table):
            if table.alias:
                unalias[table.alias] = table.name
            unalias[table.name] = table.name
            if table.name not in table_dict:
                table_dict[table.name] = Table(table.name)

        # 2. 컬럼 추출 및 테이블 연결
        cols = []
        for col in sql_ast.find_all(exp.Column):
            c = Column(col.alias_or_name)
            if col.table and col.table in unalias:
                table_dict[unalias[col.table]].add_column(c)
                cols.append(c)

        # 3. Query 객체 생성
        query_list.append(Query(i, sql, cols))

    return Workload(query_list)
```

**데이터 구조**:

```python
# workload.py
class Column:
    name: str
    table: Table

class Table:
    name: str
    columns: List[Column]

class Query:
    nr: int         # 쿼리 ID
    text: str       # SQL 문자열
    columns: List[Column]  # 인덱스 가능한 컬럼들

class Workload:
    queries: List[Query]
```

**추출되는 컬럼**:
- WHERE 절의 컬럼 (예: `WHERE users.id = 1` → `users.id`)
- JOIN 조건의 컬럼 (예: `JOIN posts ON users.id = posts.user_id`)
- ORDER BY 컬럼
- GROUP BY 컬럼

**입력**: SQL 문자열 리스트
**출력**: `Workload` 객체 (쿼리 + 파싱된 컬럼 정보)

## 3단계: 인덱스 후보 생성

**파일**: `algorithm_examples/Index/index_selection_evaluation/selection/workload.py`

**함수**: `Workload.potential_indexes()`

```python
def indexable_columns(self):
    indexable_columns = set()
    for query in self.queries:
        indexable_columns |= set(query.columns)  # 모든 쿼리의 컬럼 합집합
    return sorted(list(indexable_columns))

def potential_indexes(self):
    return sorted([Index([c]) for c in self.indexable_columns()])
```

**입력**: Workload (쿼리 집합)
**출력**: 단일 컬럼 인덱스 리스트 (예: `[Index(users.id), Index(posts.user_id), ...]`)

**복합 인덱스 생성**:
- 알고리즘 실행 중에 동적으로 생성됨 (4단계 참조)
- `ExtendAlgorithm._attach_to_indexes()`: 기존 인덱스에 컬럼 추가

## 4단계: 탐욕적 탐색 (Extend Algorithm)

**파일**: `algorithm_examples/Index/index_selection_evaluation/selection/algorithms/extend_algorithm.py`

**함수**: `ExtendAlgorithm._calculate_best_indexes(workload)`

### 알고리즘 의사코드

```python
def _calculate_best_indexes(workload):
    # 초기화
    single_candidates = workload.potential_indexes()  # 단일 컬럼 인덱스
    index_combination = []  # 선택된 인덱스 집합
    current_cost = calculate_cost(workload, [])  # 인덱스 없을 때 비용

    while True:
        best = {"combination": [], "benefit_to_size_ratio": 0, "cost": None}

        # 1. 단일 컬럼 인덱스 평가
        for candidate in single_candidates:
            if candidate not in index_combination:
                evaluate_combination(
                    index_combination + [candidate],
                    best,
                    current_cost
                )

        # 2. 복합 인덱스 생성 및 평가
        for attribute in single_candidates:
            for index in index_combination:
                if len(index.columns) < max_index_width:
                    if index.appendable_by(attribute):
                        new_index = Index(index.columns + attribute.columns)
                        new_combination = index_combination.copy()
                        new_combination.remove(index)
                        new_combination.append(new_index)
                        evaluate_combination(new_combination, best, current_cost)

        # 3. 개선이 없으면 종료
        if best["benefit_to_size_ratio"] <= 0:
            break

        # 4. 최선의 조합 채택
        index_combination = best["combination"]
        current_cost = best["cost"]

    return index_combination
```

### 핵심 함수: `_evaluate_combination()`

```python
def _evaluate_combination(index_combination, best, current_cost):
    # 1. 새 조합의 비용 계산
    cost = cost_evaluation.calculate_cost(workload, index_combination)

    # 2. 충분한 개선이 아니면 무시
    if (cost * min_cost_improvement) >= current_cost:
        return

    # 3. benefit/size ratio 계산
    benefit = current_cost - cost  # 비용 감소량
    new_index = index_combination[-1]
    new_index_size = new_index.estimated_size
    ratio = benefit / new_index_size

    # 4. 예산 내에서 최고 ratio면 채택
    total_size = sum(index.estimated_size for index in index_combination)
    if ratio > best["benefit_to_size_ratio"] and total_size <= budget:
        best["combination"] = index_combination
        best["benefit_to_size_ratio"] = ratio
        best["cost"] = cost
```

### 비용 평가: Hypothetical Index

**파일**: `algorithm_examples/Index/index_selection_evaluation/selection/cost_evaluation.py`

**함수**: `CostEvaluation.pilot_calculate_cost(workload, indexes)`

```python
def pilot_calculate_cost(workload, indexes):
    # 1. PilotIndex로 변환
    pilot_indexes = [to_pilot_index(index) for index in indexes]

    # 2. Hypothetical Index 생성
    data_interactor.push_index(pilot_indexes)  # hypopg 사용
    data_interactor.pull_estimated_cost()

    # 3. 각 쿼리의 비용 추정
    total_cost = 0
    for query in workload.queries:
        data: PilotTransData = data_interactor.execute(query.text, is_reset=False)
        cost = data.estimated_cost  # PostgreSQL 옵티마이저 비용
        total_cost += cost

    # 4. 인덱스 크기 추정
    for i, index in enumerate(indexes):
        pilot_index = pilot_indexes[i]
        index.estimated_size = db_connector.get_index_byte(pilot_index)
        index.hypopg_oid = pilot_index.hypopg_oid
        index.hypopg_name = pilot_index.hypopg_name

    return total_cost
```

**Hypothetical Index (hypopg)**:
- PostgreSQL 확장: 실제 생성 없이 인덱스 효과 추정
- `CREATE INDEX` 없이 메타데이터만 생성
- 쿼리 플래너가 이를 사용하여 비용 계산
- 빠르고 저렴한 평가 가능

**입력**:
- `workload`: 쿼리 집합
- `indexes`: 평가할 인덱스 조합

**출력**:
- `total_cost`: 모든 쿼리의 추정 비용 합계 (PostgreSQL cost units)

### 평가 지표

**Benefit**: `current_cost - new_cost` (비용 감소량, 클수록 좋음)
**Size**: `index.estimated_size` (바이트 단위)
**Ratio**: `benefit / size` (단위 크기당 이득, 클수록 좋음)

탐욕적 알고리즘은 각 단계에서 **ratio가 최대인 인덱스**를 선택합니다.

## 5단계: 선택된 인덱스 생성

**파일**: `algorithm_examples/Index/EventImplement.py`

**함수**: `IndexPeriodicModelUpdateEvent.custom_model_update()`

```python
def custom_model_update(pilot_model, db_controller, data_manager):
    # 1. 기존 인덱스 제거
    db_controller.drop_all_indexes()

    # 2. 워크로드 로딩 및 파싱
    sqls = self._load_sql()
    workload = to_workload(sqls)

    # 3. 알고리즘 실행
    parameters = {
        "benchmark_name": self.config.db,
        "budget_MB": 250,
        "max_index_width": 2
    }
    connector = DbConnector(PilotDataInteractor(config, enable_simulate_index=True))
    algo = ExtendAlgorithm(connector, parameters=parameters)
    indexes = algo.calculate_best_indexes(workload)

    # 4. 실제 인덱스 생성
    for index in indexes:
        columns = [c.name for c in index.columns]
        db_controller.create_index(
            PilotIndex(columns, index.table().name, index.index_idx())
        )
        print("create index {}".format(index))
```

**생성되는 인덱스 이름 형식**:
- 단일 컬럼: `users_id_idx`
- 복합 인덱스: `users_id_name_idx`

## 6단계: MLflow 로깅

**파일**: `algorithm_examples/Index/EventImplement.py`

```python
if self.mlflow_tracker:
    # 1. 메트릭 로깅
    self.mlflow_tracker.log_training_metrics({
        "index_optimization_time_seconds": optimization_time,
        "num_indexes_selected": len(indexes),
        "num_queries": len(sqls)
    }, step=self.update_count)

    # 2. 인덱스 구성 저장
    index_config = {
        "indexes": [
            {
                "table": index.table().name,
                "columns": [c.name for c in index.columns],
                "index_name": index.index_idx(),
                "estimated_size_bytes": index.estimated_size
            }
            for index in indexes
        ],
        "num_indexes": len(indexes),
        "optimization_time_seconds": optimization_time,
        "num_queries_analyzed": len(sqls),
        "update_iteration": self.update_count
    }
    mlflow.log_dict(index_config, f"index_config_iteration_{self.update_count}.json")
```

**로깅되는 정보**:
- 최적화 소요 시간
- 선택된 인덱스 수
- 분석한 쿼리 수
- 인덱스 상세 정보 (테이블, 컬럼, 크기)

---

## 핵심 데이터 구조 요약

### 입력 데이터

```python
# SQL 문자열
sql = "SELECT u.name FROM users u WHERE u.id = 1"

# 파싱 후
Query(
    nr=0,
    text="SELECT u.name FROM users u WHERE u.id = 1",
    columns=[Column(name="id", table=Table("users"))]
)
```

### 중간 데이터

```python
# 인덱스 후보
Index(columns=[Column("id", table=Table("users"))])
Index(columns=[Column("name", table=Table("users"))])

# 복합 인덱스 (알고리즘 중 생성)
Index(columns=[
    Column("id", table=Table("users")),
    Column("name", table=Table("users"))
])
```

### 출력 데이터

```python
# 선택된 인덱스 집합
[
    Index(columns=[Column("id", table=Table("users"))]),
    Index(columns=[Column("user_id", table=Table("posts"))]),
    Index(columns=[Column("id", table=Table("users")),
                   Column("name", table=Table("users"))])
]

# DB에 생성
# CREATE INDEX users_id_idx ON users(id)
# CREATE INDEX posts_user_id_idx ON posts(user_id)
# CREATE INDEX users_id_name_idx ON users(id, name)
```

---

## ML 모델이 아닌 이유

Index 알고리즘이 MSCN/Lero와 다른 점:

| 특성 | MSCN/Lero | Index (Extend) |
|------|-----------|----------------|
| **방법론** | 신경망 학습 | 휴리스틱 탐색 |
| **학습 필요** | 예 (수천 쿼리) | 아니오 |
| **모델 저장** | 예 (PyTorch) | 아니오 |
| **추론 단계** | 신경망 forward pass | 탐색 알고리즘 실행 |
| **비용 추정** | 학습된 모델 | PostgreSQL 옵티마이저 |
| **메모리 요구** | GPU (학습), 낮음 (추론) | 낮음 (탐색만) |
| **실행 시간** | 빠름 (추론 ms) | 느림 (탐색 초~분) |
| **입력** | 쿼리 특징 벡터 | 워크로드 (쿼리 집합) |
| **출력** | 카디널리티 추정 | 인덱스 집합 |

**Index는 "학습" 없이 매번 탐색**합니다:
- `enable_training=True`는 오해의 소지: 실제로는 "인덱스 선택 수행"을 의미
- `num_training`은 "분석할 쿼리 수"를 의미
- `load_model_id`는 적용 불가 (저장할 모델이 없음)

**왜 `PretrainingModelEvent`를 사용하나?**
- PilotScope 프레임워크 재사용을 위한 편의성
- 실제로는 `PeriodicModelUpdateEvent`로 주기적 재최적화
- "학습"이 아니라 "최적화 실행"

---

## 파라미터 설명

**IndexPresetScheduler.py**:
```python
get_index_preset_scheduler(
    config: PilotConfig,
    use_mlflow=True,
    experiment_name=None,  # MLflow 실험 이름
    dataset_name=None      # 워크로드 이름 (DB_워크로드 형식)
)
```

**IndexPeriodicModelUpdateEvent**:
```python
IndexPeriodicModelUpdateEvent(
    config,
    per_query_count=200,      # 200 쿼리마다 재최적화
    execute_on_init=True,     # 초기화 시 즉시 실행
    mlflow_tracker=tracker,   # MLflow 로거
    dataset_name="stats_tiny" # 워크로드 소스
)
```

**ExtendAlgorithm parameters**:
```python
parameters = {
    "budget_MB": 250,           # 인덱스 최대 크기 (MB)
    "max_index_width": 2,       # 인덱스 최대 컬럼 수
    "min_cost_improvement": 1.003  # 최소 개선율 (0.3%)
}
```

**주의**: `per_query_count=200`은 200개 쿼리 **실행 후** 재최적화를 의미합니다. 최적화 자체는 전체 워크로드를 사용합니다.

---

## 성능 특성

**시간 복잡도**:
- 인덱스 후보 수: O(C) (C = 컬럼 수)
- 탐색 반복: O(I) (I = 선택된 인덱스 수, 보통 < 15)
- 각 반복의 비용 평가: O(C × Q) (Q = 쿼리 수)
- 총: O(I × C × Q)

**실제 병목**:
- `calculate_cost()`: PostgreSQL 쿼리 플래너 호출 (지배적)
- Hypothetical Index 생성/삭제
- 쿼리 복잡도에 따라 변동

**최적화 기법**:
- **캐싱**: `CostEvaluation.cache` - 동일 (쿼리, 인덱스) 조합 재사용
- **조기 종료**: `min_cost_improvement` - 미미한 개선 무시
- **예산 제한**: `budget_MB` - 탐색 공간 축소
- **너비 제한**: `max_index_width` - 복합 인덱스 컬럼 수 제한

**예시 실행 시간** (stats_tiny, 20 쿼리):
- 워크로드 로딩: < 1초
- 탐색 + 비용 평가: 10~30초
- 인덱스 생성: 1~5초
- 총: ~20~40초

---

## 디버깅 팁

**인덱스가 선택되지 않는 경우**:
1. 워크로드 확인: `print(sqls)` in `_load_sql()`
2. 파싱 확인: `print(workload.indexable_columns())` in `custom_model_update()`
3. 후보 확인: `print(single_attribute_index_candidates)` in `_calculate_best_indexes()`
4. 비용 확인: `print(f"cost: {cost}, benefit: {benefit}, ratio: {ratio}")` in `_evaluate_combination()`

**비용 평가 오류**:
- Hypothetical Index 미지원: `CREATE EXTENSION hypopg;` 확인
- 통계 부족: `ANALYZE` 실행
- 타임아웃: `config.sql_execution_timeout` 증가

**성능 문제**:
- 쿼리 수 축소: `sqls = sqls[:10]` in `_load_sql()`
- 예산 감소: `budget_MB=100`
- 반복 제한: `if count > 2: break` (이미 구현됨, line 87-89)

---

## 확장 가능성

**다른 알고리즘**:
- `AutoAdminAlgorithm`: Microsoft AutoAdmin
- `DB2AdvisAlgorithm`: IBM DB2 Advisor
- `DexterAlgorithm`: Dexter (ML 기반)
- `RelaxationAlgorithm`: 완화 기반 탐색

**알고리즘 교체**:
```python
# EventImplement.py line 65
from selection.algorithms.auto_admin_algorithm import AutoAdminAlgorithm
algo = AutoAdminAlgorithm(connector, parameters=parameters)
```

**파라미터 튜닝**:
- `budget_MB`: 저장 공간에 따라 조정
- `max_index_width`: 복잡한 쿼리는 3~4로 증가
- `min_cost_improvement`: 더 엄격하려면 1.01 (1%)
- `per_query_count`: 워크로드 변화 속도에 따라 조정
