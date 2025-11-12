# Index Selection (Automatic Indexing)

Index Selection은 **인덱스 추천(Index Recommendation)**을 위한 알고리즘입니다.

## 알고리즘 개요

**목적**: 워크로드에 최적화된 인덱스 집합을 자동으로 선택

**작동 방식**:
1. 워크로드 분석하여 인덱스 후보 생성
2. 각 인덱스 조합의 비용-효과 분석
3. 최적 인덱스 집합 선택
4. 추천된 인덱스 생성

**적합한 워크로드**:
- 읽기 성능이 중요한 경우 (OLAP, 분석 쿼리)
- 특정 쿼리 패턴이 반복되는 경우
- 디스크 I/O가 병목인 경우

**고려사항**:
- 인덱스는 쓰기 성능 감소 (INSERT, UPDATE, DELETE)
- 스토리지 공간 필요
- 유지보수 비용 (VACUUM, ANALYZE)

## 파일 구조

```
Index/
├── IndexPresetScheduler.py     # 팩토리 함수 (진입점)
├── EventImplement.py           # 인덱스 선택 로직
└── index_selection_evaluation/ # 벤치마크 및 평가 도구
    └── ...
```

## 사용 방법

### 기본 사용

```python
from pilotscope.DBInteractor.PilotDataInteractor import PostgreSQLConfig
from algorithm_examples.Index.IndexPresetScheduler import get_index_preset_scheduler

# 설정
config = PostgreSQLConfig(db="your_db")

# Scheduler 생성
scheduler, tracker = get_index_preset_scheduler(
    config,
    enable_collection=True,   # 워크로드 분석
    enable_training=True,     # 인덱스 선택
    dataset_name="your_db"
)

# 추천된 인덱스 자동 생성
# (scheduler.init() 시 자동 실행)
```

### 추천된 인덱스 확인

```python
# MLflow에서 확인
# Artifacts에 index_recommendations.json 저장됨

# 또는 직접 조회
import psycopg2
conn = psycopg2.connect(...)
cur = conn.cursor()
cur.execute("SELECT * FROM pg_indexes WHERE schemaname = 'public'")
for row in cur.fetchall():
    print(row)
```

## 주요 컴포넌트

### 1. IndexPresetScheduler

**역할**: 인덱스 선택 알고리즘 설정

**MSCN/Lero/Knob과의 차이**:
- 쿼리 실행 최적화가 아닌 **스키마 변경**
- 한 번 실행하면 인덱스 생성 (지속적 효과)

**주요 파라미터**:
- `num_training`: 워크로드 쿼리 수
- `num_collection`: 분석할 쿼리 수
- `max_indexes`: 최대 생성할 인덱스 수 (기본: 10)

### 2. 인덱스 선택 알고리즘

**위치**: `EventImplement.py`

**선택 과정**:
```python
# 1. 워크로드에서 인덱스 후보 추출
candidates = []
for sql in workload_queries:
    # WHERE 절 컬럼 → 인덱스 후보
    # JOIN 키 → 인덱스 후보
    # ORDER BY 컬럼 → 인덱스 후보
    candidates.extend(extract_index_candidates(sql))

# 예: ['users(age)', 'orders(user_id)', 'orders(created_at)']

# 2. 각 인덱스의 효과 추정
for idx in candidates:
    # 가상 인덱스(Hypothetical Index) 생성
    create_hypothetical_index(idx)

    # 워크로드 재실행하여 효과 측정
    cost_with_index = estimate_workload_cost()

    # 효과 계산
    benefit = baseline_cost - cost_with_index
    cost = estimate_index_storage_size(idx)

    # 비용-효과 비율
    score = benefit / cost

# 3. 상위 N개 인덱스 선택
selected_indexes = top_n_by_score(candidates, n=max_indexes)

# 4. 실제 인덱스 생성
for idx in selected_indexes:
    execute_ddl(f"CREATE INDEX {idx}")
```

### 3. EventImplement

**역할**: 워크로드 분석 및 인덱스 생성

**데이터 수집** (워크로드 분석):
```python
def iterative_data_collection(self):
    # 워크로드 쿼리 로드
    workload_queries = dataset.read_train_sql()

    # 각 쿼리의 실행 플랜 분석
    for sql in workload_queries:
        # Seq Scan이 많이 발생하는 테이블/컬럼 파악
        plan = self.data_interactor.pull_physical_plan(sql)
        execution_time = self.data_interactor.pull_execution_time(sql)

        # 데이터 저장
        self.data_manager.save(self.data_save_table, {
            'sql': sql,
            'plan': plan,
            'execution_time': execution_time
        })
```

**인덱스 선택 및 생성**:
```python
def custom_model_training(self):
    # 1. 워크로드 데이터 로드
    workload_data = self.data_manager.read(self.data_save_table)

    # 2. 인덱스 후보 생성
    candidates = self._extract_candidates(workload_data)

    # 3. 가상 인덱스로 효과 추정
    index_scores = []
    for idx in candidates:
        self._create_hypothetical_index(idx)
        benefit = self._estimate_benefit(workload_data, idx)
        cost = self._estimate_cost(idx)
        score = benefit / cost
        index_scores.append((idx, score))

    # 4. 상위 N개 선택
    selected = sorted(index_scores, key=lambda x: x[1], reverse=True)[:self.max_indexes]

    # 5. 실제 인덱스 생성
    for idx, score in selected:
        ddl = f"CREATE INDEX idx_{idx} ON {idx}"
        self.data_interactor.execute(ddl)
        print(f"Created index: {idx} (score: {score})")

    # 6. MLflow에 기록
    self.mlflow_tracker.log_artifact("index_recommendations.json", {
        'selected_indexes': [idx for idx, _ in selected],
        'scores': [score for _, score in selected]
    })
```

### 4. Hypothetical Index (PostgreSQL)

PostgreSQL의 `hypopg` 확장을 사용하여 실제 인덱스를 생성하지 않고 효과 추정:

```sql
-- Hypothetical Index 생성
SELECT hypopg_create_index('CREATE INDEX ON users(age)');

-- 쿼리 플랜 확인 (가상 인덱스 사용)
EXPLAIN SELECT * FROM users WHERE age > 30;

-- Hypothetical Index 제거
SELECT hypopg_drop_index(indexid);
```

**장점**: 실제 인덱스를 생성하지 않고 빠르게 효과 측정

## 수정 시 주의사항

### 1. 인덱스 후보 추출 로직 변경

**위치**: `EventImplement.py`

**주의**:
- 너무 많은 후보 생성 → 탐색 시간 증가
- 너무 적은 후보 생성 → 최적 인덱스를 놓칠 수 있음

```python
# 기본 전략: WHERE, JOIN, ORDER BY 컬럼
def extract_index_candidates(self, sql):
    candidates = []

    # WHERE 절
    where_columns = parse_where_columns(sql)
    candidates.extend(where_columns)

    # JOIN 키
    join_keys = parse_join_keys(sql)
    candidates.extend(join_keys)

    # ORDER BY 컬럼
    order_columns = parse_order_by_columns(sql)
    candidates.extend(order_columns)

    # 복합 인덱스 (선택적)
    # compound_indexes = generate_compound_indexes(candidates)
    # candidates.extend(compound_indexes)

    return candidates
```

### 2. 최대 인덱스 수 제한

**위치**: `IndexPresetScheduler.py`

**주의**:
- 인덱스가 너무 많으면 쓰기 성능 저하
- 일반적으로 테이블당 5~10개 권장

```python
def get_index_preset_scheduler(..., max_indexes=10):
    # max_indexes: 생성할 최대 인덱스 수
    pass
```

### 3. 비용-효과 계산 변경

**위치**: `EventImplement.py`

**현재**: `score = benefit / cost`

**다른 방식**:
```python
# 1. 단순 효과 기준
score = benefit

# 2. 가중치 적용
score = benefit * weight_factor - cost * penalty_factor

# 3. ROI (Return on Investment)
score = (benefit - cost) / cost
```

### 4. 복합 인덱스 (Composite Index)

**주의**: 복합 인덱스는 컬럼 순서가 중요

```python
# 올바른 순서: 선택도가 높은 컬럼 먼저
CREATE INDEX idx_users_age_name ON users(age, name);  # ✅

# 잘못된 순서: 선택도가 낮은 컬럼 먼저
CREATE INDEX idx_users_name_age ON users(name, age);  # ❌ (name이 선택도 낮으면)
```

## 성능 튜닝

### 탐색 속도 개선

```python
# 1. 인덱스 후보 수 제한
max_candidates = 50  # 상위 50개만 고려

# 2. 워크로드 크기 축소
num_training = 100  # 대표 쿼리 100개만 사용

# 3. Hypothetical Index 사용 (실제 생성 X)
```

### 더 나은 인덱스 찾기

```python
# 1. 더 많은 후보 탐색
max_candidates = 200

# 2. 복합 인덱스 고려
# (단, 조합 폭발 주의)

# 3. 워크로드를 실제와 유사하게 구성
```

## Index Selection vs MSCN/Lero/Knob 비교

| 항목 | Index Selection | MSCN/Lero | KnobTuning |
|------|----------------|-----------|------------|
| **목적** | 스키마 최적화 | 쿼리 최적화 | DB 설정 최적화 |
| **적용 시점** | 사전 (스키마 변경) | 실시간 (쿼리마다) | 사전 (전역 설정) |
| **변경 빈도** | 낮음 (주기적) | 높음 (쿼리마다) | 낮음 (한 번) |
| **효과 지속성** | 지속적 | 일시적 | 지속적 |
| **부작용** | 쓰기 성능 감소 | 없음 | 없음 |

**조합 사용**:
1. Index Selection으로 인덱스 생성
2. KnobTuning으로 DB 설정 최적화
3. MSCN/Lero로 개별 쿼리 최적화

## 문제 해결

### Q1. 추천된 인덱스가 효과 없음
- 워크로드가 실제와 다름: 대표 쿼리 재선정
- 인덱스가 사용되지 않음: `ANALYZE` 실행하여 통계 업데이트
- 옵티마이저가 인덱스를 선택하지 않음: `random_page_cost` 조정

### Q2. 쓰기 성능이 저하됨
- 인덱스가 너무 많음: `max_indexes` 감소
- 불필요한 인덱스 제거: 사용되지 않는 인덱스 삭제

```sql
-- 사용되지 않는 인덱스 확인 (PostgreSQL)
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY schemaname, tablename;
```

### Q3. Hypothetical Index가 작동 안 함
- `hypopg` 확장 설치 확인:

```sql
CREATE EXTENSION IF NOT EXISTS hypopg;
SELECT * FROM hypopg_list_indexes;
```

### Q4. 디스크 공간 부족
- 인덱스는 스토리지 공간 필요
- 불필요한 인덱스 제거 또는 `max_indexes` 감소

## 인덱스 유지보수

### 인덱스 모니터링

```sql
-- 인덱스 사용 통계
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- 인덱스 크기
SELECT schemaname, tablename, indexname, pg_size_pretty(pg_relation_size(indexrelid))
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC;
```

### 인덱스 재구축

```sql
-- 인덱스가 비대해지면 재구축
REINDEX INDEX idx_users_age;

-- 또는 테이블 전체
REINDEX TABLE users;
```

### 인덱스 제거

```sql
-- 사용되지 않는 인덱스 제거
DROP INDEX idx_users_unused;
```

## 관련 문서

- **공통 패턴**: [algorithm_examples/CLAUDE.md](../CLAUDE.md)
- **상세 가이드**: [docs/PRODUCTION_OPTIMIZATION.md](../../docs/PRODUCTION_OPTIMIZATION.md)
- **벤치마크**: [index_selection_evaluation/](index_selection_evaluation/)

## 참고 논문

- Sadri et al., "Online Index Selection Using Deep Reinforcement Learning"
- PilotScope 논문: `paper/PilotScope.pdf`
