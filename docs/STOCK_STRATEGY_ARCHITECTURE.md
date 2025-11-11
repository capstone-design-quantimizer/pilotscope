# StockStrategy Dataset 아키텍처 이해하기

## 핵심 개념: 하나의 DB, 여러 워크로드

### 문제: 왜 혼란스러운가?

많은 사용자가 다음과 같은 혼란을 겪습니다:

```bash
# 이렇게 쓰면 되는 건가?
python unified_test.py --algo mscn --db stock_strategy_momentum_investing

# 아니면 이렇게?
python unified_test.py --algo mscn --db stock_strategy --workload momentum_investing
```

**정답: 두 번째가 맞습니다!**

### 이유: PilotScope의 DB vs Dataset 구조

PilotScope는 다음을 구분합니다:

1. **Database (물리적 PostgreSQL DB)**
   - 실제 PostgreSQL 서버에 존재하는 데이터베이스
   - 테이블, 인덱스, 데이터가 저장됨
   - StockStrategy의 경우: `stock_strategy` **하나만 존재**

2. **Dataset (쿼리 파일 세트)**
   - 같은 데이터베이스에 대해 실행할 SQL 쿼리 모음
   - Train/Test로 분할된 `.txt` 파일
   - StockStrategy의 경우: 3개의 워크로드 존재

## 내부 동작 원리

### unified_test.py 흐름

```python
# 1. 사용자 입력
python unified_test.py --algo mscn --db stock_strategy --workload momentum_investing

# 2. unified_test.py (line 110-113)
if workload_name is None:
    dataset_name = db_name  # "stock_strategy"
else:
    dataset_name = f"{db_name}_{workload_name}"  # "stock_strategy_momentum_investing"

# 3. unified_test.py (line 370)
config.db = db_name  # "stock_strategy" ← 실제 DB 연결용

# 4. load_test_sql() 호출
test_sqls = load_test_sql(dataset_name)  # "stock_strategy_momentum_investing"

# 5. algorithm_examples/utils.py (line 67)
def load_test_sql(db):
    if "stock_strategy_momentum_investing" == db.lower():
        return StockStrategyMomentumInvestingDataset(DatabaseEnum.POSTGRESQL).read_test_sql()
        # ↑ 이 Dataset 클래스의 created_db_name = "stock_strategy"

# 6. Scheduler 생성 (MscnPresetScheduler.py line 117)
scheduler = SchedulerFactory.create_scheduler(config)
# config.db = "stock_strategy" 사용하여 PostgreSQL 연결

# 7. 실제 DB 쿼리 실행
scheduler.execute(sql)  # "stock_strategy" DB에서 momentum_investing 쿼리 실행
```

### 핵심 정리

| 변수/설정 | 값 | 용도 |
|----------|-----|------|
| `db_name` (CLI) | `stock_strategy` | 실제 PostgreSQL DB 이름 |
| `workload_name` (CLI) | `momentum_investing` | 쿼리 파일 선택용 |
| `dataset_name` (내부) | `stock_strategy_momentum_investing` | Dataset 클래스 선택용 |
| `config.db` | `stock_strategy` | DB 연결용 (DBController) |
| `train_sql_file` | `stock_strategy_momentum_investing_train.txt` | 실제 쿼리 파일 |

## 파일 구조

```
pilotscope/Dataset/StockStrategy/
├── stock_strategy.sql                           # PostgreSQL 덤프 (DB 생성용)
│
├── stock_strategy_value_investing_train.txt     # Dataset 1: Value Investing
├── stock_strategy_value_investing_test.txt
│
├── stock_strategy_momentum_investing_train.txt  # Dataset 2: Momentum
├── stock_strategy_momentum_investing_test.txt
│
└── stock_strategy_ml_hybrid_train.txt           # Dataset 3: ML Hybrid
    stock_strategy_ml_hybrid_test.txt
```

**중요:**
- `stock_strategy.sql`에서 생성되는 DB 이름: `stock_strategy`
- 모든 `*_train.txt`, `*_test.txt` 파일은 이 **하나의 DB**를 참조
- 파일 이름에 `stock_strategy_momentum_investing`이 있다고 해서 해당 이름의 DB가 생기는 게 아님!

## StockStrategyDataset 클래스 구조

```python
# pilotscope/Dataset/StockStrategyDataset.py

class StockStrategyDataset(BaseDataset):
    """베이스 클래스 - 기본 워크로드 (value_investing)"""
    data_location_dict = {DatabaseEnum.POSTGRESQL: "stock_strategy.sql"}
    sub_dir = "StockStrategy"
    train_sql_file = "stock_strategy_value_investing_train.txt"
    test_sql_file = "stock_strategy_value_investing_test.txt"

    def __init__(self, use_db_type, created_db_name="stock_strategy", ...):
        # ↑ 모든 서브클래스가 created_db_name="stock_strategy" 사용
        super().__init__(use_db_type, created_db_name, ...)

class StockStrategyMomentumInvestingDataset(StockStrategyDataset):
    """Momentum 워크로드 - 같은 DB, 다른 쿼리 파일"""
    train_sql_file = "stock_strategy_momentum_investing_train.txt"
    test_sql_file = "stock_strategy_momentum_investing_test.txt"

    def __init__(self, use_db_type, created_db_name="stock_strategy", ...):
        # ↑ 여전히 created_db_name="stock_strategy"!
        super().__init__(use_db_type, created_db_name, ...)
```

**핵심:**
- 모든 Dataset 클래스가 `created_db_name="stock_strategy"` 사용
- 차이점은 오직 `train_sql_file`, `test_sql_file`만!

## 다른 PilotScope Dataset과의 비교

### StatsTiny + StatsTinyCustom 패턴

```bash
# StatsTiny (기본 워크로드)
--db stats_tiny
# → Database: stats_tiny, Queries: stats_train_time2int.txt

# StatsTinyCustom (커스텀 워크로드)
--db stats_tiny --workload custom
# → Database: stats_tiny (같음!), Queries: stats_custom_train.txt
```

### StockStrategy (동일 패턴)

```bash
# 기본 워크로드
--db stock_strategy
# → Database: stock_strategy, Queries: stock_strategy_value_investing_train.txt

# 커스텀 워크로드
--db stock_strategy --workload momentum_investing
# → Database: stock_strategy (같음!), Queries: stock_strategy_momentum_investing_train.txt
```

## 잘못된 사용법과 그 결과

### ❌ 잘못된 사용법

```bash
python unified_test.py --algo mscn --db stock_strategy_momentum_investing
```

**문제:**
1. `config.db = "stock_strategy_momentum_investing"` 설정됨
2. DBController가 `stock_strategy_momentum_investing` DB에 연결 시도
3. **PostgreSQL 에러**: `database "stock_strategy_momentum_investing" does not exist`

```
psql -U pilotscope -h localhost -p 5432 -l
                              List of databases
     Name       |  Owner     | Encoding | Collate | Ctype | Access privileges
----------------+------------+----------+---------+-------+-------------------
 stock_strategy | pilotscope | UTF8     | ...     | ...   |
```

실제로는 `stock_strategy`만 존재!

### ✅ 올바른 사용법

```bash
python unified_test.py --algo mscn --db stock_strategy --workload momentum_investing
```

**결과:**
1. `config.db = "stock_strategy"` 설정됨 ✅
2. `dataset_name = "stock_strategy_momentum_investing"` (쿼리 파일 로딩용) ✅
3. DBController가 `stock_strategy` DB에 성공적으로 연결 ✅
4. `stock_strategy_momentum_investing_train.txt`의 쿼리 실행 ✅

## 검증 방법

### 1. PostgreSQL에서 확인

```bash
# 존재하는 데이터베이스 확인
psql -U pilotscope -h localhost -p 5432 -l | grep stock

# 결과: stock_strategy만 존재
stock_strategy | pilotscope | UTF8
```

### 2. Python에서 확인

```python
from pilotscope.Dataset.StockStrategyDataset import (
    StockStrategyDataset,
    StockStrategyMomentumInvestingDataset
)
from pilotscope.PilotEnum import DatabaseEnum

# 각 Dataset의 created_db_name 확인
ds1 = StockStrategyDataset(DatabaseEnum.POSTGRESQL)
ds2 = StockStrategyMomentumInvestingDataset(DatabaseEnum.POSTGRESQL)

print(ds1.created_db_name)  # "stock_strategy"
print(ds2.created_db_name)  # "stock_strategy" (같음!)

print(ds1.test_sql_file)    # "stock_strategy_value_investing_test.txt"
print(ds2.test_sql_file)    # "stock_strategy_momentum_investing_test.txt" (다름!)
```

### 3. DB 연결 로그 확인

```bash
# PostgreSQL 로그에서 연결 확인 (Docker 내부)
tail -f /var/log/postgresql/postgresql-13-main.log | grep connection

# 올바른 경우:
# connection authorized: user=pilotscope database=stock_strategy

# 잘못된 경우:
# FATAL:  database "stock_strategy_momentum_investing" does not exist
```

## 왜 이렇게 설계했나?

### 장점

1. **데이터베이스 중복 방지**
   - 같은 스키마/데이터를 여러 번 로드할 필요 없음
   - 디스크 공간 절약

2. **공정한 비교**
   - 모든 워크로드가 동일한 데이터베이스 상태에서 실행
   - 인덱스, 통계, 설정이 동일하게 유지

3. **MLflow 실험 관리**
   - 같은 DB에 대한 다양한 쿼리 패턴 성능 비교 가능
   - Experiment: `mscn_stock_strategy_momentum_investing`
   - Tag: `db_name=stock_strategy`, `workload=momentum_investing`

4. **확장 용이**
   - 새 워크로드 추가 시 쿼리 파일만 추가하면 됨
   - DB 재로드 불필요

### 유사 사례

운영 환경에서도 동일한 패턴:
- **하나의 Production DB**
- **여러 애플리케이션의 서로 다른 쿼리 패턴**
  - 분석팀: 복잡한 JOIN, Aggregation
  - API 서버: 단순 SELECT, Index Scan
  - 배치 작업: 대량 UPDATE, DELETE

StockStrategy Dataset이 바로 이런 시나리오를 시뮬레이션!

## 정리

### 핵심 원칙

```
1개 PostgreSQL Database = stock_strategy
3개 Workload (쿼리 세트) = value_investing, momentum_investing, ml_hybrid
```

### 사용법 템플릿

```bash
# 기본 워크로드
python unified_test.py --algo <algorithm> --db stock_strategy

# 특정 워크로드
python unified_test.py --algo <algorithm> --db stock_strategy --workload <workload_name>
```

### 금지 사항

```bash
# ❌ 절대 하지 말 것
--db stock_strategy_<workload_name>
```

이 패턴은 **존재하지 않는 데이터베이스**를 참조하려는 시도입니다!

## 참고 자료

- [STOCK_STRATEGY_QUICKSTART.md](./STOCK_STRATEGY_QUICKSTART.md) - 빠른 시작 가이드
- [CUSTOM_WORKLOAD_GUIDE.md](./CUSTOM_WORKLOAD_GUIDE.md) - 워크로드 생성 방법
- [pilotscope/Dataset/StockStrategyDataset.py](../pilotscope/Dataset/StockStrategyDataset.py) - Dataset 구현
- [algorithm_examples/utils.py](../algorithm_examples/utils.py) - Dataset 로더