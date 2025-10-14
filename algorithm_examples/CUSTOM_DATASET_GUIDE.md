# Adding Custom Datasets to PilotScope

## Overview

PilotScope의 데이터셋 시스템은 **3가지 구성 요소**로 이루어져 있습니다:

1. **SQL 파일** (`.txt`): 학습/테스트용 SQL 쿼리 모음
2. **데이터셋 클래스** (`.py`): SQL 로딩 및 DB 초기화 로직
3. **데이터 파일** (optional): 실제 데이터 (dump 파일, CSV 등)

---

## Quick Start: SQL 파일만 추가하기 (가장 간단)

기존 데이터베이스에 새로운 SQL 워크로드만 추가하려면:

### Step 1: SQL 파일 생성

```bash
# pilotscope/Dataset/MyWorkload/ 폴더 생성
mkdir pilotscope/Dataset/MyWorkload

# SQL 파일 작성
# pilotscope/Dataset/MyWorkload/my_train.txt
```

**형식** (각 쿼리는 세미콜론으로 구분):
```sql
select count(*) from users where age > 25;
select name, email from users where created_at > '2024-01-01'::timestamp;
select u.name, count(o.id) from users u join orders o on u.id = o.user_id group by u.name;
```

### Step 2: 데이터셋 클래스 생성

```python
# pilotscope/Dataset/MyWorkloadDataset.py
from pilotscope.Dataset.BaseDataset import BaseDataset
from pilotscope.PilotEnum import DatabaseEnum
import os

class MyWorkloadDataset(BaseDataset):
    """
    Custom workload for my application.
    """
    sub_dir = "MyWorkload"
    train_sql_file = "my_train.txt"
    test_sql_file = "my_test.txt"
    file_db_type = DatabaseEnum.POSTGRESQL  # SQL 파일의 문법 타입
    
    def __init__(self, use_db_type: DatabaseEnum, created_db_name="my_database", data_dir=None):
        super().__init__(use_db_type, created_db_name, data_dir)
        # download_urls는 None (이미 존재하는 DB 사용)
        self.download_urls = None
```

### Step 3: 사용하기

```python
from pilotscope.Dataset.MyWorkloadDataset import MyWorkloadDataset
from pilotscope.PilotEnum import DatabaseEnum

# 데이터셋 로드
dataset = MyWorkloadDataset(DatabaseEnum.POSTGRESQL, created_db_name="production_db")

# SQL 가져오기
train_sqls = dataset.read_train_sql()
test_sqls = dataset.read_test_sql()

# 테스트 실행
for sql in train_sqls:
    result = scheduler.execute(sql)
```

---

## Advanced: 완전한 데이터셋 추가하기

새로운 데이터베이스 + 데이터 + SQL을 모두 포함하려면:

### Directory Structure

```
pilotscope/Dataset/
├── MyDataset/                    # 새 데이터셋 폴더
│   ├── schema.sql               # 테이블 스키마 정의
│   ├── my_train.txt             # 학습용 SQL
│   └── my_test.txt              # 테스트용 SQL
└── MyDatasetDataset.py          # 데이터셋 클래스
```

### Step 1: 스키마 파일 작성

```sql
-- pilotscope/Dataset/MyDataset/schema.sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    age INT,
    created_at TIMESTAMP
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    amount DECIMAL(10,2),
    created_at TIMESTAMP
);

CREATE INDEX idx_users_age ON users(age);
CREATE INDEX idx_orders_user_id ON orders(user_id);
```

### Step 2: SQL 워크로드 작성

```sql
-- pilotscope/Dataset/MyDataset/my_train.txt
select count(*) from users where age > 25;
select name from users where email like '%@gmail.com';
select u.name, count(o.id) from users u join orders o on u.id = o.user_id group by u.name;
```

### Step 3: 데이터셋 클래스 작성

```python
# pilotscope/Dataset/MyDatasetDataset.py
from pilotscope.Dataset.BaseDataset import BaseDataset
from pilotscope.PilotEnum import DatabaseEnum
import os

class MyDatasetDataset(BaseDataset):
    """
    My custom dataset with schema and data.
    """
    # 데이터 다운로드 URL (GitHub Release 등)
    data_location_dict = {
        DatabaseEnum.POSTGRESQL: [
            "https://github.com/myuser/myrepo/releases/download/v1.0/my_data.tar.gz"
        ],
        DatabaseEnum.SPARK: None
    }
    
    # 데이터 파일의 SHA256 해시 (무결성 검증용)
    data_sha256 = "your_sha256_hash_here"
    
    # 파일 경로
    sub_dir = "MyDataset"
    schema_file = "schema.sql"
    train_sql_file = "my_train.txt"
    test_sql_file = "my_test.txt"
    file_db_type = DatabaseEnum.POSTGRESQL
    
    def __init__(self, use_db_type: DatabaseEnum, created_db_name="my_db", data_dir=None):
        super().__init__(use_db_type, created_db_name, data_dir)
        self.download_urls = self.data_location_dict[use_db_type]
```

### Step 4: 데이터 준비 (Optional)

데이터 파일을 제공하려면:

```bash
# PostgreSQL dump 파일 생성
pg_dump my_database > my_data.dump

# tar.gz로 압축
tar -czf my_data.tar.gz my_data.dump

# GitHub Release 등에 업로드하고 URL을 data_location_dict에 추가
```

### Step 5: 사용하기

```python
from pilotscope.PilotConfig import PostgreSQLConfig
from pilotscope.Dataset.MyDatasetDataset import MyDatasetDataset
from pilotscope.PilotEnum import DatabaseEnum

# 1. 데이터셋 초기화
dataset = MyDatasetDataset(DatabaseEnum.POSTGRESQL)

# 2. 데이터베이스에 로드 (최초 1회)
config = PostgreSQLConfig()
dataset.load_to_db(config)  # 자동으로 다운로드 + DB 생성

# 3. 스키마 실행 (필요시)
schema_sqls = dataset.read_schema()
for sql in schema_sqls:
    db_controller.execute(sql)

# 4. 워크로드 실행
train_sqls = dataset.read_train_sql()
test_sqls = dataset.read_test_sql()
```

---

## Pattern 1: 기존 프로덕션 DB의 쿼리 로그 사용

실제 운영 환경에서 가장 흔한 케이스:

```python
# pilotscope/Dataset/ProductionWorkloadDataset.py
from pilotscope.Dataset.BaseDataset import BaseDataset
from pilotscope.PilotEnum import DatabaseEnum

class ProductionWorkloadDataset(BaseDataset):
    """
    Real production query logs from our application.
    """
    sub_dir = "ProductionWorkload"
    train_sql_file = "prod_queries_2024_01.txt"  # 1월 쿼리
    test_sql_file = "prod_queries_2024_02.txt"   # 2월 쿼리
    file_db_type = DatabaseEnum.POSTGRESQL
    
    def __init__(self, use_db_type: DatabaseEnum, created_db_name="production"):
        super().__init__(use_db_type, created_db_name, data_dir=None)
        self.download_urls = None  # 이미 존재하는 DB 사용

# 사용
dataset = ProductionWorkloadDataset(DatabaseEnum.POSTGRESQL)
real_queries = dataset.read_train_sql()
```

**쿼리 로그 수집 방법:**
```sql
-- PostgreSQL 쿼리 로그 활성화
ALTER SYSTEM SET log_statement = 'all';
SELECT pg_reload_conf();

-- 로그에서 SQL 추출
grep "SELECT\|INSERT\|UPDATE\|DELETE" /var/log/postgresql/postgresql.log > prod_queries.txt
```

---

## Pattern 2: 애플리케이션별 워크로드

특정 기능의 쿼리만 모아서:

```python
# pilotscope/Dataset/AnalyticsWorkloadDataset.py
class AnalyticsWorkloadDataset(BaseDataset):
    """
    Analytics dashboard queries only.
    """
    sub_dir = "AnalyticsWorkload"
    train_sql_file = "analytics_train.txt"
    test_sql_file = "analytics_test.txt"
    file_db_type = DatabaseEnum.POSTGRESQL
    
    def __init__(self, use_db_type: DatabaseEnum, created_db_name="analytics_db"):
        super().__init__(use_db_type, created_db_name)
        self.download_urls = None

# analytics_train.txt 예시:
"""
select date_trunc('day', created_at) as date, count(*) from orders group by date;
select category, sum(amount) from products p join orders o on p.id = o.product_id group by category;
select user_id, count(*) as order_count from orders where created_at > '2024-01-01' group by user_id;
"""
```

---

## Pattern 3: 동적 SQL 생성

파일 대신 코드로 SQL 생성:

```python
# pilotscope/Dataset/SyntheticWorkloadDataset.py
from pilotscope.Dataset.BaseDataset import BaseDataset
import random

class SyntheticWorkloadDataset(BaseDataset):
    """
    Dynamically generated synthetic workload.
    """
    def __init__(self, use_db_type, created_db_name="test_db"):
        super().__init__(use_db_type, created_db_name)
    
    def read_train_sql(self):
        """동적으로 SQL 생성"""
        sqls = []
        for i in range(100):
            age = random.randint(18, 80)
            sqls.append(f"select count(*) from users where age > {age}")
        return sqls
    
    def read_test_sql(self):
        """테스트용 SQL 생성"""
        sqls = []
        for i in range(50):
            limit = random.randint(10, 100)
            sqls.append(f"select * from users order by created_at desc limit {limit}")
        return sqls

# 사용
dataset = SyntheticWorkloadDataset(DatabaseEnum.POSTGRESQL, "my_db")
train_sqls = dataset.read_train_sql()  # 매번 다른 쿼리 생성
```

---

## Utility: utils.py 수정

기존 `algorithm_examples/utils.py`의 `load_test_sql()` 함수를 확장:

```python
# algorithm_examples/utils.py에 추가
from pilotscope.Dataset.MyWorkloadDataset import MyWorkloadDataset

def load_test_sql(db):
    if "stats_tiny" == db.lower():
        return StatsTinyDataset(DatabaseEnum.POSTGRESQL).read_test_sql()
    elif "stats" in db.lower():
        return StatsDataset(DatabaseEnum.POSTGRESQL).read_test_sql()
    elif "imdb" in db:
        return ImdbDataset(DatabaseEnum.POSTGRESQL).read_test_sql()
    elif "tpcds" in db.lower():
        return TpcdsDataset(DatabaseEnum).read_test_sql()
    elif "my_workload" == db.lower():  # 추가!
        return MyWorkloadDataset(DatabaseEnum.POSTGRESQL).read_test_sql()
    elif "production" == db.lower():   # 추가!
        return ProductionWorkloadDataset(DatabaseEnum.POSTGRESQL).read_test_sql()
    else:
        raise NotImplementedError
```

---

## SQL 파일 형식 규칙

### ✅ 올바른 형식

```sql
select count(*) from users where age > 25;
select name from users where email like '%@gmail.com';
select u.name, count(o.id) from users u join orders o on u.id = o.user_id group by u.name;
```

**규칙:**
- 각 쿼리는 **세미콜론(`;`)으로 종료**
- 한 줄 또는 여러 줄 모두 가능
- 주석 가능 (`--` 또는 `/* */`)

### ❌ 잘못된 형식

```sql
select count(*) from users where age > 25  # 세미콜론 없음 ❌
select name from users;; # 세미콜론 2개 ❌
```

---

## Complete Example: E-commerce Dataset

실제 전자상거래 데이터셋 예제:

```python
# pilotscope/Dataset/EcommerceDataset.py
from pilotscope.Dataset.BaseDataset import BaseDataset
from pilotscope.PilotEnum import DatabaseEnum

class EcommerceDataset(BaseDataset):
    """
    E-commerce workload with orders, products, and users.
    """
    sub_dir = "Ecommerce"
    schema_file = "ecommerce_schema.sql"
    train_sql_file = "ecommerce_train.txt"
    test_sql_file = "ecommerce_test.txt"
    file_db_type = DatabaseEnum.POSTGRESQL
    
    def __init__(self, use_db_type: DatabaseEnum, created_db_name="ecommerce"):
        super().__init__(use_db_type, created_db_name)
        self.download_urls = None

# 테스트에서 사용
from pilotscope.PilotConfig import PostgreSQLConfig

config = PostgreSQLConfig()
config.db = "ecommerce"

dataset = EcommerceDataset(DatabaseEnum.POSTGRESQL)
train_sqls = dataset.read_train_sql()

# 알고리즘 테스트
scheduler = get_mscn_preset_scheduler(config)
for sql in train_sqls:
    scheduler.execute(sql)
```

---

## Summary

### 🎯 3가지 방법

| 방법 | 언제 사용? | 필요 파일 |
|------|-----------|----------|
| **SQL만 추가** | 기존 DB에 새 워크로드 추가 | `.txt` 파일만 |
| **완전한 데이터셋** | 새 벤치마크 데이터셋 배포 | `.txt` + `schema.sql` + 데이터 |
| **동적 생성** | 합성 워크로드, 테스트 | Python 코드로 생성 |

### 📝 체크리스트

- [ ] `pilotscope/Dataset/{MyDataset}/` 폴더 생성
- [ ] `{dataset}_train.txt`, `{dataset}_test.txt` 작성
- [ ] `MyDatasetDataset.py` 클래스 작성
- [ ] `algorithm_examples/utils.py`에 로딩 로직 추가
- [ ] 테스트 실행 확인

### 🚀 Quick Template

```bash
# 1. 폴더 생성
mkdir pilotscope/Dataset/MyDataset

# 2. SQL 파일 작성
echo "select * from users;" > pilotscope/Dataset/MyDataset/my_train.txt

# 3. 데이터셋 클래스 복사 & 수정
cp pilotscope/Dataset/StatsDataset.py pilotscope/Dataset/MyDatasetDataset.py
# sub_dir, train_sql_file 등 수정

# 4. 사용!
python test_example_algorithms/test_mscn_example.py
```

**핵심**: `.txt` 파일에 SQL을 세미콜론으로 구분해서 저장하면 끝! 🎉
