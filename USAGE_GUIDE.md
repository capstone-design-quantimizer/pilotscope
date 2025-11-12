# PilotScope 활용 가이드

외부 DB와 쿼리를 사용하여 다양한 AI4DB 알고리즘을 테스트하는 실전 가이드입니다.

## 개요

PilotScope를 활용하면 **자신의 데이터베이스와 워크로드**에 대해 다양한 AI 기반 최적화 알고리즘을 쉽게 비교할 수 있습니다.

**지원 알고리즘**:
- **MSCN**: 카디널리티 추정 (Cardinality Estimation)
- **Lero**: 쿼리 플랜 최적화 (Query Plan Optimization)
- **KnobTuning**: 데이터베이스 설정 최적화
- **Index**: 인덱스 추천 (Index Selection)

## 빠른 시작 (5단계)

### 1단계: 환경 준비

```bash
# Docker 환경 시작
docker-compose up -d
docker-compose exec pilotscope-dev bash
conda activate pilotscope
```

### 2단계: 데이터베이스 준비

외부 PostgreSQL 데이터를 PilotScope의 PostgreSQL로 마이그레이션합니다.

```bash
# 방법 1: pg_dump 사용 (권장)
pg_dump -h external_host -U user -d source_db --schema-only > schema.sql
pg_dump -h external_host -U user -d source_db --data-only > data.sql

# PilotScope PostgreSQL로 복원
psql -h localhost -U postgres -d your_db < schema.sql
psql -h localhost -U postgres -d your_db < data.sql

# 방법 2: CSV 사용
# 1. 외부 DB에서 CSV 추출
# 2. pilotscope/Dataset/YourDataset/ 폴더 생성
# 3. CSV 파일 배치
```

### 3단계: 쿼리 준비

쿼리 파일을 `pilotscope/Dataset/YourDataset/` 폴더에 배치합니다.

```bash
# 폴더 생성
mkdir -p pilotscope/Dataset/YourDataset

# 쿼리 파일 작성
# train.txt: 학습용 쿼리 (한 줄에 하나씩)
# test.txt: 테스트용 쿼리
```

**예시 (pilotscope/Dataset/YourDataset/train.txt)**:
```sql
SELECT * FROM users WHERE age > 30;
SELECT COUNT(*) FROM orders WHERE status = 'completed';
SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id WHERE o.created_at > '2024-01-01';
```

**쿼리 준비 팁**:
- 한 줄에 하나의 쿼리 (줄바꿈으로 구분)
- 세미콜론(`;`)은 선택사항
- 주석은 `--`로 시작
- 학습용 쿼리: 최소 100개 권장 (많을수록 좋음)
- 테스트용 쿼리: 20~50개 권장

### 4단계: Dataset 클래스 생성

`pilotscope/Dataset/YourDataset.py` 파일을 생성합니다.

```python
from pilotscope.Dataset.BaseDataset import BaseDataset
from pilotscope.PilotEnum import DatabaseEnum

class YourDataset(BaseDataset):
    sub_dir = "YourDataset"
    train_sql_file = "train.txt"
    test_sql_file = "test.txt"
    file_db_type = DatabaseEnum.POSTGRESQL

    def __init__(self, use_db_type, created_db_name="your_db"):
        super().__init__(use_db_type, created_db_name)
        self.download_urls = None  # 다운로드 불필요
```

`algorithm_examples/utils.py`에 등록:

```python
def load_test_sql(db):
    # 기존 코드...

    # 추가
    if "yourdataset" == db.lower():
        from pilotscope.Dataset.YourDataset import YourDataset
        return YourDataset(DatabaseEnum.POSTGRESQL, "your_db").read_test_sql()

    # 나머지 코드...
```

### 5단계: 알고리즘 테스트 실행

```bash
cd test_example_algorithms

# 단일 알고리즘 테스트
python unified_test.py --algo mscn --db yourdataset

# 여러 알고리즘 비교
python unified_test.py --algo baseline mscn lero --db yourdataset --compare

# 학습 파라미터 조정
python unified_test.py --algo mscn --db yourdataset \
    --epochs 100 --training-size 500 --collection-size 500
```

## 커스텀 워크로드 사용

동일한 DB에 대해 여러 워크로드를 테스트하려면:

```bash
# 워크로드별 폴더 생성
mkdir -p pilotscope/Dataset/YourDataset/oltp
mkdir -p pilotscope/Dataset/YourDataset/olap

# 각 폴더에 쿼리 배치
# oltp/train.txt, oltp/test.txt
# olap/train.txt, olap/test.txt
```

Dataset 클래스 수정:

```python
class YourDatasetOLTP(BaseDataset):
    sub_dir = "YourDataset/oltp"
    train_sql_file = "train.txt"
    test_sql_file = "test.txt"
    file_db_type = DatabaseEnum.POSTGRESQL

    def __init__(self, use_db_type, created_db_name="your_db"):
        super().__init__(use_db_type, created_db_name)
        self.download_urls = None

class YourDatasetOLAP(BaseDataset):
    sub_dir = "YourDataset/olap"
    train_sql_file = "train.txt"
    test_sql_file = "test.txt"
    file_db_type = DatabaseEnum.POSTGRESQL

    def __init__(self, use_db_type, created_db_name="your_db"):
        super().__init__(use_db_type, created_db_name)
        self.download_urls = None
```

실행:

```bash
python unified_test.py --algo mscn --db yourdataset_oltp
python unified_test.py --algo mscn --db yourdataset_olap
```

## 결과 확인

### 실행 시간 비교

```bash
# MLflow UI 실행
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 브라우저에서 http://localhost:5000 접속
```

### 저장된 결과

```
ExampleData/
├── Mscn/
│   ├── Model/
│   │   ├── mscn_20241112_120000        # 모델 파일
│   │   └── mscn_20241112_120000.json   # 메타데이터
│   └── TrainingData/                   # 학습 데이터
└── Lero/
    └── ...
```

### 최적 모델 찾기

```bash
# MLflow에서 자동으로 찾기 (권장)
python unified_test.py --algo mscn --db yourdataset --no-training

# 또는 수동으로 지정
python unified_test.py --algo mscn --db yourdataset --no-training \
    --load-model mscn_20241112_120000
```

## 고급 활용

### 1. 하이퍼파라미터 튜닝

```bash
# epochs 변경
python unified_test.py --algo mscn --db yourdataset --epochs 200

# 학습 데이터 크기 제한
python unified_test.py --algo mscn --db yourdataset \
    --training-size 1000 --collection-size 1000
```

### 2. 학습 데이터만 수집 (학습은 나중에)

```bash
python unified_test.py --algo mscn --db yourdataset \
    --collection-only
```

### 3. 기존 데이터로 재학습

```bash
python unified_test.py --algo mscn --db yourdataset \
    --training-only --epochs 100
```

### 4. 여러 DB에서 학습한 모델 비교

```bash
# DB1에서 학습
python unified_test.py --algo mscn --db yourdataset1

# DB2에서 테스트 (DB1 모델 사용)
python unified_test.py --algo mscn --db yourdataset2 --no-training \
    --load-model mscn_yourdataset1_timestamp
```

## 문제 해결

### Q1. 쿼리 실행이 너무 느림
```bash
# 학습 쿼리 수 제한
python unified_test.py --algo mscn --db yourdataset --training-size 100

# 또는 빠른 쿼리만 필터링
# train.txt에서 긴 쿼리 제거
```

### Q2. 메모리 부족
```bash
# 배치 크기 감소 (코드 수정 필요)
# 또는 작은 데이터셋으로 시작

# 테스트 쿼리만 실행 (학습 스킵)
python unified_test.py --algo mscn --db yourdataset --no-training
```

### Q3. 모델 성능이 baseline보다 나쁨
- 학습 데이터 부족: 최소 100개 이상 권장
- 학습 쿼리와 테스트 쿼리 분포 차이: 비슷한 패턴의 쿼리로 구성
- epochs 부족: 100 → 200으로 증가
- 알고리즘이 워크로드에 맞지 않을 수 있음 (다른 알고리즘 시도)

### Q4. MLflow에서 결과를 찾을 수 없음
```bash
# MLflow 데이터베이스 확인
ls -la mlflow.db

# MLflow UI 재시작
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 실전 워크플로우

### 워크플로우 1: 새로운 DB 테스트

```bash
# 1. 데이터 준비
pg_dump -h external_host -U user -d source_db > backup.sql
psql -h localhost -U postgres -d your_db < backup.sql

# 2. Dataset 생성
# pilotscope/Dataset/YourDataset.py 작성

# 3. 쿼리 수집 (PostgreSQL 로그에서)
python scripts/extract_queries_from_log.py \
    --input /var/log/postgresql/postgresql.log \
    --output pilotscope/Dataset/YourDataset/ \
    --train-ratio 0.8

# 4. Baseline 성능 측정
python unified_test.py --algo baseline --db yourdataset

# 5. 알고리즘 비교
python unified_test.py --algo baseline mscn lero --db yourdataset --compare

# 6. MLflow에서 결과 확인
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### 워크플로우 2: 워크로드별 최적화

```bash
# 1. 워크로드 분류 (OLTP, OLAP, Mixed)
# 수동으로 쿼리 분류하여 폴더 구성

# 2. 각 워크로드 테스트
python unified_test.py --algo mscn --db yourdataset_oltp
python unified_test.py --algo mscn --db yourdataset_olap

# 3. 결과 비교
# MLflow UI에서 experiment별로 비교
```

### 워크플로우 3: 운영 배포

```bash
# 1. 최적 모델 선택
# MLflow UI에서 test_total_time 기준 최고 성능 모델 확인

# 2. 모델 저장
# ExampleData/Mscn/Model/mscn_TIMESTAMP 파일 백업

# 3. 운영 환경 테스트
python unified_test.py --algo mscn --db production_db \
    --no-training --load-model mscn_BEST_TIMESTAMP

# 4. 성능 모니터링
# MLflow에서 test_total_time, test_queries 메트릭 확인
```

## 다음 단계

- **알고리즘별 상세 가이드**: [algorithm_examples/CLAUDE.md](algorithm_examples/CLAUDE.md)
- **MSCN 커스터마이징**: [algorithm_examples/Mscn/CLAUDE.md](algorithm_examples/Mscn/CLAUDE.md)
- **Lero 커스터마이징**: [algorithm_examples/Lero/CLAUDE.md](algorithm_examples/Lero/CLAUDE.md)
- **상세 문서**: [docs/](docs/)

---

**팁**: 처음 사용한다면 작은 데이터셋(테이블 5개, 쿼리 50개)으로 시작하여 전체 워크플로우를 익힌 후, 실제 데이터로 확장하세요.
