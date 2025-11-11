# StockStrategy Dataset Quick Start

외부 운영 DB에서 덤프한 데이터를 PilotScope에 통합하여 AI4DB 알고리즘 테스트 및 MLFlow로 최적 튜닝 도출

## ⚠️ 중요: Database vs Workload

**핵심 개념:**
- **데이터베이스 (Database)**: PostgreSQL 물리적 DB = `stock_strategy` (단 1개)
- **워크로드 (Workload)**: 같은 DB에 대한 서로 다른 쿼리 세트 (3개)

```bash
# ✅ 올바른 사용법
--db stock_strategy                              # 기본 워크로드 (value_investing)
--db stock_strategy --workload momentum_investing  # momentum_investing 워크로드
--db stock_strategy --workload ml_hybrid          # ml_hybrid 워크로드

# ❌ 잘못된 사용법 (데이터베이스가 존재하지 않음!)
--db stock_strategy_momentum_investing  # 이런 DB는 없습니다!
```

**아키텍처:**
```
┌─────────────────────────────────────────┐
│  PostgreSQL Database: stock_strategy   │  ← 단 하나의 물리적 DB
│  (8 tables)                             │
└───────────┬─────────────────────────────┘
            │ 같은 DB, 다른 쿼리 패턴
            │
    ┌───────┼───────┬────────────┐
    ▼       ▼       ▼            ▼
 Value   Momentum  ML Hybrid  (미래 워크로드)
 1684개   1680개    1636개
```

---

## 🚀 5분 Quick Start

### 1. Docker 환경 진입
```bash
docker-compose exec pilotscope-dev bash
conda activate pilotscope
cd /workspace
```

### 2. Dataset 테스트
```bash
cd test_example_algorithms
python test_stock_strategy_dataset.py
```

**기대 출력:**
```
[SUCCESS] All tests passed!
- Value Investing: 1347 train / 337 test queries
- Momentum Investing: 1344 train / 336 test queries
- ML Hybrid: 1308 train / 328 test queries
```

### 3. 데이터베이스 로드 (최초 1회)
```bash
python load_stock_strategy_db.py
```

**참고:** 모든 템플릿이 같은 DB (`stock_strategy`)를 공유하므로 한 번만 로드하면 됩니다.

### 4. Baseline 테스트
```bash
# 기본 워크로드 (value_investing)
python unified_test.py --algo baseline --db stock_strategy

# 또는 명시적으로 지정
python unified_test.py --algo baseline --db stock_strategy --workload value_investing
```

### 5. MSCN 알고리즘 테스트
```bash
# 빠른 테스트 (10분)
python unified_test.py \
    --algo mscn \
    --db stock_strategy \
    --epochs 10 \
    --training-size 100 \
    --collection-size 100 \
    --use-mlflow

# 전체 테스트 (1-2시간)
python unified_test.py \
    --algo mscn \
    --db stock_strategy \
    --epochs 100 \
    --use-mlflow
```

### 6. MLFlow UI 확인
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

브라우저: `http://localhost:5000`

---

## 📊 사용 가능한 워크로드

| 워크로드 이름 | 특성 | Train | Test | 사용법 |
|-------------|-----|-------|------|--------|
| `value_investing` | 가치투자 (P/E, P/B, 배당) | 1347 | 337 | `--db stock_strategy` (기본값) |
| `momentum_investing` | 모멘텀 (RSI, 이동평균) | 1344 | 336 | `--db stock_strategy --workload momentum_investing` |
| `ml_hybrid` | ML 하이브리드 | 1308 | 328 | `--db stock_strategy --workload ml_hybrid` |

**중요:**
- **PostgreSQL 데이터베이스**: `stock_strategy` (단 하나만 존재)
- **워크로드**: 같은 DB에 대한 서로 다른 쿼리 세트
- 모든 워크로드가 동일한 8개 테이블을 참조하되, 다른 쿼리 패턴 사용

**동작 원리:**
```bash
# unified_test.py 내부적으로:
config.db = "stock_strategy"  # 실제 DB 연결
dataset_name = "stock_strategy_momentum_investing"  # 쿼리 파일 로딩용
```

---

## 🧪 알고리즘 비교 실험

### 전체 알고리즘 테스트
```bash
#!/bin/bash
# batch_test.sh

ALGORITHMS=("baseline" "mscn" "lero" "knob")

for algo in "${ALGORITHMS[@]}"; do
    echo "Testing $algo on stock_strategy (value_investing workload)..."
    python unified_test.py --algo $algo --db stock_strategy --use-mlflow
done
```

### 워크로드별 비교
```bash
# 같은 알고리즘을 3개 워크로드로 테스트
python unified_test.py --algo mscn --db stock_strategy --use-mlflow
python unified_test.py --algo mscn --db stock_strategy --workload momentum_investing --use-mlflow
python unified_test.py --algo mscn --db stock_strategy --workload ml_hybrid --use-mlflow
```

---

## 📁 프로젝트 구조

```
pilotscope/
├── pilotscope/Dataset/StockStrategy/
│   ├── stock_strategy.sql                          # PostgreSQL 덤프
│   ├── workload_02/                                # 원본 워크로드 (5000 queries)
│   ├── stock_strategy_value_investing_train.txt    # 분할된 워크로드
│   ├── stock_strategy_value_investing_test.txt
│   ├── stock_strategy_momentum_investing_train.txt
│   ├── stock_strategy_momentum_investing_test.txt
│   ├── stock_strategy_ml_hybrid_train.txt
│   └── stock_strategy_ml_hybrid_test.txt
├── pilotscope/Dataset/StockStrategyDataset.py      # Dataset 클래스
├── algorithm_examples/utils.py                     # Dataset 로더 등록
├── scripts/
│   ├── analyze_workload_templates.py               # 워크로드 분석
│   └── split_workload_by_template.py               # 워크로드 분할
└── test_example_algorithms/
    ├── test_stock_strategy_dataset.py              # Dataset 테스트
    ├── load_stock_strategy_db.py                   # DB 로드
    └── unified_test.py                             # 통합 테스트
```

---

## 🔄 워크플로우 요약

```
1. 운영 DB 덤프 (pg_dump)
   ↓
2. pilotscope/Dataset/StockStrategy/stock_strategy.sql로 복사
   ↓
3. 워크로드 생성 및 분류 (이미 완료)
   ↓
4. load_stock_strategy_db.py 실행
   ↓
5. unified_test.py로 알고리즘 테스트
   ↓
6. MLFlow UI에서 결과 분석
```

---

## 📈 MLFlow 결과 분석

### 주요 메트릭
- `test/average_query_time`: 평균 쿼리 시간
- `test/total_time`: 전체 실행 시간
- `test/query_count`: 쿼리 수

### 비교 방법
1. MLFlow UI에서 Run 선택
2. "Compare" 버튼 클릭
3. Parallel Coordinates Plot으로 시각화
4. 최적 파라미터 조합 도출

---

## ⚙️ 커스터마이징

### 새 워크로드 추가
```bash
# 1. SQL 파일 생성
pilotscope/Dataset/StockStrategy/stock_strategy_custom_template_train.txt
pilotscope/Dataset/StockStrategy/stock_strategy_custom_template_test.txt

# 2. Dataset 클래스 추가 (pilotscope/Dataset/StockStrategyDataset.py)
class StockStrategyCustomTemplateDataset(StockStrategyDataset):
    train_sql_file = "stock_strategy_custom_template_train.txt"
    test_sql_file = "stock_strategy_custom_template_test.txt"

    def __init__(self, use_db_type, created_db_name="stock_strategy", ...):
        super().__init__(use_db_type, created_db_name, ...)

# 3. algorithm_examples/utils.py에 등록
elif "stock_strategy_custom_template" == db.lower():
    return StockStrategyCustomTemplateDataset(DatabaseEnum.POSTGRESQL).read_test_sql()

# 4. 사용
python unified_test.py --algo mscn --db stock_strategy --workload custom_template
```

### 하이퍼파라미터 튜닝
```bash
# Grid search
for epochs in 10 50 100; do
  python unified_test.py --algo mscn --db stock_strategy \
    --epochs $epochs --use-mlflow
done
```

---

## 🐛 자주 발생하는 문제

### 1. DB가 이미 존재
```bash
psql -U postgres -h localhost -p 5432 \
    -c "DROP DATABASE IF EXISTS stock_strategy;"
```

### 2. 쿼리 실행 에러
- Baseline 테스트로 에러 쿼리 필터링
- SQL 파일에서 수동 수정

### 3. 학습 시간이 너무 오래 걸림
```bash
# 샘플링으로 빠른 테스트
python unified_test.py --algo mscn --db stock_strategy \
    --epochs 10 --training-size 100 --collection-size 100
```

### 4. 잘못된 사용법 (흔한 실수)
```bash
# ❌ 틀린 사용법 - 데이터베이스가 존재하지 않음!
python unified_test.py --algo mscn --db stock_strategy_momentum_investing

# ✅ 올바른 사용법 - DB와 워크로드를 분리
python unified_test.py --algo mscn --db stock_strategy --workload momentum_investing
```

**왜 틀렸나?**
- `stock_strategy_momentum_investing`이라는 PostgreSQL 데이터베이스는 존재하지 않습니다
- 실제 DB는 `stock_strategy` 하나뿐이고, `momentum_investing`은 쿼리 파일 이름입니다

---

## 📚 더 자세한 내용

- **전체 가이드**: [docs/CUSTOM_WORKLOAD_GUIDE.md](./CUSTOM_WORKLOAD_GUIDE.md)
- **PilotScope 문서**: [docs/DOCKER_GUIDE.md](./DOCKER_GUIDE.md)
- **Model 관리**: [docs/MODEL_MANAGEMENT.md](./MODEL_MANAGEMENT.md)

---

## 🎯 예상 실험 시간

| 작업 | 소요 시간 |
|-----|---------|
| Dataset 생성 | 10분 |
| DB 로드 | 5분 |
| Baseline 테스트 | 5-10분 |
| MSCN (샘플) | 10-15분 |
| MSCN (전체) | 1-2시간 |
| Lero (전체) | 3-5시간 (GPU) |
| Knob Tuning | 30분-1시간 |

**총 실험 시간** (3개 템플릿 × 4개 알고리즘): 약 12-20시간