# Custom Workload Integration Guide

**목적**: 운영 DB에서 덤프한 데이터와 생성된 워크로드를 PilotScope에 통합하여, 여러 AI4DB 알고리즘을 테스트하고 MLFlow로 최적의 튜닝 방안을 도출

## 📋 Overview

이 가이드는 다음 워크플로우를 다룹니다:

```
운영 DB 덤프 → PilotScope Dataset 생성 → 워크로드 분류 → 알고리즘 테스트 → MLFlow로 결과 분석
```

### 프로젝트 구조

```
pilotscope/
├── pilotscope/Dataset/StockStrategy/
│   ├── stock_strategy.sql                          # PostgreSQL 덤프 파일
│   ├── workload_02/                                # 원본 워크로드 (JSON 형식)
│   │   ├── pilotscope_batch_01.txt
│   │   ├── pilotscope_batch_02.txt
│   │   └── ... (50개 배치 파일)
│   ├── stock_strategy_value_investing_train.txt    # Value Investing 학습 쿼리
│   ├── stock_strategy_value_investing_test.txt     # Value Investing 테스트 쿼리
│   ├── stock_strategy_momentum_investing_train.txt # Momentum Investing 학습 쿼리
│   ├── stock_strategy_momentum_investing_test.txt  # Momentum Investing 테스트 쿼리
│   ├── stock_strategy_ml_hybrid_train.txt          # ML Hybrid 학습 쿼리
│   └── stock_strategy_ml_hybrid_test.txt           # ML Hybrid 테스트 쿼리
├── pilotscope/Dataset/StockStrategyDataset.py      # Dataset 클래스 정의
├── algorithm_examples/utils.py                     # Dataset 로더 등록
├── scripts/
│   ├── analyze_workload_templates.py               # 워크로드 분석 스크립트
│   └── split_workload_by_template.py               # 워크로드 분할 스크립트
└── test_example_algorithms/
    ├── test_stock_strategy_dataset.py              # Dataset 테스트
    ├── load_stock_strategy_db.py                   # DB 로드 스크립트
    └── unified_test.py                             # 통합 테스트 프레임워크
```

---

## 🚀 Step-by-Step Guide

### Step 1: 운영 DB 덤프 생성

```bash
# 운영 PostgreSQL 서버에서 실행
pg_dump -U your_user -d your_database \
    --no-owner --no-privileges \
    -f dump-postgres-YYYYMMDD.sql

# 또는 특정 테이블만 덤프
pg_dump -U your_user -d your_database \
    -t users -t stocks_daily_info -t fundamentals_daily \
    -t financials_quarterly -t strategies -t backtest_results \
    -f dump-postgres-YYYYMMDD.sql
```

**덤프 파일을 PilotScope로 복사:**
```bash
cp dump-postgres-YYYYMMDD.sql pilotscope/Dataset/StockStrategy/stock_strategy.sql
```

---

### Step 2: 워크로드 분석 및 분류

#### 2.1 워크로드 템플릿 분석

생성된 워크로드가 `pilotscope/Dataset/StockStrategy/workload_02/`에 있다고 가정합니다.

```bash
# Docker 컨테이너 진입
docker-compose exec pilotscope-dev bash
conda activate pilotscope
cd /workspace

# 워크로드 템플릿 분석
python scripts/analyze_workload_templates.py
```

**출력 예시:**
```
============================================================
Strategy Template Statistics
============================================================
Value Investing Style         :  1684 queries (33.68%)
Momentum Investing Style      :  1680 queries (33.60%)
ML Hybrid Style               :  1636 queries (32.72%)
------------------------------------------------------------
Total                         :  5000 queries
============================================================
```

#### 2.2 워크로드를 템플릿별로 분할

```bash
python scripts/split_workload_by_template.py \
    --input pilotscope/Dataset/StockStrategy/workload_02 \
    --output pilotscope/Dataset/StockStrategy \
    --train-ratio 0.8
```

**결과:**
- `stock_strategy_value_investing_train.txt` (1347 queries)
- `stock_strategy_value_investing_test.txt` (337 queries)
- `stock_strategy_momentum_investing_train.txt` (1344 queries)
- `stock_strategy_momentum_investing_test.txt` (336 queries)
- `stock_strategy_ml_hybrid_train.txt` (1308 queries)
- `stock_strategy_ml_hybrid_test.txt` (328 queries)

---

### Step 3: Dataset 클래스 정의 (이미 완료)

`pilotscope/Dataset/StockStrategyDataset.py`에 다음 클래스들이 정의되어 있습니다:

```python
# 파라미터화된 베이스 클래스
StockStrategyDataset(use_db_type, template='value_investing', created_db_name=None)

# 편의 클래스
StockStrategyValueInvestingDataset(use_db_type)
StockStrategyMomentumInvestingDataset(use_db_type)
StockStrategyMLHybridDataset(use_db_type)
```

**특징:**
- 하나의 DB 스키마 (`stock_strategy.sql`)를 공유
- 템플릿별로 다른 워크로드 사용
- 자동으로 DB명 설정 (`stock_strategy_{template}`)

---

### Step 4: Dataset 테스트

```bash
cd test_example_algorithms
python test_stock_strategy_dataset.py
```

**테스트 항목:**
1. Parameterized Dataset 로딩
2. Convenience Classes 로딩
3. Utils 통합 테스트
4. SQL 형식 검증

---

### Step 5: 데이터베이스 로드

```bash
# DB 로드 스크립트 실행 (최초 1회만)
python test_example_algorithms/load_stock_strategy_db.py
```

**내부 동작:**
1. PostgreSQL에 새 데이터베이스 생성 (`stock_strategy`)
2. 덤프 파일 (`stock_strategy.sql`) 복원
3. 인덱스 및 제약조건 자동 생성

**중요:**
- **모든 템플릿이 같은 DB (`stock_strategy`)를 공유**합니다
- DB는 **한 번만 로드**하면 됩니다
- 3개 템플릿(value_investing, momentum_investing, ml_hybrid)은 **워크로드(쿼리 파일)만** 다릅니다

---

### Step 6: Baseline 성능 측정

```bash
# Value Investing 워크로드로 Baseline 테스트
python unified_test.py --algo baseline --db stock_strategy_value_investing
```

**Baseline이란?**
- AI 알고리즘 없이 PostgreSQL 네이티브 옵티마이저만 사용
- 모든 알고리즘 비교의 기준점

**출력:**
```
============================================================
Algorithm: baseline | Database: stock_strategy_value_investing
============================================================
Total execution time: XX.XXs
Average query time: XX.XXs
Query count: 337
```

---

### Step 7: AI 알고리즘 테스트

#### 7.1 MSCN (Cardinality Estimation)

```bash
# 학습 + 테스트 (전체)
python unified_test.py \
    --algo mscn \
    --db stock_strategy_value_investing \
    --epochs 100 \
    --use-mlflow

# 빠른 테스트 (샘플링)
python unified_test.py \
    --algo mscn \
    --db stock_strategy_value_investing \
    --epochs 10 \
    --training-size 100 \
    --collection-size 100 \
    --use-mlflow
```

**MSCN이란?**
- Multi-Set Convolutional Network
- JOIN 카디널리티를 예측하여 쿼리 플랜 개선
- 학습 데이터: `stock_strategy_value_investing_train.txt`
- 테스트 데이터: `stock_strategy_value_investing_test.txt`

#### 7.2 Lero (Learned Optimizer)

```bash
python unified_test.py \
    --algo lero \
    --db stock_strategy_value_investing \
    --epochs 50 \
    --use-mlflow
```

**Lero란?**
- Learned Robust Optimizer
- 강화학습으로 최적의 쿼리 플랜 선택
- GPU 권장 (CPU는 느림)

#### 7.3 Knob Tuning

```bash
python unified_test.py \
    --algo knob \
    --db stock_strategy_value_investing \
    --use-mlflow
```

**Knob Tuning이란?**
- PostgreSQL 파라미터 자동 튜닝
- `shared_buffers`, `work_mem`, `effective_cache_size` 등 최적화

---

### Step 8: 템플릿별 비교 실험

3가지 워크로드 템플릿으로 동일한 알고리즘을 테스트하여 워크로드 특성 분석:

```bash
# 1. Value Investing (가치 투자 스타일)
python unified_test.py --algo mscn --db stock_strategy_value_investing --use-mlflow

# 2. Momentum Investing (모멘텀 투자 스타일)
python unified_test.py --algo mscn --db stock_strategy_momentum_investing --use-mlflow

# 3. ML Hybrid (ML 하이브리드 스타일)
python unified_test.py --algo mscn --db stock_strategy_ml_hybrid --use-mlflow
```

---

### Step 9: MLFlow로 결과 분석

#### 9.1 MLFlow UI 실행

```bash
# Docker 컨테이너 내부에서
cd /workspace/test_example_algorithms
mlflow ui --host 0.0.0.0 --port 5000
```

브라우저에서 접속: `http://localhost:5000`

#### 9.2 실험 비교

MLFlow UI에서 다음을 확인:

**메트릭:**
- `test/average_query_time`: 평균 쿼리 실행 시간
- `test/total_time`: 전체 테스트 시간
- `test/query_count`: 실행된 쿼리 수
- `train/loss`: 학습 손실 (MSCN, Lero)

**파라미터:**
- `algorithm`: 사용된 알고리즘 (baseline, mscn, lero, knob)
- `database`: 데이터셋 (stock_strategy_value_investing, etc.)
- `num_epoch`: 학습 에폭 수
- `num_training`: 학습 쿼리 수

**아티팩트:**
- 학습된 모델 파일 (MSCN: `.pt`, Lero: `.pkl`)
- 최적 Knob 설정 (Knob Tuning)

#### 9.3 Python API로 결과 조회

```python
import mlflow
import pandas as pd

# 실험 목록 조회
client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()

# 특정 실험의 런 조회
runs = mlflow.search_runs(experiment_ids=["0"], order_by=["metrics.test/total_time ASC"])

# 결과 DataFrame으로 분석
print(runs[['params.algorithm', 'params.database', 'metrics.test/total_time']])
```

---

## 📊 워크로드 템플릿 특성

### Value Investing Style (1684 queries)
**특징:**
- 재무 비율 중심 (P/E, P/B, dividend_yield)
- `financials_quarterly` 테이블 집중 사용
- 복잡한 서브쿼리 (TTM 계산)

**예시 쿼리:**
```sql
SELECT fd.per, fd.pbr, fd.dividend_yield, sdi.close_price
FROM stocks_daily_info sdi
LEFT JOIN fundamentals_daily fd ON fd.ticker = sdi.ticker
WHERE sdi.market = 'KOSPI' AND fd.pbr < 1.0
ORDER BY fd.per ASC, fd.pbr ASC;
```

### Momentum Investing Style (1680 queries)
**특징:**
- 기술적 지표 중심 (RSI, moving averages, momentum)
- `stocks_daily_info` 테이블 집중 사용
- 시계열 데이터 분석

**예시 쿼리:**
```sql
SELECT ticker, close_price, rsi_14, momentum_3m, momentum_12m
FROM stocks_daily_info
WHERE rsi_14 > 70 OR rsi_14 < 30
ORDER BY momentum_12m DESC;
```

### ML Hybrid Style (1636 queries)
**특징:**
- ML 모델 예측 + 전통적 지표 결합
- `ml_models`, `strategies`, `backtest_results` JOIN
- JSONB 컬럼 활용 (metrics, strategy_json)

**예시 쿼리:**
```sql
SELECT s.name, br.metrics->>'sharpe_ratio' as sharpe, m.name as model
FROM strategies s
JOIN backtest_results br ON s.id = br.strategy_id
JOIN ml_models m ON br.ml_model_id = m.id
ORDER BY (br.metrics->>'total_return')::float DESC;
```

---

## 🔧 고급 활용

### 1. 커스텀 워크로드 추가

새로운 워크로드 템플릿을 추가하려면:

```python
# pilotscope/Dataset/StockStrategyDataset.py에 추가
TEMPLATES = ['value_investing', 'momentum_investing', 'ml_hybrid', 'your_custom_template']
```

```bash
# 워크로드 파일 생성
# pilotscope/Dataset/StockStrategy/stock_strategy_your_custom_template_train.txt
# pilotscope/Dataset/StockStrategy/stock_strategy_your_custom_template_test.txt
```

### 2. 하이퍼파라미터 튜닝

```bash
# Grid search 예시
for epochs in 10 50 100; do
  for training_size in 100 500 1000; do
    python unified_test.py \
      --algo mscn \
      --db stock_strategy_value_investing \
      --epochs $epochs \
      --training-size $training_size \
      --use-mlflow
  done
done
```

### 3. 배치 실험 스크립트

```bash
#!/bin/bash
# batch_experiment.sh

ALGORITHMS=("baseline" "mscn" "lero" "knob")
DATASETS=("stock_strategy_value_investing" "stock_strategy_momentum_investing" "stock_strategy_ml_hybrid")

for algo in "${ALGORITHMS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo "Running $algo on $dataset..."
    python unified_test.py --algo $algo --db $dataset --use-mlflow
  done
done
```

---

## 📝 Best Practices

### 1. 실험 관리
- **명확한 실험 이름**: MLFlow에서 `mlflow.set_experiment("stock_strategy_comparison")`
- **태그 활용**: `mlflow.set_tag("workload_type", "value_investing")`
- **노트 기록**: 각 실험의 목적과 결과를 MLFlow UI에 기록

### 2. 리소스 최적화
- **샘플링 사용**: 빠른 프로토타이핑 시 `--training-size`, `--collection-size` 제한
- **GPU 활용**: Lero 학습 시 GPU 필수 (`export CUDA_VISIBLE_DEVICES=0`)
- **병렬 실험**: 독립적인 실험은 다른 포트/디렉토리에서 동시 실행 가능

### 3. 결과 재현성
- **Random seed 고정**: `split_workload_by_template.py`의 `--seed` 파라미터
- **모델 버전 관리**: MLFlow Artifacts에 자동 저장됨
- **환경 기록**: MLFlow가 자동으로 conda env, git commit 기록

---

## 🐛 Troubleshooting

### 문제 1: DB 로드 실패
```
ERROR: database "stock_strategy" already exists
```

**해결:**
```bash
psql -U postgres -h localhost -p 5432 -c "DROP DATABASE IF EXISTS stock_strategy;"
python test_example_algorithms/load_stock_strategy_db.py
```

### 문제 2: 쿼리 실행 오류
```
ERROR: column "fq.period_end" must appear in the GROUP BY clause
```

**해결:**
- 일부 워크로드 쿼리에 SQL 에러가 있을 수 있음
- Baseline 테스트로 에러 쿼리 필터링 후 수동 수정

### 문제 3: MSCN 학습 너무 느림
```
Training epoch 1/100... (30 minutes elapsed)
```

**해결:**
```bash
# 샘플링으로 빠른 테스트
python unified_test.py --algo mscn --db stock_strategy_value_investing \
    --epochs 10 --training-size 100
```

### 문제 4: MLFlow UI 접속 안 됨
```
Connection refused: http://localhost:5000
```

**해결:**
```bash
# Docker 포트 포워딩 확인
docker-compose ps  # 5000 포트가 매핑되어 있는지 확인

# 또는 호스트 IP 사용
mlflow ui --host 0.0.0.0 --port 5000
```

---

## 📚 참고 자료

- **PilotScope 공식 문서**: [GitHub](https://github.com/alibaba/pilotscope)
- **MSCN 논문**: "Learned Cardinalities: Estimating Correlated Joins with Deep Learning" (CIDR 2019)
- **Lero 논문**: "Lero: A Learning-to-Rank Query Optimizer" (SIGMOD 2023)
- **MLFlow 문서**: [mlflow.org](https://mlflow.org)

---

## 🎯 Workflow Summary

```
1. pg_dump로 운영 DB 덤프
   ↓
2. 워크로드 생성 및 분류 (scripts/split_workload_by_template.py)
   ↓
3. Dataset 클래스 정의 (pilotscope/Dataset/StockStrategyDataset.py)
   ↓
4. DB 로드 (test_example_algorithms/load_stock_strategy_db.py)
   ↓
5. Baseline 측정 (unified_test.py --algo baseline)
   ↓
6. AI 알고리즘 테스트 (unified_test.py --algo mscn/lero/knob)
   ↓
7. MLFlow UI에서 결과 비교 분석
   ↓
8. 최적 설정 도출 및 운영 환경 적용
```

**예상 소요 시간:**
- Dataset 생성: 10분
- DB 로드: 5분
- Baseline 테스트: 5-10분
- MSCN 학습 (전체): 1-2시간
- Lero 학습 (전체): 3-5시간 (GPU)
- Knob Tuning: 30분-1시간

**총 실험 시간 (3개 템플릿 × 4개 알고리즘)**: ~12-20시간