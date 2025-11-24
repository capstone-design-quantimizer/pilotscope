# MSCN Training Loss NaN Analysis

## 문제 요약

`stock_strategy_representative_momentum` 워크로드로 MSCN 학습 시 `loss: nan` 발생.
그러나 테스트 단계는 20/20 성공률로 완료되고, baseline보다 나은 성능(avg_time 감소)을 보임.

## 근본 원인

### 1. 데이터 문제: market_cap 컬럼이 모든 행에서 NULL

```sql
-- 데이터 현황
SELECT COUNT(*) as total, COUNT(market_cap) as non_null
FROM stocks_daily_info;
-- Result: total=12700, non_null=0

-- 실제 데이터 샘플
SELECT event_date, ticker, market_cap FROM stocks_daily_info LIMIT 5;
```

| event_date | ticker    | market_cap |
|------------|-----------|------------|
| 2020-08-24 | 005930.KS | NULL       |
| 2020-08-25 | 005930.KS | NULL       |
| 2020-08-26 | 005930.KS | NULL       |

### 2. 쿼리 필터 결과: 모든 subquery의 cardinality가 0

representative_momentum 워크로드의 모든 쿼리는 다음과 같은 패턴:

```sql
WITH base AS (
    SELECT sdi.ticker, sdi.company_name, ..., sdi.market_cap, ...
    FROM stocks_daily_info sdi
    WHERE sdi.event_date = DATE '2023-06-02'
      AND sdi.market_cap >= 0  -- ⚠️ 이 조건 때문에 0 rows 반환
),
scored AS (
    SELECT base.*, ... FROM base
)
SELECT * FROM scored ...;
```

**PostgreSQL의 NULL 처리**:
- `market_cap >= 0` 조건은 NULL 값을 필터링함 (NULL은 >= 0을 만족하지 않음)
- 모든 market_cap이 NULL → WHERE market_cap >= 0 returns 0 rows
- 모든 subquery의 cardinality = 0

### 3. 학습 실패 메커니즘

**Collection Phase**:
```
⚠️  WARNING: All 10 cardinalities are 0!
  min cardinality: 0
  max cardinality: 0
  mean cardinality: 0.0
```

**Training Phase - Label Normalization**:
```python
# algorithm_examples/Mscn/source/model.py
def normalize_labels(labels, min_val, max_val):
    # labels = [0, 0, 0, ..., 0]
    # min_val = 0, max_val = 0
    return (labels - min_val) / (max_val - min_val)
    # Division by zero: (0 - 0) / (0 - 0) = 0 / 0 = NaN
```

**결과**:
```
Epoch 0, loss: nan
Epoch 1, loss: nan
...
Epoch 99, loss: nan
```

## 역설적 결과: 왜 untrained MSCN이 baseline보다 나은가?

### 가설 1: PostgreSQL Fallback이 작동함

MSCN의 Partial Fallback 전략:
1. 유효한 subquery → MSCN 모델 예측 사용
2. 무효한 subquery → PostgreSQL 기본 추정 사용

**Test Phase에서 발생한 일**:
- 대부분의 subquery가 무효(PostgreSQL Anchor의 CTE 버그 등)
- MSCN 예측 실패 → PostgreSQL fallback 사용
- **결과적으로 PostgreSQL 추정치를 그대로 사용**

**왜 baseline보다 빠른가?**:
- Baseline: 매번 PostgreSQL이 새로 추정 계산
- MSCN (fallback): 캐시된 추정치 재사용 가능
- 또는 쿼리 실행 경로 최적화 효과

### 가설 2: 빈 결과셋의 빠른 실행

모든 쿼리가 `WHERE market_cap >= 0`으로 0 rows 반환:
- PostgreSQL은 빈 결과셋을 매우 빠르게 처리
- MSCN의 (잘못된) 카디널리티 추정도 결과에 큰 영향 없음
- 실제 실행 시간이 매우 짧아서 변동성 범위 내

### 가설 3: 모델 초기화 값이 우연히 유효

MSCN 모델이 학습되지 않았지만:
- 가중치 초기화 값이 일부 쿼리에 우연히 맞음
- 또는 모델이 0에 가까운 값을 예측 → 실제 cardinality(0)와 일치

## 해결 방안

### 옵션 1: market_cap 데이터 채우기 (권장하지 않음)

**이유**: representative_momentum은 momentum 기반 전략이므로 market_cap 필터가 본질적 요구사항이 아닐 수 있음

### 옵션 2: 쿼리 수정 - market_cap 조건 제거 (권장)

```sql
-- 기존
WHERE sdi.event_date = DATE '2023-06-02' AND sdi.market_cap >= 0

-- 수정안 1: market_cap 조건 제거
WHERE sdi.event_date = DATE '2023-06-02'

-- 수정안 2: 다른 유효한 컬럼으로 필터링
WHERE sdi.event_date = DATE '2023-06-02'
  AND sdi.close_price > 0  -- close_price는 NOT NULL 컬럼
```

### 옵션 3: 워크로드 데이터 재생성

1. `pilotscope/Dataset/StockStrategy/Representative/` 폴더의 원본 쿼리 수정
2. `parse_representative_queries.py` 재실행
3. 새로운 train/test 파일 생성

```bash
cd pilotscope/Dataset/StockStrategy
# 1. Representative/momentum_queries.txt 수정 (market_cap 조건 제거)
# 2. 재파싱
python parse_representative_queries.py
# 3. 테스트
cd test_example_algorithms
python unified_test.py --algo mscn --db stock_strategy_representative_momentum
```

## 데이터 품질 체크리스트

향후 유사한 문제 방지를 위한 검증 항목:

```sql
-- 1. NULL 값 비율 확인
SELECT
    column_name,
    COUNT(*) as total,
    COUNT(column_name) as non_null,
    ROUND(100.0 * COUNT(column_name) / COUNT(*), 2) as non_null_pct
FROM stocks_daily_info, information_schema.columns
WHERE table_name = 'stocks_daily_info'
GROUP BY column_name;

-- 2. 워크로드 쿼리의 실제 cardinality 확인
SELECT COUNT(*)
FROM stocks_daily_info
WHERE event_date = DATE '2023-06-02'
  AND market_cap >= 0;
-- Expected: > 0, Actual: 0 → ⚠️ 문제!

-- 3. 대체 컬럼으로 필터링 시 cardinality
SELECT COUNT(*)
FROM stocks_daily_info
WHERE event_date = DATE '2023-06-02'
  AND close_price > 0;
-- Expected: > 0 (close_price는 NOT NULL)
```

## 교훈

1. **워크로드 생성 전 데이터 품질 검증 필수**
   - 쿼리에서 사용하는 모든 컬럼의 NULL 비율 확인
   - 필터 조건이 실제 데이터와 맞는지 검증

2. **학습 단계에서 조기 경고 필요**
   - "All cardinalities are 0" 경고를 ERROR로 승격
   - min=max인 경우 normalization 전에 실패

3. **테스트 성공 ≠ 알고리즘 동작**
   - Fallback 메커니즘이 작동하여 실제 MSCN 모델이 사용되지 않았을 가능성
   - 성능 향상이 우연한 부산물일 수 있음

## 다음 단계

1. ✅ **market_cap NULL 원인 분석 완료**
2. ⬜ **쿼리 수정 또는 데이터 재생성 결정**
3. ⬜ **수정 후 재테스트로 정상 학습 확인**
4. ⬜ **데이터 검증 자동화 추가**