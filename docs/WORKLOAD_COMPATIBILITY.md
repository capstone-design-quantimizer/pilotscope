# Workload-Algorithm Compatibility Guide

알고리즘과 워크로드 간 호환성 가이드. 각 알고리즘의 요구사항과 워크로드 특성에 따른 조합 선택 방법.

## 호환성 매트릭스

| Workload | MSCN | Lero | Knob Tuning | Index Selection |
|----------|------|------|-------------|-----------------|
| **value_investing** | ✅ | ✅ | ✅ | ✅ |
| **momentum_investing** | ✅ | ✅ | ✅ | ✅ |
| **ml_hybrid** | ✅ | ✅ | ✅ | ✅ |
| **representative_momentum** | ⚠️ | ❌ | ✅ | ❌ |
| **representative_smallcap_turnover** | ⚠️ | ❌ | ✅ | ❌ |
| **representative_value_quality** | ⚠️ | ❌ | ✅ | ❌ |

**범례**:
- ✅ 완전 호환
- ⚠️ 부분 호환 (PostgreSQL Anchor 버그로 일부 subquery 무효화)
- ❌ 비호환

## PostgreSQL Anchor란?

### 개요

**PostgreSQL Anchor**는 PilotScope 연구팀이 개발한 **PostgreSQL 13.1의 커스텀 빌드(포크)**입니다. 공식 PostgreSQL 플러그인이 아니며, PilotScope를 위해 PostgreSQL 소스코드를 직접 수정한 버전입니다.

**목적**: AI4DB 알고리즘(MSCN, Lero 등)이 PostgreSQL 내부 정보(subquery, cardinality 등)에 접근하고 힌트를 주입할 수 있도록 Python ↔ PostgreSQL 통신 메커니즘 제공

### 작동 방식

**1. Python → PostgreSQL: 특수 JSON 코멘트 삽입**

PilotScope가 SQL 쿼리에 특별한 JSON 코멘트를 삽입:

```sql
/*pilotscope {
  "anchor": {
    "SUBQUERY_CARD_PULL_ANCHOR": {"enable": true}
  },
  "port": 56755,
  "url": "localhost"
} pilotscope*/
SELECT ticker, close_price
FROM stocks_daily_info
WHERE date = '2023-01-01';
```

**2. PostgreSQL Anchor → Python: HTTP POST로 데이터 전송**

수정된 PostgreSQL이:
1. 코멘트를 파싱하여 Anchor 활성화
2. 쿼리 실행 중 subquery들을 추출
3. HTTP POST로 Python 백엔드에 카디널리티 정보 전송

```json
POST http://localhost:56755
{
  "tid": "139998697207552",
  "subquery_2_card": {
    "SELECT COUNT(*) FROM stocks_daily_info WHERE date = '2023-01-01'": "100",
    "SELECT COUNT(*) FROM stocks_daily_info WHERE close_price > 50000": "50"
  }
}
```

**3. Python 처리 및 힌트 주입**

Python이 받은 데이터를:
1. MSCN 모델로 카디널리티 예측
2. 예측 결과를 다시 PostgreSQL에 힌트로 주입
3. PostgreSQL이 힌트를 반영하여 쿼리 실행 계획 수정

**4. PostgreSQL 로그 메시지**

```
INFO: There is no pilotscope comment.
INFO: Goto standard_planner!
```

→ **수정된 PostgreSQL이 출력하는 로그**. PilotScope 코멘트가 없으면 일반 플래너를 사용한다는 의미.

### 알려진 버그

PostgreSQL Anchor는 **연구용 프로토타입**이므로 다음과 같은 버그가 존재합니다:

#### 1. 중첩 CTE 처리 실패

**증상**: 두 번째 CTE가 첫 번째 CTE를 참조하면 빈 SQL 생성

```sql
WITH base AS (
    SELECT * FROM stocks_daily_info WHERE market_cap >= 0
),
scored AS (
    SELECT * FROM base  -- ← base는 CTE 이름
)
SELECT * FROM scored;
```

**Anchor가 생성한 subquery**:
- ✅ `SELECT COUNT(*) FROM stocks_daily_info WHERE market_cap >= 0`
- ❌ `SELECT COUNT(*) FROM ;`  ← **테이블명 누락!**

#### 2. AND JOIN 조건 분리 실패

**증상**: `ON a=b AND c=d` 형태의 JOIN 조건을 분리할 때 테이블명 누락

```sql
FROM stocks_daily_info sdi
LEFT JOIN fundamentals_daily fd
    ON fd.ticker = sdi.ticker AND fd.event_date = sdi.event_date
```

**Anchor가 생성한 subquery**:
```sql
SELECT COUNT(*) FROM fundamentals_daily fd
LEFT JOIN stocks_daily_info sdi ON sdi.event_date = fd.event_date
LEFT JOIN  ON sdi.ticker = fd.ticker  -- ❌ 테이블명 누락!
```

**에러**: `syntax error at or near "ON"`

#### 3. Correlated Subquery Placeholder

**증상**: Correlated subquery를 처리할 때 의미 없는 placeholder 생성

```sql
/* sdi.ticker */  -- ← correlated subquery placeholder
```

이런 placeholder는 실행 불가능하므로 필터링해야 함.

### 왜 버그가 많은가?

**1. 연구용 프로토타입**
- PilotScope 논문을 위한 PoC(Proof of Concept) 수준
- 프로덕션 품질 보장 안 됨
- 완전한 테스트 커버리지 불가능

**2. C 코드 복잡성**
- PostgreSQL Query Planner/Optimizer 내부를 수정
- Plan tree를 탐색하며 subquery 추출
- SQL 파싱 및 재구성 로직이 매우 복잡

**3. 최신 SQL 기능 지원 부족**
- CTE는 PostgreSQL 8.4(2009)에 추가된 기능
- Anchor 개발 시 CTE edge case 테스트 부족
- 중첩 CTE, window function 등의 복잡한 조합 미지원

**4. 리소스 제약**
- 학술 연구 프로젝트 → 제한된 개발 인력
- PostgreSQL 포크 유지보수 어려움
- 새로운 PostgreSQL 버전 추적 불가능

### 대응 방법

PilotScope는 Anchor 버그를 완전히 해결하는 대신 **Partial Fallback** 전략 사용:

1. **정상 subquery**: MSCN/Lero 예측 사용
2. **무효 subquery**: PostgreSQL 기본 추정치 사용 (fallback)
3. **전체 쿼리**: 성공적으로 실행 → 모든 쿼리가 평가에 포함

**결과**:
- 완벽하지 않지만 실용적
- CTE 워크로드도 부분적으로 활용 가능
- 기본 워크로드(`value_investing` 등)는 버그 없음

## 워크로드 특성

### 직접 테이블 쿼리 워크로드

**해당 워크로드**: `value_investing`, `momentum_investing`, `ml_hybrid`

**쿼리 구조**:
```sql
SELECT ticker, close_price, volume
FROM stocks_daily_info
WHERE date BETWEEN '2020-01-01' AND '2020-12-31'
  AND volume > 1000000
ORDER BY volume DESC;
```

**특징**:
- 실제 테이블(`stocks_daily_info`)에 직접 쿼리
- 모든 알고리즘과 호환
- Index 선택 알고리즘이 테이블 컬럼에 인덱스 생성 가능

### CTE 기반 워크로드

**해당 워크로드**: `representative_momentum`, `representative_smallcap_turnover`, `representative_value_quality`

**쿼리 구조**:
```sql
WITH base AS (
    SELECT ticker, date, close_price, volume
    FROM stocks_daily_info
    WHERE date BETWEEN '2020-01-01' AND '2020-12-31'
),
scored AS (
    SELECT base.*,
           LAG(close_price, 20) OVER (PARTITION BY ticker ORDER BY date) AS price_20d_ago,
           close_price / LAG(close_price, 20) OVER (PARTITION BY ticker ORDER BY date) - 1 AS momentum_score
    FROM base
)
SELECT * FROM scored WHERE momentum_score > 0.1;
```

**특징**:
- CTE(Common Table Expression) 사용 (`WITH ... AS`)
- 복잡한 집계 및 윈도우 함수 포함
- `base`, `scored` 등은 임시 결과 집합(테이블 아님)
- **Knob Tuning**: 완전 호환 ✅
- **MSCN**: 부분 호환 ⚠️ (CTE 파싱 성공, PostgreSQL Anchor 버그로 일부 subquery 무효화)
- **Lero**: 비호환 ❌ (CTE 파싱 불가, Collection 단계 실패)
- **Index Selection**: 비호환 ❌ (CTE에 인덱스 생성 불가)

## 알고리즘 요구사항

### MSCN (Cardinality Estimation) 알고리즘

**요구사항**:
- SQL 쿼리 파싱하여 테이블 및 조건 추출 (`sqlglot` 라이브러리)
- 서브쿼리 형태로 카디널리티 수집 (PostgreSQL Anchor 확장)
- 쿼리 실행 시 테이블명 추출 필요

**CTE 지원 상태**: ⚠️ **부분 호환** (2024-11-19 업데이트)

MSCN의 CTE 파싱은 성공하지만, PostgreSQL Anchor 확장의 버그로 일부 subquery가 무효화됩니다.

#### PostgreSQL Anchor 버그 상세

**발생 원인**: PostgreSQL Anchor C 확장이 중첩 CTE에서 subquery 추출 시 잘못된 SQL 생성

**예시 CTE 쿼리**:
```sql
WITH base AS (
    SELECT ticker FROM stocks_daily_info WHERE market_cap >= 0
),
scored AS (
    SELECT base.*, LAG(ticker) OVER (...) AS prev_ticker
    FROM base  -- ← 첫 번째 CTE를 참조
)
SELECT * FROM scored LIMIT 100;
```

**PostgreSQL Anchor가 추출한 subquery**:
1. ✅ `/* (stocks_daily_info sdi) */ SELECT COUNT(*) FROM stocks_daily_info WHERE market_cap >= 0;`
2. ❌ `/* () */ SELECT COUNT(*) FROM ;`  ← **버그: 잘못된 SQL**

**왜 발생하는가**:
- 첫 번째 CTE (`base`): 실제 테이블 `stocks_daily_info` 참조 → ✅ 정상 추출
- 두 번째 CTE (`scored`): 첫 번째 CTE `base` 참조 → ❌ PostgreSQL Anchor가 `base`를 해석하지 못하고 빈 SQL 생성

#### 부분 채택 (Partial Fallback) 방식

**해결 방법**: 잘못된 subquery만 필터링하고 정상 subquery는 MSCN 예측 사용

**필터링 규칙**:
```python
# 무효화되는 subquery 패턴
- "SELECT COUNT(*) FROM ;" (빈 FROM 절)
- "/* sdi.ticker */" (correlated subquery placeholder)
```

**결과**:
```
MSCN estimates OK (1 valid, 1 skipped)
✅ Successful: 20/20
```

- **정상 subquery (1개)**: MSCN 예측 사용
- **무효 subquery (1개)**: PostgreSQL estimate 유지 (fallback)
- **전체 쿼리**: 성공적으로 실행 → 학습/평가에 반영됨!

**중요**: 전체 쿼리를 skip하지 않고 부분만 fallback하므로 **모든 쿼리가 평가에 포함**됩니다.

**호환 워크로드**:
- ✅ 완전 호환: `value_investing`, `momentum_investing`, `ml_hybrid`
- ⚠️ 부분 호환: `representative_*` (CTE 워크로드)

### Lero (Learned Optimizer) 알고리즘

**요구사항**:
- 쿼리 실행 계획 생성 및 비용 추정
- 여러 실행 계획 비교
- SQL 쿼리 파싱하여 테이블 및 서브쿼리 추출 (`sqlglot` 라이브러리 사용)

**제약사항**:
```sql
-- ❌ 불가능: CTE를 파싱하지 못해 테이블명 추출 실패
WITH base AS (SELECT * FROM stocks_daily_info WHERE ...)
SELECT * FROM base;

-- ✅ 가능: 직접 테이블 쿼리
SELECT * FROM stocks_daily_info WHERE ...;
```

**에러 예시**:
```
sqlglot.errors.ParseError: Expected table name but got None. Line 1, Col: 30.
/* () */ SELECT COUNT(*) FROM ;
```

**호환 워크로드**: `value_investing`, `momentum_investing`, `ml_hybrid`

**테스트 결과**:
- **Collection 단계: 실패** (첫 번째 쿼리부터 파싱 에러)
- MSCN보다 더 빨리 실패 (Collection 초반에 즉시 중단)

### Index Selection 알고리즘

**요구사항**:
- 실제 데이터베이스 테이블 필요
- PostgreSQL의 `hypopg` 확장 사용하여 가상 인덱스 생성
- `CREATE INDEX` 구문이 실제 테이블에 적용되어야 함

**제약사항**:
```sql
-- ❌ 불가능: CTE는 임시 결과이므로 인덱스 생성 불가
CREATE INDEX ON base (ticker, date);

-- ✅ 가능: 실제 테이블에 인덱스 생성 가능
CREATE INDEX ON stocks_daily_info (ticker, date);
```

**에러 예시**:
```
psycopg2.errors.SyntaxError: syntax error at or near "*"
LINE 1: select * from hypopg_create_index( 'create index on base (*)...
```

**호환 워크로드**: `value_investing`, `momentum_investing`, `ml_hybrid`

### Knob Tuning 알고리즘

**요구사항**:
- 쿼리 실행 시간 측정
- PostgreSQL 설정 파라미터 조정

**제약사항**: 없음 (모든 워크로드 지원)

**호환 워크로드**: 모든 워크로드

### MSCN / Lero 알고리즘

**요구사항**:
- 쿼리 실행 계획 및 카디널리티 정보
- 통계 데이터 수집

**제약사항**: 없음 (모든 워크로드 지원)

**호환 워크로드**: 모든 워크로드

## 사용 예시

### MSCN 알고리즘 - 호환 워크로드

```bash
# ✅ 성공: 직접 테이블 쿼리 워크로드
python unified_test.py --algo mscn --db stock_strategy --workload momentum_investing
```

### MSCN 알고리즘 - 부분 호환 워크로드

```bash
# ⚠️ 부분 성공: CTE 기반 워크로드 (PostgreSQL Anchor 버그로 일부 subquery 무효화)
python unified_test.py --algo mscn --db stock_strategy --workload representative_momentum
# Collection: 성공, Training: 성공, Test: 성공 (20/20 쿼리, 일부 subquery는 PostgreSQL estimate 사용)
# 출력: "MSCN estimates OK (1 valid, 1 skipped)" - 부분 채택 방식으로 모든 쿼리 성공
```

### Lero 알고리즘 - 호환 워크로드

```bash
# ✅ 성공: 직접 테이블 쿼리 워크로드
python unified_test.py --algo lero --db stock_strategy --workload momentum_investing --timeout 900
```

### Lero 알고리즘 - 비호환 워크로드

```bash
# ❌ 실패: CTE 기반 워크로드
python unified_test.py --algo lero --db stock_strategy --workload representative_momentum --timeout 900
# Collection: 실패 (첫 번째 쿼리부터 파싱 에러)
# 에러: Expected table name but got None (CTE를 파싱하지 못함)
```

### Index 알고리즘 - 호환 워크로드

```bash
# ✅ 성공: 직접 테이블 쿼리 워크로드
python unified_test.py --algo index --db stock_strategy --workload momentum_investing
```

### Index 알고리즘 - 비호환 워크로드

```bash
# ❌ 실패: CTE 기반 워크로드
python unified_test.py --algo index --db stock_strategy --workload representative_momentum
# 에러: syntax error at or near "*" (CTE 'base'에 인덱스 생성 시도)
```

### Knob Tuning - 모든 워크로드 지원

```bash
# ✅ 성공: 직접 테이블 쿼리
python unified_test.py --algo knob --db stock_strategy --workload momentum_investing

# ✅ 성공: CTE 기반 쿼리
python unified_test.py --algo knob --db stock_strategy --workload representative_momentum
```

## 워크로드 선택 가이드

### Index Selection, MSCN, 또는 Lero 실험 시

**권장 워크로드** (완전 호환):
- `value_investing`: 가치 투자 전략 (PER, PBR 기반)
- `momentum_investing`: 모멘텀 투자 전략 (가격 추세 기반)
- `ml_hybrid`: ML 기반 하이브리드 전략

**부분 호환 워크로드** (MSCN만):
- `representative_*`: MSCN은 부분 호환 (PostgreSQL Anchor 버그로 일부 subquery 무효화)
  - 주의: 일부 성능 저하 가능 (무효 subquery는 PostgreSQL estimate 사용)

**비권장 워크로드**:
- `representative_*`: Lero/Index Selection은 완전 비호환

### 복잡한 분석 쿼리 테스트 시

**권장 워크로드**:
- `representative_momentum`: 윈도우 함수 포함 모멘텀 분석
- `representative_smallcap_turnover`: 소형주 거래량 분석
- `representative_value_quality`: 가치/품질 복합 분석

**지원 알고리즘**:
- ✅ **Knob Tuning**: 완전 호환 (쿼리 실행만 필요, 파싱 불필요)
- ⚠️ **MSCN**: 부분 호환 (CTE 파싱 성공, PostgreSQL Anchor 버그로 일부 subquery 무효화)
- ❌ **Lero**: 비호환 (CTE 파싱 불가, Collection 즉시 실패)
- ❌ **Index Selection**: 비호환 (CTE에 인덱스 생성 불가)

## 문제 해결

### MSCN 알고리즘 CTE 워크로드 부분 호환 (2024-11-19 업데이트)

**현재 상태**: ⚠️ **부분 호환** (PostgreSQL Anchor 버그로 일부 subquery 무효화)

**증상** (내부 로그):
```
MSCN estimates OK (1 valid, 1 skipped)
✅ Successful: 20/20
```

**근본 원인**: PostgreSQL Anchor C 확장의 버그
- MSCN 자체는 CTE를 정상적으로 파싱 (`mscn_utils.py`에서 CTE 지원 구현됨)
- PostgreSQL Anchor가 중첩 CTE에서 subquery 추출 시 잘못된 SQL 생성
- 예: `/* () */ SELECT COUNT(*) FROM ;` (빈 FROM 절)

**해결 방법**: 부분 채택 (Partial Fallback)
1. 정상 subquery → MSCN 예측 사용
2. 무효 subquery → PostgreSQL estimate 유지
3. 전체 쿼리는 성공적으로 실행 → **모든 쿼리가 평가에 포함**

**구현 위치**: `algorithm_examples/Mscn/MscnParadigmCardAnchorHandler.py`

**성능 영향**:
- 일부 subquery는 PostgreSQL estimate 사용 (MSCN 예측 미사용)
- 하지만 주요 테이블 join은 MSCN 예측 활용 가능
- 전체 쿼리는 정상 실행되므로 성능 비교 가능

**완전 해결 방법**: PostgreSQL Anchor C 코드 수정 (고급, 미구현)

### Lero 알고리즘이 CTE 워크로드에서 실패

**증상**:
```
sqlglot.errors.ParseError: Expected table name but got None. Line 1, Col: 30.
/* () */ SELECT COUNT(*) FROM ;
```

**원인**: Lero가 `sqlglot` 라이브러리로 CTE를 파싱하지 못해 테이블명 추출 실패

**해결책**: 직접 테이블 쿼리 워크로드(`*_investing`, `ml_hybrid`) 사용

**상세 설명**:
- Collection 단계 초반에 즉시 실패 (첫 번째 쿼리부터 파싱 불가)
- MSCN보다 더 빨리 실패 (MSCN은 Collection은 성공하지만 Lero는 Collection 자체가 실패)

### Index 알고리즘이 CTE 워크로드에서 실패

**증상**:
```
psycopg2.errors.SyntaxError: syntax error at or near "*"
LINE 1: select * from hypopg_create_index( 'create index on base (*)...
```

**원인**: CTE `base`는 임시 결과 집합이므로 인덱스 생성 불가

**해결책**: 직접 테이블 쿼리 워크로드(`*_investing`, `ml_hybrid`) 사용

### 모든 알고리즘이 특정 워크로드에서 실패

**확인 사항**:
1. 쿼리 파일 라인 엔딩: LF(Unix) 사용 (CRLF 금지)
2. 빈 줄 필터링: 쿼리 파싱 시 empty line 제거
3. PostgreSQL 권한: `pilotscope` 사용자로 실행

**디버깅 명령**:
```bash
# 쿼리 파일 라인 엔딩 확인
file pilotscope/Dataset/StockStrategy/stock_strategy_*_train.txt

# 빈 줄 확인
cat -A pilotscope/Dataset/StockStrategy/stock_strategy_*_train.txt | grep '^$'
```

## 참고

- **USAGE_GUIDE.md**: 외부 DB/워크로드 추가 방법
- **algorithm_examples/Index/CLAUDE.md**: Index 알고리즘 상세 가이드
- **pilotscope/Dataset/StockStrategy/README.md**: 워크로드 생성 방법