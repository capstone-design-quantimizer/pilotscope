# CTE 지원 구현 가이드

PilotScope에서 CTE(Common Table Expression) 기반 쿼리를 지원하기 위한 구현 가이드.

## 문제 요약

**현재 상황**:
- MSCN, Lero, Index 알고리즘이 CTE 쿼리를 파싱하지 못함
- Representative 워크로드 (`WITH base AS (...)`) 사용 불가
- Knob Tuning만 유일하게 작동 (쿼리 파싱 불필요)

**근본 원인**:
- PilotScope의 `QueryMetaData._parse_table()` 메서드가 `exp.Table`만 검색
- CTE는 `exp.CTE` 노드로 파싱되어 감지되지 않음
- `sqlglot` 라이브러리는 CTE를 완벽히 지원하지만 PilotScope가 활용하지 않음

## 영향받는 파일

| 알고리즘 | 파일 경로 | 클래스 | 메서드 |
|---------|----------|--------|--------|
| **MSCN** | `algorithm_examples/Mscn/source/mscn_utils.py` | `QueryMetaData` | `_parse_table()` (line 34) |
| **Lero** | `algorithm_examples/Lero/LeroPilotAdapter.py` | `QueryMetaData` | `_parse_table()` (line 62) |
| **Index** | `algorithm_examples/Index/IndexPilotAdapter.py` | - | 근본적으로 다른 문제 (후술) |

## 수정 전략

### Phase 1: MSCN 수정 (난이도: 중)

MSCN은 Collection/Training은 성공하고 Test만 실패하므로 상대적으로 수정이 쉽습니다.

#### 1.1 파일: `algorithm_examples/Mscn/source/mscn_utils.py`

**현재 코드** (line 34-40):
```python
def _parse_table(self):
    for table in self.expression.find_all(exp.Table, bfs=False):
        self.tables.append(table.name)
        if table.alias:
            self.table_alias.append(table.alias)
            self.names_to_alias[table.name] = table.alias
            self.alias_to_names[table.alias] = table.name
```

**수정된 코드**:
```python
def _parse_table(self):
    # 1. 기존 테이블 파싱
    for table in self.expression.find_all(exp.Table, bfs=False):
        self.tables.append(table.name)
        if table.alias:
            self.table_alias.append(table.alias)
            self.names_to_alias[table.name] = table.alias
            self.alias_to_names[table.alias] = table.name

    # 2. CTE 파싱 추가
    for cte in self.expression.find_all(exp.CTE, bfs=False):
        # CTE alias (e.g., "base", "scored")
        cte_alias = cte.alias

        # CTE 내부의 실제 테이블 찾기
        cte_query = cte.this  # CTE의 SELECT 쿼리
        for table in cte_query.find_all(exp.Table, bfs=False):
            # CTE 내부 테이블을 실제 테이블로 취급
            if table.name not in self.tables:
                self.tables.append(table.name)

        # CTE alias를 테이블 alias로 등록
        if cte_alias and len(self.tables) > 0:
            # CTE가 참조하는 주요 테이블을 찾아서 매핑
            # 간단한 경우: 첫 번째 테이블을 대표로 사용
            representative_table = self.tables[0]
            self.table_alias.append(cte_alias)
            self.alias_to_names[cte_alias] = representative_table
            if representative_table not in self.names_to_alias:
                self.names_to_alias[representative_table] = cte_alias
```

**설명**:
- CTE 내부의 실제 테이블(`stocks_daily_info`)을 추출
- CTE alias(`base`, `scored`)를 테이블 alias처럼 처리
- `FROM base` → `FROM stocks_daily_info` 변환 가능

#### 1.2 테스트 방법

```python
# 테스트 스크립트: test_cte_parsing.py
from algorithm_examples.Mscn.source.mscn_utils import QueryMetaData

# CTE 쿼리
cte_query = """
WITH base AS (
    SELECT ticker, close_price, volume
    FROM stocks_daily_info
    WHERE date > '2020-01-01'
)
SELECT * FROM base WHERE volume > 1000000
"""

# 파싱 테스트
qmd = QueryMetaData(cte_query)
print(f"Tables: {qmd.tables}")  # ['stocks_daily_info']
print(f"Aliases: {qmd.alias_to_names}")  # {'base': 'stocks_daily_info'}

# 서브쿼리 생성 테스트
# MSCN이 이 정보를 사용하여 카디널리티 수집
```

### Phase 2: Lero 수정 (난이도: 중)

Lero는 Collection 단계에서 바로 실패하므로 MSCN과 동일한 수정이 필요합니다.

#### 2.1 파일: `algorithm_examples/Lero/LeroPilotAdapter.py`

**현재 코드** (line 62-68):
```python
def _parse_table(self):
    for table in self.expression.find_all(exp.Table, bfs=False):
        self.tables.append(table.name)
        if table.alias:
            self.table_alias.append(table.alias)
            self.names_to_alias[table.name] = table.alias
            self.alias_to_names[table.alias] = table.name
```

**수정된 코드**:
MSCN과 동일한 패턴으로 수정 (위의 1.1 참조)

#### 2.2 추가 고려사항

Lero는 실행 계획 생성이 필요하므로:

**파일**: `algorithm_examples/Lero/EventImplement.py`

**수정 위치**: `iterative_data_collection()` 메서드 (line 75)

```python
# 현재 코드
cards_picker = CardsPickerModel(subquery_2_card.keys(), subquery_2_card.values())

# 문제: CTE 쿼리를 파싱할 때 테이블명이 없어서 실패
# 해결: QueryMetaData가 CTE를 올바르게 파싱하면 자동 해결됨
```

### Phase 3: Index Selection (난이도: 높음, 근본적 제약)

Index Selection은 **구조적으로 CTE 지원 불가능**합니다.

**이유**:
```sql
-- CTE는 쿼리 실행 중에만 존재하는 임시 결과
WITH base AS (SELECT * FROM stocks_daily_info WHERE ...)
SELECT * FROM base;

-- 인덱스는 영구적인 테이블에만 생성 가능
CREATE INDEX idx_base ON base (ticker);  -- ❌ 불가능 (base는 테이블이 아님)
CREATE INDEX idx_stocks ON stocks_daily_info (ticker);  -- ✅ 가능
```

**대안 (고급)**:
CTE를 언폴딩(unfolding)하여 실제 테이블에 인덱스 제안:

```python
# algorithm_examples/Index/IndexPilotAdapter.py에 추가

def unfold_cte_to_base_tables(query_with_cte):
    """
    CTE 쿼리를 분석하여 실제 테이블과 컬럼 추출

    Example:
        WITH base AS (SELECT ticker, close_price FROM stocks_daily_info WHERE ...)
        SELECT * FROM base WHERE ticker = 'AAPL'

        → stocks_daily_info 테이블의 ticker 컬럼에 인덱스 제안
    """
    expression = parse_one(query_with_cte)

    # CTE 정의 찾기
    cte_definitions = {}
    for cte in expression.find_all(exp.CTE, bfs=False):
        cte_alias = cte.alias
        cte_query = cte.this

        # CTE 내부 테이블 추출
        base_tables = [t.name for t in cte_query.find_all(exp.Table, bfs=False)]
        cte_definitions[cte_alias] = {
            'base_tables': base_tables,
            'query': cte_query
        }

    # 메인 쿼리에서 CTE 참조 찾기
    main_query = expression.this  # CTE 이후의 메인 SELECT

    # 조건절(WHERE)에서 사용된 컬럼 추출
    predicates = []
    for where in main_query.find_all(exp.Where, bfs=False):
        for condition in where.find_all(exp.Column, bfs=False):
            predicates.append({
                'column': condition.name,
                'table': condition.table  # CTE alias일 수 있음
            })

    # CTE alias를 실제 테이블로 변환
    index_suggestions = []
    for pred in predicates:
        if pred['table'] in cte_definitions:
            # CTE alias → base table 변환
            base_tables = cte_definitions[pred['table']]['base_tables']
            for base_table in base_tables:
                index_suggestions.append({
                    'table': base_table,
                    'column': pred['column']
                })

    return index_suggestions
```

**한계**:
- CTE가 복잡한 JOIN이나 집계를 포함하면 매핑이 어려움
- 윈도우 함수나 복잡한 계산 컬럼은 인덱스 제안 불가능
- Representative 워크로드는 이런 복잡한 케이스이므로 **여전히 제한적**

## 구현 우선순위

### 1단계: MSCN 수정 (2-3시간)
- ✅ 영향 범위 작음 (1개 파일, 1개 메서드)
- ✅ 테스트가 명확함 (Test 단계 성공 여부)
- ✅ 즉시 효과 확인 가능

### 2단계: Lero 수정 (2-3시간)
- ✅ MSCN과 동일한 패턴
- ✅ Collection 단계부터 성공 필요

### 3단계: Index 언폴딩 (선택, 8-10시간)
- ⚠️ 복잡도 높음
- ⚠️ Representative 워크로드에는 여전히 한계
- ⚠️ ROI(투자 대비 효과) 낮음

## 테스트 체크리스트

### MSCN 테스트
```bash
# 1. 기존 워크로드 (회귀 테스트)
python unified_test.py --algo mscn --db stock_strategy --workload momentum_investing

# 2. CTE 워크로드 (신규 기능)
python unified_test.py --algo mscn --db stock_strategy --workload representative_momentum

# 3. 검증 포인트
# - Collection: 80개 쿼리 수집 성공
# - Training: Loss가 수렴 (nan 아님)
# - Test: 20개 쿼리 모두 성공 (파싱 에러 없음)
```

### Lero 테스트
```bash
# 1. 기존 워크로드
python unified_test.py --algo lero --db stock_strategy --workload momentum_investing --timeout 900

# 2. CTE 워크로드
python unified_test.py --algo lero --db stock_strategy --workload representative_momentum --timeout 900

# 3. 검증 포인트
# - Collection: 100개 쿼리 수집 성공 (첫 쿼리부터)
# - Training: 모델 학습 완료
# - Test: 쿼리 실행 성공
```

## 예상 문제 및 해결

### 문제 1: 중첩된 CTE
```sql
WITH base AS (...),
     scored AS (SELECT * FROM base),  -- CTE가 다른 CTE 참조
     filtered AS (SELECT * FROM scored)
SELECT * FROM filtered;
```

**해결**: CTE 간 참조를 재귀적으로 추적

```python
def resolve_cte_chain(cte_alias, cte_definitions):
    """CTE 체인을 따라가며 최종 base table 찾기"""
    if cte_alias not in cte_definitions:
        return [cte_alias]  # 실제 테이블

    base_tables = []
    cte_query = cte_definitions[cte_alias]['query']
    for table in cte_query.find_all(exp.Table, bfs=False):
        # 재귀적으로 CTE 체인 따라가기
        base_tables.extend(resolve_cte_chain(table.name, cte_definitions))

    return base_tables
```

### 문제 2: CTE와 실제 테이블 혼합
```sql
WITH base AS (SELECT * FROM stocks_daily_info)
SELECT b.*, f.value
FROM base b
JOIN financials_quarterly f ON b.ticker = f.ticker;
```

**해결**: 이미 위의 코드로 처리됨 (모든 테이블 수집)

### 문제 3: 서브쿼리 생성 시 CTE 유지
MSCN이 카디널리티 수집을 위해 서브쿼리를 생성할 때 CTE 구조 유지 필요.

**파일**: `pilotscope/PilotDataInteractor.py` (subquery 생성 로직)

**현재 문제**: CTE가 제거되고 테이블만 남음

**해결**:
```python
def generate_cardinality_query(query_metadata):
    # CTE가 있는 경우 원본 CTE 구조 유지
    if query_metadata.has_cte:
        # CTE 정의 부분 추출
        cte_part = extract_cte_definition(query_metadata.raw)
        # 메인 쿼리 부분을 COUNT로 변환
        main_query = f"SELECT COUNT(*) FROM {query_metadata.tables[0]}"
        return f"{cte_part}\n{main_query}"
    else:
        # 기존 로직
        return f"SELECT COUNT(*) FROM {query_metadata.tables[0]}"
```

## 구현 후 예상 효과

| 알고리즘 | 수정 전 | 수정 후 |
|---------|---------|---------|
| **MSCN** | Test 20/20 실패 | Test 20/20 성공 (예상) |
| **Lero** | Collection 즉시 실패 | Collection 성공 → Training/Test 가능 |
| **Index** | Collection 실패 | 여전히 제한적 (언폴딩 구현 시 부분 개선) |
| **Knob** | 완전 호환 ✅ | 변화 없음 (이미 완벽) |

## 장기 계획

### Option 1: PilotScope 업스트림 기여
수정 사항을 PilotScope 원본 프로젝트에 PR (Pull Request) 제출
- 다른 사용자들도 혜택
- 코드 리뷰 및 개선
- 유지보수 부담 감소

### Option 2: 포크 유지
자체 포크 유지하며 독자적 기능 추가
- CTE 지원 외에도 추가 기능 개발 가능
- 빠른 실험 및 배포
- 업스트림 변경 사항 주기적 머지 필요

## 참고 자료

**sqlglot 문서**:
- CTE 파싱: https://sqlglot.com/sqlglot.html#CTE
- Expression 노드: https://sqlglot.com/sqlglot/expressions.html

**PilotScope 이슈**:
- CTE 지원 요청 이슈를 생성하여 커뮤니티 의견 수렴 고려

**관련 논문**:
- MSCN 논문: CTE에 대한 제약 언급 없음 (구현 문제)
- Lero 논문: 복잡한 쿼리 지원 가능하다고 명시

## 다음 단계

1. **결정**: MSCN/Lero CTE 지원을 구현할지 결정
2. **브랜치**: `feature/cte-support` 브랜치 생성
3. **구현**: MSCN부터 단계적 수정
4. **테스트**: 기존 워크로드 회귀 테스트 + CTE 워크로드 신규 테스트
5. **문서화**: 호환성 매트릭스 업데이트 (❌ → ✅)

---

**작성일**: 2025-11-19
**작성자**: Claude
**버전**: 1.0