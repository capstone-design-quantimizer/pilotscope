# Index Selection

인덱스 자동 추천. 워크로드 분석하여 최적 인덱스 집합 선택.

## 작동 방식

1. 워크로드 분석 → 인덱스 후보 생성
2. 각 인덱스 조합의 비용-효과 분석
3. 최적 인덱스 집합 선택
4. 추천된 인덱스 생성

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
