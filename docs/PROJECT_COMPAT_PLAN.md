# StockStrategy + PilotScope Compatibility Plan

목표: StockStrategy 단일 DB(`stock_strategy`)에 여러 워크로드(`*_investing`, `representative_*`)를 적용하고, PilotScope 기반 AI4DB 알고리즘(MSCN, Lero, Knob, Index)을 MLflow로 비교·추적한다. 대표 워크로드(representative_*)의 CTE 기반 쿼리가 PostgreSQL Anchor와 충돌하는 문제를 해소하거나 우회한다.

## 아키텍처 개요
- 컨테이너: `docker-compose.yml`의 `pilotscope-dev`가 커스텀 PostgreSQL Anchor(13.1 포크) + Conda + MLflow(54321) 포함.
- 코어: 앵커(`pilotscope/Anchor/*`), DB 컨트롤러, 데이터 인터랙터, 스케줄러(`pilotscope/PilotScheduler.py`), 데이터셋 API(`pilotscope/Dataset/*`).
- 알고리즘 예제: `algorithm_examples/` (MSCN, Lero, Knob, Index). MSCN 카드 주입 핸들러는 `algorithm_examples/Mscn/MscnParadigmCardAnchorHandler.py`.
- 실행 허브: `test_example_algorithms/unified_test.py`가 CLI/JSON 설정을 받아 스케줄러 생성, MLflow 로깅, 워크로드 로딩(`algorithm_examples/utils.py`).
- 데이터셋: `pilotscope/Dataset/StockStrategyDataset.py`가 모든 워크로드에 동일한 DB 이름(`stock_strategy`)을 사용하고 쿼리 파일만 변경. 대표 워크로드 쿼리는 `pilotscope/Dataset/StockStrategy/stock_strategy_representative_*_*.txt`.

## 현황 문제(대표 워크로드 × PostgreSQL Anchor)
- CTE 기반 쿼리에서 Anchor가 중첩 CTE를 잘못 파싱해 `SELECT COUNT(*) FROM ;` 등의 무효 서브쿼리를 생성 (`docs/WORKLOAD_COMPATIBILITY.md` 참조).
- MSCN: `MscnParadigmCardAnchorHandler.py`가 무효 서브쿼리를 건너뛰고 유효분만 예측(Partial Fallback). 정확도 저하 가능.
- Lero: `LeroPilotAdapter.py`의 `QueryMetaData`가 CTE 미지원 → Collection 1쿼리 차에서 파싱 오류.
- Index: CTE 결과셋에 인덱스 생성 불가 → HypoPG 구문 오류.
- Knob: 쿼리 파싱 의존도가 낮아 모든 워크로드 정상 동작.

## 실행 계획
1) 재현·계측
   - `python unified_test.py --algo <algo> --db stock_strategy --workload representative_momentum --timeout 900 --use-mlflow` 등으로 알고리즘/워크로드별 로그 확보.
   - Anchor 반환 `subquery_2_card` 덤프하여 무효 패턴(빈 FROM, correlated placeholder 등) 리스트업.
2) MSCN 안정화(빠른 승수)
   - `MscnParadigmCardAnchorHandler.py` 필터링 로직 기준으로 테스트 케이스 보강(유효/무효 혼재).
   - `algorithm_examples/Mscn/source/mscn_utils.py` CTE 파싱 추가 변경 필요 시 보완.
3) Lero CTE 지원
   - `LeroPilotAdapter.py`의 `QueryMetaData._parse_table`과 alias 처리에 MSCN과 동일한 CTE 지원·밸리데이션 적용.
   - 무효 서브쿼리 필터링을 MSCN처럼 수행, 유효분이 없을 때 PostgreSQL 추정치로 fallback 후 Collection 지속.
4) Index Selection 우회/부분 지원
   - CTE 워크로드에서는 HypoPG 인덱스 생성을 스킵하거나 base table unfold 가능 시에만 시도. 실패 시 로그 후 계속 진행.
   - 호환성 표(`docs/WORKLOAD_COMPATIBILITY.md`)를 실제 동작 기준으로 업데이트.
5) Anchor 근본 수정(장기)
   - Anchor C 소스(커스텀 PostgreSQL 빌드)에서 CTE 추출 로직 점검: CTE alias 해석, JOIN 조건 분해, correlated placeholder 제거.
   - 수정 후 Docker 이미지 재빌드 → `docker-compose up --build`로 검증. 대안: Anchor 없이 EXPLAIN 기반 수집 백업 경로 연구.
6) 검증·관측
   - 단위: CTE 쿼리 파싱 유닛 테스트 추가(MSCN/Lero QueryMetaData).
   - 통합: `unified_test.py` 워크로드별 스모크 테스트, MLflow 런 상태(FINISHED/FAILED) 확인.
   - 메트릭: 성공 쿼리 수, DB execution time, fallback 비율 등을 로그/MLflow에 기록.

## 주요 파일
- 실행 허브: `test_example_algorithms/unified_test.py`
- 데이터셋 매핑: `pilotscope/Dataset/StockStrategyDataset.py`, 쿼리 파일 `pilotscope/Dataset/StockStrategy/stock_strategy_representative_*_*.txt`
- MSCN 카드 주입: `algorithm_examples/Mscn/MscnParadigmCardAnchorHandler.py`
- Lero 파싱: `algorithm_examples/Lero/LeroPilotAdapter.py`
- 호환성 문서: `docs/WORKLOAD_COMPATIBILITY.md`, CTE 가이드 `docs/CTE_SUPPORT_IMPLEMENTATION.md`
- 환경: `docker-compose.yml`

## 리스크/오픈 이슈
- Anchor C 코드 위치/빌드 파이프라인 확인 필요(레포 내 미포함으로 추정).
- representative_* 쿼리는 window 함수·중첩 CTE가 많아 sqlglot/Anchor 모두 스트레스가 큼 → 추가 예외 대비 필요.
- MLflow 로그에 fallback 비율 등 커스텀 메트릭을 넣어야 재현·비교 용이.

## 다음 행동 제안
1. Lero CTE 파싱 지원부터 착수 → Collection 통과 여부 확인.
2. Anchor C 소스 경로/빌드 방식 확인(공유 필요 시 요청).
3. 호환성 표/문서를 실제 실행 결과로 갱신.
