# CLAUDE.md

이 문서는 Claude Code가 PilotScope 프로젝트 작업 시 참조하는 핵심 가이드입니다.

## 프로젝트 개요

**PilotScope**는 AI4DB(AI for Database) 알고리즘을 실제 데이터베이스에 적용하는 미들웨어입니다.

핵심 개념: 데이터베이스 엔진을 수정하지 않고, 쿼리 실행 중간에 AI 모델을 삽입하여 카디널리티 추정, 쿼리 플랜 최적화, 인덱스 선택, 노브 튜닝 등을 수행합니다.

```
쿼리 입력 → PilotScope 인터셉트 → AI 최적화 적용 → DB 실행 → 결과 수집
```

## 빠른 시작

```bash
# Docker 환경 시작
docker-compose up -d
docker-compose exec pilotscope-dev bash

# 샘플 테스트
conda activate pilotscope
cd test_example_algorithms
python test_mscn_example.py
```

## 문서 구조

### 사용자 가이드
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - 외부 DB와 쿼리를 활용한 알고리즘 테스트 가이드 ⭐

### 알고리즘 개발
- **[algorithm_examples/CLAUDE.md](algorithm_examples/CLAUDE.md)** - 알고리즘 공통 구현 패턴
- **[algorithm_examples/Mscn/CLAUDE.md](algorithm_examples/Mscn/CLAUDE.md)** - MSCN 카디널리티 추정
- **[algorithm_examples/Lero/CLAUDE.md](algorithm_examples/Lero/CLAUDE.md)** - Lero 쿼리 플랜 최적화
- **[algorithm_examples/KnobTuning/CLAUDE.md](algorithm_examples/KnobTuning/CLAUDE.md)** - DB 노브 튜닝
- **[algorithm_examples/Index/CLAUDE.md](algorithm_examples/Index/CLAUDE.md)** - 인덱스 선택

### 상세 가이드
- **[docs/](docs/)** - Docker, MLflow, 모델 관리, 운영 최적화 등 상세 문서

## 핵심 아키텍처

```
┌────────────────────────────────────┐
│  알고리즘 (MSCN, Lero, Knob, Index) │
└────────────┬───────────────────────┘
             │
┌────────────▼───────────────────────┐
│  PilotScope 미들웨어               │
│  - Scheduler: 실행 오케스트레이션  │
│  - Handler: AI 로직 주입           │
│  - Event: 학습/데이터 수집 트리거  │
└────────────┬───────────────────────┘
             │
┌────────────▼───────────────────────┐
│  PostgreSQL (Anchor 패치 버전)     │
└────────────────────────────────────┘
```

## 주요 컴포넌트

### PilotScheduler
- 쿼리 실행 오케스트레이션
- 호출 흐름: `init()` → `execute(sql)` → handler 트리거 → 결과 반환

### PresetScheduler (Factory)
- 알고리즘별 팩토리 함수 (`get_*_preset_scheduler()`)
- 주요 파라미터:
  - `enable_collection`: 학습 데이터 수집
  - `enable_training`: 모델 학습
  - `load_model_id`: 기존 모델 로드
  - `dataset_name`: 데이터셋/워크로드 구분

### Handler
- `BasePushHandler`: DB에 힌트 주입 (카디널리티, 플랜 등)
- `BasePullHandler`: DB에서 데이터 수집 (실행 시간, 플랜 등)

### Event
- `PretrainingModelEvent`: `scheduler.init()` 시 학습 실행
- 데이터 수집 및 모델 학습 로직 구현

## 환경 설정

### Docker (권장)
- Volume mount 방식: 코드 변경 즉시 반영
- PostgreSQL: `localhost:5432` (컨테이너 내부), `localhost:54323` (호스트)

### Non-Docker
```bash
pip install -e .  # 개발 모드 설치
```

**요구사항**: Python 3.8, PostgreSQL 13.1 (Anchor 패치 버전)

## 테스트 실행

```bash
# 단일 알고리즘
python test_mscn_example.py

# 여러 알고리즘 비교
python unified_test.py --algo baseline mscn lero --db stats_tiny --compare

# 기존 모델 로드
python unified_test.py --algo mscn --db production --no-training --load-model mscn_20241019_103000
```

## 파일 구조

```
pilotscope/
├── pilotscope/                 # 코어 미들웨어
├── algorithm_examples/         # 알고리즘 구현 (MSCN, Lero, Knob, Index)
├── test_example_algorithms/    # 테스트 스크립트
├── ExampleData/                # 학습된 모델
├── docs/                       # 상세 가이드
└── scripts/                    # 유틸리티 스크립트
```

## 디버깅

```bash
# 실행 시간 디버깅
export DEBUG_EXECUTION_TIME=1
python your_test.py

# PilotTransData 속성 확인
# - execution_time: 쿼리 실행 시간
# - estimated_cost: 옵티마이저 비용 추정
# - subquery_2_card: 서브쿼리별 카디널리티
# - physical_plan: 실행 플랜
```

## 중요 제약사항

- PostgreSQL 13.1 (pilotscope-postgresql branch의 Anchor 패치 버전 필수)
- Python 3.8
- Docker 환경 권장 (의존성 관리 복잡성)

## 관련 리소스

- 공식 문서: https://woodybryant.github.io/PilotScopeDoc.io/
- 논문: `paper/PilotScope.pdf`

---

**마지막 업데이트**: 2024-11
