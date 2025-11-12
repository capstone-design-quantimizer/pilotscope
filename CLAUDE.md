# PilotScope

AI4DB 미들웨어. 쿼리 실행 중 AI 모델 주입하여 카디널리티 추정, 플랜 최적화, 인덱스 선택, 노브 튜닝 수행.

## 문서 맵

- **USAGE_GUIDE.md**: 외부 DB/쿼리 활용 가이드
- **algorithm_examples/CLAUDE.md**: 알고리즘 공통 패턴
- **algorithm_examples/{Mscn,Lero,KnobTuning,Index}/CLAUDE.md**: 알고리즘별 가이드
- **docs/**: 상세 문서 (Docker, MLflow, 모델 관리 등)

## 아키텍처

```
알고리즘 (MSCN, Lero, Knob, Index)
    ↓
Scheduler → Handler → Event
    ↓
PostgreSQL (Anchor 패치)
```

## 핵심 컴포넌트

**PilotScheduler**: `init()` → `execute(sql)` → handler 트리거 → 결과 반환

**PresetScheduler**: 팩토리 함수 `get_*_preset_scheduler(config, enable_collection, enable_training, ...)`

**Handler**:
- `BasePushHandler`: DB 힌트 주입
- `BasePullHandler`: DB 데이터 수집

**Event**: `PretrainingModelEvent` - `scheduler.init()` 시 학습 실행

**파라미터**:
- `enable_collection`: 데이터 수집 여부
- `enable_training`: 모델 학습 여부
- `num_collection`: 수집 쿼리 수 (-1: 전체)
- `num_training`: 학습 쿼리 수 (-1: 전체)
- `num_epoch`: 학습 에포크
- `load_model_id`: 기존 모델 ID
- `dataset_name`: 데이터셋/워크로드 구분

## 빠른 시작

```bash
docker-compose up -d
docker-compose exec pilotscope-dev bash
conda activate pilotscope
cd test_example_algorithms
python test_mscn_example.py
```

## 테스트

```bash
# 단일 알고리즘
python test_mscn_example.py

# 여러 알고리즘 비교
python unified_test.py --algo baseline mscn lero --db stats_tiny --compare

# 기존 모델 로드
python unified_test.py --algo mscn --no-training --load-model mscn_20241019_103000
```

## 파일 구조

```
pilotscope/                 # 코어 미들웨어
algorithm_examples/         # 알고리즘 (MSCN, Lero, Knob, Index)
test_example_algorithms/    # 테스트 스크립트
ExampleData/                # 모델 저장
docs/                       # 상세 가이드
```

## 환경

- Python 3.8
- PostgreSQL 13.1 (Anchor 패치 필수)
- Docker 권장

## 디버깅

```bash
export DEBUG_EXECUTION_TIME=1
```

PilotTransData 속성: `execution_time`, `estimated_cost`, `subquery_2_card`, `physical_plan`
