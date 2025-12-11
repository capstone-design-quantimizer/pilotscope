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

**중요**: 컨테이너 내에서 실행하세요!

```bash
# 컨테이너 접속
docker-compose exec pilotscope-dev bash
conda activate pilotscope
cd test_example_algorithms

# 단일 알고리즘
python test_mscn_example.py

# unified_test.py로 테스트 (결과는 MLflow에 자동 저장)
python unified_test.py --algo baseline --db stats_tiny
python unified_test.py --algo mscn --db stats_tiny
python unified_test.py --algo lero --db stats_tiny --timeout 900

# 기존 모델 로드 (학습 없이)
python unified_test.py --algo mscn --db stats_tiny --no-training

# 파라미터 조정
python unified_test.py --algo mscn --db stats_tiny --epochs 50 --training-size 100

# MLflow UI에서 결과 확인
# 브라우저: http://localhost:54321
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
- Docker 필수

**중요**: 모든 개발과 테스트는 Docker 컨테이너 내에서 진행합니다. 의존성(sqlalchemy, torch 등)이 컨테이너 내부에만 설치되어 있습니다.

### 컨테이너 접속 방법
```bash
# 컨테이너 시작
docker-compose up -d

# 컨테이너 접속 (중요: pilotscope 사용자로 접속)
docker-compose exec -u pilotscope pilotscope-dev bash

# Conda 환경 활성화
conda activate pilotscope

# 작업 디렉토리로 이동
cd /home/pilotscope/workspace/test_example_algorithms
```

**중요**:
- 반드시 `-u pilotscope` 옵션으로 pilotscope 사용자로 접속해야 합니다
- root로 접속하면 PostgreSQL 관련 작업(특히 knob 알고리즘)에서 권한 오류 발생
- 작업 디렉토리: `/home/pilotscope/workspace`

### 코드 수정
- 로컬에서 코드 수정 (volume mount로 실시간 반영)
- 컨테이너 내에서 테스트 실행

### 외부에서 직접 실행 (docker-compose exec -T 사용)
```bash
# 올바른 방법: pilotscope 사용자로 실행 + conda activate
docker-compose exec -T -u pilotscope pilotscope-dev bash -c "source /home/pilotscope/miniconda3/etc/profile.d/conda.sh && conda activate pilotscope && cd /home/pilotscope/workspace/test_example_algorithms && python unified_test.py --algo baseline --db stats_tiny"

# 잘못된 예시들
# ❌ root 사용자로 실행 (PostgreSQL 권한 오류)
# docker-compose exec -T pilotscope-dev bash -c "..."

# ❌ conda run 사용 (환경 설정 불완전)
# docker-compose exec -T -u pilotscope pilotscope-dev bash -c "conda run -n pilotscope python ..."
```

## 디버깅

```bash
export DEBUG_EXECUTION_TIME=1
```

PilotTransData 속성: `execution_time`, `estimated_cost`, `subquery_2_card`, `physical_plan`
