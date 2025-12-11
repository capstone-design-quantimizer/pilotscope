# PilotScope Experiment API

**목적**: MLflow 실험 결과를 PostgreSQL 테이블로 저장하고, FastAPI로 조회/관리 제공 (데모용 최소 기능)

**핵심 원칙**:
- MLflow는 그대로 사용 (기존 워크플로우 유지)
- 수동 동기화로 MLflow → PostgreSQL 복사
- 조회 API만 제공 (실행 트리거 없음)
- 이름 변경 등 간단한 관리 기능만 추가

---

## 1. 아키텍처

```
unified_test.py 실행
    ↓
MLflow에 자동 저장 (기존 방식)
    ↓
수동 동기화 스크립트 실행 (scripts/sync_mlflow_to_db.py)
    ↓
PostgreSQL experiment_logs 테이블
    ↓
FastAPI 조회 (GET /api/experiments)
```

**동기화 타이밍**:
- 개발 중: 테스트 후 수동 실행
- 프로덕션: cron 또는 systemd timer로 주기적 실행 (예: 1분마다)

---

## 2. 데이터베이스 스키마

### 테이블: `experiment_logs`

**위치**: 내장 PostgreSQL, DB 이름 `PilotScopeUserData` (PilotConfig.user_data_db_name 기본값)

```sql
CREATE TABLE IF NOT EXISTS experiment_logs (
    id SERIAL PRIMARY KEY,

    -- 사용자 관리 필드
    experiment_name VARCHAR(255),          -- 사용자 지정 이름 (NULL 가능, 기본: run_name)

    -- MLflow 메타데이터
    run_id VARCHAR(255) UNIQUE NOT NULL,  -- MLflow run ID
    run_name VARCHAR(255),                 -- MLflow run 이름 (algo_dataset_timestamp)
    mlflow_experiment VARCHAR(100),        -- MLflow experiment 이름 (예: "pilotscope")

    -- 실험 정보 (raw 값)
    algorithm VARCHAR(50) NOT NULL,        -- mscn, lero, knob, index, baseline
    dataset VARCHAR(100) NOT NULL,         -- stats_tiny, imdb, etc.
    workload VARCHAR(100),                 -- default, custom, etc.
    model_id VARCHAR(255),                 -- 생성된 모델 ID (있는 경우)

    -- UI 표시용 이름 (동기화 시 자동 매핑)
    algorithm_display VARCHAR(100),        -- MSCN (카디널리티), Lero (학습 옵티마이저)
    dataset_display VARCHAR(100),          -- 통계 DB (Small), IMDB
    workload_display VARCHAR(100),         -- 기본, 사용자 정의

    -- 실행 결과
    status VARCHAR(20) NOT NULL,           -- FINISHED, FAILED, RUNNING
    execution_time FLOAT,                  -- 총 실행 시간 (초) - test_total_time
    average_time FLOAT,                    -- 평균 쿼리 시간 (초) - test_average_time

    -- 상세 데이터
    metrics JSONB,                         -- 모든 메트릭 (test_*, train_* 등)
    parameters JSONB,                      -- 하이퍼파라미터
    best_config JSONB,                     -- knob/index 추천 설정 (있는 경우)

    -- 타임스탬프
    started_at TIMESTAMP,                  -- 실험 시작 시간
    completed_at TIMESTAMP,                -- 실험 완료 시간
    synced_at TIMESTAMP DEFAULT NOW(),     -- DB 동기화 시간

    -- 인덱스
    CONSTRAINT experiment_logs_run_id_key UNIQUE (run_id)
);

-- 성능 인덱스
CREATE INDEX IF NOT EXISTS idx_experiment_logs_algorithm ON experiment_logs(algorithm);
CREATE INDEX IF NOT EXISTS idx_experiment_logs_dataset ON experiment_logs(dataset);
CREATE INDEX IF NOT EXISTS idx_experiment_logs_started_at ON experiment_logs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_experiment_logs_status ON experiment_logs(status);
```

### 필드 설명

| 필드 | 타입 | 설명 | 예시 |
|------|------|------|------|
| `experiment_name` | VARCHAR | 사용자 지정 이름 (수정 가능) | "통계 DB 카디널리티 개선" |
| `run_id` | VARCHAR | MLflow run ID (고유값) | "abc123def456" |
| `run_name` | VARCHAR | MLflow 자동 생성 이름 | "mscn_stats_tiny_20250104_153022" |
| `algorithm` | VARCHAR | 알고리즘 이름 (raw, 필터링용) | "mscn", "lero", "knob" |
| `algorithm_display` | VARCHAR | 알고리즘 표시 이름 (UI용) | "MSCN (카디널리티)" |
| `dataset` | VARCHAR | 데이터셋 이름 (raw, 필터링용) | "stats_tiny", "imdb" |
| `dataset_display` | VARCHAR | 데이터셋 표시 이름 (UI용) | "통계 DB (Small)" |
| `workload_display` | VARCHAR | 워크로드 표시 이름 (UI용) | "기본", "사용자 정의" |
| `execution_time` | FLOAT | 총 실행 시간 (초) | 123.45 |
| `metrics` | JSONB | 모든 메트릭 | `{"test_total_time": 123.45, "test_q_error_95th": 1.23}` |
| `best_config` | JSONB | 추천 설정 (knob/index) | `{"shared_buffers": "2GB"}` |

---

## 3. 동기화 스크립트

### 파일: `scripts/sync_mlflow_to_db.py`

MLflow 데이터를 PostgreSQL로 복사하는 스크립트입니다.

```python
#!/usr/bin/env python3
"""
MLflow 실험 결과를 PostgreSQL experiment_logs 테이블로 동기화

사용법:
    python scripts/sync_mlflow_to_db.py                    # 전체 동기화
    python scripts/sync_mlflow_to_db.py --experiment mscn  # 특정 실험만
    python scripts/sync_mlflow_to_db.py --since 1h         # 최근 1시간만
"""

import mlflow
import psycopg2
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Display name 매핑 (동기화 시 자동 적용)
DISPLAY_NAMES = {
    "algorithm": {
        "mscn": "MSCN (카디널리티)",
        "lero": "Lero (학습 옵티마이저)",
        "knob": "Knob Tuning",
        "index": "Index Selection",
        "baseline": "PostgreSQL 기본"
    },
    "dataset": {
        "stats_tiny": "통계 DB (Small)",
        "stats": "통계 DB",
        "imdb": "IMDB",
        "tpch": "TPC-H"
    },
    "workload": {
        "default": "기본",
        "custom": "사용자 정의",
        "join": "조인 중심",
        "representative": "대표 쿼리"
    }
}


class MLflowToPostgresSync:
    def __init__(self, mlflow_tracking_uri: str = None, pg_config: dict = None):
        # MLflow 설정
        if mlflow_tracking_uri is None:
            project_root = Path(__file__).parent.parent.resolve()
            mlflow_tracking_uri = str((project_root / "mlruns").absolute())
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # PostgreSQL 설정 (내장 DB)
        self.pg_config = pg_config or {
            'host': 'localhost',
            'port': 5432,
            'database': 'PilotScopeUserData',
            'user': 'pilotscope',
            'password': 'pilotscope'
        }

    def init_table(self):
        """테이블이 없으면 생성"""
        conn = psycopg2.connect(**self.pg_config)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS experiment_logs (
                id SERIAL PRIMARY KEY,
                experiment_name VARCHAR(255),
                run_id VARCHAR(255) UNIQUE NOT NULL,
                run_name VARCHAR(255),
                mlflow_experiment VARCHAR(100),
                algorithm VARCHAR(50) NOT NULL,
                dataset VARCHAR(100) NOT NULL,
                workload VARCHAR(100),
                model_id VARCHAR(255),
                algorithm_display VARCHAR(100),
                dataset_display VARCHAR(100),
                workload_display VARCHAR(100),
                status VARCHAR(20) NOT NULL,
                execution_time FLOAT,
                average_time FLOAT,
                metrics JSONB,
                parameters JSONB,
                best_config JSONB,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                synced_at TIMESTAMP DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_experiment_logs_algorithm
                ON experiment_logs(algorithm);
            CREATE INDEX IF NOT EXISTS idx_experiment_logs_dataset
                ON experiment_logs(dataset);
            CREATE INDEX IF NOT EXISTS idx_experiment_logs_started_at
                ON experiment_logs(started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_experiment_logs_status
                ON experiment_logs(status);
        """)

        conn.commit()
        cur.close()
        conn.close()
        logger.info("Table initialized")

    def sync_all(self, experiment_name: str = None, since_hours: int = None):
        """
        MLflow runs를 PostgreSQL로 동기화

        Args:
            experiment_name: 특정 실험만 동기화 (None이면 전체)
            since_hours: 최근 N시간 내 runs만 동기화 (None이면 전체)
        """
        # 실험 목록 가져오기
        if experiment_name:
            experiments = [mlflow.get_experiment_by_name(experiment_name)]
            if experiments[0] is None:
                logger.warning(f"Experiment not found: {experiment_name}")
                return
        else:
            experiments = mlflow.search_experiments()

        conn = psycopg2.connect(**self.pg_config)
        synced_count = 0

        for exp in experiments:
            if exp is None or exp.lifecycle_stage == "deleted":
                continue

            # Runs 조회
            filter_string = ""
            if since_hours:
                since_time = datetime.now() - timedelta(hours=since_hours)
                since_ms = int(since_time.timestamp() * 1000)
                filter_string = f"attributes.start_time > {since_ms}"

            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=filter_string,
                order_by=["start_time DESC"]
            )

            for _, run in runs.iterrows():
                try:
                    self._upsert_run(conn, exp.name, run)
                    synced_count += 1
                except Exception as e:
                    logger.error(f"Failed to sync run {run.run_id}: {e}")

        conn.commit()
        conn.close()
        logger.info(f"Synced {synced_count} runs")

    def _upsert_run(self, conn, experiment_name: str, run):
        """단일 run을 DB에 upsert"""
        cur = conn.cursor()

        # MLflow 데이터 추출
        run_id = run.run_id
        run_name = run.get('tags.mlflow.runName', run_id)
        algorithm = run.get('params.algorithm', 'unknown')
        dataset = run.get('params.dataset', 'unknown')
        workload = run.get('tags.dataset.workload', 'default')
        model_id = run.get('params.model_id') or run.get('tags.model_id')
        status = run.status

        # Display name 매핑
        algorithm_display = DISPLAY_NAMES["algorithm"].get(algorithm, algorithm)
        dataset_display = DISPLAY_NAMES["dataset"].get(dataset, dataset)
        workload_display = DISPLAY_NAMES["workload"].get(workload, workload) if workload else None

        # 타임스탬프 (밀리초 → datetime)
        started_at = datetime.fromtimestamp(run.start_time / 1000) if run.start_time else None
        completed_at = datetime.fromtimestamp(run.end_time / 1000) if run.end_time else None

        # 메트릭 수집
        metrics = {}
        execution_time = None
        average_time = None
        for col in run.index:
            if col.startswith('metrics.'):
                metric_name = col.replace('metrics.', '')
                metrics[metric_name] = run[col]

                if metric_name == 'test_total_time':
                    execution_time = run[col]
                elif metric_name == 'test_average_time':
                    average_time = run[col]

        # 파라미터 수집
        parameters = {}
        for col in run.index:
            if col.startswith('params.') and col != 'params.algorithm' and col != 'params.dataset':
                param_name = col.replace('params.', '')
                parameters[param_name] = run[col]

        # best_config 추출 (knob/index 알고리즘인 경우 - 향후 확장 가능)
        best_config = None
        # TODO: knob/index 결과를 별도로 저장하는 로직 추가 가능

        # UPSERT (ON CONFLICT UPDATE)
        cur.execute("""
            INSERT INTO experiment_logs (
                run_id, run_name, mlflow_experiment,
                algorithm, dataset, workload, model_id,
                algorithm_display, dataset_display, workload_display,
                status, execution_time, average_time,
                metrics, parameters, best_config,
                started_at, completed_at, synced_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
            )
            ON CONFLICT (run_id) DO UPDATE SET
                run_name = EXCLUDED.run_name,
                algorithm_display = EXCLUDED.algorithm_display,
                dataset_display = EXCLUDED.dataset_display,
                workload_display = EXCLUDED.workload_display,
                status = EXCLUDED.status,
                execution_time = EXCLUDED.execution_time,
                average_time = EXCLUDED.average_time,
                metrics = EXCLUDED.metrics,
                parameters = EXCLUDED.parameters,
                best_config = EXCLUDED.best_config,
                completed_at = EXCLUDED.completed_at,
                synced_at = NOW()
        """, (
            run_id, run_name, experiment_name,
            algorithm, dataset, workload, model_id,
            algorithm_display, dataset_display, workload_display,
            status, execution_time, average_time,
            json.dumps(metrics), json.dumps(parameters), json.dumps(best_config) if best_config else None,
            started_at, completed_at
        ))

        cur.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync MLflow to PostgreSQL")
    parser.add_argument("--experiment", help="Specific experiment name to sync")
    parser.add_argument("--since", help="Sync only recent runs (e.g., '1h', '24h')")
    parser.add_argument("--init", action="store_true", help="Initialize table")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    syncer = MLflowToPostgresSync()

    if args.init:
        syncer.init_table()

    # Parse since argument
    since_hours = None
    if args.since:
        if args.since.endswith('h'):
            since_hours = int(args.since[:-1])
        elif args.since.endswith('d'):
            since_hours = int(args.since[:-1]) * 24

    syncer.sync_all(experiment_name=args.experiment, since_hours=since_hours)
    print("Sync completed!")
```

### 사용 예시

```bash
# 컨테이너 접속
docker-compose exec -u pilotscope pilotscope-dev bash
conda activate pilotscope

# 초기 테이블 생성
python scripts/sync_mlflow_to_db.py --init

# 전체 동기화
python scripts/sync_mlflow_to_db.py

# 최근 1시간 데이터만
python scripts/sync_mlflow_to_db.py --since 1h

# 특정 실험만
python scripts/sync_mlflow_to_db.py --experiment pilotscope
```

### 자동 동기화 (옵션)

cron으로 주기적 실행:

```bash
# crontab -e
# 1분마다 동기화
* * * * * cd /home/pilotscope/workspace && /home/pilotscope/miniconda3/envs/pilotscope/bin/python scripts/sync_mlflow_to_db.py --since 1h >> /tmp/mlflow_sync.log 2>&1
```

---

## 4. FastAPI 서버

### 파일 구조

```
pilotscope/api/
├── __init__.py
├── main.py              # FastAPI 앱
├── models.py            # Pydantic 모델
└── database.py          # DB 연결 관리
```

### 4.1. FastAPI 엔드포인트

**기본 URL**: `http://localhost:8000`

#### GET /api/experiments

실험 목록 조회 (필터/정렬 지원)

**Query Parameters**:
- `algorithm`: 알고리즘 필터 (예: `mscn`, `lero`)
- `dataset`: 데이터셋 필터 (예: `stats_tiny`)
- `status`: 상태 필터 (예: `FINISHED`)
- `limit`: 최대 결과 수 (기본값: 50)
- `offset`: 페이지네이션 오프셋 (기본값: 0)
- `sort`: 정렬 필드 (기본값: `started_at`)
- `order`: 정렬 순서 (`asc` 또는 `desc`, 기본값: `desc`)

**Response 200**:
```json
{
  "total": 25,
  "experiments": [
    {
      "id": 1,
      "experiment_name": "통계 DB 카디널리티 개선",
      "run_id": "abc123",
      "run_name": "mscn_stats_tiny_20250104_153022",
      "algorithm": "mscn",
      "algorithm_display": "MSCN (카디널리티)",
      "dataset": "stats_tiny",
      "dataset_display": "통계 DB (Small)",
      "workload": "default",
      "workload_display": "기본",
      "model_id": "mscn_20250104_153022",
      "status": "FINISHED",
      "execution_time": 123.45,
      "average_time": 0.62,
      "started_at": "2025-01-04T15:30:22",
      "completed_at": "2025-01-04T15:32:25"
    }
  ]
}
```

**예시**:
```bash
# 전체 조회
curl http://localhost:8000/api/experiments

# MSCN 알고리즘만
curl "http://localhost:8000/api/experiments?algorithm=mscn"

# 최신 10개, 실행시간 기준 정렬
curl "http://localhost:8000/api/experiments?limit=10&sort=execution_time&order=asc"
```

---

#### GET /api/experiments/{id}

실험 상세 조회

**Response 200**:
```json
{
  "id": 1,
  "experiment_name": "통계 DB 카디널리티 개선",
  "run_id": "abc123",
  "run_name": "mscn_stats_tiny_20250104_153022",
  "mlflow_experiment": "pilotscope",
  "algorithm": "mscn",
  "algorithm_display": "MSCN (카디널리티)",
  "dataset": "stats_tiny",
  "dataset_display": "통계 DB (Small)",
  "workload": "default",
  "workload_display": "기본",
  "model_id": "mscn_20250104_153022",
  "status": "FINISHED",
  "execution_time": 123.45,
  "average_time": 0.62,
  "metrics": {
    "test_total_time": 123.45,
    "test_average_time": 0.62,
    "test_q_error_50th": 1.05,
    "test_q_error_95th": 1.23,
    "test_q_error_max": 2.45
  },
  "parameters": {
    "enable_collection": true,
    "num_collection": 100,
    "num_training": 500,
    "num_epoch": 50
  },
  "best_config": null,
  "started_at": "2025-01-04T15:30:22",
  "completed_at": "2025-01-04T15:32:25",
  "synced_at": "2025-01-04T15:35:00"
}
```

**Response 404**:
```json
{
  "detail": "Experiment not found"
}
```

---

#### PATCH /api/experiments/{id}

실험 이름 변경

**Request Body**:
```json
{
  "experiment_name": "새로운 실험 이름"
}
```

**Response 200**:
```json
{
  "id": 1,
  "experiment_name": "새로운 실험 이름",
  "run_id": "abc123"
}
```

---

#### POST /api/experiments/sync

MLflow → PostgreSQL 동기화 트리거 (API에서 직접 실행)

**Request Body** (optional):
```json
{
  "experiment_name": "pilotscope",
  "since_hours": 1
}
```

**Response 200**:
```json
{
  "status": "success",
  "synced_count": 5,
  "message": "Synced 5 experiments"
}
```

---

### 4.2. FastAPI 구현 예시

**파일: `api/main.py`**

```python
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import psycopg2
import psycopg2.extras
import json
from datetime import datetime

app = FastAPI(title="PilotScope Experiment API", version="1.0.0")

# CORS 설정 (프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PostgreSQL 설정
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'PilotScopeUserData',
    'user': 'pilotscope',
    'password': 'pilotscope'
}


def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)


@app.get("/")
def root():
    return {"message": "PilotScope Experiment API", "version": "1.0.0"}


@app.get("/api/experiments")
def list_experiments(
    algorithm: Optional[str] = None,
    dataset: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort: str = Query("started_at", regex="^(started_at|execution_time|algorithm|dataset)$"),
    order: str = Query("desc", regex="^(asc|desc)$")
):
    """실험 목록 조회"""
    conn = get_db_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # WHERE 조건 구성
    where_clauses = []
    params = []

    if algorithm:
        where_clauses.append("algorithm = %s")
        params.append(algorithm)
    if dataset:
        where_clauses.append("dataset = %s")
        params.append(dataset)
    if status:
        where_clauses.append("status = %s")
        params.append(status)

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    # COUNT 쿼리
    count_query = f"SELECT COUNT(*) FROM experiment_logs {where_sql}"
    cur.execute(count_query, params)
    total = cur.fetchone()['count']

    # SELECT 쿼리
    select_query = f"""
        SELECT
            id, experiment_name, run_id, run_name,
            algorithm, algorithm_display,
            dataset, dataset_display,
            workload, workload_display,
            model_id, status, execution_time, average_time,
            started_at, completed_at
        FROM experiment_logs
        {where_sql}
        ORDER BY {sort} {order.upper()}
        LIMIT %s OFFSET %s
    """
    cur.execute(select_query, params + [limit, offset])
    experiments = cur.fetchall()

    cur.close()
    conn.close()

    return {
        "total": total,
        "experiments": experiments
    }


@app.get("/api/experiments/{exp_id}")
def get_experiment(exp_id: int):
    """실험 상세 조회"""
    conn = get_db_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("""
        SELECT * FROM experiment_logs WHERE id = %s
    """, (exp_id,))

    experiment = cur.fetchone()
    cur.close()
    conn.close()

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return experiment


@app.patch("/api/experiments/{exp_id}")
def update_experiment(exp_id: int, body: dict):
    """실험 이름 변경"""
    if "experiment_name" not in body:
        raise HTTPException(status_code=400, detail="experiment_name is required")

    conn = get_db_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("""
        UPDATE experiment_logs
        SET experiment_name = %s
        WHERE id = %s
        RETURNING id, experiment_name, run_id
    """, (body["experiment_name"], exp_id))

    result = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()

    if not result:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return result


@app.post("/api/experiments/sync")
def sync_experiments(body: dict = None):
    """MLflow → PostgreSQL 동기화"""
    try:
        from scripts.sync_mlflow_to_db import MLflowToPostgresSync

        syncer = MLflowToPostgresSync()

        experiment_name = body.get("experiment_name") if body else None
        since_hours = body.get("since_hours") if body else None

        syncer.sync_all(experiment_name=experiment_name, since_hours=since_hours)

        return {
            "status": "success",
            "message": "Sync completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 5. 테스트 워크플로우

### 5.1. 초기 설정

```bash
# 컨테이너 접속
docker-compose exec -u pilotscope pilotscope-dev bash
conda activate pilotscope

# 테이블 초기화
python scripts/sync_mlflow_to_db.py --init
```

### 5.2. 실험 실행 → 동기화 → 조회

```bash
# 1) 실험 실행 (MLflow에 자동 저장)
cd test_example_algorithms
python unified_test.py --algo mscn --db stats_tiny
python unified_test.py --algo lero --db stats_tiny --timeout 900

# 2) PostgreSQL로 동기화
cd ..
python scripts/sync_mlflow_to_db.py

# 3) FastAPI 서버 시작 (별도 터미널)
cd api
python main.py

# 4) API 테스트
curl http://localhost:8000/api/experiments
curl http://localhost:8000/api/experiments/1
curl -X PATCH http://localhost:8000/api/experiments/1 \
  -H "Content-Type: application/json" \
  -d '{"experiment_name": "MSCN 통계 DB 테스트"}'
```

### 5.3. 지속적 동기화 (개발 중)

**터미널 1**: 실험 실행
```bash
python unified_test.py --algo baseline --db stats_tiny
```

**터미널 2**: 동기화 watch 모드
```bash
watch -n 60 python scripts/sync_mlflow_to_db.py --since 1h
```

**터미널 3**: FastAPI 서버
```bash
cd api && python main.py
```

---

## 6. 문제 해결

### PostgreSQL 연결 오류

```bash
# 컨테이너 내부에서 PostgreSQL 확인
psql -U pilotscope -d PilotScopeUserData -c "SELECT 1"

# 권한 확인
psql -U pilotscope -d PilotScopeUserData -c "\dt"
```

### MLflow 데이터 없음

```bash
# MLflow UI에서 확인
# 브라우저: http://localhost:54321

# mlruns 디렉토리 확인
ls -la mlruns/
```

### 동기화 스크립트 오류

```bash
# 테이블 수동 생성
psql -U pilotscope -d PilotScopeUserData -f scripts/init_experiment_logs.sql

# 로그 확인
python scripts/sync_mlflow_to_db.py 2>&1 | tee sync.log
```

---

## 7. 요약

- **MLflow**: 기존 워크플로우 유지 (자동 저장)
- **동기화**: `sync_mlflow_to_db.py`로 수동 또는 cron 실행
- **PostgreSQL**: 단일 테이블 `experiment_logs`에 저장
- **FastAPI**: 조회/이름변경 API만 제공
