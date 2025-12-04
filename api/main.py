"""
PilotScope Experiment API Server

FastAPI 서버 - MLflow 실험 결과 조회 및 관리

실행:
    python api/main.py

    또는

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException, Query, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import psycopg2
import psycopg2.extras
import sys
import os
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.models import (
    ExperimentBase,
    ExperimentDetail,
    ExperimentListResponse,
    ExperimentUpdateRequest,
    ExperimentUpdateResponse,
    # SyncRequest,
    # SyncResponse,
    HealthResponse,
    ErrorResponse
)

app = FastAPI(
    title="PilotScope Experiment API",
    version="1.0.0",
    description="""
## PilotScope 실험 조회 API

PilotScope 실험 결과를 조회하고 관리하는 읽기 전용 API입니다.

### 주요 기능
- 📊 실험 목록 조회 (필터링, 정렬, 페이지네이션)
- 🔍 실험 상세 정보 조회 (메트릭, 파라미터, best_config 포함)
- ✏️ 실험 이름 변경

### 사용 가능한 알고리즘
- **mscn**: MSCN (카디널리티 추정)
- **lero**: Lero (학습 기반 옵티마이저)
- **llamatune**: LlamaTune - SMAC (Knob Tuning)
- **knob**: Knob Tuning
- **index**: Index Selection
- **baseline**: PostgreSQL 기본
    """,
    contact={
        "name": "PilotScope Team",
        "url": "https://github.com/alibaba/pilotscope"
    }
)

# CORS 설정 (프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PostgreSQL 설정 (환경변수 지원)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': os.getenv('DB_NAME', 'PilotScopeUserData'),
    'user': os.getenv('DB_USER', 'pilotscope'),
    'password': os.getenv('DB_PASSWORD', 'pilotscope')
}


def get_db_conn():
    """PostgreSQL 연결 생성"""
    return psycopg2.connect(**DB_CONFIG)


@app.get("/")
def root():
    """API 루트"""
    return {
        "message": "PilotScope Experiment API",
        "version": "1.0.0",
        "endpoints": {
            "experiments": "/api/experiments",
            "experiment_detail": "/api/experiments/{id}",
            "health": "/health"
        }
    }


@app.get(
    "/api/experiments",
    response_model=ExperimentListResponse,
    summary="실험 목록 조회",
    description="필터링, 정렬, 페이지네이션을 지원하는 실험 목록 조회",
    responses={
        200: {
            "description": "성공적으로 실험 목록 반환",
            "content": {
                "application/json": {
                    "example": {
                        "total": 70,
                        "limit": 50,
                        "offset": 0,
                        "experiments": [
                            {
                                "id": 1,
                                "experiment_name": "MSCN 카디널리티 추정",
                                "run_id": "abc123",
                                "run_name": "mscn_stats_tiny_20250104_153022",
                                "algorithm": "mscn",
                                "algorithm_display": "MSCN (카디널리티)",
                                "dataset": "stats_tiny",
                                "dataset_display": "통계 DB (Small)",
                                "status": "FINISHED",
                                "execution_time": 123.45
                            }
                        ]
                    }
                }
            }
        },
        500: {"model": ErrorResponse}
    }
)
def list_experiments(
    algorithm: Optional[str] = Query(None, description="알고리즘 필터", example="mscn"),
    dataset: Optional[str] = Query(None, description="데이터셋 필터", example="stats_tiny"),
    status: Optional[str] = Query(None, description="상태 필터 (FINISHED, FAILED, RUNNING)", example="FINISHED"),
    limit: int = Query(50, ge=1, le=500, description="페이지 크기"),
    offset: int = Query(0, ge=0, description="페이지 오프셋"),
    sort: str = Query("started_at", regex="^(started_at|execution_time|algorithm|dataset)$", description="정렬 필드"),
    order: str = Query("desc", regex="^(asc|desc)$", description="정렬 순서")
):
    """실험 목록 조회"""
    try:
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
                status, execution_time, average_time,
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
            "limit": limit,
            "offset": offset,
            "experiments": experiments
        }
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/experiments/{exp_id}",
    response_model=ExperimentDetail,
    summary="실험 상세 조회",
    description="실험 ID로 메트릭과 파라미터를 포함한 상세 정보 조회",
    responses={
        200: {"description": "실험 상세 정보"},
        404: {"model": ErrorResponse, "description": "실험을 찾을 수 없음"},
        500: {"model": ErrorResponse}
    }
)
def get_experiment(exp_id: int = PathParam(..., description="실험 ID", example=1)):
    """실험 상세 조회"""
    try:
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
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch(
    "/api/experiments/{exp_id}",
    response_model=ExperimentUpdateResponse,
    summary="실험 이름 변경",
    description="실험의 사용자 지정 이름을 변경합니다",
    responses={
        200: {"description": "성공적으로 이름 변경"},
        404: {"model": ErrorResponse, "description": "실험을 찾을 수 없음"},
        500: {"model": ErrorResponse}
    }
)
def update_experiment(
    exp_id: int = PathParam(..., description="실험 ID", example=1),
    body: ExperimentUpdateRequest = None
):
    """실험 이름 변경"""
    if not body or not body.experiment_name:
        raise HTTPException(status_code=400, detail="experiment_name is required")

    try:
        conn = get_db_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            UPDATE experiment_logs
            SET experiment_name = %s
            WHERE id = %s
            RETURNING id, experiment_name, run_id
        """, (body.experiment_name, exp_id))

        result = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()

        if not result:
            raise HTTPException(status_code=404, detail="Experiment not found")

        return result
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# MLflow 동기화 엔드포인트 (deployment에서는 사용 불가 - MLflow 없음)
# @app.post(
#     "/api/experiments/sync",
#     response_model=SyncResponse,
#     summary="MLflow 동기화",
#     description="MLflow 실험 결과를 PostgreSQL로 동기화합니다",
#     responses={
#         200: {"description": "동기화 성공"},
#         500: {"model": ErrorResponse}
#     }
# )
# def sync_experiments(body: Optional[SyncRequest] = None):
#     """MLflow → PostgreSQL 동기화"""
#     try:
#         from scripts.sync_mlflow_to_db import MLflowToPostgresSync
#
#         syncer = MLflowToPostgresSync()
#
#         experiment_name = body.experiment_name if body else None
#         since_hours = body.since_hours if body else None
#
#         syncer.sync_all(experiment_name=experiment_name, since_hours=since_hours)
#
#         return {
#             "status": "success",
#             "message": "Sync completed"
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="헬스 체크",
    description="API 서버와 데이터베이스 연결 상태를 확인합니다"
)
def health_check():
    """헬스 체크"""
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    print("Starting PilotScope Experiment API server...")
    print("API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
