"""
Pydantic models for API request/response
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class ExperimentBase(BaseModel):
    """실험 기본 정보"""
    id: int = Field(..., description="실험 ID", example=1)
    experiment_name: Optional[str] = Field(None, description="사용자 지정 실험 이름", example="MSCN 카디널리티 추정 테스트")
    run_id: str = Field(..., description="MLflow run ID", example="abc123def456")
    run_name: str = Field(..., description="MLflow 자동 생성 이름", example="mscn_stats_tiny_20250104_153022")
    algorithm: str = Field(..., description="알고리즘 (raw 값)", example="mscn")
    algorithm_display: Optional[str] = Field(None, description="알고리즘 표시 이름", example="MSCN (카디널리티)")
    dataset: str = Field(..., description="데이터셋 (raw 값)", example="stock_strategy")
    dataset_display: Optional[str] = Field(None, description="데이터셋 표시 이름", example="Stock Strategy")
    workload: Optional[str] = Field(None, description="워크로드 (raw 값)", example="representative_momentum")
    workload_display: Optional[str] = Field(None, description="워크로드 표시 이름", example="Representative - Momentum")
    status: str = Field(..., description="실험 상태", example="FINISHED")
    execution_time: Optional[float] = Field(None, description="총 실행 시간 (초)", example=123.45)
    average_time: Optional[float] = Field(None, description="평균 쿼리 시간 (초)", example=0.62)
    started_at: Optional[datetime] = Field(None, description="실험 시작 시간")
    completed_at: Optional[datetime] = Field(None, description="실험 완료 시간")


class ExperimentDetail(ExperimentBase):
    """실험 상세 정보 (메트릭, 파라미터 포함)"""
    mlflow_experiment: Optional[str] = Field(None, description="MLflow 실험 이름", example="pilotscope")
    metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="실험 메트릭",
        example={
            "test_total_time": 123.45,
            "test_average_time": 0.62,
            "test_q_error_50th": 1.05,
            "test_q_error_95th": 1.23,
            "test_q_error_max": 2.45
        }
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="하이퍼파라미터",
        example={
            "enable_collection": "true",
            "num_collection": "100",
            "num_training": "500",
            "num_epoch": "50"
        }
    )
    best_config: Optional[Dict[str, Any]] = Field(
        None,
        description="추천 설정 (knob/index)",
        example={"shared_buffers": "2GB", "work_mem": "128MB"}
    )
    synced_at: Optional[datetime] = Field(None, description="DB 동기화 시간")


class ExperimentListResponse(BaseModel):
    """실험 목록 응답"""
    total: int = Field(..., description="전체 실험 수", example=70)
    limit: int = Field(..., description="페이지 크기", example=50)
    offset: int = Field(..., description="페이지 오프셋", example=0)
    experiments: List[ExperimentBase] = Field(..., description="실험 목록")


class ExperimentUpdateRequest(BaseModel):
    """실험 이름 변경 요청"""
    experiment_name: str = Field(
        ...,
        description="새로운 실험 이름",
        min_length=1,
        max_length=255,
        example="MSCN 주식 전략 카디널리티 개선"
    )


class ExperimentUpdateResponse(BaseModel):
    """실험 이름 변경 응답"""
    id: int = Field(..., description="실험 ID", example=1)
    experiment_name: str = Field(..., description="변경된 실험 이름", example="MSCN 주식 전략 카디널리티 개선")
    run_id: str = Field(..., description="MLflow run ID", example="abc123def456")


class SyncRequest(BaseModel):
    """동기화 요청"""
    experiment_name: Optional[str] = Field(None, description="특정 실험만 동기화 (None이면 전체)", example="pilotscope")
    since_hours: Optional[int] = Field(None, description="최근 N시간 내 runs만 동기화 (None이면 전체)", example=1, ge=1)


class SyncResponse(BaseModel):
    """동기화 응답"""
    status: str = Field(..., description="동기화 상태", example="success")
    message: str = Field(..., description="메시지", example="Sync completed")


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str = Field(..., description="서비스 상태", example="healthy")
    database: str = Field(..., description="데이터베이스 연결 상태", example="connected")


class ErrorResponse(BaseModel):
    """에러 응답"""
    detail: str = Field(..., description="에러 메시지", example="Experiment not found")
