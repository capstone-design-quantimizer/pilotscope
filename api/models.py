"""
Pydantic models for API request/response
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class ExperimentBase(BaseModel):
    """실험 기본 정보"""
    id: int = Field(..., description="실험 ID", example=1)
    experiment_name: Optional[str] = Field(None, description="사용자 지정 실험 이름", example="LlamaTune Knob 최적화")
    run_id: str = Field(..., description="MLflow run ID", example="dfcac42e49584984be74230d72e04d25")
    run_name: str = Field(..., description="MLflow 자동 생성 이름", example="knob_tuning_stock_strategy_representative_value_quality_20251125_194527")
    algorithm: str = Field(..., description="알고리즘 (raw 값)", example="llamatune")
    algorithm_display: Optional[str] = Field(None, description="알고리즘 표시 이름", example="LlamaTune - SMAC (Knob)")
    dataset: str = Field(..., description="데이터셋 (raw 값)", example="stock_strategy")
    dataset_display: Optional[str] = Field(None, description="데이터셋 표시 이름", example="Stock Strategy")
    workload: Optional[str] = Field(None, description="워크로드 (raw 값)", example="representative_value_quality")
    workload_display: Optional[str] = Field(None, description="워크로드 표시 이름", example="Representative - Value Quality")
    status: str = Field(..., description="실험 상태", example="FINISHED")
    execution_time: Optional[float] = Field(None, description="총 실행 시간 (초)", example=0.12)
    average_time: Optional[float] = Field(None, description="평균 쿼리 시간 (초)", example=0.006)
    started_at: Optional[datetime] = Field(None, description="실험 시작 시간")
    completed_at: Optional[datetime] = Field(None, description="실험 완료 시간")


class ExperimentDetail(ExperimentBase):
    """실험 상세 정보 (메트릭, 파라미터 포함)"""
    mlflow_experiment: Optional[str] = Field(None, description="MLflow 실험 이름", example="knob_stock_strategy_representative_value_quality")
    metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="실험 메트릭",
        example={
            "num_knobs_tuned": 112.0,
            "test_total_time": 0.12,
            "best_performance": 40753.95,
            "test_query_count": 20.0,
            "test_average_time": 0.006,
            "knob_optimization_time_seconds": 91.74
        }
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="하이퍼파라미터",
        example={
            "optimizer": "smac",
            "periodic_update": "True",
            "update_interval": "200"
        }
    )
    best_config: Optional[Dict[str, Any]] = Field(
        None,
        description="최적 설정 (llamatune 알고리즘만 해당)",
        example={
            "target_metric": "throughput",
            "best_performance": 40753.95,
            "best_configuration": {
                "shared_buffers": 448066,
                "work_mem": 121669,
                "effective_cache_size": 991115
            }
        }
    )
    synced_at: Optional[datetime] = Field(None, description="DB 동기화 시간")


class ExperimentListResponse(BaseModel):
    """실험 목록 응답"""
    total: int = Field(..., description="전체 실험 수", example=35)
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
        example="LlamaTune 주식 전략 Knob 최적화 실험"
    )


class ExperimentUpdateResponse(BaseModel):
    """실험 이름 변경 응답"""
    id: int = Field(..., description="실험 ID", example=297)
    experiment_name: str = Field(..., description="변경된 실험 이름", example="LlamaTune 주식 전략 Knob 최적화 실험")
    run_id: str = Field(..., description="MLflow run ID", example="dfcac42e49584984be74230d72e04d25")


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
