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
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Display name 매핑 (동기화 시 자동 적용)
DISPLAY_NAMES = {
    "algorithm": {
        "mscn": "MSCN (카디널리티)",
        "lero": "Lero (실행계획)",
        "knob": "Knob Tuning",
        "llamatune": "LlamaTune - SMAC (Knob)",
        "index": "Index Selection",
        "baseline": "Baseline"
    },
    "dataset": {
        "stats_tiny": "Stats (Tiny)",
        "stats": "Stats",
        "imdb": "IMDB",
        "tpch": "TPC-H",
        "stock_strategy": "Stock Strategy"
    },
    "workload": {
        "default": "Default",
        "custom": "Custom",
        "representative": "Representative",
        "representative_momentum": "Representative - Momentum",
        "representative_value_quality": "Representative - Value Quality",
        "representative_smallcap_turnover": "Representative - Smallcap Turnover",
        "momentum_investing": "Momentum Investing",
        "value_investing": "Value Investing",
        "value_quality": "Value Quality",
        "ml_hybrid": "ML Hybrid"
    }
}


class MLflowToPostgresSync:
    def __init__(self, mlflow_tracking_uri: str = None, pg_config: dict = None):
        # MLflow 설정
        if mlflow_tracking_uri is None:
            project_root = Path(__file__).parent.parent.resolve()
            mlflow_tracking_uri = str((project_root / "mlruns").absolute())
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # MLflow Client (artifact 접근용)
        self.client = mlflow.tracking.MlflowClient()

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

    def sync_all(self, experiment_name: str = None, since_hours: int = None, cleanup_deleted: bool = True):
        """
        MLflow runs를 PostgreSQL로 동기화

        Args:
            experiment_name: 특정 실험만 동기화 (None이면 전체)
            since_hours: 최근 N시간 내 runs만 동기화 (None이면 전체)
            cleanup_deleted: MLflow에서 삭제된 runs를 DB에서도 제거 (default: True)
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
        deleted_count = 0

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

            # 1. MLflow runs를 DB에 upsert
            for _, run in runs.iterrows():
                try:
                    self._upsert_run(conn, exp.name, run)
                    conn.commit()  # 각 run마다 commit
                    synced_count += 1
                except Exception as e:
                    conn.rollback()  # 실패 시 rollback
                    logger.error(f"Failed to sync run {run.run_id}: {e}")

            # 2. MLflow에 없는 runs를 DB에서 삭제 (cleanup_deleted=True인 경우)
            if cleanup_deleted:
                mlflow_run_ids = set(runs['run_id'].tolist()) if not runs.empty else set()
                deleted = self._cleanup_deleted_runs(conn, exp.name, mlflow_run_ids)
                deleted_count += deleted

        # 3. 전역 cleanup: MLflow에 존재하지 않는 모든 run_id 제거
        if cleanup_deleted:
            global_deleted = self._global_cleanup(conn, experiments)
            deleted_count += global_deleted

        conn.close()
        logger.info(f"Synced {synced_count} runs, Deleted {deleted_count} orphaned runs")

    def _upsert_run(self, conn, experiment_name: str, run):
        """단일 run을 DB에 upsert"""
        cur = conn.cursor()

        # MLflow 데이터 추출
        run_id = run.run_id
        run_name = run.get('tags.mlflow.runName', run_id)
        algorithm = run.get('params.algorithm', 'unknown')
        dataset_raw = run.get('params.dataset', 'unknown')
        status = run.status

        # Dataset 파싱: "stock_strategy_워크로드명" → dataset="stock_strategy", workload="워크로드명"
        if '_' in dataset_raw and dataset_raw.startswith('stock_strategy_'):
            parts = dataset_raw.split('_', 2)  # "stock_strategy_워크로드명"를 3개로 split
            if len(parts) >= 3:
                dataset = f"{parts[0]}_{parts[1]}"  # "stock_strategy"
                workload = parts[2]  # "워크로드명"
            else:
                dataset = dataset_raw
                workload = 'default'
        else:
            dataset = dataset_raw
            workload = run.get('tags.dataset.workload', 'default')

        # Display name 매핑
        algorithm_display = DISPLAY_NAMES["algorithm"].get(algorithm, algorithm)
        dataset_display = DISPLAY_NAMES["dataset"].get(dataset, dataset)
        workload_display = DISPLAY_NAMES["workload"].get(workload, workload) if workload else None

        # 타임스탬프 (pandas Timestamp → datetime)
        started_at = pd.to_datetime(run.start_time) if pd.notna(run.start_time) else None
        completed_at = pd.to_datetime(run.end_time) if pd.notna(run.end_time) else None

        # 메트릭 수집 (NaN 처리)
        metrics = {}
        execution_time = None
        average_time = None
        for col in run.index:
            if col.startswith('metrics.'):
                metric_name = col.replace('metrics.', '')
                value = run[col]
                # NaN 체크 및 변환
                if pd.notna(value):
                    metrics[metric_name] = float(value)
                    if metric_name == 'test_total_time':
                        execution_time = float(value)
                    elif metric_name == 'test_average_time':
                        average_time = float(value)

        # 파라미터 수집 (NaN 처리)
        parameters = {}
        for col in run.index:
            if col.startswith('params.') and col != 'params.algorithm' and col != 'params.dataset':
                param_name = col.replace('params.', '')
                value = run[col]
                # NaN이 아닌 경우만 저장
                if pd.notna(value):
                    parameters[param_name] = str(value)

        # best_config 추출 (knob/index 알고리즘인 경우)
        best_config = None
        if algorithm in ['knob', 'llamatune', 'index']:
            try:
                artifacts = self.client.list_artifacts(run_id)
                # knob_config_iteration_*.json 또는 index_config_*.json 찾기
                config_files = [art for art in artifacts if 'config' in art.path.lower() and art.path.endswith('.json')]

                if config_files:
                    # 가장 마지막 iteration 또는 첫 번째 파일 선택
                    config_file = config_files[-1] if len(config_files) > 1 else config_files[0]

                    # artifact 다운로드 및 읽기
                    artifact_path = self.client.download_artifacts(run_id, config_file.path)
                    with open(artifact_path, 'r') as f:
                        best_config = json.load(f)
            except Exception as e:
                logger.debug(f"Could not read best_config for run {run_id}: {e}")

        # UPSERT (ON CONFLICT UPDATE)
        cur.execute("""
            INSERT INTO experiment_logs (
                run_id, run_name, mlflow_experiment,
                algorithm, dataset, workload,
                algorithm_display, dataset_display, workload_display,
                status, execution_time, average_time,
                metrics, parameters, best_config,
                started_at, completed_at, synced_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
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
            algorithm, dataset, workload,
            algorithm_display, dataset_display, workload_display,
            status, execution_time, average_time,
            json.dumps(metrics), json.dumps(parameters), json.dumps(best_config) if best_config else None,
            started_at, completed_at
        ))

        cur.close()

    def _cleanup_deleted_runs(self, conn, experiment_name: str, mlflow_run_ids: set) -> int:
        """
        MLflow에서 삭제된 runs를 PostgreSQL에서도 제거

        Args:
            conn: DB 연결
            experiment_name: 실험 이름
            mlflow_run_ids: MLflow에 존재하는 run_id 집합

        Returns:
            삭제된 runs 수
        """
        cur = conn.cursor()

        # DB에서 해당 실험의 모든 run_id 조회
        cur.execute("""
            SELECT run_id FROM experiment_logs
            WHERE mlflow_experiment = %s
        """, (experiment_name,))

        db_run_ids = {row[0] for row in cur.fetchall()}

        # MLflow에 없는 run_id 찾기
        orphaned_run_ids = db_run_ids - mlflow_run_ids

        if orphaned_run_ids:
            logger.info(f"Deleting {len(orphaned_run_ids)} orphaned runs from {experiment_name}")
            # 삭제 실행
            cur.execute("""
                DELETE FROM experiment_logs
                WHERE run_id = ANY(%s)
            """, (list(orphaned_run_ids),))
            conn.commit()

        cur.close()
        return len(orphaned_run_ids)

    def _global_cleanup(self, conn, experiments) -> int:
        """
        MLflow 전체에서 존재하지 않는 runs를 PostgreSQL에서 제거

        Args:
            conn: DB 연결
            experiments: MLflow 실험 리스트

        Returns:
            삭제된 runs 수
        """
        cur = conn.cursor()

        # MLflow의 모든 run_id 수집
        all_mlflow_run_ids = set()
        for exp in experiments:
            if exp is None or exp.lifecycle_stage == "deleted":
                continue

            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"]
            )

            if not runs.empty:
                all_mlflow_run_ids.update(runs['run_id'].tolist())

        # DB의 모든 run_id 조회
        cur.execute("SELECT run_id FROM experiment_logs")
        db_run_ids = {row[0] for row in cur.fetchall()}

        # MLflow에 없는 run_id 찾기
        orphaned_run_ids = db_run_ids - all_mlflow_run_ids

        if orphaned_run_ids:
            logger.info(f"Global cleanup: Deleting {len(orphaned_run_ids)} orphaned runs")
            # 삭제 실행
            cur.execute("""
                DELETE FROM experiment_logs
                WHERE run_id = ANY(%s)
            """, (list(orphaned_run_ids),))
            conn.commit()

        cur.close()
        return len(orphaned_run_ids)


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
