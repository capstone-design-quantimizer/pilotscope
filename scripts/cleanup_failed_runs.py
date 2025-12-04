#!/usr/bin/env python3
"""
MLflow에서 실패한 실험(runs)을 삭제하는 스크립트

사용법:
    python scripts/cleanup_failed_runs.py                  # 실패한 runs 조회만
    python scripts/cleanup_failed_runs.py --delete         # 실제 삭제
    python scripts/cleanup_failed_runs.py --experiment mscn  # 특정 실험만
"""

import mlflow
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cleanup_failed_runs(delete: bool = False, experiment_name: str = None):
    """
    실패한 MLflow runs를 삭제

    Args:
        delete: True면 실제 삭제, False면 조회만
        experiment_name: 특정 실험만 처리 (None이면 전체)
    """
    # MLflow 설정
    project_root = Path(__file__).parent.parent.resolve()
    mlflow_tracking_uri = str((project_root / "mlruns").absolute())
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # 실험 목록 가져오기
    if experiment_name:
        experiments = [mlflow.get_experiment_by_name(experiment_name)]
        if experiments[0] is None:
            logger.error(f"Experiment not found: {experiment_name}")
            return
    else:
        experiments = mlflow.search_experiments()

    total_failed = 0
    deleted_count = 0

    for exp in experiments:
        if exp is None or exp.lifecycle_stage == "deleted":
            continue

        logger.info(f"\n검사 중: {exp.name}")

        # FAILED 상태인 runs 조회
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'FAILED'",
            order_by=["start_time DESC"]
        )

        if len(runs) == 0:
            logger.info("  ✓ 실패한 실험 없음")
            continue

        logger.info(f"  ⚠️  실패한 실험 {len(runs)}개 발견:")

        for _, run in runs.iterrows():
            run_id = run.run_id
            run_name = run.get('tags.mlflow.runName', run_id[:8])
            started_at = run.start_time

            total_failed += 1
            logger.info(f"    - {run_name} ({run_id[:8]}...) - {started_at}")

            if delete:
                try:
                    mlflow.delete_run(run_id)
                    deleted_count += 1
                    logger.info(f"      ✓ 삭제됨")
                except Exception as e:
                    logger.error(f"      ✗ 삭제 실패: {e}")

    logger.info(f"\n{'='*60}")
    logger.info(f"총 실패한 실험: {total_failed}개")

    if delete:
        logger.info(f"삭제된 실험: {deleted_count}개")
        logger.info(f"\n✓ 삭제 완료!")
        logger.info(f"\n다음 단계:")
        logger.info(f"1. PostgreSQL 동기화: python scripts/sync_mlflow_to_db.py")
        logger.info(f"2. MLflow UI에서 확인: http://localhost:54321")
    else:
        logger.info(f"\n위 실험들을 삭제하려면 --delete 옵션을 사용하세요:")
        logger.info(f"python scripts/cleanup_failed_runs.py --delete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow 실패한 실험 정리")
    parser.add_argument("--delete", action="store_true", help="실제로 삭제 (없으면 조회만)")
    parser.add_argument("--experiment", help="특정 실험만 처리")
    args = parser.parse_args()

    if not args.delete:
        logger.info("=" * 60)
        logger.info("조회 모드 (실제 삭제하지 않음)")
        logger.info("삭제하려면 --delete 옵션을 사용하세요")
        logger.info("=" * 60)

    cleanup_failed_runs(delete=args.delete, experiment_name=args.experiment)
