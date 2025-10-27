"""
Baseline Preset Scheduler

No AI algorithm, just direct database execution for performance comparison.
"""

import sys
sys.path.append("../")

from pilotscope.Factory.SchedulerFactory import SchedulerFactory
from pilotscope.PilotScheduler import PilotScheduler
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor
from pilotscope.Common.MLflowTracker import MLflowTracker


def get_baseline_preset_scheduler(config, use_mlflow=True, **kwargs) -> tuple:
    """
    Create baseline scheduler (no AI, direct execution)

    Args:
        config: PilotConfig instance
        use_mlflow: MLflow tracking enable (default: True)
        **kwargs: Ignored (for compatibility with other schedulers)

    Returns:
        Tuple of (PilotScheduler, MLflowTracker or None)
    """
    # Initialize MLflow tracker for baseline (no training, only testing)
    mlflow_tracker = None
    if use_mlflow:
        mlflow_tracker = MLflowTracker(experiment_name=f"baseline_{config.db}")
        # Start run for testing only (no training phase for baseline)
        mlflow_tracker.start_training(
            algo_name="baseline",
            dataset=config.db,
            params={"no_ai": True, "direct_execution": True}
        )

    # Create scheduler without any AI handlers
    scheduler: PilotScheduler = SchedulerFactory.create_scheduler(config)

    # Register data collection for fair comparison with AI algorithms
    # Collect execution time to measure actual DB query performance
    test_data_save_table = "baseline_data_table"
    scheduler.register_required_data(test_data_save_table, pull_execution_time=True)

    # No AI handlers - this is the key difference from MSCN/Lero
    # No events, no card estimation, no query optimization
    # Just pure database execution through PilotScope infrastructure

    # Attach tracker to scheduler
    scheduler.mlflow_tracker = mlflow_tracker

    # Start the scheduler
    scheduler.init()
    return scheduler, mlflow_tracker

