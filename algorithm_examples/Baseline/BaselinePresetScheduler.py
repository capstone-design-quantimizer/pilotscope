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


def get_baseline_preset_scheduler(config, use_mlflow=True, experiment_name=None, dataset_name=None, **kwargs) -> tuple:
    """
    Create baseline scheduler (no AI, direct execution)

    Args:
        config: PilotConfig instance
        use_mlflow: MLflow tracking enable (default: True)
        experiment_name: MLflow experiment name (optional)
        dataset_name: Dataset name for logging (optional)
        **kwargs: Ignored (for compatibility with other schedulers)

    Returns:
        Tuple of (PilotScheduler, MLflowTracker or None)
    """
    # Initialize MLflow tracker for baseline (no training, only testing)
    mlflow_tracker = None
    if use_mlflow:
        # Use provided experiment name or fallback to default
        exp_name = experiment_name if experiment_name else f"baseline_{config.db}"
        mlflow_tracker = MLflowTracker(experiment_name=exp_name)

        # Start run for testing only (no training phase for baseline)
        # Extract workload from dataset_name
        workload = None
        db_name = config.db
        if dataset_name and "_" in dataset_name:
            # e.g., "stats_tiny_custom" -> db="stats_tiny", workload="custom"
            parts = dataset_name.rsplit("_", 1)
            if len(parts) == 2 and parts[0] == config.db:
                workload = parts[1]

        mlflow_tracker.start_training(
            algo_name="baseline",
            dataset=dataset_name if dataset_name else config.db,
            params={"no_ai": True, "direct_execution": True},
            db_name=db_name,
            workload=workload,
            num_queries=0  # No training for baseline
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

