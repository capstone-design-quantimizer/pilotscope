import sys

sys.path.append("../")
sys.path.append("../algorithm_examples/Index/index_selection_evaluation")
from pilotscope.Factory.SchedulerFactory import SchedulerFactory
from pilotscope.PilotConfig import PilotConfig
from pilotscope.PilotScheduler import PilotScheduler
from pilotscope.Common.MLflowTracker import MLflowTracker
from algorithm_examples.Index.EventImplement import IndexPeriodicModelUpdateEvent


def get_index_preset_scheduler(config: PilotConfig, use_mlflow=True, experiment_name=None, dataset_name=None, **kwargs) -> tuple:
    test_data_table = "extend_{}_test_data_table".format(config.db)
    config.sql_execution_timeout = config.once_request_timeout = 50000
    config.print()

    # Initialize MLflow tracker
    mlflow_tracker = None
    if use_mlflow:
        # Use provided experiment name or fallback to default
        exp_name = experiment_name if experiment_name else f"index_{config.db}"
        mlflow_tracker = MLflowTracker(experiment_name=exp_name)

        # Extract workload from dataset_name
        workload = None
        db_name = config.db
        if dataset_name and "_" in dataset_name:
            # e.g., "stats_tiny_custom" -> db="stats_tiny", workload="custom"
            parts = dataset_name.rsplit("_", 1)
            if len(parts) == 2 and parts[0] == config.db:
                workload = parts[1]

        # Start run for index selection (iterative optimization)
        mlflow_tracker.start_training(
            algo_name="index_selection",
            dataset=dataset_name if dataset_name else config.db,
            params={
                "algorithm": "extend",
                "periodic_update": True,
                "update_interval": 200
            },
            db_name=db_name,
            workload=workload
        )

    # core
    scheduler: PilotScheduler = SchedulerFactory.create_scheduler(config)

    # allow to pretrain model
    periodic_model_update_event = IndexPeriodicModelUpdateEvent(config, 200, execute_on_init=True, mlflow_tracker=mlflow_tracker, dataset_name=dataset_name)
    scheduler.register_events([periodic_model_update_event])
    scheduler.register_required_data(test_data_table, pull_execution_time=True)

    # Attach tracker to scheduler
    scheduler.mlflow_tracker = mlflow_tracker

    # start
    scheduler.init()
    return scheduler, mlflow_tracker
