import sys

sys.path.append("../")
from pilotscope.Factory.SchedulerFactory import SchedulerFactory
from pilotscope.PilotConfig import PilotConfig
from pilotscope.PilotScheduler import PilotScheduler
from pilotscope.Common.MLflowTracker import MLflowTracker
from algorithm_examples.KnobTuning.EventImplement import KnobPeriodicModelUpdateEvent


def get_knob_preset_scheduler(config: PilotConfig, use_mlflow=True, experiment_name=None, dataset_name=None, **kwargs) -> tuple:
    # config.db = "stats_tiny"
    config.sql_execution_timeout = 300000
    config.once_request_timeout = 300000

    # Initialize MLflow tracker
    mlflow_tracker = None
    if use_mlflow:
        # Use provided experiment name or fallback to default
        exp_name = experiment_name if experiment_name else f"knob_{config.db}"
        mlflow_tracker = MLflowTracker(experiment_name=exp_name)

        # Extract workload from dataset_name
        workload = None
        db_name = config.db
        if dataset_name and "_" in dataset_name:
            # e.g., "stats_tiny_custom" -> db="stats_tiny", workload="custom"
            parts = dataset_name.rsplit("_", 1)
            if len(parts) == 2 and parts[0] == config.db:
                workload = parts[1]

        # Start run for knob tuning (iterative optimization)
        mlflow_tracker.start_training(
            algo_name="knob_tuning",
            dataset=dataset_name if dataset_name else config.db,
            params={
                "algorithm": "llamatune",
                "optimizer": "smac",
                "periodic_update": True,
                "update_interval": 200
            },
            db_name=db_name,
            workload=workload
        )

    # core
    scheduler: PilotScheduler = SchedulerFactory.create_scheduler(config)
    scheduler.db_controller.backup_config()

    # allow to pretrain model
    periodic_model_update_event = KnobPeriodicModelUpdateEvent(config, 200,
                                                               execute_on_init=True,
                                                               llamatune_config_file="../algorithm_examples/KnobTuning/llamatune/configs/llama_config.ini",
                                                               optimizer_type="smac",
                                                               mlflow_tracker=mlflow_tracker)
    scheduler.register_events([periodic_model_update_event])
    scheduler.register_required_data("llamatune_data", pull_execution_time=True)

    # Attach tracker to scheduler
    scheduler.mlflow_tracker = mlflow_tracker

    # TimeStatistic.print()
    # start
    scheduler.init()
    return scheduler, mlflow_tracker
