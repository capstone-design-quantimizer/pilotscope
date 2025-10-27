import sys

sys.path.append("../")
from pilotscope.Factory.SchedulerFactory import SchedulerFactory
from pilotscope.PilotConfig import PilotConfig
from pilotscope.PilotScheduler import PilotScheduler
from pilotscope.Common.MLflowTracker import MLflowTracker
from algorithm_examples.KnobTuning.EventImplement import KnobPeriodicModelUpdateEvent


def get_knob_preset_scheduler(config: PilotConfig, use_mlflow=True, **kwargs) -> tuple:
    # config.db = "stats_tiny"
    config.sql_execution_timeout = 300000
    config.once_request_timeout = 300000

    # Initialize MLflow tracker
    mlflow_tracker = None
    if use_mlflow:
        mlflow_tracker = MLflowTracker(experiment_name=f"knob_{config.db}")
        # Start run for knob tuning (iterative optimization)
        mlflow_tracker.start_training(
            algo_name="knob_tuning",
            dataset=config.db,
            params={
                "algorithm": "llamatune",
                "optimizer": "smac",
                "periodic_update": True,
                "update_interval": 200
            }
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
