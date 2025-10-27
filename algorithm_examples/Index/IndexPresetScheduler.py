import sys

sys.path.append("../")
sys.path.append("../algorithm_examples/Index/index_selection_evaluation")
from pilotscope.Factory.SchedulerFactory import SchedulerFactory
from pilotscope.PilotConfig import PilotConfig
from pilotscope.PilotScheduler import PilotScheduler
from pilotscope.Common.MLflowTracker import MLflowTracker
from algorithm_examples.Index.EventImplement import IndexPeriodicModelUpdateEvent


def get_index_preset_scheduler(config: PilotConfig, use_mlflow=True, **kwargs) -> tuple:
    test_data_table = "extend_{}_test_data_table".format(config.db)
    config.sql_execution_timeout = config.once_request_timeout = 50000
    config.print()

    # Initialize MLflow tracker
    mlflow_tracker = None
    if use_mlflow:
        mlflow_tracker = MLflowTracker(experiment_name=f"index_{config.db}")
        # Start run for index selection (iterative optimization)
        mlflow_tracker.start_training(
            algo_name="index_selection",
            dataset=config.db,
            params={
                "algorithm": "extend",
                "periodic_update": True,
                "update_interval": 200
            }
        )

    # core
    scheduler: PilotScheduler = SchedulerFactory.create_scheduler(config)

    # allow to pretrain model
    periodic_model_update_event = IndexPeriodicModelUpdateEvent(config, 200, execute_on_init=True, mlflow_tracker=mlflow_tracker)
    scheduler.register_events([periodic_model_update_event])
    scheduler.register_required_data(test_data_table, pull_execution_time=True)

    # Attach tracker to scheduler
    scheduler.mlflow_tracker = mlflow_tracker

    # start
    scheduler.init()
    return scheduler, mlflow_tracker
