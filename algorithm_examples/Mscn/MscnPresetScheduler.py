import sys

sys.path.append("../")

from pilotscope.Factory.SchedulerFactory import SchedulerFactory
from pilotscope.PilotModel import PilotModel
from pilotscope.PilotScheduler import PilotScheduler
from pilotscope.Common.MLflowTracker import MLflowTracker
from algorithm_examples.Mscn.EventImplement import MscnPretrainingModelEvent
from algorithm_examples.Mscn.MscnParadigmCardAnchorHandler import MscnCardPushHandler
from algorithm_examples.Mscn.MscnPilotModel import MscnPilotModel


def get_mscn_preset_scheduler(config, enable_collection, enable_training, num_collection = -1, num_training = -1, num_epoch = 100, load_model_id=None, use_mlflow=True, experiment_name=None, dataset_name=None) -> tuple:
    if type(enable_collection) == str:
        enable_collection = eval(enable_collection)
    if type(enable_training) == str:
        enable_training = eval(enable_training)
    if type(num_collection) == str:
        num_collection = int(num_collection)
    if type(num_training) == str:
        num_training = int(num_training)
    if type(num_epoch) == str:
        num_epoch = int(num_epoch)

    model_name = "mscn"

    # Initialize MLflow tracker
    mlflow_tracker = None
    if use_mlflow and enable_training:
        # Use provided experiment name or fallback to default
        exp_name = experiment_name if experiment_name else f"mscn_{config.db}"
        mlflow_tracker = MLflowTracker(experiment_name=exp_name)

        # Start MLflow run for training
        hyperparams = {
            "num_epoch": num_epoch,
            "num_training": num_training,
            "num_collection": num_collection,
            "enable_collection": enable_collection,
            "enable_training": enable_training
        }
        # Extract workload from dataset_name
        workload = None
        db_name = config.db
        if dataset_name and "_" in dataset_name:
            # e.g., "stats_tiny_custom" -> db="stats_tiny", workload="custom"
            parts = dataset_name.rsplit("_", 1)
            if len(parts) == 2 and parts[0] == config.db:
                workload = parts[1]

        mlflow_tracker.start_training(
            algo_name="mscn",
            dataset=dataset_name if dataset_name else config.db,
            params=hyperparams,
            db_name=db_name,
            workload=workload,
            num_queries=num_training if num_training > 0 else None
        )

    # Model loading logic
    if load_model_id:
        # Load specific model by ID
        print(f"📂 Loading specific model: {load_model_id}")
        mscn_pilot_model: PilotModel = MscnPilotModel.load_model(load_model_id, "mscn")
    elif not enable_training:
        # If not training, try to load best model from MLflow first
        if use_mlflow:
            exp_name = experiment_name if experiment_name else f"mscn_{config.db}"
            best_run = MLflowTracker.get_best_run(
                experiment_name=exp_name,
                metric="test_total_time",
                ascending=True
            )
            if best_run:
                print(f"📊 Loading best model from MLflow: {best_run['run_name']}")
                # Extract model_id from run parameters
                model_id = best_run['params'].get('model_id', None)
                if model_id:
                    mscn_pilot_model = MscnPilotModel.load_model(model_id, "mscn")
                else:
                    print("⚠️  No model_id in MLflow run, creating new model")
                    mscn_pilot_model: PilotModel = MscnPilotModel(model_name, mlflow_tracker=mlflow_tracker, save_to_local=False)
                    mscn_pilot_model._load_model_impl()
            else:
                print("⚠️  No trained models found in MLflow, creating new model")
                mscn_pilot_model: PilotModel = MscnPilotModel(model_name, mlflow_tracker=mlflow_tracker, save_to_local=False)
                mscn_pilot_model._load_model_impl()
        else:
            # Fallback to old registry method
            from pilotscope.ModelRegistry import ModelRegistry
            registry = ModelRegistry()
            best = registry.get_best_model("mscn", test_dataset=dataset_name if dataset_name else config.db)
            if best:
                print(f"📊 Loading best model for {dataset_name if dataset_name else config.db}: {best['model_id']}")
                mscn_pilot_model = MscnPilotModel.load_model(best['model_id'], "mscn")
            else:
                print("⚠️  No trained models found, creating new model")
                mscn_pilot_model: PilotModel = MscnPilotModel(model_name, mlflow_tracker=mlflow_tracker, save_to_local=False)
                mscn_pilot_model._load_model_impl()
    else:
        # Create new model for training
        mscn_pilot_model: PilotModel = MscnPilotModel(model_name, mlflow_tracker=mlflow_tracker, save_to_local=False)
        mscn_pilot_model._load_model_impl()

        # Set training metadata
        hyperparams = {
            "num_epoch": num_epoch,
            "num_training": num_training,
            "num_collection": num_collection,
            "enable_collection": enable_collection,
            "enable_training": enable_training
        }
        mscn_pilot_model.set_training_info(dataset_name if dataset_name else config.db, hyperparams)

    # core
    scheduler: PilotScheduler = SchedulerFactory.create_scheduler(config)

    # register a pretraining model event, which will prepare training set during init.
    if enable_collection or enable_training:
        # Use dataset_name to separate data for different workloads
        data_table = f"mscn_pretraining_{dataset_name if dataset_name else config.db}"
        event = MscnPretrainingModelEvent(config, mscn_pilot_model, data_table,
                                          enable_collection=enable_collection,
                                          enable_training=enable_training,
                                          num_collection=num_collection,
                                          num_training=num_training,
                                          num_epoch=num_epoch,
                                          mlflow_tracker=mlflow_tracker,
                                          dataset_name=dataset_name)
        scheduler.register_events([event])

    # register a card push handler
    scheduler.register_custom_handlers([MscnCardPushHandler(mscn_pilot_model, config)])

    # register required data (execution time collection for test phase)
    test_data_table = "{}_test_data_table".format(model_name)
    scheduler.register_required_data(test_data_table, pull_execution_time=True)

    # Attach model and tracker to scheduler for later access
    scheduler.pilot_model = mscn_pilot_model
    scheduler.mlflow_tracker = mlflow_tracker

    # start
    scheduler.init()
    return scheduler, mlflow_tracker