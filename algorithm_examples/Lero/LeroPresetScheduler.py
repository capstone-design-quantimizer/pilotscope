import sys

from pilotscope.DataManager.DataManager import DataManager

sys.path.append("../")
sys.path.append("../algorithm_examples/Lero/source")

from pilotscope.Factory.SchedulerFactory import SchedulerFactory
from pilotscope.PilotModel import PilotModel
from pilotscope.PilotScheduler import PilotScheduler
from pilotscope.Common.MLflowTracker import MLflowTracker
from algorithm_examples.Lero.EventImplement import LeroPretrainingModelEvent, LeroPeriodicCollectEvent, \
    LeroPeriodicModelUpdateEvent
from algorithm_examples.Lero.LeroParadigmCardAnchorHandler import LeroCardPushHandler
from algorithm_examples.Lero.LeroPilotModel import LeroPilotModel


def get_lero_preset_scheduler(config, enable_collection, enable_training, num_collection = -1, num_training = -1, num_epoch = 100, load_model_id=None, use_mlflow=True, experiment_name=None, dataset_name=None) -> tuple:
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

    model_name = "lero_pair"
    test_data_table = "{}_test_data_table".format(model_name)
    # Use dataset_name to separate data for different workloads
    pretraining_data_table = f"lero_pretraining_{dataset_name if dataset_name else config.db}"

    data_manager = DataManager(config)
    if enable_collection: # if enable_collection, drop old data and collect new data. otherwise use old data to train.
        data_manager.remove_table_and_tracker(pretraining_data_table)

    # Initialize MLflow tracker
    mlflow_tracker = None
    if use_mlflow and enable_training:
        # Use provided experiment name or fallback to default
        exp_name = experiment_name if experiment_name else f"lero_{config.db}"
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
            algo_name="lero",
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
        lero_pilot_model: PilotModel = LeroPilotModel.load_model(load_model_id, "lero")
    elif not enable_training:
        # If not training, try to load best model from MLflow first
        if use_mlflow:
            exp_name = experiment_name if experiment_name else f"lero_{config.db}"
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
                    lero_pilot_model = LeroPilotModel.load_model(model_id, "lero")
                else:
                    print("⚠️  No model_id in MLflow run, creating new model")
                    lero_pilot_model: PilotModel = LeroPilotModel(model_name, mlflow_tracker=mlflow_tracker, save_to_local=False)
                    lero_pilot_model._load_model_impl()
            else:
                print("⚠️  No trained models found in MLflow, creating new model")
                lero_pilot_model: PilotModel = LeroPilotModel(model_name, mlflow_tracker=mlflow_tracker, save_to_local=False)
                lero_pilot_model._load_model_impl()
        else:
            # Fallback to old registry method
            from pilotscope.ModelRegistry import ModelRegistry
            registry = ModelRegistry()
            best = registry.get_best_model("lero", test_dataset=dataset_name if dataset_name else config.db)
            if best:
                print(f"📊 Loading best model for {dataset_name if dataset_name else config.db}: {best['model_id']}")
                lero_pilot_model = LeroPilotModel.load_model(best['model_id'], "lero")
            else:
                print("⚠️  No trained models found, creating new model")
                lero_pilot_model: PilotModel = LeroPilotModel(model_name, mlflow_tracker=mlflow_tracker, save_to_local=False)
                lero_pilot_model._load_model_impl()
    else:
        # Create new model for training
        lero_pilot_model: PilotModel = LeroPilotModel(model_name, mlflow_tracker=mlflow_tracker, save_to_local=False)
        lero_pilot_model._load_model_impl()

        # Set training metadata
        hyperparams = {
            "num_epoch": num_epoch,
            "num_training": num_training,
            "num_collection": num_collection,
            "enable_collection": enable_collection,
            "enable_training": enable_training
        }
        lero_pilot_model.set_training_info(dataset_name if dataset_name else config.db, hyperparams)
    
    lero_handler = LeroCardPushHandler(lero_pilot_model, config)

    # core
    scheduler: PilotScheduler = SchedulerFactory.create_scheduler(config)
    scheduler.register_custom_handlers([lero_handler])
    scheduler.register_required_data(test_data_table, pull_execution_time=True, pull_physical_plan=True)

    # allow to pretrain model
    pretraining_event = LeroPretrainingModelEvent(config, lero_pilot_model, pretraining_data_table,
                                                  enable_collection=enable_collection, enable_training=enable_training, num_collection = num_collection,\
                                                  num_training = num_training, num_epoch = num_epoch,
                                                  mlflow_tracker=mlflow_tracker,
                                                  dataset_name=dataset_name)
    scheduler.register_events([pretraining_event])

    # Attach model and tracker to scheduler for later access
    scheduler.pilot_model = lero_pilot_model
    scheduler.mlflow_tracker = mlflow_tracker

    # start
    scheduler.init()
    return scheduler, mlflow_tracker


def get_lero_dynamic_preset_scheduler(config, dataset_name=None) -> PilotScheduler:
    model_name = "lero_pair"

    import torch
    print(torch.version.cuda)
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")

    model_name = "lero_pair"  # This test can only work when existing a model
    lero_pilot_model: PilotModel = LeroPilotModel(model_name)
    lero_pilot_model.load_model()
    lero_handler = LeroCardPushHandler(lero_pilot_model, config)

    # core
    training_data_save_table = "{}_data_table".format(model_name)
    scheduler: PilotScheduler = SchedulerFactory.create_scheduler(config)
    scheduler.register_custom_handlers([lero_handler])
    scheduler.register_required_data(training_data_save_table, pull_execution_time=True, pull_physical_plan=True)

    # dynamically collect data
    dynamic_training_data_save_table = "{}_period_training_data_table".format(model_name)
    period_collect_event = LeroPeriodicCollectEvent(dynamic_training_data_save_table, config, 100, dataset_name=dataset_name)
    # dynamically update model
    period_train_event = LeroPeriodicModelUpdateEvent(dynamic_training_data_save_table, config, 100,
                                                      lero_pilot_model)
    scheduler.register_events([period_collect_event, period_train_event])

    # start
    scheduler.init()
    return scheduler
