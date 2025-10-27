import sys

sys.path.append("../")

from pilotscope.Factory.SchedulerFactory import SchedulerFactory
from pilotscope.PilotModel import PilotModel
from pilotscope.PilotScheduler import PilotScheduler
from pilotscope.Common.MLflowTracker import MLflowTracker
from algorithm_examples.Mscn.EventImplement import MscnPretrainingModelEvent
from algorithm_examples.Mscn.MscnParadigmCardAnchorHandler import MscnCardPushHandler
from algorithm_examples.Mscn.MscnPilotModel import MscnPilotModel


def get_mscn_preset_scheduler(config, enable_collection, enable_training, num_collection = -1, num_training = -1, num_epoch = 100, load_model_id=None, use_mlflow=True) -> tuple:
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
        mlflow_tracker = MLflowTracker(experiment_name=f"mscn_{config.db}")

        # Start MLflow run for training
        hyperparams = {
            "num_epoch": num_epoch,
            "num_training": num_training,
            "num_collection": num_collection,
            "enable_collection": enable_collection,
            "enable_training": enable_training
        }
        mlflow_tracker.start_training(
            algo_name="mscn",
            dataset=config.db,
            params=hyperparams
        )

    # Model loading logic
    if load_model_id:
        # Load specific model by ID
        print(f"📂 Loading specific model: {load_model_id}")
        mscn_pilot_model: PilotModel = MscnPilotModel.load_model(load_model_id, "mscn")
    elif not enable_training:
        # If not training, try to load best model from MLflow first
        if use_mlflow:
            best_run = MLflowTracker.get_best_run(
                experiment_name=f"mscn_{config.db}",
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
                    mscn_pilot_model: PilotModel = MscnPilotModel(model_name)
                    mscn_pilot_model._load_model_impl()
            else:
                print("⚠️  No trained models found in MLflow, creating new model")
                mscn_pilot_model: PilotModel = MscnPilotModel(model_name)
                mscn_pilot_model._load_model_impl()
        else:
            # Fallback to old registry method
            from pilotscope.ModelRegistry import ModelRegistry
            registry = ModelRegistry()
            best = registry.get_best_model("mscn", test_dataset=config.db)
            if best:
                print(f"📊 Loading best model for {config.db}: {best['model_id']}")
                mscn_pilot_model = MscnPilotModel.load_model(best['model_id'], "mscn")
            else:
                print("⚠️  No trained models found, creating new model")
                mscn_pilot_model: PilotModel = MscnPilotModel(model_name)
                mscn_pilot_model._load_model_impl()
    else:
        # Create new model for training
        mscn_pilot_model: PilotModel = MscnPilotModel(model_name)
        mscn_pilot_model._load_model_impl()

        # Set training metadata
        hyperparams = {
            "num_epoch": num_epoch,
            "num_training": num_training,
            "num_collection": num_collection,
            "enable_collection": enable_collection,
            "enable_training": enable_training
        }
        mscn_pilot_model.set_training_info(config.db, hyperparams)
    
    mscn_handler = MscnCardPushHandler(mscn_pilot_model, config)

    # core
    test_data_save_table = "{}_data_table".format(model_name)
    pretrain_data_save_table = "{}_pretrain_data_table".format(model_name)
    scheduler: PilotScheduler = SchedulerFactory.create_scheduler(config)
    scheduler.register_custom_handlers([mscn_handler])
    scheduler.register_required_data(test_data_save_table, pull_execution_time=True)
    # allow to pretrain model
    pretraining_event = MscnPretrainingModelEvent(config, mscn_pilot_model, pretrain_data_save_table,
                                                  enable_collection=enable_collection,
                                                  enable_training=enable_training,
                                                  training_data_file=None, num_collection = num_collection,
                                                  num_training = num_training, num_epoch = num_epoch,
                                                  mlflow_tracker=mlflow_tracker)
    # If training_data_file is None, training data will be collected in this run
    scheduler.register_events([pretraining_event])

    # Attach model and tracker to scheduler for later access
    scheduler.pilot_model = mscn_pilot_model
    scheduler.mlflow_tracker = mlflow_tracker

    # start
    scheduler.init()
    return scheduler, mlflow_tracker
