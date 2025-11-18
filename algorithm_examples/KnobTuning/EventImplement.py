import random
from functools import partial
from pathlib import Path

import pandas as pd

from pilotscope.DBController.BaseDBController import BaseDBController
from pilotscope.DataManager.DataManager import DataManager
from pilotscope.PilotEvent import PeriodicModelUpdateEvent
from pilotscope.PilotModel import PilotModel

pd.set_option('display.max_columns', None)
import numpy as np
import sys

sys.path.append("../algorithm_examples/KnobTuning/llamatune")
from config import config
from executors.executor import ExecutorFactory
from optimizer import get_smac_optimizer
from space import ConfigSpaceGenerator
from storage import StorageFactory
import run_smac
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def llamatune(conf):
    config.update_from_file(conf["conf_filepath"])
    config.seed = conf["seed"]
    ### number of DBMS internal metrics being sampled
    config.num_dbms_metrics = 60

    # Set global random state
    random.seed(config.seed)
    np.random.seed(config.seed)

    # init input & output space
    spaces = ConfigSpaceGenerator.from_config(config)
    target_metric = spaces.target_metric

    # init storage class
    perf_label = 'Throughput' if target_metric == 'throughput' else 'Latency'
    columns = ['Iteration', perf_label, 'Optimum', 'Runtime']

    benchmark, workload = (
        config['benchmark_info']['name'], config['benchmark_info']['workload'])

    inner_path = Path(f'{benchmark}.{workload}') / f'seed{config.seed}'
    storage = StorageFactory.from_config(config, columns=columns, inner_path=inner_path)

    # store dbms & benchmark info in experiment state object
    benchmark_info_config = config.benchmark_info
    dbms_info_config = config.dbms_info
    results_path = Path(config['storage']['outdir']) / inner_path

    # init executor
    executor = ExecutorFactory.from_config(config, spaces, storage, parse_metrics=(conf["optimizer"] == "ddpg"),
                                           num_dbms_metrics=config.num_dbms_metrics)

    exp_state = run_smac.ExperimentState(
        dbms_info_config, benchmark_info_config, results_path, target_metric)
    optimizer = get_smac_optimizer(config, spaces,
                                   partial(run_smac.evaluate_dbms_conf, spaces, executor, storage, columns),
                                   exp_state)

    # evaluate on default config
    default_config = spaces.get_default_configuration()

    logger.info('Evaluating Default Configuration')
    logger.debug(default_config)

    perf = run_smac.evaluate_dbms_conf(spaces, executor, storage, columns, default_config, state=exp_state)
    perf = perf if exp_state.minimize else -perf
    assert perf >= 0, \
        f'Performance should not be negative: perf={perf}, metric={target_metric}'

    # set starting point for worse performance
    exp_state.worse_perf = perf * 4 if exp_state.minimize else perf / 4

    optimizer.optimize()

    # Print final stats
    logger.info(f'\nBest Configuration:\n{exp_state.best_conf}')
    if target_metric == 'throughput':
        logger.info(f'Throughput: {exp_state.best_perf} ops/sec')
    else:
        logger.info(f'95-th Latency: {exp_state.best_perf} milliseconds')
    logger.info(f'Saved @ {storage.outdir}')
    return exp_state


class KnobPeriodicModelUpdateEvent(PeriodicModelUpdateEvent):
    def __init__(self, config, per_query_count, llamatune_config_file, execute_on_init=True,
                 optimizer_type="smac", mlflow_tracker=None, dataset_name=None):
        super().__init__(config, per_query_count, execute_on_init=execute_on_init)
        self.optimizer_type = optimizer_type
        self.llamatune_config_file = llamatune_config_file
        self.mlflow_tracker = mlflow_tracker
        self.update_count = 0
        self.dataset_name = dataset_name if dataset_name else config.db

    def custom_model_update(self, pilot_model: PilotModel, db_controller: BaseDBController,
                            data_manager: DataManager):
        db_controller.recover_config()
        db_controller.restart()

        # Create temporary config file with dynamic SQL path
        import tempfile
        import configparser
        from algorithm_examples.utils import load_training_sql

        # Load training SQLs for the dataset
        train_sqls = load_training_sql(self.dataset_name)

        # Create temporary SQL file
        temp_sql_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        for sql in train_sqls:
            temp_sql_file.write(sql + '\n')
        temp_sql_file.close()

        # Read template config
        template_config = configparser.ConfigParser()
        template_config.read(self.llamatune_config_file)

        # Update executor section with dynamic paths
        template_config.set('executor', 'sqls_file_path', temp_sql_file.name)
        template_config.set('executor', 'db_name', self.config.db)

        # Create temporary config file
        temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
        template_config.write(temp_config_file)
        temp_config_file.close()

        conf = {
            "conf_filepath": temp_config_file.name,
            "seed": int(time.time()),
            "optimizer": self.optimizer_type
        }

        start_time = time.time()
        exp_state = llamatune(conf)
        optimization_time = time.time() - start_time

        # Cleanup temporary files
        import os
        try:
            os.unlink(temp_sql_file.name)
            os.unlink(temp_config_file.name)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")

        # Log to MLflow
        if self.mlflow_tracker:
            self.mlflow_tracker.log_training_metrics({
                "knob_optimization_time_seconds": optimization_time,
                "best_performance": exp_state.best_perf,
                "num_knobs_tuned": len(exp_state.best_conf),
                "target_metric": exp_state.target_metric
            }, step=self.update_count)

            # Log knob configuration as artifact (JSON file)
            import json
            import mlflow
            knob_config = {
                "best_configuration": dict(exp_state.best_conf),
                "best_performance": exp_state.best_perf,
                "target_metric": exp_state.target_metric,
                "optimization_time_seconds": optimization_time,
                "update_iteration": self.update_count
            }
            mlflow.log_dict(knob_config, f"knob_config_iteration_{self.update_count}.json")

        db_controller.write_knob_to_file(dict(exp_state.best_conf))
        db_controller.restart()

        self.update_count += 1
