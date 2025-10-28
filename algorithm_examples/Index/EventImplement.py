import random

from algorithm_examples.Index.index_selection_evaluation.selection.algorithms.extend_algorithm import ExtendAlgorithm
from algorithm_examples.Index.index_selection_evaluation.selection.index_selection_evaluation import to_workload
from algorithm_examples.Index.index_selection_evaluation.selection.workload import Query
from algorithm_examples.utils import to_pilot_index, load_test_sql
from pilotscope.Common.Index import Index as PilotIndex
from pilotscope.DBController.BaseDBController import BaseDBController
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor
from pilotscope.DataManager.DataManager import DataManager
from pilotscope.PilotEvent import PeriodicModelUpdateEvent
from pilotscope.PilotModel import PilotModel


class DbConnector:

    def __init__(self, data_interactor: PilotDataInteractor):
        super().__init__()
        self.data_interactor = data_interactor
        self.db_controller = data_interactor.db_controller

    def get_cost(self, query: Query):
        return self.db_controller.get_estimated_cost(query.text)

    def drop_indexes(self):
        self.db_controller.drop_all_indexes()

    def drop_index(self, index):
        self.db_controller.drop_index(to_pilot_index(index))

    def get_index_byte(self, index):
        return self.db_controller.get_index_byte(index)

    def get_config(self):
        return self.db_controller.config


class IndexPeriodicModelUpdateEvent(PeriodicModelUpdateEvent):

    def __init__(self, config, per_query_count, execute_on_init=False, mlflow_tracker=None):
        super().__init__(config, per_query_count, None)
        self.execute_on_init = execute_on_init
        self.mlflow_tracker = mlflow_tracker
        self.update_count = 0

    def _load_sql(self):
        sqls: list = load_test_sql(self.config.db)
        random.shuffle(sqls)
        if "imdb" in self.config.db:
            sqls = sqls[0:len(sqls) // 2]
        return sqls

    def custom_model_update(self, pilot_model: PilotModel, db_controller: BaseDBController,
                            data_manager: DataManager):
        print("start to update model and set index by Extend algorithm")
        db_controller.drop_all_indexes()
        sqls = self._load_sql()
        workload = to_workload(sqls)
        parameters = {
            "benchmark_name": self.config.db, "budget_MB": 250, "max_index_width": 2
        }
        # only for adapt the origin ML code with minimal change. Actually, the user need not provide a DbConnector.
        connector = DbConnector(PilotDataInteractor(self.config, enable_simulate_index=True))
        algo = ExtendAlgorithm(connector, parameters=parameters)
        print("start to search best indexes")

        import time
        start_time = time.time()
        indexes = algo.calculate_best_indexes(workload)
        optimization_time = time.time() - start_time

        # Log to MLflow
        if self.mlflow_tracker:
            self.mlflow_tracker.log_training_metrics({
                "index_optimization_time_seconds": optimization_time,
                "num_indexes_selected": len(indexes),
                "num_queries": len(sqls)
            }, step=self.update_count)

            # Log index list as artifact (JSON file)
            import mlflow
            index_list = []
            for index in indexes:
                columns = [c.name for c in index.columns]
                index_list.append({
                    "table": index.table().name,
                    "columns": columns,
                    "index_name": index.index_idx(),
                    "estimated_size_bytes": index.estimated_size if hasattr(index, 'estimated_size') else None
                })

            index_config = {
                "indexes": index_list,
                "num_indexes": len(indexes),
                "optimization_time_seconds": optimization_time,
                "num_queries_analyzed": len(sqls),
                "update_iteration": self.update_count
            }
            mlflow.log_dict(index_config, f"index_config_iteration_{self.update_count}.json")

        for index in indexes:
            columns = [c.name for c in index.columns]
            db_controller.create_index(PilotIndex(columns, index.table().name, index.index_idx()))
            print("create index {}".format(index))

        self.update_count += 1
