from pandas import DataFrame

from algorithm_examples.Mscn.source.mscn_model import MscnModel
from algorithm_examples.Mscn.source.mscn_utils import load_tokens, parse_queries, load_schema
from algorithm_examples.utils import load_training_sql
from pilotscope.DBController.BaseDBController import BaseDBController
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor
from pilotscope.DataManager.DataManager import DataManager
from pilotscope.PilotConfig import PilotConfig
from pilotscope.PilotEvent import PretrainingModelEvent
from pilotscope.PilotModel import PilotModel
from pilotscope.PilotTransData import PilotTransData


class MscnPretrainingModelEvent(PretrainingModelEvent):

    def __init__(self, config: PilotConfig, bind_pilot_model: PilotModel, data_saving_table, enable_collection=True,
                 enable_training=True, training_data_file=None, num_collection = -1, num_training = -1, num_epoch = 100,
                 mlflow_tracker=None, dataset_name=None):
        super().__init__(config, bind_pilot_model, data_saving_table, enable_collection, enable_training)
        self.sqls = []
        self.config.once_request_timeout = 60
        self.config.sql_execution_timeout = 60
        self.pilot_data_interactor = PilotDataInteractor(self.config)
        self.training_data_file = training_data_file
        self.num_collection = num_collection
        self.num_training = num_training
        self.num_epoch = num_epoch
        self.mlflow_tracker = mlflow_tracker
        self.dataset_name = dataset_name if dataset_name else config.db

    def iterative_data_collection(self, db_controller: BaseDBController, train_data_manager: DataManager):
        print("start to collect data for MSCN algorithms")
        self.sqls = load_training_sql(self.dataset_name)
        if self.num_collection > 0:
            train_sqls = self.sqls[:self.num_collection]
        else:
            train_sqls = self.sqls
        column_2_value_list = []
        skipped_count = 0
        debug_printed = False  # Global flag to print first successful collection only

        for i, sql in enumerate(train_sqls):
            # print per 10
            if i % 10 == 0:
                print("current is the {}-th sql, total is {}. (print per 10)".format(i, len(train_sqls)))

            self.pilot_data_interactor.pull_subquery_card()
            data: PilotTransData = self.pilot_data_interactor.execute(sql)

            import re
            # Debug stats for collection
            total_subqueries = len(data.subquery_2_card)
            skipped_correlated = 0
            skipped_empty = 0
            collected = 0

            for sub_sql in data.subquery_2_card.keys():
                # Skip correlated subqueries (contain column placeholders like /* sdi.ticker */)
                # Check for pattern: /* <identifier> */ (not just table comments)
                # Table comments are like /* (table_name alias) */, correlated refs are like /* alias.column */
                if re.search(r'/\*\s*\w+\.\w+\s*\*/', sub_sql):
                    # This is a correlated subquery reference, skip it
                    skipped_correlated += 1
                    continue

                # Skip empty/malformed subqueries (e.g., "SELECT COUNT(*) FROM ;")
                # These are generated when correlated subqueries are extracted
                if re.search(r'FROM\s*[;)]', sub_sql):
                    # Empty FROM clause, skip it
                    skipped_empty += 1
                    continue

                # Execute the subquery (which is "SELECT COUNT(*) FROM ... WHERE ...") to get cardinality
                self.pilot_data_interactor.pull_record()
                sub_data: PilotTransData = self.pilot_data_interactor.execute(sub_sql)

                # Debug: print first successful subquery details (ANY query, not just i==0)
                if not debug_printed and sub_data.records is not None:
                    print(f"\n=== DEBUG: First successful collection (query {i}) ===")
                    print(f"Subquery SQL: {sub_sql[:300]}")
                    print(f"records type: {type(sub_data.records)}")
                    if hasattr(sub_data.records, 'values'):
                        print(f"records.values shape: {sub_data.records.values.shape if hasattr(sub_data.records.values, 'shape') else 'NO SHAPE'}")
                        print(f"records.values: {sub_data.records.values}")
                        if len(sub_data.records.values) > 0 and len(sub_data.records.values[0]) > 0:
                            print(f"records.values[0][0]: {sub_data.records.values[0][0]} (type: {type(sub_data.records.values[0][0])})")
                    else:
                        print(f"records has NO 'values' attribute!")
                    print("=" * 60 + "\n")
                    debug_printed = True

                if (not sub_data.records is None):
                    card = int(sub_data.records.values[0][0])

                    # Debug: print first 5 NON-ZERO cardinalities to verify data exists
                    if card > 0 and len([c for c in column_2_value_list if c['card'] > 0]) < 5:
                        print(f"🎯 Non-zero cardinality found! Query {i}, card={card}")
                        print(f"   SQL: {sub_sql[:150]}...")

                    column_2_value = {"query": sub_sql, "card": card}
                    column_2_value_list.append(column_2_value)
                    collected += 1

            # Print stats every 10 queries
            if (i + 1) % 10 == 0:
                print(f"Query {i+1}/{len(self.sqls)}: subqueries={total_subqueries}, collected={collected}, skipped_correlated={skipped_correlated}, skipped_empty={skipped_empty}")

        return column_2_value_list, True

    def custom_model_training(self, bind_pilot_model, db_controller: BaseDBController,
                              data_manager: DataManager):
        print("start to train model in {}".format(MscnPretrainingModelEvent))
        if not self.training_data_file is None:
            tokens, labels = load_tokens(self.training_data_file, self.training_data_file + ".token")
            schema = load_schema(self.pilot_data_interactor.db_controller)
            model = MscnModel()
            model.fit(tokens, labels + 1, schema, mlflow_tracker=self.mlflow_tracker)
        else:
            # Check if data exists
            try:
                data: DataFrame = data_manager.read_all(self.data_saving_table)
            except Exception as e:
                print(f"\n❌ Error: No collected data found in table '{self.data_saving_table}'")
                print(f"   Please run data collection first:")
                print(f"   python unified_test.py --algo mscn --db {self.config.db} \\")
                print(f"       --collection-size -1 --no-training")
                print(f"\n   Error details: {e}")
                raise RuntimeError(f"No training data available. Run collection first.") from e

            # Check if data is empty
            if data.shape[0] == 0:
                print(f"\n❌ Error: Table '{self.data_saving_table}' exists but contains no data")
                print(f"   Please collect data first:")
                print(f"   python unified_test.py --algo mscn --db {self.config.db} \\")
                print(f"       --collection-size -1 --no-training")
                raise RuntimeError(f"No training data available (0 rows in table).")

            print(f"📂 Loaded {data.shape[0]} collected sql-card pairs from '{self.data_saving_table}' (dataset: {self.dataset_name})")

            if self.num_training > 0:
                if self.num_training > data.shape[0]:
                    print(f"⚠️  Warning: Requested training size ({self.num_training}) > available data ({data.shape[0]})")
                    print(f"   Using all {data.shape[0]} available pairs")
                    self.num_training = data.shape[0]
                data = data[:self.num_training]

            print(f"🎓 Training MSCN on {data.shape[0]} sql-card pairs")

            # Debug: print cardinality distribution
            import numpy as np
            cards = data["card"].values
            print(f"\n📊 Cardinality Distribution:")
            print(f"   Min: {np.min(cards)}, Max: {np.max(cards)}, Mean: {np.mean(cards):.2f}, Median: {np.median(cards):.2f}")
            print(f"   Unique values: {len(np.unique(cards))}")
            print(f"   Card=0: {np.sum(cards == 0)}, Card=1: {np.sum(cards == 1)}, Card>1: {np.sum(cards > 1)}")
            if len(np.unique(cards)) <= 20:
                print(f"   All unique cards: {sorted(np.unique(cards).tolist())}")

            # Debug: print sample queries with different cardinalities
            if np.sum(cards > 0) > 0:
                print(f"\n📝 Sample queries with card > 0:")
                non_zero_data = data[data["card"] > 0]
                for idx in range(min(3, len(non_zero_data))):
                    row = non_zero_data.iloc[idx]
                    print(f"   Card={row['card']}: {row['query'][:100]}...")
            else:
                print(f"\n⚠️  WARNING: All {len(cards)} cardinalities are 0!")
                print(f"   Sample queries:")
                for idx in range(min(3, len(data))):
                    print(f"   {data.iloc[idx]['query'][:100]}...")
            print()

            tables, joins, predicates = parse_queries(data["query"].values)
            schema = load_schema(self.pilot_data_interactor.db_controller)
            model = MscnModel()
            # Mscn can only handler card that is larger than 0, so we add 1 to all cards. In prediction we minus it by 1.
            model.fit((tables, joins, predicates), data["card"].values + 1, schema, num_epochs = self.num_epoch, mlflow_tracker=self.mlflow_tracker)
        return model
