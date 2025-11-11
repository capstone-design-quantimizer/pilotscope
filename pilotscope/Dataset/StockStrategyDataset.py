from pilotscope.Dataset.BaseDataset import BaseDataset
from pilotscope.DBController import BaseDBController
import os

from pilotscope.Factory.DBControllerFectory import DBControllerFactory
from pilotscope.PilotConfig import PilotConfig
from pilotscope.PilotEnum import DatabaseEnum


class StockStrategyDataset(BaseDataset):
    """
    Base dataset for stock trading strategy analysis.
    Uses default database and default workload (value_investing).

    IMPORTANT: All workloads share the SAME database 'stock_strategy'.
    Only the query files differ per workload.

    Correct Usage:
        python unified_test.py --algo baseline --db stock_strategy
        # Uses default workload (value_investing)

        python unified_test.py --algo baseline --db stock_strategy --workload momentum_investing
        python unified_test.py --algo baseline --db stock_strategy --workload ml_hybrid
        # Uses custom workloads on the SAME database

    WRONG Usage (will fail):
        python unified_test.py --algo baseline --db stock_strategy_value_investing
        # This tries to connect to a non-existent database!
    """
    data_location_dict = {DatabaseEnum.POSTGRESQL: "stock_strategy.sql",
                          DatabaseEnum.SPARK: None}
    sub_dir = "StockStrategy"
    train_sql_file = "stock_strategy_value_investing_train.txt"  # default workload
    test_sql_file = "stock_strategy_value_investing_test.txt"
    now_path = os.path.join(os.path.dirname(__file__), sub_dir)
    file_db_type = DatabaseEnum.POSTGRESQL

    def __init__(self, use_db_type: DatabaseEnum, created_db_name="stock_strategy", data_dir=None) -> None:
        super().__init__(use_db_type, created_db_name, data_dir)
        self.data_file = self.data_location_dict[use_db_type]

    def load_to_db(self, config: PilotConfig):
        """Load the database dump file into PostgreSQL

        Note: Requires plain text SQL format dump (--format=plain)
        Custom format dumps have version compatibility issues.
        """
        config.db = self.created_db_name
        db_controller = DBControllerFactory.get_db_controller(config)
        self._load_dump(os.path.join(self.now_path, self.data_file), db_controller)


# Workload-specific datasets (same DB, different query files)

class StockStrategyValueInvestingDataset(StockStrategyDataset):
    """
    Stock strategy dataset with Value Investing workload.
    Focuses on fundamental value metrics (P/E, P/B, dividend yield).

    Usage:
        --db stock_strategy  # This is the default, uses value_investing queries
        --db stock_strategy --workload value_investing  # Explicit

    Database: 'stock_strategy' (shared with all workloads)
    """
    train_sql_file = "stock_strategy_value_investing_train.txt"
    test_sql_file = "stock_strategy_value_investing_test.txt"

    def __init__(self, use_db_type: DatabaseEnum, created_db_name="stock_strategy", data_dir=None) -> None:
        super().__init__(use_db_type, created_db_name, data_dir)


class StockStrategyMomentumInvestingDataset(StockStrategyDataset):
    """
    Stock strategy dataset with Momentum Investing workload.
    Focuses on price momentum, RSI, and moving averages.

    Usage:
        --db stock_strategy --workload momentum_investing

    Database: 'stock_strategy' (shared with all workloads)
    """
    train_sql_file = "stock_strategy_momentum_investing_train.txt"
    test_sql_file = "stock_strategy_momentum_investing_test.txt"

    def __init__(self, use_db_type: DatabaseEnum, created_db_name="stock_strategy", data_dir=None) -> None:
        super().__init__(use_db_type, created_db_name, data_dir)


class StockStrategyMLHybridDataset(StockStrategyDataset):
    """
    Stock strategy dataset with ML Hybrid workload.
    Combines ML model predictions with traditional metrics.

    Usage:
        --db stock_strategy --workload ml_hybrid

    Database: 'stock_strategy' (shared with all workloads)
    """
    train_sql_file = "stock_strategy_ml_hybrid_train.txt"
    test_sql_file = "stock_strategy_ml_hybrid_test.txt"

    def __init__(self, use_db_type: DatabaseEnum, created_db_name="stock_strategy", data_dir=None) -> None:
        super().__init__(use_db_type, created_db_name, data_dir)