from pilotscope.Dataset.BaseDataset import BaseDataset
from pilotscope.DBController import BaseDBController
import os

from pilotscope.Factory.DBControllerFectory import DBControllerFactory
from pilotscope.PilotConfig import PilotConfig
from pilotscope.PilotEnum import DatabaseEnum


class StatsTinyCustomDataset(BaseDataset):
    """
    Custom StatsTiny dataset with workload queries from stats_pilotscope_input.txt.

    This dataset contains 5000 workload queries generated using various strategies
    (Post Interaction Deep Dive, User Activity Analysis, etc.) and is split into:
    - Training set: 4000 queries (80%)
    - Test set: 1000 queries (20%)

    The dataset uses the same schema and database as StatsTiny (stats_tiny DB).
    Only the workload queries are different from the original StatsTiny dataset.
    """
    data_location_dict = {DatabaseEnum.POSTGRESQL: "stats_tiny.sql",
                          DatabaseEnum.SPARK: None}
    sub_dir = "StatsTiny"
    train_sql_file = "stats_custom_train.txt"
    test_sql_file = "stats_custom_test.txt"
    now_path = os.path.join(os.path.dirname(__file__), sub_dir)
    file_db_type = DatabaseEnum.POSTGRESQL

    def __init__(self, use_db_type: DatabaseEnum, created_db_name="stats_tiny", data_dir=None) -> None:
        super().__init__(use_db_type, created_db_name, data_dir)
        self.data_file = self.data_location_dict[use_db_type]

    def load_to_db(self, config: PilotConfig):  # Overload
        """
        Load StatsTiny schema to the database.

        Note: This uses the same schema as StatsTiny dataset.
        Only the workload queries are different.
        """
        config.db = self.created_db_name
        db_controller = DBControllerFactory.get_db_controller(config)
        self._load_dump(os.path.join(self.now_path, self.data_file), db_controller)