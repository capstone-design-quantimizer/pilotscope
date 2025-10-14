import sys

sys.path.append("../")

import unittest

from pilotscope.Common.Util import pilotscope_exit
from pilotscope.Common.Drawer import Drawer
from pilotscope.Common.TimeStatistic import TimeStatistic
from pilotscope.PilotConfig import PilotConfig, PostgreSQLConfig
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor
from algorithm_examples.utils import load_test_sql, save_test_result
from algorithm_examples.ExampleConfig import get_time_statistic_img_path


class BaselineTest(unittest.TestCase):
    def setUp(self):
        self.config: PilotConfig = PostgreSQLConfig()
        self.config.db = "stats_tiny"
        self.algo = "baseline"

    def test_baseline(self):
        try:
            config = self.config
            # Create data interactor without any push handlers (no AI algorithms)
            data_interactor = PilotDataInteractor(config)
            
            print("start to test sql")
            sqls = load_test_sql(config.db)
            for i, sql in enumerate(sqls):
                if i % 10 == 0:
                    print("current is the {}-th sql, and total is {}".format(i, len(sqls)))
                TimeStatistic.start('Baseline')
                data_interactor.execute(sql)
                TimeStatistic.end('Baseline')
            
            name_2_value = TimeStatistic.get_sum_data()
            
            # Save results in JSON format for easy comparison
            save_test_result(self.algo, self.config.db)
            
            # Also save visualization
            Drawer.draw_bar(name_2_value, get_time_statistic_img_path(self.algo, self.config.db), is_rotation=False)
        finally:
            pilotscope_exit()


if __name__ == '__main__':
    unittest.main()
