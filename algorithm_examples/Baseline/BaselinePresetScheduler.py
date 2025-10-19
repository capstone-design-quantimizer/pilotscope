"""
Baseline Preset Scheduler

No AI algorithm, just direct database execution for performance comparison.
"""

import sys
sys.path.append("../")

from pilotscope.Factory.SchedulerFactory import SchedulerFactory
from pilotscope.PilotScheduler import PilotScheduler
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor


def get_baseline_preset_scheduler(config, **kwargs) -> PilotScheduler:
    """
    Create baseline scheduler (no AI, direct execution)
    
    Args:
        config: PilotConfig instance
        **kwargs: Ignored (for compatibility with other schedulers)
    
    Returns:
        PilotScheduler with no AI algorithms
    """
    # Create basic scheduler without any AI handlers
    scheduler: PilotScheduler = SchedulerFactory.create_scheduler(config)
    
    # Register minimal data collection
    test_data_save_table = "baseline_data_table"
    scheduler.register_required_data(test_data_save_table, pull_execution_time=True)
    
    # No AI handlers, no events - just pure database execution
    # This provides the baseline performance for comparison
    
    # start
    scheduler.init()
    return scheduler

