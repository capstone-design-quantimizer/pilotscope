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
        PilotScheduler with no AI handlers (for fair comparison)
    """
    # Create scheduler without any AI handlers
    scheduler: PilotScheduler = SchedulerFactory.create_scheduler(config)
    
    # Register data collection for fair comparison with AI algorithms
    # Collect execution time to measure actual DB query performance
    test_data_save_table = "baseline_data_table"
    scheduler.register_required_data(test_data_save_table, pull_execution_time=True)
    
    # No AI handlers - this is the key difference from MSCN/Lero
    # No events, no card estimation, no query optimization
    # Just pure database execution through PilotScope infrastructure
    
    # Start the scheduler
    scheduler.init()
    return scheduler

