import os
import sys

from pilotscope.DataManager.DataManager import DataManager
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from pilotscope.EnhancedPilotModel import EnhancedPilotModel
from algorithm_examples.Mscn.source.mscn_model import MscnModel


class MscnPilotModel(EnhancedPilotModel):
    """
    MSCN model with enhanced versioning and metadata
    
    Usage:
        # Create new model with training info
        model = MscnPilotModel()
        model.set_training_info("production", {"num_epoch": 100, ...})
        model.train(...)
        model.save_model()
        
        # Load specific model
        model = MscnPilotModel.load_model("mscn_20241019_103000", "mscn")
        
        # Add test results
        model.add_test_result("stats_tiny", 80, {"total_time": 35.2})
        model.save_model()
    """

    def __init__(self, model_name="mscn", mlflow_tracker=None, save_to_local=False):
        model_save_dir = os.path.join(
            os.path.dirname(__file__),
            "../ExampleData/Mscn/Model"
        )
        super().__init__(model_name, "mscn", model_save_dir, mlflow_tracker, save_to_local)

    def _save_model_impl(self):
        """Save MSCN model"""
        self.model.save(self.model_path)

    def _load_model_impl(self):
        """Load MSCN model"""
        try:
            model = MscnModel()
            model.load(self.model_path)
            print(f"   MSCN model loaded from: {self.model_path}")
        except:
            print("   ⚠️  MSCN model file not found, creating new model")
            model = MscnModel()
        self.model = model
