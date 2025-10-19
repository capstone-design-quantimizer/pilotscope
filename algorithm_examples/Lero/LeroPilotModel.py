import os
import sys

from algorithm_examples.Lero.source.model import LeroModelPairWise
from pilotscope.DataManager.DataManager import DataManager
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from pilotscope.EnhancedPilotModel import EnhancedPilotModel


class LeroPilotModel(EnhancedPilotModel):
    """
    Lero model with enhanced versioning and metadata
    
    Usage:
        # Create new model with training info
        model = LeroPilotModel()
        model.set_training_info("production", {"num_epoch": 50, ...})
        model.train(...)
        model.save_model()
        
        # Load specific model
        model = LeroPilotModel.load_model("lero_20241019_103000", "lero")
        
        # Add test results
        model.add_test_result("stats_tiny", 80, {"total_time": 35.2})
        model.save_model()
    """

    def __init__(self, model_name="lero"):
        model_save_dir = os.path.join(
            os.path.dirname(__file__),
            "../ExampleData/Lero/Model"
        )
        super().__init__(model_name, "lero", model_save_dir)

    def train(self, data_manager: DataManager):
        print("enter LeroPilotModel.train")

    def update(self, data_manager: DataManager):
        print("enter LeroPilotModel.update")

    def _save_model_impl(self):
        """Save Lero model"""
        self.model.save(self.model_path)

    def _load_model_impl(self):
        """Load Lero model"""
        try:
            lero_model = LeroModelPairWise(None)
            lero_model.load(self.model_path)
            print(f"   Lero model loaded from: {self.model_path}")
        except FileNotFoundError:
            print("   ⚠️  Lero model file not found, creating new model")
            lero_model = LeroModelPairWise(None)
        self.model = lero_model
