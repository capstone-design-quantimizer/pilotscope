"""
Enhanced PilotModel with timestamp-based versioning and rich metadata

Features:
- Timestamp-based model naming (no name conflicts)
- Rich metadata tracking (training, testing, performance)
- Support for separate training and testing
- Multiple dataset testing on same model
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from pilotscope.PilotModel import PilotModel


class EnhancedPilotModel(PilotModel):
    """
    Enhanced model management with timestamp and metadata
    
    Usage:
        # Training
        model = MscnPilotModel()
        model.set_training_info("production", {"num_epoch": 100, ...})
        model.train(...)
        model.save_model()
        
        # Testing on multiple datasets
        model.add_test_result("stats_tiny", 80, {"total_time": 35.2})
        model.add_test_result("imdb", 100, {"total_time": 120.5})
        model.save_model()  # Update metadata
        
        # Loading
        model = MscnPilotModel.load_model("mscn_20241019_103000", "mscn")
    """
    
    def __init__(self, model_name: str, algorithm_type: str,
                 model_save_dir: str = None, mlflow_tracker=None,
                 save_to_local: bool = False):
        """
        Args:
            model_name: Base model name (e.g., "mscn")
            algorithm_type: Algorithm type for directory organization
            model_save_dir: Custom save directory (optional, for local storage)
            mlflow_tracker: MLflowTracker instance (optional, for MLflow storage)
            save_to_local: Whether to save to local files (default: False, MLflow only)
        """
        super().__init__(model_name)
        self.algorithm_type = algorithm_type
        self.mlflow_tracker = mlflow_tracker
        self.save_to_local = save_to_local

        # Default save directory (for local storage if enabled)
        if model_save_dir is None:
            self.model_save_dir = os.path.join(
                os.path.dirname(__file__),
                f"../algorithm_examples/ExampleData/{algorithm_type.capitalize()}/Model"
            )
        else:
            self.model_save_dir = model_save_dir

        # Create timestamp-based ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_id = f"{model_name}_{timestamp}"
        self.model_path = os.path.join(self.model_save_dir, self.model_id)
        self.metadata_path = self.model_path + ".json"

        # Initialize metadata
        self.metadata = {
            "model_id": self.model_id,
            "algorithm": self.algorithm_type,
            "model_path": self.model_path,
            "training": None,
            "testing": [],
            "tags": [],
            "notes": "",
            "created_at": datetime.now().isoformat()
        }
    
    def set_training_info(self, dataset: str, hyperparams: Dict, 
                         num_queries: int = None):
        """
        Set training information
        
        Args:
            dataset: Training dataset name
            hyperparams: Training hyperparameters
            num_queries: Number of queries used for training
        """
        self.metadata["training"] = {
            "enabled": True,
            "dataset": dataset,
            "num_queries": num_queries,
            "hyperparams": hyperparams,
            "trained_at": None,  # Will be set during save
            "training_time": None  # Will be set during save
        }
    
    def add_test_result(self, dataset: str, num_queries: int, 
                       performance: Dict):
        """
        Add test result (can be called multiple times for different datasets)
        
        Args:
            dataset: Test dataset name
            num_queries: Number of test queries
            performance: Performance metrics dict
                Example: {
                    "total_time": 45.23,
                    "average_time": 0.45,
                    "median_time": 0.38,
                    "p95_time": 1.2
                }
        """
        test_result = {
            "dataset": dataset,
            "num_queries": num_queries,
            "tested_at": datetime.now().isoformat(),
            "performance": performance
        }
        self.metadata["testing"].append(test_result)
    
    def add_tags(self, *tags):
        """
        Add tags to model
        
        Args:
            *tags: Tags to add (e.g., "production", "best", "experiment")
        """
        for tag in tags:
            if tag not in self.metadata["tags"]:
                self.metadata["tags"].append(tag)
    
    def set_notes(self, notes: str):
        """Set notes for this model"""
        self.metadata["notes"] = notes
    
    def save_model(self):
        """
        Save model and metadata

        Priority:
        1. MLflow (if tracker provided) - PRIMARY
        2. Local files (if save_to_local=True) - OPTIONAL/LEGACY

        This will:
        1. Call _save_model_impl() to save actual model to local path
        2. Upload to MLflow if tracker is available
        3. Optionally keep local copy if save_to_local=True
        4. Update training time and save metadata
        """
        # Ensure directory exists (needed for temporary save)
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Save actual model to local path (implemented by subclass)
        # This is needed as a temporary file even for MLflow-only mode
        self._save_model_impl()

        # Update training timestamp if this is first save after training
        if self.metadata["training"] and not self.metadata["training"]["trained_at"]:
            self.metadata["training"]["trained_at"] = datetime.now().isoformat()

        # Save metadata to local file (temporary or permanent)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        # Primary: Save to MLflow
        if self.mlflow_tracker:
            success = self.mlflow_tracker.save_model_artifact(
                self.model_path,
                model_id=self.model_id,
                metadata=self.metadata
            )
            if success:
                print(f"✅ Model saved to MLflow: {self.model_id}")
                print(f"   Run ID: {self.mlflow_tracker.run_id}")

                # If MLflow save succeeded and local storage is disabled, clean up
                if not self.save_to_local:
                    try:
                        import shutil
                        if os.path.exists(self.model_path):
                            if os.path.isfile(self.model_path):
                                os.remove(self.model_path)
                            else:
                                shutil.rmtree(self.model_path)
                        if os.path.exists(self.metadata_path):
                            os.remove(self.metadata_path)
                        print(f"   Local files cleaned up (MLflow-only mode)")
                    except Exception as e:
                        print(f"   ⚠️  Warning: Could not clean up local files: {e}")
        else:
            print(f"⚠️  No MLflow tracker - saving to local only")

        # Secondary: Keep local copy if requested
        if self.save_to_local or not self.mlflow_tracker:
            print(f"✅ Model saved locally: {self.model_path}")
            print(f"✅ Metadata saved locally: {self.metadata_path}")
    
    def _save_model_impl(self):
        """
        Actual model saving implementation
        Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement _save_model_impl()")
    
    @classmethod
    def load_model(cls, model_id: str = None, algorithm_type: str = None,
                   model_save_dir: str = None, mlflow_run_id: str = None):
        """
        Load a specific model by ID or from MLflow run

        Args:
            model_id: Model ID (e.g., "mscn_20241019_103000") - for local loading
            algorithm_type: Algorithm type
            model_save_dir: Custom save directory (optional) - for local loading
            mlflow_run_id: MLflow run ID (optional) - for MLflow loading

        Returns:
            Loaded model instance

        Usage:
            # Load from MLflow
            model = MscnPilotModel.load_model(mlflow_run_id="abc123...")

            # Load from local (legacy)
            model = MscnPilotModel.load_model("mscn_20241019_103000", "mscn")
        """
        from pilotscope.Common.MLflowTracker import MLflowTracker

        # Option 1: Load from MLflow
        if mlflow_run_id:
            print(f"📥 Loading model from MLflow run: {mlflow_run_id}")

            try:
                # Download model artifact from MLflow
                import tempfile
                temp_dir = tempfile.mkdtemp(prefix="pilotscope_model_")
                model_path = MLflowTracker.download_model_artifact(mlflow_run_id, temp_dir)

                if model_path is None:
                    raise FileNotFoundError(f"No model artifact found in run {mlflow_run_id}")

                # Try to load metadata
                metadata_path = model_path + ".json"
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    model_id = metadata.get("model_id", "mlflow_model")
                    model_name = model_id.split('_')[0]
                    algorithm_type = metadata.get("algorithm", "unknown")
                else:
                    # Fallback: try to infer from filename
                    model_id = Path(model_path).stem
                    model_name = model_id.split('_')[0]
                    algorithm_type = algorithm_type or "unknown"
                    metadata = {}

                # Create instance
                instance = cls(model_name, algorithm_type, temp_dir, save_to_local=False)

                # Override with loaded values
                instance.model_id = model_id
                instance.model_path = model_path
                instance.metadata_path = metadata_path
                instance.metadata = metadata

                # Load actual model
                instance._load_model_impl()

                print(f"✅ Model loaded from MLflow: {model_id}")
                if metadata.get('training'):
                    train = metadata['training']
                    print(f"   Training: {train.get('dataset', 'N/A')} "
                          f"(epoch={train.get('hyperparams', {}).get('num_epoch', 'N/A')})")
                print(f"   Tests: {len(metadata.get('testing', []))}")

                return instance

            except Exception as e:
                print(f"❌ Failed to load from MLflow: {e}")
                raise

        # Option 2: Load from local files (legacy)
        elif model_id and algorithm_type:
            print(f"📥 Loading model from local files: {model_id}")

            # Determine save directory
            if model_save_dir is None:
                base_path = os.path.dirname(__file__)
                model_save_dir = os.path.join(
                    base_path,
                    f"../algorithm_examples/ExampleData/{algorithm_type.capitalize()}/Model"
                )

            model_path = os.path.join(model_save_dir, model_id)
            metadata_path = model_path + ".json"

            # Load metadata
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata not found: {metadata_path}")

            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Create instance
            model_name = model_id.split('_')[0]
            instance = cls(model_name, algorithm_type, model_save_dir, save_to_local=True)

            # Override with loaded values
            instance.model_id = model_id
            instance.model_path = model_path
            instance.metadata_path = metadata_path
            instance.metadata = metadata

            # Load actual model
            instance._load_model_impl()

            print(f"✅ Model loaded from local: {model_id}")
            if metadata.get('training'):
                train = metadata['training']
                print(f"   Training: {train.get('dataset', 'N/A')} "
                      f"(epoch={train.get('hyperparams', {}).get('num_epoch', 'N/A')})")
            print(f"   Tests: {len(metadata.get('testing', []))}")

            return instance

        else:
            raise ValueError("Must provide either (model_id + algorithm_type) or mlflow_run_id")
    
    def _load_model_impl(self):
        """
        Actual model loading implementation
        Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement _load_model_impl()")
    
    @classmethod
    def list_available_models(cls, algorithm_type: str, 
                             model_save_dir: str = None) -> List[str]:
        """
        List all available model IDs for this algorithm
        
        Args:
            algorithm_type: Algorithm type
            model_save_dir: Custom save directory (optional)
        
        Returns:
            List of model IDs
        """
        if model_save_dir is None:
            base_path = os.path.dirname(__file__)
            model_save_dir = os.path.join(
                base_path,
                f"../algorithm_examples/ExampleData/{algorithm_type.capitalize()}/Model"
            )
        
        if not os.path.exists(model_save_dir):
            return []
        
        # Find all .json metadata files
        model_ids = []
        for filename in os.listdir(model_save_dir):
            if filename.endswith('.json'):
                model_id = filename[:-5]  # Remove .json
                model_ids.append(model_id)
        
        return sorted(model_ids, reverse=True)  # Latest first
    
    def get_metadata_summary(self) -> str:
        """Get human-readable metadata summary"""
        lines = []
        lines.append(f"Model ID: {self.model_id}")
        lines.append(f"Algorithm: {self.algorithm_type}")
        
        if self.metadata.get('training'):
            train = self.metadata['training']
            lines.append(f"Training Dataset: {train.get('dataset', 'N/A')}")
            hyperparams = train.get('hyperparams', {})
            lines.append(f"Hyperparameters: {hyperparams}")
            if train.get('training_time'):
                lines.append(f"Training Time: {train['training_time']:.1f}s")
        
        if self.metadata.get('testing'):
            lines.append(f"Test Results:")
            for test in self.metadata['testing']:
                perf = test['performance']
                lines.append(f"  - {test['dataset']}: "
                           f"total={perf.get('total_time', 0):.2f}s, "
                           f"avg={perf.get('average_time', 0):.4f}s")
        
        if self.metadata.get('tags'):
            lines.append(f"Tags: {', '.join(self.metadata['tags'])}")
        
        if self.metadata.get('notes'):
            lines.append(f"Notes: {self.metadata['notes']}")
        
        return '\n'.join(lines)

