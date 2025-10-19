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
                 model_save_dir: str = None):
        """
        Args:
            model_name: Base model name (e.g., "mscn")
            algorithm_type: Algorithm type for directory organization
            model_save_dir: Custom save directory (optional)
        """
        super().__init__(model_name)
        self.algorithm_type = algorithm_type
        
        # Default save directory
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
        
        This will:
        1. Call _save_model_impl() to save actual model
        2. Update training time if applicable
        3. Save metadata JSON
        """
        # Ensure directory exists
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Save actual model (implemented by subclass)
        self._save_model_impl()
        
        # Update training timestamp if this is first save after training
        if self.metadata["training"] and not self.metadata["training"]["trained_at"]:
            self.metadata["training"]["trained_at"] = datetime.now().isoformat()
        
        # Save metadata
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Model saved: {self.model_path}")
        print(f"✅ Metadata saved: {self.metadata_path}")
    
    def _save_model_impl(self):
        """
        Actual model saving implementation
        Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement _save_model_impl()")
    
    @classmethod
    def load_model(cls, model_id: str, algorithm_type: str, 
                   model_save_dir: str = None):
        """
        Load a specific model by ID
        
        Args:
            model_id: Model ID (e.g., "mscn_20241019_103000")
            algorithm_type: Algorithm type
            model_save_dir: Custom save directory (optional)
        
        Returns:
            Loaded model instance
        """
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
        instance = cls(model_name, algorithm_type, model_save_dir)
        
        # Override with loaded values
        instance.model_id = model_id
        instance.model_path = model_path
        instance.metadata_path = metadata_path
        instance.metadata = metadata
        
        # Load actual model
        instance._load_model_impl()
        
        print(f"✅ Model loaded: {model_id}")
        if metadata.get('training'):
            train = metadata['training']
            print(f"   Training: {train.get('dataset', 'N/A')} "
                  f"(epoch={train.get('hyperparams', {}).get('num_epoch', 'N/A')})")
        print(f"   Tests: {len(metadata.get('testing', []))}")
        
        return instance
    
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

