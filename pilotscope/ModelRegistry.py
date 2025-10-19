"""
Model Registry for managing trained models

Features:
- Centralized model metadata management
- Find best models by performance
- Compare multiple models
- Clean up old models
- Tag management
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


class ModelRegistry:
    """
    Centralized registry for model management
    
    Usage:
        registry = ModelRegistry()
        
        # Register model
        registry.register_model(model.metadata)
        
        # Find best model
        best = registry.get_best_model("mscn", dataset="production")
        
        # Compare models
        registry.compare_models(["mscn_20241019_103000", "mscn_20241019_110000"])
        
        # Cleanup old models
        registry.cleanup_old_models("mscn", keep_top_n=5)
    """
    
    def __init__(self, registry_file: str = None):
        """
        Args:
            registry_file: Path to registry JSON file
        """
        if registry_file is None:
            base_path = os.path.dirname(__file__)
            registry_file = os.path.join(
                base_path,
                "../algorithm_examples/ExampleData/model_registry.json"
            )
        
        self.registry_file = registry_file
        self._ensure_registry_exists()
    
    def _ensure_registry_exists(self):
        """Create registry file if not exists"""
        if not os.path.exists(self.registry_file):
            os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
    
    def _load_registry(self) -> Dict:
        """Load registry from file"""
        with open(self.registry_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_registry(self, registry: Dict):
        """Save registry to file"""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
    
    def register_model(self, metadata: Dict):
        """
        Register a model in the registry
        
        Args:
            metadata: Model metadata dict
        """
        registry = self._load_registry()
        model_id = metadata["model_id"]
        registry[model_id] = metadata
        self._save_registry(registry)
        print(f"✅ Model registered: {model_id}")
    
    def get_model(self, model_id: str) -> Optional[Dict]:
        """
        Get model metadata by ID
        
        Args:
            model_id: Model ID
        
        Returns:
            Model metadata dict or None
        """
        registry = self._load_registry()
        return registry.get(model_id)
    
    def list_models(self, 
                   algorithm: str = None,
                   dataset: str = None,
                   tags: List[str] = None,
                   sort_by: str = "trained_at") -> List[Dict]:
        """
        List models with filters
        
        Args:
            algorithm: Filter by algorithm
            dataset: Filter by training dataset
            tags: Filter by tags (all tags must match)
            sort_by: Sort criteria ("trained_at", "performance")
        
        Returns:
            List of model metadata dicts
        """
        registry = self._load_registry()
        models = []
        
        for model_id, metadata in registry.items():
            # Filter by algorithm
            if algorithm and metadata.get("algorithm") != algorithm:
                continue
            
            # Filter by training dataset
            if dataset:
                training = metadata.get("training", {})
                if training and training.get("dataset") != dataset:
                    continue
            
            # Filter by tags
            if tags:
                model_tags = set(metadata.get("tags", []))
                if not set(tags).issubset(model_tags):
                    continue
            
            models.append(metadata)
        
        # Sort
        if sort_by == "trained_at":
            models.sort(
                key=lambda x: x.get("training", {}).get("trained_at", ""),
                reverse=True
            )
        elif sort_by == "performance":
            def get_avg_performance(m):
                tests = m.get("testing", [])
                if not tests:
                    return float('inf')
                total = sum(t["performance"].get("total_time", 0) for t in tests)
                return total / len(tests)
            
            models.sort(key=get_avg_performance)
        
        return models
    
    def get_best_model(self, 
                      algorithm: str,
                      test_dataset: str = None,
                      metric: str = "total_time") -> Optional[Dict]:
        """
        Find best model by performance
        
        Args:
            algorithm: Algorithm to search
            test_dataset: Filter by test dataset (optional)
            metric: Performance metric to optimize
        
        Returns:
            Best model metadata dict or None
        """
        models = self.list_models(algorithm=algorithm)
        
        if not models:
            return None
        
        # Collect candidates with performance data
        candidates = []
        for model in models:
            for test in model.get("testing", []):
                # Filter by test dataset if specified
                if test_dataset and test["dataset"] != test_dataset:
                    continue
                
                perf_value = test["performance"].get(metric)
                if perf_value is not None:
                    candidates.append({
                        "model": model,
                        "performance": perf_value,
                        "test_dataset": test["dataset"]
                    })
        
        if not candidates:
            return None
        
        # Find minimum (lower is better for time metrics)
        best = min(candidates, key=lambda x: x["performance"])
        return best["model"]
    
    def compare_models(self, 
                      model_ids: List[str],
                      test_dataset: str = None):
        """
        Print comparison table for multiple models
        
        Args:
            model_ids: List of model IDs to compare
            test_dataset: Filter by test dataset (optional)
        """
        registry = self._load_registry()
        
        print("\n" + "="*100)
        print("Model Comparison")
        print("="*100)
        print(f"{'Model ID':<30} {'Train Dataset':<15} {'Test Dataset':<15} "
              f"{'Total Time':<12} {'Avg Time':<10} {'Epochs':<8}")
        print("-"*100)
        
        for model_id in model_ids:
            metadata = registry.get(model_id)
            if not metadata:
                print(f"{model_id:<30} {'Not found':<15}")
                continue
            
            # Training info
            training = metadata.get("training", {})
            train_dataset = training.get("dataset", "N/A") if training else "N/A"
            epochs = training.get("hyperparams", {}).get("num_epoch", "N/A") if training else "N/A"
            
            # Testing info
            tests = metadata.get("testing", [])
            if test_dataset:
                tests = [t for t in tests if t["dataset"] == test_dataset]
            
            if not tests:
                print(f"{model_id:<30} {train_dataset:<15} {'No tests':<15} "
                      f"{'-':<12} {'-':<10} {epochs:<8}")
            else:
                for i, test in enumerate(tests):
                    perf = test["performance"]
                    total_time = perf.get("total_time", 0)
                    avg_time = perf.get("average_time", 0)
                    test_ds = test["dataset"]
                    
                    # Only show model ID on first row
                    model_id_display = model_id if i == 0 else ""
                    train_ds_display = train_dataset if i == 0 else ""
                    epochs_display = epochs if i == 0 else ""
                    
                    print(f"{model_id_display:<30} {train_ds_display:<15} {test_ds:<15} "
                          f"{total_time:<12.2f} {avg_time:<10.4f} {epochs_display:<8}")
        
        print("="*100 + "\n")
    
    def cleanup_old_models(self, 
                          algorithm: str,
                          keep_top_n: int = 5,
                          by_metric: str = "total_time") -> List[str]:
        """
        Delete old models, keep only top N
        
        Args:
            algorithm: Algorithm to clean up
            keep_top_n: Number of models to keep
            by_metric: Metric for ranking
        
        Returns:
            List of deleted model IDs
        """
        models = self.list_models(algorithm=algorithm, sort_by="performance")
        
        if len(models) <= keep_top_n:
            print(f"ℹ️  Only {len(models)} models exist, no cleanup needed")
            return []
        
        # Models to delete (bottom performers)
        to_delete = models[keep_top_n:]
        deleted_ids = []
        
        registry = self._load_registry()
        
        for model in to_delete:
            model_id = model["model_id"]
            model_path = model["model_path"]
            
            # Delete files
            try:
                # Delete model file
                if os.path.exists(model_path):
                    os.remove(model_path)
                
                # Delete metadata file
                metadata_path = model_path + ".json"
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                
                # Remove from registry
                del registry[model_id]
                deleted_ids.append(model_id)
                print(f"🗑️  Deleted: {model_id}")
            
            except Exception as e:
                print(f"⚠️  Failed to delete {model_id}: {e}")
        
        self._save_registry(registry)
        print(f"\n✅ Cleanup complete: {len(deleted_ids)} models deleted, {keep_top_n} kept")
        
        return deleted_ids
    
    def tag_model(self, model_id: str, *tags):
        """
        Add tags to a model
        
        Args:
            model_id: Model ID
            *tags: Tags to add
        """
        registry = self._load_registry()
        
        if model_id not in registry:
            print(f"❌ Model not found: {model_id}")
            return
        
        existing_tags = set(registry[model_id].get("tags", []))
        existing_tags.update(tags)
        registry[model_id]["tags"] = list(existing_tags)
        
        self._save_registry(registry)
        print(f"✅ Tags added to {model_id}: {tags}")
    
    def remove_tag(self, model_id: str, *tags):
        """
        Remove tags from a model
        
        Args:
            model_id: Model ID
            *tags: Tags to remove
        """
        registry = self._load_registry()
        
        if model_id not in registry:
            print(f"❌ Model not found: {model_id}")
            return
        
        existing_tags = set(registry[model_id].get("tags", []))
        existing_tags.difference_update(tags)
        registry[model_id]["tags"] = list(existing_tags)
        
        self._save_registry(registry)
        print(f"✅ Tags removed from {model_id}: {tags}")
    
    def set_notes(self, model_id: str, notes: str):
        """
        Set notes for a model
        
        Args:
            model_id: Model ID
            notes: Notes text
        """
        registry = self._load_registry()
        
        if model_id not in registry:
            print(f"❌ Model not found: {model_id}")
            return
        
        registry[model_id]["notes"] = notes
        self._save_registry(registry)
        print(f"✅ Notes updated for {model_id}")
    
    def print_summary(self, algorithm: str = None):
        """
        Print registry summary
        
        Args:
            algorithm: Filter by algorithm (optional)
        """
        models = self.list_models(algorithm=algorithm)
        
        if not models:
            print("📭 No models in registry")
            return
        
        print("\n" + "="*80)
        print(f"Model Registry Summary ({len(models)} models)")
        print("="*80)
        
        # Group by algorithm
        by_algo = {}
        for model in models:
            algo = model["algorithm"]
            if algo not in by_algo:
                by_algo[algo] = []
            by_algo[algo].append(model)
        
        for algo, algo_models in sorted(by_algo.items()):
            print(f"\n{algo.upper()}: {len(algo_models)} models")
            print("-" * 80)
            
            for model in algo_models[:5]:  # Show top 5
                model_id = model["model_id"]
                training = model.get("training", {})
                tests = model.get("testing", [])
                
                train_info = "No training" if not training else \
                    f"{training.get('dataset')} (epoch={training.get('hyperparams', {}).get('num_epoch', '?')})"
                
                test_info = f"{len(tests)} tests" if tests else "No tests"
                
                tags_info = f"[{', '.join(model.get('tags', []))}]" if model.get('tags') else ""
                
                print(f"  • {model_id}: {train_info} | {test_info} {tags_info}")
            
            if len(algo_models) > 5:
                print(f"  ... and {len(algo_models) - 5} more")
        
        print("="*80 + "\n")

