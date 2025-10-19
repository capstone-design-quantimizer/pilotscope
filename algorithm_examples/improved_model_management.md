# 개선된 모델 관리 시스템: 타임스탬프 + 메타데이터

> "임시 저장 느낌"으로 부담 없이 실험하고, 나중에 최적 모델만 선택

---

## 🎯 설계 원칙

1. **타임스탬프로 저장**: 이름 충돌 없음, 부담 없이 저장
2. **풍부한 메타데이터**: 모든 변수 (파라미터, 데이터셋, 성능) 기록
3. **Train/Test 분리**: 학습과 테스트를 독립적으로 관리
4. **레지스트리로 관리**: 최적 모델 찾기, 비교, 정리

---

## 📂 파일 구조

```
ExampleData/
├── Mscn/
│   ├── Model/
│   │   ├── mscn_20241019_103000          # 타임스탬프 모델
│   │   ├── mscn_20241019_103000.json     # 메타데이터
│   │   ├── mscn_20241019_150000
│   │   ├── mscn_20241019_150000.json
│   │   └── ...
│   └── training_registry.json            # 학습 이력
├── Lero/
│   ├── Model/
│   │   ├── lero_20241019_110000
│   │   └── ...
│   └── training_registry.json
└── model_registry.json                   # 전체 모델 레지스트리
```

---

## 💾 메타데이터 스키마

```json
{
  "model_id": "mscn_20241019_103000",
  "algorithm": "mscn",
  "model_path": "ExampleData/Mscn/Model/mscn_20241019_103000",
  
  "training": {
    "enabled": true,
    "dataset": "production",
    "num_queries": 500,
    "hyperparams": {
      "num_epoch": 100,
      "num_training": 500,
      "num_collection": 500,
      "learning_rate": 0.001
    },
    "trained_at": "2024-10-19T10:30:00",
    "training_time": 1800.5
  },
  
  "testing": [
    {
      "dataset": "production",
      "num_queries": 100,
      "tested_at": "2024-10-19T12:00:00",
      "performance": {
        "total_time": 45.23,
        "average_time": 0.45,
        "median_time": 0.38,
        "p95_time": 1.2
      }
    },
    {
      "dataset": "stats_tiny",
      "num_queries": 80,
      "tested_at": "2024-10-19T13:00:00",
      "performance": {
        "total_time": 35.12,
        "average_time": 0.44
      }
    }
  ],
  
  "tags": ["production", "experiment", "best"],
  "notes": "Production 데이터로 학습, 성능 좋음"
}
```

---

## 🔧 구현: Enhanced PilotModel

```python
# algorithm_examples/enhanced_pilot_model.py

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from pilotscope.PilotModel import PilotModel


class EnhancedPilotModel(PilotModel):
    """
    타임스탬프 기반 모델 저장 + 풍부한 메타데이터
    """
    
    def __init__(self, model_name, algorithm_type):
        super().__init__(model_name)
        self.algorithm_type = algorithm_type  # "mscn", "lero", etc.
        self.model_save_dir = f"../algorithm_examples/ExampleData/{algorithm_type.capitalize()}/Model"
        
        # 타임스탬프 기반 ID 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_id = f"{model_name}_{timestamp}"
        self.model_path = os.path.join(self.model_save_dir, self.model_id)
        self.metadata_path = self.model_path + ".json"
        
        # 메타데이터
        self.metadata = {
            "model_id": self.model_id,
            "algorithm": self.algorithm_type,
            "model_path": self.model_path,
            "training": None,
            "testing": [],
            "tags": [],
            "notes": ""
        }
    
    def set_training_info(self, dataset: str, hyperparams: Dict, 
                         num_queries: int = None):
        """
        학습 정보 설정
        
        Args:
            dataset: 학습 데이터셋 이름
            hyperparams: 하이퍼파라미터
            num_queries: 학습에 사용한 쿼리 수
        """
        self.metadata["training"] = {
            "enabled": True,
            "dataset": dataset,
            "num_queries": num_queries,
            "hyperparams": hyperparams,
            "trained_at": datetime.now().isoformat(),
            "training_time": None  # 나중에 업데이트
        }
    
    def add_test_result(self, dataset: str, num_queries: int, 
                       performance: Dict):
        """
        테스트 결과 추가 (여러 데이터셋에 대해 누적 가능)
        
        Args:
            dataset: 테스트 데이터셋
            num_queries: 테스트 쿼리 수
            performance: 성능 메트릭
        """
        test_result = {
            "dataset": dataset,
            "num_queries": num_queries,
            "tested_at": datetime.now().isoformat(),
            "performance": performance
        }
        self.metadata["testing"].append(test_result)
    
    def add_tags(self, *tags):
        """태그 추가 (예: "production", "best", "experiment")"""
        self.metadata["tags"].extend(tags)
    
    def set_notes(self, notes: str):
        """메모 추가"""
        self.metadata["notes"] = notes
    
    def save_model(self):
        """모델 + 메타데이터 저장"""
        import time
        start_time = time.time()
        
        # 1. 모델 저장 (자식 클래스에서 구현)
        self._save_model_impl()
        
        # 2. 학습 시간 업데이트
        if self.metadata["training"]:
            self.metadata["training"]["training_time"] = time.time() - start_time
        
        # 3. 메타데이터 저장
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Model saved: {self.model_path}")
        print(f"✅ Metadata saved: {self.metadata_path}")
    
    def _save_model_impl(self):
        """실제 모델 저장 (자식 클래스에서 구현)"""
        raise NotImplementedError
    
    @classmethod
    def load_model(cls, model_id: str, algorithm_type: str):
        """
        특정 모델 로드
        
        Args:
            model_id: 타임스탬프 ID (예: "mscn_20241019_103000")
            algorithm_type: 알고리즘 타입
        """
        model_save_dir = f"../algorithm_examples/ExampleData/{algorithm_type.capitalize()}/Model"
        model_path = os.path.join(model_save_dir, model_id)
        metadata_path = model_path + ".json"
        
        # 메타데이터 로드
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 모델 인스턴스 생성
        instance = cls(model_id.split('_')[0], algorithm_type)
        instance.model_id = model_id
        instance.model_path = model_path
        instance.metadata = metadata
        
        # 실제 모델 로드
        instance._load_model_impl()
        
        print(f"✅ Model loaded: {model_id}")
        print(f"   Training: {metadata['training']['dataset'] if metadata['training'] else 'N/A'}")
        print(f"   Tests: {len(metadata['testing'])}")
        
        return instance
    
    def _load_model_impl(self):
        """실제 모델 로드 (자식 클래스에서 구현)"""
        raise NotImplementedError


# MSCN 구현
from algorithm_examples.Mscn.source.mscn_model import MscnModel

class MscnPilotModel(EnhancedPilotModel):
    """MSCN용 Enhanced Model"""
    
    def __init__(self, model_name="mscn"):
        super().__init__(model_name, "mscn")
    
    def _save_model_impl(self):
        """MSCN 모델 저장"""
        self.model.save(self.model_path)
    
    def _load_model_impl(self):
        """MSCN 모델 로드"""
        try:
            model = MscnModel()
            model.load(self.model_path)
        except:
            print("⚠️  Model file not found, creating new model")
            model = MscnModel()
        self.model = model


# Lero 구현
from algorithm_examples.Lero.source.model import LeroModelPairWise

class LeroPilotModel(EnhancedPilotModel):
    """Lero용 Enhanced Model"""
    
    def __init__(self, model_name="lero"):
        super().__init__(model_name, "lero")
    
    def _save_model_impl(self):
        """Lero 모델 저장"""
        self.model.save(self.model_path)
    
    def _load_model_impl(self):
        """Lero 모델 로드"""
        try:
            model = LeroModelPairWise(None)
            model.load(self.model_path)
        except FileNotFoundError:
            print("⚠️  Model file not found, creating new model")
            model = LeroModelPairWise(None)
        self.model = model
```

---

## 🗂️ 모델 레지스트리

```python
# algorithm_examples/model_registry.py

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class ModelRegistry:
    """
    모델 관리 레지스트리
    - 모델 조회
    - 최적 모델 찾기
    - 모델 비교
    - 오래된 모델 정리
    """
    
    def __init__(self, registry_file="algorithm_examples/ExampleData/model_registry.json"):
        self.registry_file = registry_file
        self._ensure_registry_exists()
    
    def _ensure_registry_exists(self):
        """레지스트리 파일 생성"""
        if not os.path.exists(self.registry_file):
            os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
            with open(self.registry_file, 'w') as f:
                json.dump({}, f)
    
    def _load_registry(self) -> Dict:
        """레지스트리 로드"""
        with open(self.registry_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_registry(self, registry: Dict):
        """레지스트리 저장"""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
    
    def register_model(self, metadata: Dict):
        """모델 등록"""
        registry = self._load_registry()
        model_id = metadata["model_id"]
        registry[model_id] = metadata
        self._save_registry(registry)
        print(f"✅ Model registered: {model_id}")
    
    def list_models(self, algorithm: str = None, 
                   dataset: str = None,
                   tags: List[str] = None,
                   sort_by: str = "trained_at") -> List[Dict]:
        """
        모델 목록 조회
        
        Args:
            algorithm: 알고리즘 필터
            dataset: 학습 데이터셋 필터
            tags: 태그 필터
            sort_by: 정렬 ("trained_at", "performance")
        """
        registry = self._load_registry()
        models = []
        
        for model_id, metadata in registry.items():
            # 필터링
            if algorithm and metadata.get("algorithm") != algorithm:
                continue
            
            if dataset:
                training = metadata.get("training", {})
                if training.get("dataset") != dataset:
                    continue
            
            if tags:
                model_tags = set(metadata.get("tags", []))
                if not set(tags).issubset(model_tags):
                    continue
            
            models.append(metadata)
        
        # 정렬
        if sort_by == "trained_at":
            models.sort(key=lambda x: x.get("training", {}).get("trained_at", ""), 
                       reverse=True)
        elif sort_by == "performance":
            def get_avg_performance(m):
                tests = m.get("testing", [])
                if not tests:
                    return float('inf')
                avg_time = sum(t["performance"].get("total_time", 0) for t in tests) / len(tests)
                return avg_time
            models.sort(key=get_avg_performance)
        
        return models
    
    def get_best_model(self, algorithm: str, 
                      dataset: str = None,
                      metric: str = "total_time") -> Optional[Dict]:
        """
        최적 모델 조회
        
        Args:
            algorithm: 알고리즘
            dataset: 특정 데이터셋에서 테스트된 모델만
            metric: 최적화 기준
        """
        models = self.list_models(algorithm=algorithm)
        
        if not models:
            return None
        
        # 각 모델의 특정 데이터셋 성능 추출
        candidates = []
        for model in models:
            for test in model.get("testing", []):
                if dataset and test["dataset"] != dataset:
                    continue
                
                perf = test["performance"].get(metric)
                if perf is not None:
                    candidates.append({
                        "model": model,
                        "performance": perf,
                        "test_dataset": test["dataset"]
                    })
        
        if not candidates:
            return None
        
        # 최소값 (시간이 적을수록 좋음)
        best = min(candidates, key=lambda x: x["performance"])
        return best["model"]
    
    def compare_models(self, model_ids: List[str], 
                      test_dataset: str = None) -> None:
        """
        여러 모델 비교 출력
        
        Args:
            model_ids: 비교할 모델 ID 리스트
            test_dataset: 특정 테스트 데이터셋 기준
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
                continue
            
            # 학습 정보
            training = metadata.get("training", {})
            train_dataset = training.get("dataset", "N/A")
            epochs = training.get("hyperparams", {}).get("num_epoch", "N/A")
            
            # 테스트 정보
            tests = metadata.get("testing", [])
            if test_dataset:
                tests = [t for t in tests if t["dataset"] == test_dataset]
            
            if not tests:
                print(f"{model_id:<30} {train_dataset:<15} {'No tests':<15} "
                      f"{'-':<12} {'-':<10} {epochs:<8}")
            else:
                for test in tests:
                    perf = test["performance"]
                    total_time = perf.get("total_time", 0)
                    avg_time = perf.get("average_time", 0)
                    test_ds = test["dataset"]
                    
                    print(f"{model_id:<30} {train_dataset:<15} {test_ds:<15} "
                          f"{total_time:<12.2f} {avg_time:<10.4f} {epochs:<8}")
        
        print("="*100)
    
    def cleanup_old_models(self, algorithm: str, keep_top_n: int = 5,
                          by_metric: str = "total_time") -> List[str]:
        """
        오래된 모델 정리 (상위 N개만 유지)
        
        Args:
            algorithm: 알고리즘
            keep_top_n: 유지할 모델 개수
            by_metric: 정렬 기준
        
        Returns:
            삭제된 모델 ID 리스트
        """
        models = self.list_models(algorithm=algorithm, sort_by="performance")
        
        if len(models) <= keep_top_n:
            print(f"ℹ️  Only {len(models)} models, no cleanup needed")
            return []
        
        # 하위 모델 삭제
        to_delete = models[keep_top_n:]
        deleted_ids = []
        
        registry = self._load_registry()
        
        for model in to_delete:
            model_id = model["model_id"]
            model_path = model["model_path"]
            
            # 파일 삭제
            try:
                if os.path.exists(model_path):
                    os.remove(model_path)
                metadata_path = model_path + ".json"
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                
                # 레지스트리에서 제거
                del registry[model_id]
                deleted_ids.append(model_id)
                print(f"🗑️  Deleted: {model_id}")
            except Exception as e:
                print(f"⚠️  Failed to delete {model_id}: {e}")
        
        self._save_registry(registry)
        print(f"\n✅ Cleanup complete: {len(deleted_ids)} models deleted")
        
        return deleted_ids
    
    def tag_model(self, model_id: str, *tags):
        """모델에 태그 추가"""
        registry = self._load_registry()
        
        if model_id in registry:
            existing_tags = set(registry[model_id].get("tags", []))
            existing_tags.update(tags)
            registry[model_id]["tags"] = list(existing_tags)
            self._save_registry(registry)
            print(f"✅ Tags added to {model_id}: {tags}")
        else:
            print(f"❌ Model not found: {model_id}")
```

---

## 🔗 PresetScheduler 통합

```python
# algorithm_examples/Mscn/MscnPresetScheduler.py (수정)

from algorithm_examples.enhanced_pilot_model import MscnPilotModel
from algorithm_examples.model_registry import ModelRegistry

def get_mscn_preset_scheduler(config, enable_collection, enable_training,
                             num_collection=-1, num_training=-1, num_epoch=100,
                             load_model_id=None):  # 추가!
    """
    Args:
        load_model_id: 특정 모델 로드 (None이면 새 모델 또는 최적 모델)
    """
    
    # 1. 모델 생성 또는 로드
    if load_model_id:
        # 특정 모델 로드
        mscn_pilot_model = MscnPilotModel.load_model(load_model_id, "mscn")
    elif not enable_training:
        # 학습 안하면 최적 모델 로드
        registry = ModelRegistry()
        best = registry.get_best_model("mscn", dataset=config.db)
        if best:
            print(f"📊 Loading best model for {config.db}")
            mscn_pilot_model = MscnPilotModel.load_model(best["model_id"], "mscn")
        else:
            print("⚠️  No trained models, creating new model")
            mscn_pilot_model = MscnPilotModel()
    else:
        # 새 모델 생성
        mscn_pilot_model = MscnPilotModel()
        
        # 학습 정보 설정
        hyperparams = {
            "num_epoch": num_epoch,
            "num_training": num_training,
            "num_collection": num_collection
        }
        mscn_pilot_model.set_training_info(
            dataset=config.db,
            hyperparams=hyperparams
        )
    
    # ... 나머지 스케줄러 생성 코드 ...
    
    return scheduler, mscn_pilot_model  # 모델도 반환
```

---

## 🧪 unified_test.py 통합

```python
# test_example_algorithms/unified_test.py (수정)

def run_single_test(config, algo_name, dataset_name, algo_params=None):
    """단일 테스트 실행 + 자동 메타데이터 저장"""
    
    # ... 기존 테스트 실행 코드 ...
    
    # 테스트 완료 후
    if algo_name != "baseline":
        # 성능 메트릭 계산
        performance = {
            "total_time": elapsed,
            "average_time": elapsed / len(test_sqls),
            "median_time": ...,  # 계산 가능하면
            "p95_time": ...,
        }
        
        # 모델에 테스트 결과 추가
        if hasattr(scheduler_or_interactor, 'pilot_model'):
            model = scheduler_or_interactor.pilot_model
            model.add_test_result(
                dataset=dataset_name,
                num_queries=len(test_sqls),
                performance=performance
            )
            
            # 모델 저장 (테스트 결과 포함)
            model.save_model()
            
            # 레지스트리에 등록
            registry = ModelRegistry()
            registry.register_model(model.metadata)
    
    return result
```

---

## 🛠️ CLI 도구

```python
# scripts/model_manager.py

import argparse
from algorithm_examples.model_registry import ModelRegistry

def main():
    parser = argparse.ArgumentParser(description='Model Management Tool')
    subparsers = parser.add_subparsers(dest='command')
    
    # list 명령
    list_parser = subparsers.add_parser('list', help='List models')
    list_parser.add_argument('--algo', help='Algorithm filter')
    list_parser.add_argument('--dataset', help='Dataset filter')
    list_parser.add_argument('--tags', nargs='+', help='Tag filter')
    
    # best 명령
    best_parser = subparsers.add_parser('best', help='Find best model')
    best_parser.add_argument('--algo', required=True)
    best_parser.add_argument('--dataset', help='Test dataset')
    
    # compare 명령
    compare_parser = subparsers.add_parser('compare', help='Compare models')
    compare_parser.add_argument('models', nargs='+', help='Model IDs')
    compare_parser.add_argument('--dataset', help='Test dataset filter')
    
    # cleanup 명령
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean old models')
    cleanup_parser.add_argument('--algo', required=True)
    cleanup_parser.add_argument('--keep', type=int, default=5, help='Keep top N')
    
    # tag 명령
    tag_parser = subparsers.add_parser('tag', help='Add tags to model')
    tag_parser.add_argument('model_id')
    tag_parser.add_argument('tags', nargs='+')
    
    args = parser.parse_args()
    registry = ModelRegistry()
    
    if args.command == 'list':
        models = registry.list_models(
            algorithm=args.algo,
            dataset=args.dataset,
            tags=args.tags
        )
        
        print(f"\nFound {len(models)} models:\n")
        for model in models:
            print(f"{'='*80}")
            print(f"Model ID: {model['model_id']}")
            print(f"Algorithm: {model['algorithm']}")
            
            if model['training']:
                train = model['training']
                print(f"Training: {train['dataset']} "
                      f"(epoch={train['hyperparams'].get('num_epoch')}, "
                      f"time={train.get('training_time', 0):.1f}s)")
            
            if model['testing']:
                print(f"Tests: {len(model['testing'])}")
                for test in model['testing']:
                    print(f"  - {test['dataset']}: "
                          f"{test['performance'].get('total_time', 0):.2f}s")
            
            if model.get('tags'):
                print(f"Tags: {', '.join(model['tags'])}")
    
    elif args.command == 'best':
        best = registry.get_best_model(args.algo, dataset=args.dataset)
        if best:
            print(f"\n🏆 Best Model: {best['model_id']}")
            print(f"Training: {best['training']['dataset']}")
            print(f"Performance:")
            for test in best['testing']:
                if not args.dataset or test['dataset'] == args.dataset:
                    print(f"  {test['dataset']}: "
                          f"{test['performance']['total_time']:.2f}s")
        else:
            print("❌ No models found")
    
    elif args.command == 'compare':
        registry.compare_models(args.models, test_dataset=args.dataset)
    
    elif args.command == 'cleanup':
        deleted = registry.cleanup_old_models(args.algo, keep_top_n=args.keep)
        print(f"Deleted {len(deleted)} models")
    
    elif args.command == 'tag':
        registry.tag_model(args.model_id, *args.tags)

if __name__ == '__main__':
    main()
```

---

## 📝 사용 시나리오

### 시나리오 1: 실험적 학습

```bash
# 1. production 데이터로 학습 (epoch=100)
python unified_test.py --algo mscn --db production --epochs 100
# → 자동 저장: mscn_20241019_103000

# 2. 결과 안좋으면 다른 파라미터 시도
python unified_test.py --algo mscn --db production --epochs 200
# → 자동 저장: mscn_20241019_110000

# 3. 모델 비교
python scripts/model_manager.py compare \
    mscn_20241019_103000 mscn_20241019_110000

# 4. 안좋은 모델 정리
python scripts/model_manager.py cleanup --algo mscn --keep 3
```

### 시나리오 2: Train/Test 분리

```bash
# 1. 오전: production 데이터로 학습
python unified_test.py --algo mscn --db production --epochs 100
# → 저장: mscn_20241019_100000

# 2. 오후: 그 모델로 stats_tiny 테스트 (학습 안함)
python unified_test.py \
    --algo mscn \
    --db stats_tiny \
    --no-training \
    --load-model mscn_20241019_100000
# → 메타데이터 업데이트: testing 배열에 stats_tiny 결과 추가

# 3. 저녁: 같은 모델로 imdb 테스트
python unified_test.py \
    --algo mscn \
    --db imdb \
    --no-training \
    --load-model mscn_20241019_100000
# → 메타데이터 업데이트: testing 배열에 imdb 결과 추가

# 4. 결과 확인
python scripts/model_manager.py list --algo mscn
# Model: mscn_20241019_100000
#   Training: production (epoch=100)
#   Tests:
#     - production: 45.2s
#     - stats_tiny: 35.1s
#     - imdb: 120.5s
```

### 시나리오 3: 최적 모델 찾기

```bash
# 1. 여러 번 실험
python unified_test.py --algo mscn --db production --epochs 50
python unified_test.py --algo mscn --db production --epochs 100
python unified_test.py --algo mscn --db production --epochs 200

# 2. 최적 모델 조회
python scripts/model_manager.py best --algo mscn --dataset production
# 🏆 Best Model: mscn_20241019_110000
#   Performance: 42.5s (epoch=100)

# 3. 최적 모델에 태그
python scripts/model_manager.py tag mscn_20241019_110000 best production

# 4. 다음에 자동으로 최적 모델 로드
python unified_test.py --algo mscn --db production --no-training
# → 자동으로 mscn_20241019_110000 로드
```

---

## ✅ 장점 요약

| 기능 | 파라미터 기반 이름 | 타임스탬프 + 메타데이터 |
|------|------------------|----------------------|
| 이름 충돌 | ❌ (데이터셋 다르면 충돌) | ✅ 없음 |
| 임시 저장 | ❌ (이름 설계 필요) | ✅ 부담 없음 |
| Train/Test 분리 | ❌ | ✅ 완벽 지원 |
| 여러 데이터셋 테스트 | ❌ | ✅ 누적 가능 |
| 최적 모델 찾기 | ⚠️ (수동) | ✅ 자동 |
| 모델 정리 | 어려움 | ✅ 쉬움 |
| 실험 추적 | 제한적 | ✅ 완벽 |

---

## 🎯 결론

당신의 통찰이 정확합니다:

1. **타임스탬프**: 이름 충돌 없이 "임시 저장" 느낌으로 부담 없이 실험
2. **풍부한 메타데이터**: 모든 변수 (파라미터, 데이터셋, 성능) 기록
3. **Train/Test 분리**: 한 모델을 여러 데이터셋에서 테스트 가능
4. **레지스트리**: 최적 모델 찾기, 비교, 정리 자동화

이 방식이 실험 단계에서 훨씬 실용적입니다! 🎉

