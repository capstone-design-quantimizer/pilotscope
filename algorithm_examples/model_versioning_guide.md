# 모델 버전 관리 시스템 개선안

## 문제점 분석

### 현재 방식
```python
# MscnPilotModel.py
model_path = "ExampleData/Mscn/Model/mscn"  # 항상 같은 경로
self.model.save(self.model_path)  # 덮어쓰기!
```

### 문제
1. ❌ 파라미터 정보 없음
2. ❌ 이전 모델 손실
3. ❌ 실험 재현 불가
4. ❌ 최적 모델 추적 어려움

---

## 개선안: 파라미터 기반 버전 관리

### 방안 1: 파라미터 기반 모델 이름 (추천 ⭐⭐⭐)

```python
# 개선된 MscnPilotModel.py
class MscnPilotModel(PilotModel):
    def __init__(self, model_name, hyperparams=None):
        super().__init__(model_name)
        self.hyperparams = hyperparams or {}
        self.model_save_dir = "../algorithm_examples/ExampleData/Mscn/Model"
        
        # 파라미터 기반 고유 이름 생성
        self.model_path = self._get_versioned_path()
        self.metadata_path = self.model_path + ".json"
    
    def _get_versioned_path(self):
        """
        파라미터 기반으로 고유한 모델 경로 생성
        예: mscn_epoch100_train500_collection500
        """
        if not self.hyperparams:
            return os.path.join(self.model_save_dir, self.model_name)
        
        # 중요 파라미터만 이름에 포함
        param_str = "_".join([
            f"{k}{v}" for k, v in sorted(self.hyperparams.items())
            if k in ['num_epoch', 'num_training', 'num_collection']
        ])
        
        versioned_name = f"{self.model_name}_{param_str}"
        return os.path.join(self.model_save_dir, versioned_name)
    
    def save_model(self):
        """모델과 메타데이터를 함께 저장"""
        import json
        from datetime import datetime
        
        # 1. 모델 저장
        self.model.save(self.model_path)
        
        # 2. 메타데이터 저장
        metadata = {
            "model_name": self.model_name,
            "hyperparams": self.hyperparams,
            "saved_at": datetime.now().isoformat(),
            "model_path": self.model_path
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model saved: {self.model_path}")
        print(f"✅ Metadata saved: {self.metadata_path}")
    
    def load_model(self, hyperparams=None):
        """
        특정 파라미터의 모델 로드
        hyperparams=None이면 최신 모델 로드
        """
        if hyperparams:
            self.hyperparams = hyperparams
            self.model_path = self._get_versioned_path()
        
        try:
            model = MscnModel()
            model.load(self.model_path)
            print(f"✅ Model loaded: {self.model_path}")
            
            # 메타데이터 로드
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"   Hyperparams: {metadata['hyperparams']}")
        except:
            print(f"⚠️  Model not found, creating new model")
            model = MscnModel()
        
        self.model = model
```

**사용 예시**:
```python
# 학습 시
hyperparams = {
    "num_epoch": 100,
    "num_training": 500,
    "num_collection": 500
}
model = MscnPilotModel("mscn", hyperparams)
model.train(...)
model.save_model()
# 저장 경로: ExampleData/Mscn/Model/mscn_epoch100_training500_collection500

# 로드 시
model = MscnPilotModel("mscn")
model.load_model(hyperparams={"num_epoch": 100, ...})
```

---

### 방안 2: 타임스탬프 기반 (차선 ⭐⭐)

```python
def _get_versioned_path(self):
    """타임스탬프 기반 버전 관리"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_name = f"{self.model_name}_{timestamp}"
    return os.path.join(self.model_save_dir, versioned_name)
```

**장점**: 간단  
**단점**: 어떤 파라미터인지 이름만으로 알 수 없음

---

### 방안 3: 모델 레지스트리 (고급 ⭐⭐⭐⭐)

```python
# algorithm_examples/model_registry.py
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

class ModelRegistry:
    """
    모든 학습된 모델의 메타데이터 관리
    """
    
    def __init__(self, registry_file="algorithm_examples/ExampleData/model_registry.json"):
        self.registry_file = registry_file
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """레지스트리 로드"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """레지스트리 저장"""
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_id: str, metadata: Dict):
        """
        모델 등록
        
        Args:
            model_id: 고유 ID (예: mscn_epoch100_train500)
            metadata: {
                "algorithm": "mscn",
                "hyperparams": {...},
                "model_path": "...",
                "performance": {"total_time": 45.2, ...},
                "trained_at": "2024-01-01T10:00:00"
            }
        """
        self.registry[model_id] = metadata
        self._save_registry()
        print(f"✅ Model registered: {model_id}")
    
    def get_model(self, model_id: str) -> Optional[Dict]:
        """특정 모델 메타데이터 조회"""
        return self.registry.get(model_id)
    
    def list_models(self, algorithm: str = None, 
                   sort_by: str = "performance") -> List[Dict]:
        """
        모델 목록 조회
        
        Args:
            algorithm: 알고리즘 필터 (예: "mscn")
            sort_by: 정렬 기준 ("performance", "trained_at")
        """
        models = []
        for model_id, metadata in self.registry.items():
            if algorithm and metadata.get("algorithm") != algorithm:
                continue
            models.append({"id": model_id, **metadata})
        
        # 정렬
        if sort_by == "performance":
            models.sort(key=lambda x: x.get("performance", {}).get("total_time", float('inf')))
        elif sort_by == "trained_at":
            models.sort(key=lambda x: x.get("trained_at", ""), reverse=True)
        
        return models
    
    def get_best_model(self, algorithm: str, 
                      metric: str = "total_time") -> Optional[Dict]:
        """
        최적 모델 조회
        
        Args:
            algorithm: 알고리즘 이름
            metric: 최적화 기준 (예: "total_time")
        """
        models = self.list_models(algorithm=algorithm)
        if not models:
            return None
        
        # 최소값 찾기 (시간이므로)
        best = min(models, 
                  key=lambda x: x.get("performance", {}).get(metric, float('inf')))
        return best
    
    def compare_models(self, model_ids: List[str]) -> None:
        """여러 모델 비교 출력"""
        print("\n" + "="*80)
        print("Model Comparison")
        print("="*80)
        print(f"{'Model ID':<40} {'Algorithm':<10} {'Total Time':<12} {'Epochs':<8}")
        print("-"*80)
        
        for model_id in model_ids:
            metadata = self.get_model(model_id)
            if metadata:
                algo = metadata.get("algorithm", "N/A")
                perf = metadata.get("performance", {})
                time = perf.get("total_time", "N/A")
                epochs = metadata.get("hyperparams", {}).get("num_epoch", "N/A")
                print(f"{model_id:<40} {algo:<10} {time:<12.2f} {epochs:<8}")
        print("="*80)


# 통합된 MscnPilotModel (모델 레지스트리 사용)
class MscnPilotModel(PilotModel):
    def __init__(self, model_name, hyperparams=None):
        super().__init__(model_name)
        self.hyperparams = hyperparams or {}
        self.model_save_dir = "../algorithm_examples/ExampleData/Mscn/Model"
        self.registry = ModelRegistry()
        
        # 모델 ID 생성
        self.model_id = self._generate_model_id()
        self.model_path = os.path.join(self.model_save_dir, self.model_id)
    
    def _generate_model_id(self):
        """고유한 모델 ID 생성"""
        if not self.hyperparams:
            return self.model_name
        
        param_str = "_".join([
            f"{k}{v}" for k, v in sorted(self.hyperparams.items())
            if k in ['num_epoch', 'num_training', 'num_collection']
        ])
        return f"{self.model_name}_{param_str}"
    
    def save_model(self, performance_metrics: Dict = None):
        """
        모델 저장 + 레지스트리 등록
        
        Args:
            performance_metrics: {"total_time": 45.2, "average_time": 0.5, ...}
        """
        from datetime import datetime
        
        # 1. 모델 저장
        self.model.save(self.model_path)
        
        # 2. 레지스트리 등록
        metadata = {
            "algorithm": "mscn",
            "hyperparams": self.hyperparams,
            "model_path": self.model_path,
            "performance": performance_metrics or {},
            "trained_at": datetime.now().isoformat()
        }
        
        self.registry.register_model(self.model_id, metadata)
        
        print(f"✅ Model saved: {self.model_path}")
        print(f"✅ Registry updated")
    
    @classmethod
    def load_best_model(cls, model_name="mscn"):
        """최적 모델 자동 로드"""
        registry = ModelRegistry()
        best = registry.get_best_model("mscn")
        
        if not best:
            print("⚠️  No trained models found")
            return cls(model_name)
        
        print(f"📊 Loading best model: {best['id']}")
        print(f"   Performance: {best['performance']}")
        
        model = cls(model_name, best['hyperparams'])
        model.model_id = best['id']
        model.model_path = best['model_path']
        model.load_model()
        return model
```

**사용 예시**:
```python
# 학습 후 저장
model = MscnPilotModel("mscn", hyperparams={
    "num_epoch": 100,
    "num_training": 500
})
model.train(...)

# 성능 메트릭과 함께 저장
performance = {"total_time": 45.2, "average_time": 0.5}
model.save_model(performance_metrics=performance)

# 최적 모델 자동 로드
best_model = MscnPilotModel.load_best_model()

# 모델 비교
registry = ModelRegistry()
print(registry.list_models(algorithm="mscn", sort_by="performance"))
```

---

## 통합: PresetScheduler와 연동

```python
# MscnPresetScheduler.py (수정)
def get_mscn_preset_scheduler(config, enable_collection, enable_training,
                             num_collection=-1, num_training=-1, num_epoch=100):
    
    # 1. 하이퍼파라미터 정의
    hyperparams = {
        "num_epoch": num_epoch,
        "num_training": num_training,
        "num_collection": num_collection,
        "enable_collection": enable_collection,
        "enable_training": enable_training
    }
    
    # 2. 파라미터 기반 모델 생성
    model_name = "mscn"
    mscn_pilot_model = MscnPilotModel(model_name, hyperparams)
    
    # 3. 최적 모델 자동 로드 (선택사항)
    if not enable_training:
        mscn_pilot_model = MscnPilotModel.load_best_model()
    else:
        mscn_pilot_model.load_model()
    
    # ... 나머지 코드 동일 ...
```

---

## unified_test.py 통합

```python
# unified_test.py에서 성능 메트릭 자동 저장
def run_single_test(config, algo_name, dataset_name, algo_params=None):
    # ... 기존 코드 ...
    
    # 테스트 완료 후
    performance_metrics = {
        "total_time": elapsed,
        "average_time": elapsed / len(test_sqls),
        "num_queries": len(test_sqls),
        "dataset": dataset_name
    }
    
    # 모델에 성능 메트릭 전달 (스케줄러에서 처리)
    if hasattr(scheduler_or_interactor, 'save_model_with_metrics'):
        scheduler_or_interactor.save_model_with_metrics(performance_metrics)
```

---

## 비교: 3가지 방안

| 특징 | 방안 1 (파라미터 기반) | 방안 2 (타임스탬프) | 방안 3 (레지스트리) |
|------|---------------------|------------------|------------------|
| 구현 난이도 | ⭐⭐ 보통 | ⭐ 쉬움 | ⭐⭐⭐ 어려움 |
| 파라미터 추적 | ✅ | ⚠️ (별도 파일 필요) | ✅ |
| 최적 모델 찾기 | ⚠️ (수동) | ❌ | ✅ 자동 |
| 모델 비교 | ⚠️ | ❌ | ✅ |
| 재현성 | ✅ | ⚠️ | ✅ |
| 권장도 | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |

---

## 추천 구현 순서

### Phase 1: 빠른 개선 (1-2시간)
→ **방안 1** 구현
- MscnPilotModel, LeroPilotModel 수정
- 파라미터 기반 이름 생성
- 메타데이터 JSON 저장

### Phase 2: 완전한 시스템 (4-6시간)
→ **방안 3** 구현
- ModelRegistry 클래스 작성
- 모든 PilotModel 통합
- unified_test.py 연동

---

## 예상 결과

### Before (현재)
```
ExampleData/Mscn/Model/
  mscn                    # 항상 덮어쓰기
```

### After (개선)
```
ExampleData/Mscn/Model/
  mscn_epoch50_train500_collection500
  mscn_epoch50_train500_collection500.json
  mscn_epoch100_train1000_collection1000
  mscn_epoch100_train1000_collection1000.json
  mscn_epoch200_train500_collection500
  mscn_epoch200_train500_collection500.json

ExampleData/
  model_registry.json     # 전체 모델 메타데이터
```

### model_registry.json 예시
```json
{
  "mscn_epoch100_train500_collection500": {
    "algorithm": "mscn",
    "hyperparams": {
      "num_epoch": 100,
      "num_training": 500,
      "num_collection": 500
    },
    "model_path": "ExampleData/Mscn/Model/mscn_epoch100_train500_collection500",
    "performance": {
      "total_time": 45.23,
      "average_time": 0.52,
      "dataset": "production"
    },
    "trained_at": "2024-10-19T10:30:00"
  }
}
```

---

## CLI 도구 추가

```python
# scripts/model_manager.py
import argparse
from algorithm_examples.model_registry import ModelRegistry

def main():
    parser = argparse.ArgumentParser(description='Model management tool')
    parser.add_argument('command', choices=['list', 'best', 'compare', 'delete'])
    parser.add_argument('--algo', help='Algorithm name')
    parser.add_argument('--models', nargs='+', help='Model IDs to compare')
    
    args = parser.parse_args()
    registry = ModelRegistry()
    
    if args.command == 'list':
        models = registry.list_models(algorithm=args.algo)
        for model in models:
            print(f"{model['id']}: {model['performance']}")
    
    elif args.command == 'best':
        best = registry.get_best_model(args.algo)
        print(f"Best model: {best['id']}")
        print(f"Performance: {best['performance']}")
    
    elif args.command == 'compare':
        registry.compare_models(args.models)

if __name__ == '__main__':
    main()
```

**사용 예시**:
```bash
# 모든 MSCN 모델 조회
python scripts/model_manager.py list --algo mscn

# 최적 모델 찾기
python scripts/model_manager.py best --algo mscn

# 여러 모델 비교
python scripts/model_manager.py compare \
    --models mscn_epoch50_train500 mscn_epoch100_train500 mscn_epoch200_train500
```

---

## 결론

**추천**: 
1. 빠른 개선이 필요하면 → **방안 1** (파라미터 기반 이름)
2. 완전한 시스템이 필요하면 → **방안 3** (모델 레지스트리)

두 방안 모두 **파라미터 정보를 저장**하고 **모델 버전 관리**를 가능하게 합니다.

