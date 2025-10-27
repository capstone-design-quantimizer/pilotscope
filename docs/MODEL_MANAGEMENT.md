# 모델 관리 시스템 가이드

> 타임스탬프 기반 모델 버전 관리 + 풍부한 메타데이터

---

## 목차
1. [빠른 시작](#빠른-시작)
2. [설계 원칙](#설계-원칙)
3. [구현 상세](#구현-상세)
4. [CLI 도구 사용법](#cli-도구-사용법)
5. [실전 시나리오](#실전-시나리오)

---

## 빠른 시작

### 학습 및 자동 저장

```bash
# MSCN 학습 (자동으로 타임스탬프 이름 생성 및 메타데이터 저장)
cd test_example_algorithms
python unified_test.py --algo mscn --db production --epochs 100

# 출력:
# ✅ Model saved: ExampleData/Mscn/Model/mscn_20241019_103000
# ✅ Metadata saved: mscn_20241019_103000.json
# ✅ Model registered: mscn_20241019_103000
```

### 같은 모델로 여러 데이터셋 테스트

```bash
# 1. Production 데이터로 학습
python unified_test.py --algo mscn --db production --epochs 100
# → 저장: mscn_20241019_100000

# 2. 그 모델로 stats_tiny 테스트 (학습 안함)
python unified_test.py \
    --algo mscn \
    --db stats_tiny \
    --no-training \
    --load-model mscn_20241019_100000
# → 메타데이터 업데이트: testing 배열에 stats_tiny 결과 추가

# 3. 같은 모델로 imdb 테스트
python unified_test.py \
    --algo mscn \
    --db imdb \
    --no-training \
    --load-model mscn_20241019_100000
# → 메타데이터 업데이트: testing 배열에 imdb 결과 추가
```

### 최적 모델 자동 로드

```bash
# 학습 안하고 테스트만 하면 자동으로 최적 모델 로드
python unified_test.py --algo mscn --db production --no-training

# 출력:
# 📊 Loading best model for production: mscn_20241019_110000
```

---

## 설계 원칙

### 주요 특징

1. **타임스탬프로 저장**: 이름 충돌 없음, 부담 없이 저장
2. **풍부한 메타데이터**: 모든 변수 (파라미터, 데이터셋, 성능) 기록
3. **Train/Test 분리**: 학습과 테스트를 독립적으로 관리
4. **레지스트리로 관리**: 최적 모델 찾기, 비교, 정리
5. **CLI 도구**: 터미널에서 모델 관리

### 파일 구조

```
ExampleData/
├── Mscn/
│   ├── Model/
│   │   ├── mscn_20241019_103000          # 타임스탬프 모델
│   │   ├── mscn_20241019_103000.json     # 메타데이터
│   │   ├── mscn_20241019_150000
│   │   ├── mscn_20241019_150000.json
│   │   └── ...
│   └── training_registry.json
├── Lero/
│   ├── Model/
│   │   ├── lero_20241019_110000
│   │   └── ...
│   └── training_registry.json
└── model_registry.json                   # 중앙 레지스트리
```

### 메타데이터 구조

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
      "enable_collection": true,
      "enable_training": true
    },
    "trained_at": "2024-10-19T10:45:00",
    "training_time": 1800.5
  },

  "testing": [
    {
      "dataset": "production",
      "num_queries": 100,
      "tested_at": "2024-10-19T12:00:00",
      "performance": {
        "total_time": 45.23,
        "average_time": 0.4523,
        "num_queries": 100
      }
    },
    {
      "dataset": "stats_tiny",
      "num_queries": 80,
      "tested_at": "2024-10-19T13:00:00",
      "performance": {
        "total_time": 36.12,
        "average_time": 0.4515,
        "num_queries": 80
      }
    }
  ],

  "tags": ["production", "best"],
  "notes": "Production 데이터로 학습, 여러 데이터셋에서 좋은 성능",
  "created_at": "2024-10-19T10:30:00"
}
```

---

## 구현 상세

### EnhancedPilotModel 클래스

```python
# pilotscope/EnhancedPilotModel.py

import os
import json
from datetime import datetime
from pilotscope.PilotModel import PilotModel

class EnhancedPilotModel(PilotModel):
    """타임스탬프 기반 모델 저장 + 풍부한 메타데이터"""

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

    def set_training_info(self, dataset, hyperparams, num_queries=None):
        """학습 정보 설정"""
        self.metadata["training"] = {
            "enabled": True,
            "dataset": dataset,
            "num_queries": num_queries,
            "hyperparams": hyperparams,
            "trained_at": datetime.now().isoformat(),
            "training_time": None
        }

    def add_test_result(self, dataset, num_queries, performance):
        """테스트 결과 추가 (여러 데이터셋에 대해 누적 가능)"""
        test_result = {
            "dataset": dataset,
            "num_queries": num_queries,
            "tested_at": datetime.now().isoformat(),
            "performance": performance
        }
        self.metadata["testing"].append(test_result)

    def add_tags(self, *tags):
        """태그 추가"""
        self.metadata["tags"].extend(tags)

    def save_model(self):
        """모델 + 메타데이터 저장"""
        # 1. 모델 저장 (자식 클래스에서 구현)
        self._save_model_impl()

        # 2. 메타데이터 저장
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ Model saved: {self.model_path}")
        print(f"✅ Metadata saved: {self.metadata_path}")

    @classmethod
    def load_model(cls, model_id, algorithm_type):
        """특정 모델 로드"""
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
```

### ModelRegistry 클래스

```python
# pilotscope/ModelRegistry.py

import json
import os
from typing import Dict, List, Optional

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

    def register_model(self, metadata):
        """모델 등록"""
        registry = self._load_registry()
        model_id = metadata["model_id"]
        registry[model_id] = metadata
        self._save_registry(registry)
        print(f"✅ Model registered: {model_id}")

    def list_models(self, algorithm=None, dataset=None, tags=None):
        """모델 목록 조회"""
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

        return models

    def get_best_model(self, algorithm, test_dataset=None, metric="total_time"):
        """최적 모델 조회"""
        models = self.list_models(algorithm=algorithm)

        if not models:
            return None

        # 각 모델의 특정 데이터셋 성능 추출
        candidates = []
        for model in models:
            for test in model.get("testing", []):
                if test_dataset and test["dataset"] != test_dataset:
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

    def cleanup_old_models(self, algorithm, keep_top_n=5):
        """
        오래된 모델 정리 (상위 N개만 유지)
        Returns: 삭제된 모델 ID 리스트
        """
        models = self.list_models(algorithm=algorithm)

        if len(models) <= keep_top_n:
            print(f"ℹ️  Only {len(models)} models, no cleanup needed")
            return []

        # 성능 순으로 정렬
        models.sort(key=lambda x: min(
            t["performance"].get("total_time", float('inf'))
            for t in x.get("testing", [{}])
        ))

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
```

---

## CLI 도구 사용법

### 모델 목록 조회

```bash
cd scripts

# 모든 모델 조회
python model_manager.py list

# MSCN 모델만 조회
python model_manager.py list --algo mscn

# Production 데이터로 학습한 모델만
python model_manager.py list --algo mscn --dataset production

# 특정 태그가 있는 모델만
python model_manager.py list --tags production best
```

**출력 예시**:
```
====================================================================================================
Found 5 models
====================================================================================================
Model ID                         Algorithm  Train Dataset    Tests  Tags
----------------------------------------------------------------------------------------------------
mscn_20241019_150000             mscn       production       3      production, best
mscn_20241019_140000             mscn       production       2      production
mscn_20241019_130000             mscn       stats_tiny       1      test
lero_20241019_120000             lero       production       2      production
lero_20241019_110000             lero       imdb             1      -
====================================================================================================
```

### 최적 모델 찾기

```bash
# MSCN 최적 모델
python model_manager.py best --algo mscn

# Production 데이터셋에서 최적 모델
python model_manager.py best --algo mscn --dataset production
```

**출력 예시**:
```
================================================================================
🏆 Best Model: mscn_20241019_110000
================================================================================

Training:
  Dataset:      production
  Hyperparams:  {'num_epoch': 100, 'num_training': 500, 'num_collection': 500}
  Training Time: 1800.5s

Test Results:
  production:
    Total Time:   42.15s
    Average Time: 0.4215s
    Queries:      100
  stats_tiny:
    Total Time:   35.20s
    Average Time: 0.4400s
    Queries:      80

Tags: production, best
================================================================================
```

### 모델 비교

```bash
python model_manager.py compare \
    mscn_20241019_103000 \
    mscn_20241019_110000 \
    mscn_20241019_120000
```

**출력 예시**:
```
====================================================================================================
Model Comparison
====================================================================================================
Model ID                       Train Dataset   Test Dataset    Total Time   Avg Time   Epochs
----------------------------------------------------------------------------------------------------
mscn_20241019_103000           production      production      45.23        0.4523     100
                                               stats_tiny      36.12        0.4515
mscn_20241019_110000           production      production      42.15        0.4215     100
                                               stats_tiny      35.20        0.4400
mscn_20241019_120000           production      production      48.56        0.4856     50
====================================================================================================
```

### 모델 상세 정보

```bash
python model_manager.py show mscn_20241019_103000
```

### 태그 관리

```bash
# 태그 추가
python model_manager.py tag mscn_20241019_103000 production best

# 태그 제거
python model_manager.py tag mscn_20241019_103000 experiment --remove
```

### 오래된 모델 정리

```bash
# 상위 5개만 유지하고 나머지 삭제
python model_manager.py cleanup --algo mscn --keep 5

# 확인 없이 바로 삭제
python model_manager.py cleanup --algo mscn --keep 5 --yes
```

**출력 예시**:
```
🗑️  Cleaning up mscn models...
   Keeping top 5 models by total_time

Proceed? (yes/no): yes

🗑️  Deleted: mscn_20241018_100000
🗑️  Deleted: mscn_20241018_110000
🗑️  Deleted: mscn_20241018_120000

✅ Cleanup complete: 3 models deleted, 5 kept
```

### 요약 보기

```bash
# 전체 요약
python model_manager.py summary

# 특정 알고리즘만
python model_manager.py summary --algo mscn
```

---

## 실전 시나리오

### 시나리오 1: 실험적 학습 (임시 저장)

```bash
# 1. 여러 파라미터 조합 시도 (부담 없이 저장)
python unified_test.py --algo mscn --db production --epochs 50
python unified_test.py --algo mscn --db production --epochs 100
python unified_test.py --algo mscn --db production --epochs 200

# 2. 결과 비교
python scripts/model_manager.py list --algo mscn

# 3. 최적 모델 태그
python scripts/model_manager.py best --algo mscn
# 🏆 Best Model: mscn_20241019_110000

python scripts/model_manager.py tag mscn_20241019_110000 best production

# 4. 안좋은 모델 정리
python scripts/model_manager.py cleanup --algo mscn --keep 3
```

### 시나리오 2: Cross-Dataset 평가

```bash
# 1. IMDB 데이터로 학습
python unified_test.py --algo mscn --db imdb --epochs 100
# → mscn_20241019_100000

# 2. 그 모델로 여러 데이터셋 테스트
python unified_test.py --algo mscn --db production \
    --no-training --load-model mscn_20241019_100000

python unified_test.py --algo mscn --db stats_tiny \
    --no-training --load-model mscn_20241019_100000

# 3. 결과 확인
python scripts/model_manager.py show mscn_20241019_100000

# Output:
# Training: imdb
# Test Results:
#   - imdb: 120.5s
#   - production: 45.2s
#   - stats_tiny: 35.1s
```

### 시나리오 3: 프로덕션 배포

```bash
# 1. 최적 모델 찾기
python scripts/model_manager.py best --algo mscn --dataset production

# 2. 프로덕션 태그 추가
python scripts/model_manager.py tag mscn_20241019_110000 production deployed

# 3. 프로덕션 환경에서 사용
python unified_test.py --algo mscn --db production --no-training
# → 자동으로 mscn_20241019_110000 로드

# 4. 주기적으로 재평가
python unified_test.py --algo mscn --db production --no-training
# → 새 데이터로 테스트, 메타데이터 업데이트
```

---

## Python API 사용

### 프로그래밍 방식으로 모델 관리

```python
from pilotscope.ModelRegistry import ModelRegistry
from pilotscope.EnhancedPilotModel import EnhancedPilotModel
from algorithm_examples.Mscn.MscnPilotModel import MscnPilotModel

# 1. 레지스트리 생성
registry = ModelRegistry()

# 2. 최적 모델 찾기
best = registry.get_best_model("mscn", test_dataset="production")
print(f"Best model: {best['model_id']}")

# 3. 모델 로드
model = MscnPilotModel.load_model(best['model_id'], "mscn")

# 4. 새로운 데이터셋에서 테스트 결과 추가
model.add_test_result("new_dataset", 100, {
    "total_time": 50.5,
    "average_time": 0.505
})
model.save_model()

# 5. 레지스트리 업데이트
registry.register_model(model.metadata)

# 6. 태그 추가
registry.tag_model(model.model_id, "validated", "production")

# 7. 정리
deleted = registry.cleanup_old_models("mscn", keep_top_n=5)
print(f"Deleted {len(deleted)} models")
```

---

## 장점 요약

| 기능 | 이전 (파라미터 기반) | 현재 (타임스탬프 + 메타데이터) |
|------|---------------------|------------------------------|
| 이름 충돌 | ❌ 데이터셋 다르면 덮어쓰기 | ✅ 절대 없음 |
| 임시 저장 | ❌ 이름 설계 필요 | ✅ 부담 없이 저장 |
| Train/Test 분리 | ❌ 불가능 | ✅ 완벽 지원 |
| 여러 데이터셋 테스트 | ❌ 매번 새 모델 | ✅ 한 모델에 누적 |
| 최적 모델 찾기 | ⚠️ 수동 | ✅ 자동 |
| 모델 비교 | ⚠️ 어려움 | ✅ 쉬움 |
| 모델 정리 | ❌ 수동 | ✅ 자동 |
| 실험 추적 | ⚠️ 제한적 | ✅ 완벽 |

---

## 구현 파일 목록

### 완료 ✅
1. `pilotscope/EnhancedPilotModel.py` - 타임스탬프 + 메타데이터
2. `pilotscope/ModelRegistry.py` - 중앙 레지스트리
3. `algorithm_examples/Mscn/MscnPilotModel.py` - Enhanced 버전
4. `algorithm_examples/Lero/LeroPilotModel.py` - Enhanced 버전
5. `scripts/model_manager.py` - CLI 관리 도구

### 사용 흐름

```
학습 → 타임스탬프 모델 생성 → 메타데이터 저장 → 레지스트리 등록
  ↓
테스트 → 기존 모델 로드 → 결과 추가 → 메타데이터 업데이트
  ↓
관리 → CLI로 조회/비교/정리 → 최적 모델 선택 → 프로덕션 배포
```

---

**🎉 이제 부담 없이 실험하고, 최적 모델을 찾고, 효율적으로 관리할 수 있습니다!**
