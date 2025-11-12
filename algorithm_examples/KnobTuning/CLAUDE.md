# KnobTuning

DB 설정 파라미터(Knob) 자동 튜닝. Bayesian Optimization으로 최적 설정 탐색.

## 작동 방식

1. 다양한 Knob 설정으로 워크로드 실행
2. 각 설정의 성능 측정
3. 최적 Knob 조합 탐색
4. 최적 설정 적용

## 적합 워크로드

- 워크로드가 명확하게 정의된 경우
- DB 설정 조정으로 성능 개선 원함
- OLTP, OLAP 모두 가능

## 튜닝 대상 Knob

PostgreSQL: `shared_buffers`, `work_mem`, `effective_cache_size`, `random_page_cost` 등 100+ knobs

## 파일

```
KnobTuning/
├── KnobPresetScheduler.py      # PostgreSQL용
├── SparkKnobPresetScheduler.py # Spark용
├── EventImplement.py           # 탐색 로직
└── llamatune/                  # LlamaTune (Bayesian Optimization)
```

## 사용

```python
from algorithm_examples.KnobTuning.KnobPresetScheduler import get_knob_preset_scheduler

scheduler, tracker = get_knob_preset_scheduler(
    config,
    enable_collection=True,
    enable_training=True,
    dataset_name="your_db"
)
# scheduler.init() 시 최적 Knob 자동 탐색 및 적용
```

## 주요 컴포넌트

**EventImplement**: 워크로드 실행 및 Knob 탐색 (Bayesian Optimization)

**LlamaTune**: Bayesian Optimization 기반 탐색 알고리즘

**특이사항**:
- 모델 학습 아닌 **탐색(Search)** 수행
- 워크로드 전체를 반복 실행 (num_epoch 횟수)

## KnobTuning vs MSCN/Lero

| | KnobTuning | MSCN/Lero |
|---|---|---|
| 목적 | DB 설정 최적화 | 쿼리 최적화 |
| 적용 시점 | DB 시작 시 (전역) | 쿼리 실행 시 (쿼리별) |
| 변경 빈도 | 낮음 (한 번) | 높음 (쿼리마다) |
| 방식 | 탐색 | 학습 |

**조합 사용**: KnobTuning (DB 전역) + MSCN/Lero (개별 쿼리)

## 수정 시 주의

**탐색 Knob 범위**: 너무 넓으면 탐색 시간 증가, 너무 좁으면 최적 못 찾음

**워크로드 정의**: 대표 워크로드가 중요, 학습 쿼리와 실제 유사해야

**탐색 알고리즘 변경**: LlamaTune 외 Grid Search, Random Search, Genetic Algorithm 등 가능

## 성능 튜닝

**탐색 속도**: `num_epoch` 감소, 워크로드 크기 축소 (`num_training`), 탐색 공간 축소

**더 나은 Knob**: 더 많은 반복 (`num_epoch=200`), 더 넓은 탐색 공간, 실제와 유사한 워크로드

## 문제 해결

- 탐색 느림 → `num_epoch`, `num_training` 감소, 탐색 Knob 수 감소
- Baseline보다 나쁨 → 탐색 부족 (epoch 증가), 탐색 공간 확대, 워크로드 재선정
- Knob 적용 안 됨 → PostgreSQL 재시작 필요한 Knob 확인, 권한 확인 (`ALTER SYSTEM`)
- 워크로드마다 다름 → 정상 (Knob은 워크로드별 튜닝 필요)
