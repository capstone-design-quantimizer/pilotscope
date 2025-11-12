# Lero (Learning to Rewrite Optimizer)

쿼리 플랜 최적화. 강화학습으로 최적 플랜 선택.

## 작동 방식

1. PostgreSQL에서 여러 플랜 생성 (hint 변경)
2. Lero 모델로 각 플랜 비용 예측
3. 최소 비용 플랜 선택
4. 선택된 플랜으로 실행

## 적합 워크로드

- 플랜이 다양하게 나올 수 있는 경우
- JOIN 순서가 성능에 큰 영향
- OLAP (복잡한 분석 쿼리)

## MSCN vs Lero

- MSCN: 카디널리티 추정 (중간 결과 크기)
- Lero: 플랜 선택 (전체 실행 비용)

| | MSCN | Lero |
|---|---|---|
| AI 기술 | 딥러닝 (CNN) | 강화학습 (RL) |
| 학습 방식 | 지도학습 | 강화학습 |
| 추론 속도 | 빠름 | 느림 (여러 플랜 탐색) |
| GPU | 선택 | 권장 (PyTorch) |
| 데이터 관리 | 누적 가능 | 자동 삭제 |

## 파일

```
Lero/
├── LeroPresetScheduler.py          # 팩토리
├── LeroPilotModel.py               # 모델 래퍼 (PyTorch)
├── LeroParadigmCardAnchorHandler.py # 플랜 힌트 주입
├── LeroPilotAdapter.py             # Lero 원본 어댑터
├── EventImplement.py               # 학습/수집
└── source/                         # Lero 모델 (PyTorch)
```

## 사용

```python
from algorithm_examples.Lero.LeroPresetScheduler import get_lero_preset_scheduler

scheduler, tracker = get_lero_preset_scheduler(
    config,
    enable_collection=True,
    enable_training=True,
    num_epoch=100,
    dataset_name="your_db"
)
```

## 주요 컴포넌트

**LeroCardPushHandler**: 여러 플랜 생성 → 비용 예측 → 최적 플랜 선택 → PostgreSQL 주입

**LeroPretrainingModelEvent**: 여러 플랜 실행 및 비용 측정 (강화학습 데이터)

**특이사항**:
- `enable_collection=True` 시 기존 데이터 **자동 삭제** (새로운 탐색 필요)
- Physical plan 함께 수집 (`pull_physical_plan=True`)
- GPU 지원 (PyTorch)
- 모델 저장: `ExampleData/Lero/Model/lero_{timestamp}`

## 동적 학습

주기적 모델 업데이트 지원 (MSCN 미지원):

```python
scheduler = get_lero_dynamic_preset_scheduler(config, dataset_name="your_db")
# 100개 쿼리마다 자동 재학습
```

## 수정 시 주의

**플랜 피처 변경**: 플랜 트리 구조 피처 사용, 형식 변경 시 기존 모델 비호환

**Hint Space 변경**: 크면 수집 시간 증가, 작으면 최적 플랜 못 찾음

**데이터 자동 삭제**: 강화학습 특성상 이전 데이터가 탐색에 방해 (삭제 로직 제거 비권장)

## 성능 튜닝

**학습 속도**: GPU 사용, Hint space 축소, `num_collection`/`num_training` 감소

**정확도**: 더 많은 데이터/epoch, Hint space 확대

## 문제 해결

- 학습 매우 느림 → GPU 확인 (`torch.cuda.is_available()`), Hint space 감소
- 메모리 부족 → GPU 메모리 부족 시 배치 크기 감소 또는 CPU 사용
- Baseline보다 나쁨 → 학습 초기 (탐색 단계), 더 많은 epoch (200+)
- 추론 느림 → Hint space 감소 또는 MSCN 고려
