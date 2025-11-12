# MSCN (Multi-Set Convolutional Network)

카디널리티 추정. PostgreSQL 옵티마이저의 카디널리티 추정을 AI 모델로 대체.

## 작동 방식

1. SQL → 피처 벡터 변환
2. MSCN 모델 → 카디널리티 예측
3. 예측값 PostgreSQL에 힌트 주입
4. PostgreSQL이 AI 예측으로 플랜 생성

## 적합 워크로드

- 복잡한 JOIN 쿼리
- 옵티마이저 카디널리티 추정 부정확
- OLAP (분석 쿼리)

## 파일

```
Mscn/
├── MscnPresetScheduler.py          # 팩토리
├── MscnPilotModel.py               # 모델 래퍼
├── MscnParadigmCardAnchorHandler.py # 카디널리티 힌트 주입
├── EventImplement.py               # 학습/수집
└── source/mscn.py                  # MSCN 모델
```

## 사용

```python
from algorithm_examples.Mscn.MscnPresetScheduler import get_mscn_preset_scheduler

scheduler, tracker = get_mscn_preset_scheduler(
    config,
    enable_collection=True,
    enable_training=True,
    num_epoch=100,
    dataset_name="your_db"
)
```

## 주요 컴포넌트

**MscnCardPushHandler**: 카디널리티 예측 → PostgreSQL 주입

**MscnPretrainingModelEvent**: 학습 데이터 수집 (ground truth 카디널리티)

**특이사항**:
- Lero와 달리 데이터 수집 시 기존 데이터 자동 삭제 안 함 (누적)
- 모델 저장: `ExampleData/Mscn/Model/mscn_{timestamp}`

## 수정 시 주의

**피처 추출 변경**: 새 피처 형식은 별도 `model_name` 사용 (기존 모델 비호환)

**하이퍼파라미터**: 하드코딩 대신 `**kwargs`로 받기

**데이터 테이블 변경**: 기존 데이터 마이그레이션 필요

## 성능 튜닝

**학습 속도**: `num_collection`, `num_training`, `num_epoch` 감소

**정확도**: 더 많은 데이터, 더 많은 epoch

## 문제 해결

- 학습 느림 → 데이터 크기 감소, 복잡한 쿼리 제외
- 예측 부정확 → 데이터 부족 (500개 이상 권장), 학습/테스트 분포 유사하게
- Baseline보다 나쁨 → MSCN은 복잡한 JOIN에 효과적, 단순 쿼리는 오버헤드
