# PilotScope 문서 가이드

> PilotScope를 효과적으로 사용하기 위한 통합 문서

---

## 📚 문서 구조

### 개발 환경
- **[DOCKER_GUIDE.md](DOCKER_GUIDE.md)** - Docker 환경 구성 및 최적화
  - 빠른 시작
  - Volume mount 패턴
  - GPU 설정
  - PostgreSQL 최적화
  - 문제 해결

### 운영 최적화
- **[PRODUCTION_OPTIMIZATION.md](PRODUCTION_OPTIMIZATION.md)** - 운영 데이터 기반 쿼리 최적화
  - 5단계 빠른 시작 (30분)
  - 아키텍처 개요
  - ProductionDataset 구현
  - Config Sweep
  - Cross-Dataset Training

### 모델 관리
- **[MODEL_MANAGEMENT.md](MODEL_MANAGEMENT.md)** - 모델 버전 관리 시스템
  - 타임스탬프 기반 저장
  - 메타데이터 관리
  - CLI 도구 사용법
  - 실전 시나리오

---

## 🚀 시작하기

### 처음 사용자
1. [DOCKER_GUIDE.md](DOCKER_GUIDE.md) - Docker 환경 설정
2. [PRODUCTION_OPTIMIZATION.md](PRODUCTION_OPTIMIZATION.md) - 빠른 시작 섹션

### 운영 데이터로 최적화
1. [PRODUCTION_OPTIMIZATION.md](PRODUCTION_OPTIMIZATION.md) - 전체 가이드
2. [MODEL_MANAGEMENT.md](MODEL_MANAGEMENT.md) - 모델 관리

### 고급 사용자
- Config Sweep으로 최적 파라미터 탐색
- Cross-Dataset Training
- 모델 레지스트리 활용

---

## 📂 관련 문서

### algorithm_examples 폴더
- `CUSTOM_DATASET_GUIDE.md` - 임의의 데이터셋 추가
- `PRODUCTION_USAGE.md` - 실전 사용 가이드
- `README_RESULTS.md` - 결과 관리

### 외부 리소스
- **공식 문서**: https://woodybryant.github.io/PilotScopeDoc.io/
- **논문**: [paper/PilotScope.pdf](../paper/PilotScope.pdf)

---

## 🎯 주요 기능

### 1. Docker 개발 환경
- Volume mount로 즉각적인 코드 반영
- GPU 지원
- PostgreSQL 자동 최적화
- 일관된 개발 환경

### 2. 운영 데이터 최적화
- PostgreSQL 로그에서 쿼리 자동 추출
- 여러 AI 알고리즘 (MSCN, Lero) 비교
- Config Sweep으로 최적 파라미터 탐색
- Cross-Dataset Training

### 3. 모델 관리
- 타임스탬프 기반 자동 버전 관리
- Train/Test 결과 분리 추적
- CLI 도구로 쉬운 관리
- 최적 모델 자동 선택

---

## 💡 실전 워크플로우

### 워크플로우 1: 처음 시작
```bash
# 1. Docker 환경 시작
docker-compose up -d
docker-compose exec pilotscope-dev bash

# 2. 샘플 테스트
conda activate pilotscope
cd test_example_algorithms
python unified_test.py --algo baseline mscn --db stats_tiny --compare
```

### 워크플로우 2: 운영 데이터 최적화
```bash
# 1. 로그 추출
python scripts/extract_queries_from_log.py \
    --input /var/log/postgresql/postgresql.log \
    --output pilotscope/Dataset/Production/

# 2. ProductionDataset 생성 (코드 수정 필요)

# 3. 테스트 실행
python unified_test.py --algo baseline mscn lero --db production --compare

# 4. 최적 모델 확인
python scripts/model_manager.py best --algo mscn --dataset production
```

### 워크플로우 3: 최적 파라미터 탐색
```bash
# Config Sweep 실행
python algorithm_examples/config_sweep.py \
    --algo mscn \
    --db production \
    --param num_epoch 50 100 200 \
    --param num_training 500 1000 2000

# 결과 확인
python scripts/model_manager.py list --algo mscn
```

---

## 🔧 문제 해결

### Docker 관련
→ [DOCKER_GUIDE.md - 문제 해결](DOCKER_GUIDE.md#문제-해결)

### 운영 최적화 관련
→ [PRODUCTION_OPTIMIZATION.md - 문제 해결](PRODUCTION_OPTIMIZATION.md#문제-해결)

### 모델 관리 관련
→ [MODEL_MANAGEMENT.md](MODEL_MANAGEMENT.md)

---

## 📊 문서 변경 이력

### 2024-10 - 문서 통합 및 정리
- Docker 관련 문서 통합 (DOCKER.md + DOCKER_PERFORMANCE.md)
- 운영 최적화 문서 통합 (4개 문서 → 1개)
- 모델 관리 문서 통합 (3개 문서 → 1개)
- docs 폴더 구조로 재구성

### 이전 문서 (제거됨)
- ~~DOCKER.md~~
- ~~DOCKER_PERFORMANCE.md~~
- ~~README_운영최적화.md~~
- ~~빠른_시작_가이드.md~~
- ~~운영_데이터_기반_최적화_가이드.md~~
- ~~구현_요약.md~~
- ~~improved_model_management.md~~
- ~~model_versioning_guide.md~~
- ~~모델_관리_시스템_사용_가이드.md~~
- ~~WARP.md~~

---

## 🤝 기여하기

문서 개선 제안이나 오류 발견 시 이슈를 올려주세요!

---

**마지막 업데이트**: 2024-10
**버전**: 2.0 (통합 버전)
