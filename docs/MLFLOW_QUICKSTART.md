# MLflow Quick Start Guide

MLflow 통합으로 실험 관리가 자동화되었습니다. 5분 안에 시작할 수 있습니다.

## 1단계: 환경 설정 (Docker)

```bash
# Docker 빌드 (MLflow 자동 설치됨)
docker-compose build

# 컨테이너 시작
docker-compose up -d

# 컨테이너 접속
docker-compose exec pilotscope-dev bash
```

## 2단계: 실험 실행

```bash
# Conda 환경 활성화
conda activate pilotscope

# 테스트 실행 (MLflow 자동 추적)
cd test_example_algorithms
python unified_test.py --algo mscn --db stats_tiny
```

**자동으로 기록되는 정보:**
- ✅ 하이퍼파라미터 (num_epoch, learning_rate 등)
- ✅ 학습 메트릭 (epoch별 loss)
- ✅ 테스트 결과 (total_time, average_time 등)
- ✅ 모델 파일 백업

## 3단계: 결과 확인

**방법 1: 웹 UI (권장)**

```bash
# MLflow UI 시작
./scripts/mlflow_ui.sh

# 브라우저에서 http://localhost:5000 접속
```

**방법 2: CLI**

```bash
# 모든 실험 목록
python scripts/mlflow_query.py list

# 특정 실험의 run 목록
python scripts/mlflow_query.py runs mscn_stats_tiny

# 최고 성능 모델 찾기
python scripts/mlflow_query.py best mscn_stats_tiny

# 두 run 비교
python scripts/mlflow_query.py compare <run_id1> <run_id2>
```

## 주요 기능

### 자동 실험 추적
```python
# 단순히 테스트 실행만 하면 자동으로 모든 정보가 기록됨
python unified_test.py --algo mscn --db stats_tiny
```

### 최고 모델 자동 로드
```python
# 학습 없이 최고 모델만 사용
python unified_test.py --algo mscn --db stats_tiny --no-training
# MLflow에서 자동으로 최고 성능 모델 로드
```

### 여러 알고리즘 비교
```bash
# 여러 알고리즘 개별 실행 (결과는 MLflow에 자동 저장)
python unified_test.py --algo baseline --db stats_tiny
python unified_test.py --algo mscn --db stats_tiny
python unified_test.py --algo lero --db stats_tiny --timeout 900

# MLflow UI에서 시각적으로 비교
# 브라우저: http://localhost:54321
```

## 디렉토리 구조

```
pilotscope/
├── mlruns/                    # MLflow 실험 데이터 (자동 생성)
│   ├── 1/                    # mscn_stats_tiny 실험
│   └── 2/                    # lero_stats_tiny 실험
├── algorithm_examples/
│   └── ExampleData/          # 기존 모델 파일 (여전히 사용됨)
└── test_example_algorithms/
    └── results/              # JSON 결과 (호환성 유지)
```

## 주요 명령어

```bash
# 실험 실행
python unified_test.py --algo mscn --db stats_tiny

# MLflow UI 시작
./scripts/mlflow_ui.sh

# 실험 조회
python scripts/mlflow_query.py list
python scripts/mlflow_query.py runs mscn_stats_tiny
python scripts/mlflow_query.py best mscn_stats_tiny

# 비교
python scripts/mlflow_query.py compare <run1> <run2>
```

## 다음 단계

- 📚 상세 가이드: [MLFLOW_GUIDE.md](./MLFLOW_GUIDE.md)
- 🐳 Docker 환경: [DOCKER_GUIDE.md](./DOCKER_GUIDE.md)
- 🤖 알고리즘 예제: [CLAUDE.md](../CLAUDE.md)

## 문제 해결

### MLflow UI 접속 불가
```bash
# 포트 확인
docker-compose ps
# 5000:5000 매핑 확인

# 재시작
docker-compose restart
```

### 이전 JSON 파일은?
기존 `results/*.json` 파일도 여전히 생성됩니다. 호환성 유지됨.

```bash
# 기존 방식도 계속 사용 가능
python algorithm_examples/compare_results.py --latest mscn lero
```
