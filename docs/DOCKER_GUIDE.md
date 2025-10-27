# Docker 개발 및 성능 최적화 가이드

> PilotScope를 Docker로 개발하고 최적의 성능을 내는 완전한 가이드

---

## 목차
1. [빠른 시작](#빠른-시작)
2. [개발 환경 구성](#개발-환경-구성)
3. [성능 최적화](#성능-최적화)
4. [문제 해결](#문제-해결)

---

## 빠른 시작

### Docker Compose 사용 (권장)

```powershell
# 1. 컨테이너 시작
docker-compose up -d

# 2. 컨테이너 접속
docker-compose exec pilotscope-dev bash

# 3. 환경 활성화 및 실행
conda activate pilotscope
python test_example_algorithms/simple_baseline.py

# 4. 종료
docker-compose down
```

### 수동 실행

```powershell
# 1. 이미지 빌드 (최초 1회)
docker build -f Dockerfile.dev -t pilotscope:dev `
  --build-arg enable_postgresql=true .

# 2. 컨테이너 실행
docker run -it --rm `
  -v ${PWD}:/home/pilotscope/workspace `
  -w /home/pilotscope/workspace `
  pilotscope:dev bash

# 3. 컨테이너 내부
conda activate pilotscope
python test_example_algorithms/simple_baseline.py
```

---

## 개발 환경 구성

### Volume Mount 방식

현재 프로젝트는 **volume mount** 패턴을 사용합니다:

```
호스트 (Windows)              Docker 컨테이너 (Linux)
─────────────────            ───────────────────────
pilotscope/                  /home/pilotscope/workspace/
  ├── pilotscope/    <──┐      ├── pilotscope/
  └── test_*.py         └────>  └── test_*.py

호스트에서 편집               컨테이너에 즉시 반영!
(VS Code, IDE)               (재빌드 불필요)
```

### 장점
1. **즉각적인 코드 반영** - 이미지 재빌드 없이 코드 수정이 바로 적용됨
2. **일관된 환경** - PostgreSQL, Conda 등 복잡한 의존성이 항상 동일한 환경에서 실행됨
3. **호스트 독립성** - Windows에서 개발해도 Linux 환경에서 실행됨

### 두 가지 Dockerfile

- **`Dockerfile`** (프로덕션)
  - GitHub에서 코드를 clone
  - 배포용

- **`Dockerfile.dev`** (개발)
  - 로컬 파일을 volume mount로 사용
  - 빠른 개발 사이클

### 재빌드가 필요한 경우

#### ❌ 재빌드 불필요 (99%)
- Python 코드 수정
- 새 .py 파일 추가
- 설정 파일 변경

#### ✅ 재빌드 필요
- `requirements.txt` 수정
- `Dockerfile.dev` 변경
- PostgreSQL/Spark 설정 변경

```powershell
# 재빌드 명령
docker-compose build
# 또는
docker build -f Dockerfile.dev -t pilotscope:dev .
```

### 빌드 옵션

```powershell
# PostgreSQL만 (빠름, ~10분)
docker build -f Dockerfile.dev -t pilotscope:dev `
  --build-arg enable_postgresql=true `
  --build-arg enable_spark=false .

# Spark 포함 (느림, ~30분)
docker build -f Dockerfile.dev -t pilotscope:dev `
  --build-arg enable_postgresql=true `
  --build-arg enable_spark=true .
```

### Entrypoint 자동화

컨테이너 시작 시 [docker-entrypoint.sh](../docker-entrypoint.sh)가 자동으로:
1. PostgreSQL 시작
2. Conda 환경 활성화
3. `requirements.txt` 동기화 (변경사항 있을 시)

---

## 성능 최적화

### 1. GPU 설정

#### 사전 요구사항

**호스트 시스템에 필요:**
```bash
# 1. NVIDIA Driver 설치
sudo apt-get install -y nvidia-driver-535

# 2. NVIDIA Container Toolkit 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 3. 확인
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

#### Docker Compose 설정

```yaml
# docker-compose.yml에 이미 포함됨
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all  # 모든 GPU 사용
          capabilities: [gpu]
```

#### GPU 없는 환경
GPU가 없는 경우 `docker-compose-no-gpu.yml` 사용:
```bash
docker-compose -f docker-compose-no-gpu.yml up -d
```

#### GPU 확인
```bash
# 컨테이너 접속
docker exec -it pilotscope-dev bash
conda activate pilotscope

# GPU 사용 가능 여부 확인
python scripts/check_gpu.py

# 성능 벤치마크
python scripts/check_gpu.py --benchmark
```

**예상 출력:**
```
✅ PyTorch CUDA 사용 가능
   GPU 개수: 1
   GPU 0: NVIDIA GeForce RTX 3080 (10.00 GB)

🖥️  CPU 테스트: 45.23초 (4.523초/회)
🚀 GPU 테스트: 2.15초 (0.215초/회)
📊 GPU 속도 향상: 21.0x 빠름
```

### 2. 리소스 최적화

#### 현재 설정
```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '0'      # 무제한
      memory: 32G    # 최대 32GB
    reservations:
      cpus: '4'      # 최소 4 코어
      memory: 8G     # 최소 8GB
```

#### 시스템별 권장 설정

**고사양 워크스테이션 (64GB+ RAM)**
```yaml
limits:
  memory: 48G
reservations:
  cpus: '8'
  memory: 16G
```

**중사양 PC (32GB RAM)**
```yaml
limits:
  memory: 24G
reservations:
  cpus: '4'
  memory: 8G
```

**저사양 PC (16GB RAM)**
```yaml
limits:
  memory: 12G
reservations:
  cpus: '2'
  memory: 4G
```

#### 리소스 모니터링
```bash
# 실시간 사용량 확인
docker stats pilotscope-dev

# 출력 예시:
# CONTAINER ID   NAME             CPU %     MEM USAGE / LIMIT   MEM %
# abc123def456   pilotscope-dev   350.12%   8.5GiB / 32GiB     26.56%
```

### 3. PostgreSQL 최적화

#### 자동 최적화 (권장)
```bash
docker exec -it pilotscope-dev bash
bash scripts/optimize_postgresql.sh
```

#### 수동 최적화

```bash
# PostgreSQL 설정 파일 편집
vim /home/pilotscope/pg_data/postgresql.conf
```

**권장 설정 (32GB RAM 기준):**
```ini
# 메모리
shared_buffers = 8GB              # 전체 RAM의 25%
effective_cache_size = 24GB       # 전체 RAM의 75%
work_mem = 256MB                  # 쿼리당 메모리
maintenance_work_mem = 2GB        # VACUUM, CREATE INDEX

# 성능
random_page_cost = 1.1            # SSD 사용 시
effective_io_concurrency = 200    # SSD 병렬 I/O
default_statistics_target = 100   # 통계 정확도

# 병렬 쿼리
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_worker_processes = 8

# 체크포인트
checkpoint_completion_target = 0.9
wal_buffers = 16MB
max_wal_size = 4GB
```

**적용:**
```bash
pg_ctl restart -D /home/pilotscope/pg_data
```

#### PostgreSQL 성능 확인
```bash
# 현재 설정 확인
psql -d pilotscope -c "
SELECT name, setting, unit
FROM pg_settings
WHERE name IN ('shared_buffers', 'work_mem', 'max_parallel_workers');
"

# 활성 연결 수
psql -d pilotscope -c "SELECT count(*) FROM pg_stat_activity;"

# 느린 쿼리 확인 (1초 이상)
psql -d pilotscope -c "
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
WHERE mean_time > 1000
ORDER BY total_time DESC
LIMIT 10;
"
```

### 4. 성능 벤치마크

#### GPU 성능 비교
```bash
# Lero 학습 (GPU 사용)
python unified_test.py --algo lero --db stats_tiny \
    --collection-size 100 \
    --training-size 500 \
    --epochs 100
# 예상 시간: 1-2시간

# CPU만 사용
export CUDA_VISIBLE_DEVICES=""
python unified_test.py --algo lero --db stats_tiny \
    --collection-size 100 \
    --training-size 500 \
    --epochs 100
# 예상 시간: 5-10시간 (2-5배 느림)
```

#### PostgreSQL 쿼리 성능
```bash
psql -d stats_tiny -c "
EXPLAIN ANALYZE
SELECT count(*)
FROM badges b
JOIN posts p ON b.userid = p.owneruserid
WHERE p.score > 10;
"
```

---

## 문제 해결

### GPU 인식 안됨

#### 확인 1: 호스트에서 GPU 확인
```bash
nvidia-smi
```
→ 오류 발생 시: NVIDIA Driver 재설치

#### 확인 2: Docker GPU 지원 확인
```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```
→ 오류 발생 시: nvidia-docker2 재설치

#### 확인 3: Docker Compose 설정 확인
```bash
grep -A 5 "devices:" docker-compose.yml
```

### 메모리 부족

**증상:**
```
docker: Error response from daemon: OCI runtime create failed
```

**해결:**
```yaml
# docker-compose.yml 수정
deploy:
  resources:
    limits:
      memory: 16G  # 메모리 제한 줄이기
```

### PostgreSQL 연결 실패

```bash
# PostgreSQL 상태 확인
docker exec -it pilotscope-dev bash
ps aux | grep postgres

# 로그 확인
cat /home/pilotscope/pg_data/logfile

# 재시작
pg_ctl restart -D /home/pilotscope/pg_data
```

### Port 충돌

```powershell
# 다른 포트 사용
docker run -it -v ${PWD}:/workspace -p 5433:5432 pilotscope:dev bash
```

### 코드 변경이 반영 안됨

**확인사항:**
- Volume mount가 올바른가? `-v ${PWD}:/workspace`
- 올바른 경로에서 파일을 수정하고 있는가?
- 컨테이너 재시작 필요

```bash
docker-compose restart
```

### Line Ending 오류 (`\r` 오류)

Windows (CRLF) vs Linux (LF) 차이로 발생:

```powershell
# ✅ 단순한 명령 사용
docker run -it -v ${PWD}:/workspace pilotscope:dev bash

# ❌ 여러 줄 명령 피하기
docker run ... bash -c "multiple
lines
here"
```

---

## 성능 최적화 체크리스트

학습/테스트 전 확인:

- [ ] GPU 사용 가능 확인 (`python scripts/check_gpu.py`)
- [ ] Docker 리소스 할당 확인 (`docker stats`)
- [ ] PostgreSQL 최적화 적용 (`bash scripts/optimize_postgresql.sh`)
- [ ] 충분한 디스크 공간 (최소 50GB)
- [ ] Shared memory 설정 (`shm_size: 8gb`)

---

## 권장 워크플로우

### Phase 1: 환경 확인 (1회)

```bash
# GPU 확인
docker exec -it pilotscope-dev bash -c \
  "conda activate pilotscope && python scripts/check_gpu.py"

# PostgreSQL 최적화
docker exec -it pilotscope-dev bash -c \
  "bash scripts/optimize_postgresql.sh"
```

### Phase 2: 빠른 테스트

```bash
# 작은 데이터로 GPU/CPU 성능 비교
python unified_test.py --algo mscn --db stats_tiny \
    --collection-size 100 \
    --training-size 500 \
    --epochs 10
```

### Phase 3: 전체 학습

```bash
# GPU 사용하여 전체 데이터 학습
python unified_test.py --algo lero --db production \
    --collection-size 500 \
    --training-size 2000 \
    --epochs 100 \
    --timeout 1800
```

---

## 파일 구조

```
pilotscope/
├── Dockerfile              # 프로덕션용
├── Dockerfile.dev          # 개발용
├── docker-compose.yml      # Docker Compose 설정
├── docker-entrypoint.sh    # 자동 초기화 스크립트
├── .dockerignore          # Docker 빌드 제외 파일
└── docs/
    └── DOCKER_GUIDE.md    # 이 문서
```

---

## 참고 자료

- **NVIDIA Docker**: https://github.com/NVIDIA/nvidia-docker
- **PostgreSQL 튜닝**: https://pgtune.leopard.in.ua/
- **Docker 리소스 관리**: https://docs.docker.com/config/containers/resource_constraints/

---

**문제가 계속되면 시스템 사양을 확인하거나 이슈를 올려주세요! 🚀**
