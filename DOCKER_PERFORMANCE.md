# Docker 컨테이너 성능 최적화 가이드

## 목차
1. [GPU 설정](#gpu-설정)
2. [리소스 최적화](#리소스-최적화)
3. [PostgreSQL 최적화](#postgresql-최적화)
4. [성능 벤치마크](#성능-벤치마크)

---

## GPU 설정

### 사전 요구사항

#### 1. NVIDIA Driver 설치 (호스트 시스템)

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y nvidia-driver-535  # 버전은 GPU에 맞게 조정

# 확인
nvidia-smi
```

#### 2. NVIDIA Container Toolkit 설치

```bash
# Docker에서 GPU를 사용하기 위한 toolkit 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 확인
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Docker Compose 설정

#### GPU 사용 (기본)

```bash
# GPU를 사용하는 설정으로 시작
docker-compose up -d
```

`docker-compose.yml`에 이미 GPU 설정이 포함되어 있습니다:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all  # 모든 GPU 사용
          capabilities: [gpu]
```

#### GPU 없이 사용

```bash
# GPU가 없는 환경에서는 no-gpu 버전 사용
docker-compose -f docker-compose-no-gpu.yml up -d
```

### GPU 사용 확인

```bash
# 컨테이너 접속
docker exec -it pilotscope-dev bash

# conda 환경 활성화
conda activate pilotscope

# GPU 확인 스크립트 실행
python scripts/check_gpu.py

# 성능 벤치마크 (선택)
python scripts/check_gpu.py --benchmark
```

**예상 출력:**
```
============================================================
GPU 및 CUDA 설정 확인
============================================================

✅ PyTorch 버전: 2.0.1
   CUDA 사용 가능: ✅ YES
   CUDA 버전: 11.7
   GPU 개수: 1

   📊 GPU 0:
      이름: NVIDIA GeForce RTX 3080
      메모리: 10.00 GB
      사용 중: 0.12 GB
      예약됨: 0.50 GB

   🧪 GPU 테스트 중...
   ✅ GPU 연산 테스트 성공!
============================================================
```

---

## 리소스 최적화

### 현재 설정 (docker-compose.yml)

```yaml
deploy:
  resources:
    limits:
      cpus: '0'      # 무제한 (모든 CPU 사용)
      memory: 32G    # 최대 32GB
    reservations:
      cpus: '4'      # 최소 4 CPU 코어
      memory: 8G     # 최소 8GB
```

### 시스템별 권장 설정

#### 고사양 워크스테이션 (64GB+ RAM)
```yaml
deploy:
  resources:
    limits:
      memory: 48G
    reservations:
      cpus: '8'
      memory: 16G
```

#### 중사양 PC (32GB RAM)
```yaml
deploy:
  resources:
    limits:
      memory: 24G
    reservations:
      cpus: '4'
      memory: 8G
```

#### 저사양 PC (16GB RAM)
```yaml
deploy:
  resources:
    limits:
      memory: 12G
    reservations:
      cpus: '2'
      memory: 4G
```

### 리소스 사용량 모니터링

```bash
# 컨테이너 리소스 사용량 실시간 확인
docker stats pilotscope-dev

# 예시 출력:
# CONTAINER ID   NAME             CPU %     MEM USAGE / LIMIT   MEM %
# abc123def456   pilotscope-dev   350.12%   8.5GiB / 32GiB     26.56%
```

---

## PostgreSQL 최적화

### 자동 최적화 스크립트 실행

```bash
# 컨테이너 접속
docker exec -it pilotscope-dev bash

# 최적화 스크립트 실행
bash scripts/optimize_postgresql.sh
```

### 수동 최적화

컨테이너 내부에서:

```bash
# PostgreSQL 설정 파일 편집
vim /home/pilotscope/pg_data/postgresql.conf
```

**권장 설정 (32GB RAM 기준):**

```ini
# 메모리 설정
shared_buffers = 8GB              # 전체 RAM의 25%
effective_cache_size = 24GB       # 전체 RAM의 75%
work_mem = 256MB                  # 쿼리당 메모리
maintenance_work_mem = 2GB        # VACUUM, CREATE INDEX

# 성능 최적화
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
# PostgreSQL 재시작
pg_ctl restart -D /home/pilotscope/pg_data
```

### PostgreSQL 성능 확인

```bash
# 현재 설정 확인
psql -d pilotscope -c "
SELECT name, setting, unit 
FROM pg_settings 
WHERE name IN (
    'shared_buffers', 
    'work_mem', 
    'max_parallel_workers'
);
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

---

## 성능 벤치마크

### 1. GPU 성능 테스트

```bash
docker exec -it pilotscope-dev bash
conda activate pilotscope
python scripts/check_gpu.py --benchmark
```

**예상 결과:**
```
🖥️  CPU 테스트 (행렬 곱셈 5000x5000, 10회)...
   소요 시간: 45.23초 (4.523초/회)

🚀 GPU 테스트 (행렬 곱셈 5000x5000, 10회)...
   소요 시간: 2.15초 (0.215초/회)

📊 결과:
   GPU 속도 향상: 21.0x 빠름
   ✅ GPU가 매우 효과적으로 작동하고 있습니다!
```

### 2. Lero/MSCN 학습 속도 비교

#### GPU 사용 (예상)
```bash
# Lero 학습
python unified_test.py --algo lero --db stats_tiny \
    --collection-size 100 \
    --training-size 500 \
    --epochs 100

# 예상 시간: 1-2시간
```

#### CPU만 사용 (예상)
```bash
# GPU 비활성화
export CUDA_VISIBLE_DEVICES=""

python unified_test.py --algo lero --db stats_tiny \
    --collection-size 100 \
    --training-size 500 \
    --epochs 100

# 예상 시간: 5-10시간 (2-5배 느림)
```

### 3. PostgreSQL 쿼리 성능

```bash
# 쿼리 실행 계획 분석
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

### GPU가 인식되지 않음

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
# GPU 설정이 있는지 확인
grep -A 5 "devices:" docker-compose.yml
```

### 메모리 부족

#### 증상
```
docker: Error response from daemon: OCI runtime create failed
```

#### 해결
```yaml
# docker-compose.yml 수정
deploy:
  resources:
    limits:
      memory: 16G  # 줄이기
```

### PostgreSQL 연결 실패

```bash
# PostgreSQL 상태 확인
docker exec -it pilotscope-dev bash
ps aux | grep postgres

# 로그 확인
cat /tmp/postgres.log

# 재시작
pg_ctl restart -D /home/pilotscope/pg_data
```

---

## 성능 최적화 체크리스트

학습/테스트 전 확인:

- [ ] GPU 사용 가능 확인 (`python scripts/check_gpu.py`)
- [ ] Docker 리소스 할당 확인 (`docker stats`)
- [ ] PostgreSQL 최적화 적용 (`bash scripts/optimize_postgresql.sh`)
- [ ] 충분한 디스크 공간 (최소 50GB)
- [ ] Shared memory 설정 (shm_size: 8gb)

---

## 권장 워크플로우

### Phase 1: 환경 확인 (1회)

```bash
# GPU 확인
docker exec -it pilotscope-dev bash -c "conda activate pilotscope && python scripts/check_gpu.py"

# PostgreSQL 최적화
docker exec -it pilotscope-dev bash -c "bash scripts/optimize_postgresql.sh"
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

## 참고 자료

- **NVIDIA Docker**: https://github.com/NVIDIA/nvidia-docker
- **PostgreSQL 튜닝**: https://pgtune.leopard.in.ua/
- **Docker 리소스 관리**: https://docs.docker.com/config/containers/resource_constraints/
- **Lero 최적화**: `algorithm_examples/LERO_TROUBLESHOOTING.md`

---

**성능 문제가 계속되면 이슈를 올리거나 시스템 사양을 확인하세요! 🚀**

