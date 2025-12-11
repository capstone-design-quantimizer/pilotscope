# PilotScope Experiment API - Deployment Package

MLflow 실험 결과를 조회하는 경량 FastAPI 서버 배포 패키지입니다.

## 패키지 구성

```
deployment/
├── docker-compose.yml          # PostgreSQL + FastAPI 스택
├── Dockerfile                  # FastAPI 이미지 빌드
├── requirements.txt            # 최소 의존성 (FastAPI, PostgreSQL 클라이언트만)
├── experiment_logs_dump.sql    # 실험 데이터 (35개 실험)
└── README.md                   # 이 파일
```

## 배포 방법 (EC2)

### 1. 파일 전송

```bash
# 로컬에서 실행
scp -r deployment/ ec2-user@<EC2-IP>:~/pilotscope-api/
```

또는 Git으로:

```bash
# EC2에서 실행
git clone <repository-url>
cd pilotscope/deployment
```

### 2. Docker 설치 (EC2에 없는 경우)

```bash
# Amazon Linux 2
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Docker Compose 설치
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 재로그인하여 docker 권한 적용
exit
```

### 3. 배포 실행

```bash
cd deployment
docker-compose up -d
```

### 4. 상태 확인

```bash
# 컨테이너 상태
docker-compose ps

# API 헬스 체크
curl http://localhost:8000/health
# 응답: {"status":"healthy","database":"connected"}

# 실험 목록 조회
curl "http://localhost:8000/api/experiments?limit=5"
```

### 5. 로그 확인

```bash
# API 로그
docker-compose logs -f api

# PostgreSQL 로그
docker-compose logs -f postgres
```

## API 엔드포인트

### 헬스 체크
```bash
GET /health
```

### 실험 목록 조회
```bash
GET /api/experiments?algorithm=mscn&dataset=stock_strategy&limit=50&offset=0
```

**파라미터:**
- `algorithm`: 알고리즘 필터 (mscn, lero, llamatune, knob, index, baseline)
- `dataset`: 데이터셋 필터 (stock_strategy, stats_tiny, imdb, tpch)
- `status`: 상태 필터 (FINISHED, FAILED, RUNNING)
- `limit`: 페이지 크기 (1-500, default: 50)
- `offset`: 페이지 오프셋 (default: 0)
- `sort`: 정렬 필드 (started_at, execution_time, algorithm, dataset)
- `order`: 정렬 순서 (asc, desc)

### 실험 상세 조회
```bash
GET /api/experiments/{id}
```

best_config, metrics, parameters를 포함한 전체 정보 반환

### 실험 이름 변경
```bash
PATCH /api/experiments/{id}
Content-Type: application/json

{
  "experiment_name": "새로운 이름"
}
```

### 문서 (Swagger UI)
```
http://<EC2-IP>:8000/docs
```

## 포트 및 보안 그룹

**EC2 인바운드 규칙:**
- 8000 (TCP): FastAPI (0.0.0.0/0 또는 특정 IP)
- 5432 (TCP): PostgreSQL (컨테이너 내부 전용, 외부 개방 불필요)

## 컨테이너 관리

### 중지
```bash
docker-compose down
```

### 재시작
```bash
docker-compose restart
```

### 로그 초기화하며 재시작
```bash
docker-compose down -v  # 주의: PostgreSQL 데이터도 삭제됨
docker-compose up -d
```

### 이미지 재빌드
```bash
docker-compose build --no-cache
docker-compose up -d
```

## 데이터베이스 정보

- **DB 이름**: PilotScopeUserData
- **테이블**: experiment_logs
- **User**: pilotscope / pilotscope
- **데이터**: 35개 실험 (2025-11-16 ~ 2025-11-25)

**포함된 데이터:**
- MSCN 카디널리티 추정 실험
- Lero 실행계획 최적화 실험
- LlamaTune/Knob Tuning 실험 (best_config 포함)
- Baseline 실험
- 다양한 워크로드 (momentum investing, value quality, smallcap turnover 등)

## 트러블슈팅

### 포트 충돌
```bash
# 사용 중인 포트 확인
sudo netstat -tlnp | grep 8000

# docker-compose.yml에서 포트 변경
ports:
  - "8001:8000"  # 호스트:컨테이너
```

### PostgreSQL 초기화 실패
```bash
# 볼륨 삭제 후 재시작
docker-compose down -v
docker-compose up -d

# 초기화 로그 확인
docker-compose logs postgres | grep "database system is ready"
```

### API 시작 실패
```bash
# API 로그 확인
docker-compose logs api

# 흔한 원인:
# 1. PostgreSQL이 아직 준비되지 않음 -> healthcheck가 자동 대기
# 2. 의존성 문제 -> requirements.txt 확인
# 3. Python 경로 문제 -> Dockerfile의 COPY 경로 확인
```

## 환경 변수 (선택)

`docker-compose.yml`에서 수정 가능:

```yaml
environment:
  DB_HOST: postgres          # PostgreSQL 호스트
  DB_PORT: 5432             # PostgreSQL 포트
  DB_NAME: PilotScopeUserData
  DB_USER: pilotscope
  DB_PASSWORD: pilotscope
```

## 리소스 요구사항

- **CPU**: 1 vCPU (최소)
- **메모리**: 1 GB (최소), 2 GB (권장)
- **디스크**: 500 MB (이미지 + 데이터)
- **EC2 인스턴스**: t2.micro 이상

## 참고사항

- 이 패키지는 **읽기 전용** 조회용입니다
- MLflow 동기화 기능은 포함되지 않음
- 새로운 실험 데이터 추가는 지원하지 않음
- 프로덕션 사용 시 HTTPS/인증 추가 권장
- PostgreSQL 비밀번호는 운영 환경에서 변경 필요
