# 배포용 가벼운 FastAPI 이미지
FROM python:3.9-slim

WORKDIR /app

# PostgreSQL 클라이언트 라이브러리만 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# 최소 의존성만 설치
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# API 코드 복사
COPY api/ /app/api/

# 포트 노출
EXPOSE 8000

# API 서버 실행
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
