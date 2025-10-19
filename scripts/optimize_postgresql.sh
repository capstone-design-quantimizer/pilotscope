#!/bin/bash
# PostgreSQL 성능 최적화 스크립트
# Docker 컨테이너 내부에서 실행하세요

set -e

echo "==========================================="
echo "PostgreSQL 성능 최적화 시작"
echo "==========================================="

PG_DATA="/home/pilotscope/pg_data"
PG_CONF="${PG_DATA}/postgresql.conf"

# 백업 생성
if [ ! -f "${PG_CONF}.backup" ]; then
    echo "📦 설정 파일 백업 중..."
    cp "${PG_CONF}" "${PG_CONF}.backup"
fi

# PostgreSQL 중지
echo "⏸️  PostgreSQL 중지 중..."
${PG_PATH}/bin/pg_ctl stop -D ${PG_DATA} || true
sleep 2

echo "⚙️  PostgreSQL 설정 최적화 중..."

# 성능 최적화 설정 추가
cat >> "${PG_CONF}" << 'EOF'

# =========================================
# PilotScope 성능 최적화 설정
# =========================================

# 메모리 설정
shared_buffers = 2GB              # 공유 버퍼 (전체 RAM의 25%)
effective_cache_size = 6GB        # 디스크 캐시 (전체 RAM의 75%)
work_mem = 128MB                  # 정렬/해시 작업 메모리
maintenance_work_mem = 512MB      # VACUUM, CREATE INDEX 메모리

# 쿼리 플래너 설정
random_page_cost = 1.1            # SSD 최적화 (HDD는 4.0)
effective_io_concurrency = 200    # SSD 최적화
default_statistics_target = 100   # 통계 정확도 향상

# 체크포인트 설정
checkpoint_completion_target = 0.9
wal_buffers = 16MB
max_wal_size = 4GB
min_wal_size = 1GB

# 병렬 쿼리 설정
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_worker_processes = 8

# 연결 설정
max_connections = 200

# 로깅 (필요시 활성화)
# log_min_duration_statement = 1000  # 1초 이상 쿼리 로깅
# log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
# log_statement = 'all'

# PilotScope 필수 설정
shared_preload_libraries = 'pg_hint_plan'
listen_addresses = '*'

EOF

echo "✅ 설정 완료!"
echo ""
echo "📝 적용된 주요 설정:"
echo "  - shared_buffers: 2GB"
echo "  - effective_cache_size: 6GB"
echo "  - work_mem: 128MB"
echo "  - max_parallel_workers: 8"

# PostgreSQL 재시작
echo ""
echo "🔄 PostgreSQL 재시작 중..."
${PG_PATH}/bin/pg_ctl start -D ${PG_DATA} -l /tmp/postgres.log

# 설정 확인
sleep 3
echo ""
echo "✅ PostgreSQL 성능 최적화 완료!"
echo ""
echo "📊 현재 설정 확인:"
${PG_PATH}/bin/psql -d pilotscope -c "
SELECT name, setting, unit 
FROM pg_settings 
WHERE name IN (
    'shared_buffers', 
    'effective_cache_size', 
    'work_mem', 
    'maintenance_work_mem',
    'max_parallel_workers',
    'random_page_cost'
)
ORDER BY name;
"

echo ""
echo "==========================================="
echo "💡 성능 튜닝 팁:"
echo "  1. Lero/MSCN 학습 중 메모리 사용량 모니터링"
echo "  2. 복잡한 쿼리는 EXPLAIN ANALYZE로 분석"
echo "  3. 주기적으로 VACUUM ANALYZE 실행"
echo "==========================================="

