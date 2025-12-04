#!/bin/bash
# 현재 experiment_logs 데이터를 SQL 파일로 덤프

DUMP_FILE="deployment/experiment_logs_dump.sql"

echo "Exporting experiment_logs table..."

# 디렉토리 생성
mkdir -p deployment

# 테이블 스키마 + 데이터 덤프
pg_dump -U pilotscope -d PilotScopeUserData \
    -t experiment_logs \
    --clean \
    --if-exists \
    > "$DUMP_FILE"

echo "✓ Exported to $DUMP_FILE"
echo ""
echo "Row count:"
psql -U pilotscope -d PilotScopeUserData -c "SELECT COUNT(*) FROM experiment_logs;"
