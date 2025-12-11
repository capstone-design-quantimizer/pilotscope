#!/bin/bash
# PostgreSQL 설정 완전 초기화 스크립트
# Knob Tuning 등으로 변경된 모든 설정을 기본값으로 복원합니다.

set -e  # 에러 발생 시 즉시 중단

echo "========================================="
echo "PostgreSQL 설정 초기화 시작"
echo "========================================="

# 백업 생성
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "1. 현재 설정 백업 중..."
cp /home/pilotscope/pg_data/postgresql.conf /home/pilotscope/pg_data/postgresql.conf.backup_${TIMESTAMP}
echo "   백업 완료: postgresql.conf.backup_${TIMESTAMP}"

# 원본 템플릿 복사
echo "2. 원본 템플릿으로 복원 중..."
cp /home/pilotscope/pgsql/share/postgresql.conf.sample /home/pilotscope/pg_data/postgresql.conf
echo "   복원 완료"

# PostgreSQL 재시작
echo "3. PostgreSQL 재시작 중..."
pg_ctl -D /home/pilotscope/pg_data restart
echo "   재시작 완료"

# 설정 확인
echo ""
echo "========================================="
echo "초기화 완료! 현재 설정 확인:"
echo "========================================="
psql -U pilotscope -d stock_strategy -c "
SELECT
    name,
    setting,
    unit,
    CASE
        WHEN unit = '8kB' THEN (setting::bigint * 8 / 1024)::text || ' MB'
        WHEN unit = 'kB' THEN (setting::bigint / 1024)::text || ' MB'
        ELSE setting || COALESCE(' ' || unit, '')
    END as readable_value
FROM pg_settings
WHERE name IN ('shared_buffers', 'work_mem', 'random_page_cost', 'seq_page_cost', 'effective_cache_size', 'maintenance_work_mem')
ORDER BY name;
"

echo ""
echo "✅ PostgreSQL 설정이 기본값으로 초기화되었습니다!"
echo "   백업 파일: /home/pilotscope/pg_data/postgresql.conf.backup_${TIMESTAMP}"