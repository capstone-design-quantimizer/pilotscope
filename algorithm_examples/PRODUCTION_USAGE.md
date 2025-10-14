# PilotScope in Production Environments

## Overview

**테스트 환경 vs 프로덕션 환경의 차이점**

### 🧪 테스트 환경 (현재 예제들)
```python
# .txt 파일에서 SQL 로드
sqls = load_test_sql("stats_tiny")  # imdb_train.txt, stats_test.txt 등에서 로드
for sql in sqls:
    scheduler.execute(sql)
```
- **목적**: 알고리즘 성능 벤치마킹, 연구용
- **SQL 소스**: 사전에 준비된 `.txt` 파일
- **실행 방식**: 배치 처리 (모든 쿼리 한번에 실행)

### 🏭 프로덕션 환경 (실제 운영)
```python
# 실시간 들어오는 SQL을 처리
sql = receive_from_application()  # 앱에서 실시간 수신
result = scheduler.execute(sql)
return result
```
- **목적**: 실제 서비스에서 AI 최적화 적용
- **SQL 소스**: 애플리케이션에서 실시간으로 전달
- **실행 방식**: 실시간 처리 (쿼리마다 즉시 응답)

---

## Production Deployment Patterns

### Pattern 1: Middleware Proxy (가장 일반적)

애플리케이션과 DB 사이에 PilotScope를 프록시로 배치:

```python
# production_proxy.py
from pilotscope.PilotScheduler import PilotScheduler
from pilotscope.PilotConfig import PostgreSQLConfig

class PilotScopeProxy:
    def __init__(self):
        config = PostgreSQLConfig()
        config.db = "production_db"
        
        # AI 알고리즘 스케줄러 설정
        self.scheduler = get_mscn_preset_scheduler(config)
        self.scheduler.init()
    
    def execute_query(self, sql: str):
        """
        애플리케이션에서 받은 SQL을 AI 최적화 적용하여 실행
        """
        # AI 알고리즘이 자동으로 적용됨
        result = self.scheduler.execute(sql)
        return result

# Flask/FastAPI 등과 통합
from flask import Flask, request, jsonify
app = Flask(__name__)
proxy = PilotScopeProxy()

@app.route('/query', methods=['POST'])
def handle_query():
    sql = request.json['sql']
    result = proxy.execute_query(sql)
    return jsonify({'data': result})
```

**구조:**
```
Application → HTTP/API → PilotScope Proxy → Database
                            (AI 최적화)
```

---

### Pattern 2: Direct Integration (Direct 통합)

애플리케이션 코드에 직접 PilotScope 통합:

```python
# application_service.py
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor
from pilotscope.PilotConfig import PostgreSQLConfig

class UserService:
    def __init__(self):
        config = PostgreSQLConfig()
        config.db = "production_db"
        
        # AI 알고리즘 핸들러 등록
        self.data_interactor = PilotDataInteractor(config)
        self.data_interactor.push_card(ml_model.predict)  # ML 모델 적용
    
    def get_user_stats(self, user_id):
        # 비즈니스 로직에서 생성한 SQL
        sql = f"SELECT * FROM users WHERE id = {user_id}"
        
        # AI 최적화 적용되어 실행
        result = self.data_interactor.execute(sql)
        return result.records
```

**구조:**
```
Application Code → PilotDataInteractor → Database
                      (AI 최적화)
```

---

### Pattern 3: Scheduler-Based Service (스케줄러 기반 서비스)

장기 실행 서비스로 배포:

```python
# production_scheduler_service.py
from pilotscope.PilotScheduler import PilotScheduler
from pilotscope.PilotConfig import PostgreSQLConfig
import queue
import threading

class PilotScopeService:
    def __init__(self):
        config = PostgreSQLConfig()
        config.db = "production_db"
        
        self.scheduler = PilotScheduler(config)
        
        # AI 알고리즘 핸들러 등록
        self.scheduler.register_custom_handlers([
            CardPushHandler(),  # 카디널리티 예측
            IndexPushHandler()  # 인덱스 추천
        ])
        
        # 데이터 수집 설정
        self.scheduler.register_required_data(
            table_name_for_store_data="query_metrics",
            pull_execution_time=True,
            pull_estimated_cost=True
        )
        
        # 이벤트 등록 (모델 재학습 등)
        self.scheduler.register_events([
            PeriodicModelUpdateEvent(interval_minutes=60)
        ])
        
        self.scheduler.init()
        
        # 쿼리 큐
        self.query_queue = queue.Queue()
        
        # 백그라운드 워커 시작
        self.worker_thread = threading.Thread(target=self._process_queries)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def submit_query(self, sql: str):
        """쿼리를 큐에 추가하고 비동기 처리"""
        future = queue.Queue()
        self.query_queue.put((sql, future))
        return future.get()  # 결과 대기
    
    def _process_queries(self):
        """백그라운드에서 쿼리 처리"""
        while True:
            sql, future = self.query_queue.get()
            try:
                result = self.scheduler.execute(sql)
                future.put(('success', result))
            except Exception as e:
                future.put(('error', str(e)))

# 서비스 시작
service = PilotScopeService()

# 사용 예
result = service.submit_query("SELECT * FROM users WHERE age > 25")
```

**구조:**
```
Application → Queue → PilotScope Service (Background) → Database
                         (AI 최적화 + 모델 업데이트)
```

---

## Real-World Example: E-commerce Analytics

실제 전자상거래 분석 시스템 예제:

```python
# ecommerce_analytics.py
from pilotscope.PilotScheduler import PilotScheduler
from pilotscope.PilotConfig import PostgreSQLConfig
from algorithm_examples.Mscn.MscnPresetScheduler import get_mscn_preset_scheduler

class EcommerceAnalytics:
    def __init__(self):
        config = PostgreSQLConfig()
        config.db = "ecommerce_prod"
        config.sql_execution_timeout = 60000  # 1분 타임아웃
        
        # MSCN 알고리즘 적용 (카디널리티 예측)
        self.scheduler = get_mscn_preset_scheduler(
            config,
            enable_collection=True,   # 실행 데이터 수집
            enable_training=False     # 프로덕션에서는 학습 비활성화
        )
        self.scheduler.init()
    
    def get_daily_sales_report(self, date):
        """일일 매출 리포트 - 복잡한 조인 쿼리"""
        sql = f"""
        SELECT 
            p.category,
            COUNT(DISTINCT o.order_id) as order_count,
            SUM(oi.quantity * oi.price) as total_revenue
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        WHERE o.order_date = '{date}'
        GROUP BY p.category
        ORDER BY total_revenue DESC
        """
        
        # MSCN 알고리즘이 자동으로 카디널리티 예측 → 더 나은 쿼리 플랜
        result = self.scheduler.execute(sql)
        return result
    
    def get_customer_segments(self, min_orders=5):
        """고객 세그먼트 분석 - 매우 복잡한 쿼리"""
        sql = f"""
        WITH customer_stats AS (
            SELECT 
                c.customer_id,
                COUNT(o.order_id) as order_count,
                AVG(o.total_amount) as avg_order_value,
                MAX(o.order_date) as last_order_date
            FROM customers c
            LEFT JOIN orders o ON c.customer_id = o.customer_id
            GROUP BY c.customer_id
            HAVING COUNT(o.order_id) >= {min_orders}
        )
        SELECT * FROM customer_stats
        ORDER BY avg_order_value DESC
        LIMIT 1000
        """
        
        # AI 최적화 적용되어 실행
        result = self.scheduler.execute(sql)
        return result

# 프로덕션 사용
analytics = EcommerceAnalytics()

# 실시간 리포트 생성 (AI 최적화 자동 적용)
sales_report = analytics.get_daily_sales_report('2024-01-15')
customer_segments = analytics.get_customer_segments(min_orders=10)
```

---

## Key Differences: Test vs Production

| 항목 | 테스트 환경 | 프로덕션 환경 |
|------|------------|---------------|
| **SQL 소스** | `.txt` 파일 (고정) | 애플리케이션 (동적) |
| **실행 방식** | 배치 (loop) | 실시간 (per request) |
| **목적** | 벤치마킹, 연구 | 서비스 제공 |
| **데이터 수집** | 항상 활성화 | 선택적 (성능 고려) |
| **모델 학습** | 실행 중 학습 가능 | 별도 프로세스로 분리 |
| **에러 처리** | 간단 (실패 무시) | 철저 (재시도, 로깅) |
| **모니터링** | 선택적 | 필수 (메트릭, 알람) |

---

## Production Deployment Checklist

### ✅ Configuration
- [ ] 프로덕션 DB 연결 정보 설정
- [ ] 타임아웃 설정 (`sql_execution_timeout`)
- [ ] 커넥션 풀 설정

### ✅ AI Algorithm
- [ ] 사전 학습된 모델 로드
- [ ] 모델 업데이트 전략 수립 (주기적 재학습)
- [ ] 폴백 메커니즘 (AI 실패 시 일반 실행)

### ✅ Monitoring & Logging
- [ ] 쿼리 실행 시간 로깅
- [ ] AI 예측 정확도 모니터링
- [ ] 에러 로깅 및 알람

### ✅ Performance
- [ ] 데이터 수집 최소화 (프로덕션 부하 고려)
- [ ] 비동기 처리 (긴 쿼리)
- [ ] 캐싱 전략

### ✅ Security
- [ ] SQL 인젝션 방지
- [ ] DB 크레덴셜 안전한 관리 (환경변수, Vault)
- [ ] 접근 제어

---

## Common Patterns

### Pattern: Graceful Degradation (AI 실패 시 폴백)

```python
def execute_with_fallback(scheduler, sql):
    try:
        # AI 최적화 시도
        return scheduler.execute(sql)
    except Exception as e:
        logging.error(f"AI optimization failed: {e}")
        # AI 실패 시 일반 실행
        return db_controller.execute(sql)
```

### Pattern: Async Processing (비동기 처리)

```python
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=10)

def execute_async(sql):
    future = executor.submit(scheduler.execute, sql)
    return future  # 비동기 결과 반환
```

### Pattern: Model Update Service (모델 업데이트 서비스)

```python
# 별도 프로세스로 모델 학습
class ModelUpdateService:
    def __init__(self):
        self.scheduler = get_mscn_preset_scheduler(config)
        
    def retrain_daily(self):
        """매일 새로운 쿼리 데이터로 모델 재학습"""
        # 수집된 데이터로 학습
        self.scheduler.register_events([
            PeriodicModelUpdateEvent(interval_minutes=1440)  # 24시간
        ])
```

---

## Summary

**핵심 포인트:**

1. **테스트**: `.txt` 파일에서 SQL 로드 → 배치 벤치마킹
2. **프로덕션**: 애플리케이션에서 실시간 SQL 수신 → 즉시 AI 최적화 적용
3. **통합 방식**:
   - Middleware Proxy (권장)
   - Direct Integration
   - Background Service
4. **동일한 API**: `scheduler.execute(sql)` 또는 `data_interactor.execute(sql)`
5. **차이점**: SQL의 **출처**만 다름 (파일 vs 애플리케이션)

PilotScope는 **SQL 출처에 무관하게** 동일한 방식으로 동작합니다! 🎯
