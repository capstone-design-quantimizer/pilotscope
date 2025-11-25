# PilotScope Backend API (Recommend-Only, Embedded DB)

운영 안정성 최우선: 운영 DB는 건드리지 않고, 내장 PostgreSQL(운영 DB 복제본)에서만 튜닝/실행을 돌린 뒤 **저장된 결과만** API로 노출한다. 백엔드는 실행을 트리거하지 않고, 저장된 메타스토어에서 읽기만 한다.

## 핵심 원칙
- 내장 DB만 사용(`_is_local=True`), 원격 접근/재시작/DDL 없음.
- 알고리즘/데이터셋/DB 선택은 서버 내부 정책으로 고정. 클라이언트는 최소 입력만 보낸다.
- 메타스토어는 내장 PostgreSQL의 `PilotScopeUserData` DB 내 테이블을 사용한다. `DataManager`가 이 DB에 읽기/쓰기를 수행한다.

## 메타스토어 (위치, 스키마, I/O)
- **위치**: 내장 PostgreSQL 인스턴스, DB 이름 `PilotScopeUserData` (PilotConfig.user_data_db_name 기본값).
- **테이블 스키마 예시**
  - `knob_reco(id serial PK, run_id text, algo text, recommended_knobs jsonb, apply_snippet text, metrics jsonb, created_at timestamptz default now(), source text)`
  - `index_reco(id serial PK, run_id text, algo text, recommended_indexes jsonb, ddl_snippet text[], metrics jsonb, created_at timestamptz default now(), source text)`
  - `model_jobs(id serial PK, algo text, dataset text, status text, run_id text, metrics jsonb, model_id text, created_at timestamptz default now())`
- **쓰기**: 튜닝 스크립트에서 `DataManager.save_data()` 사용. `DataManager`는 `config.user_data_db_name`으로 DB를 바꿔 접속한 뒤 테이블 생성/insert를 수행한다.
- **읽기**: API는 `SELECT * FROM knob_reco ORDER BY created_at DESC LIMIT 1` 같은 쿼리로 최신 레코드를 읽어 응답을 만든다 (`DataManager.read_all()`로도 가능).

## 튜닝/학습 실행은 어디서 어떻게 돌리는가
- **Knob 튜닝 배치 예** (추천만 저장):
  ```
  python test_example_algorithms/unified_test.py --algo knob --db stats_tiny --timeout 900
  ```
  실행 후 배치 스크립트가 `best_conf`/metrics를 `knob_reco`에 기록 (`DataManager.save_data` 호출로 후처리 추가 필요).
- **Index 튜닝 배치 예**:
  ```
  python test_example_algorithms/unified_test.py --algo index --db stats_tiny --timeout 900
  ```
  실행 후 추천 인덱스/DDL/metrics를 `index_reco`에 기록.
- **Lero/MSCN 학습(옵션)**:
  ```
  python test_example_algorithms/unified_test.py --algo lero --db stats_tiny --epochs 50 --training-size 500
  ```
  모델/메트릭을 `model_jobs`에 기록, 모델 아티팩트는 MLflow 또는 로컬 디렉터리 관리.
- 실행 트리거는 크론/워크플로 스케줄러에서 관리. 백엔드 API는 이 저장 결과만 조회한다.

## API 개요 (조회만)
- Knob 추천 조회: `GET /api/reco/knob`
- Index 추천 조회: `GET /api/reco/index`
- 쿼리 가속(Lero/MSCN): `POST /api/query` (실시간 실행 필요 시)
- 학습/수집(옵션): `POST /api/models/{algo}/train` → `GET /api/models/{run_id}` (트리거/조회만)

---
## 1A) Knob 추천 조회 API
튜닝 실행은 배치/스케줄러가 미리 수행하고 `knob_reco`에 저장해 둔다. API는 최신 저장값만 반환한다.

### GET /api/reco/knob
```json
{
  "recommended_knobs": { "shared_buffers": "2GB", "work_mem": "128MB" },
  "apply_snippet": "shared_buffers = '2GB'\nwork_mem = '128MB'\n",
  "metric": { "latency_ms": 123.4, "throughput_qps": 45.6 },
  "updated_at": "...",
  "source": "latest_successful_run"
}
```

### 내부 운용 (knob)
1) 배치/스케줄러가 내장 DB 대상으로 knob 튜닝 실행(`knob_preset`, recommend-only).  
2) `best_conf`/메트릭/스니펫을 `PilotScopeUserData.knob_reco`에 `DataManager.save_data()`로 저장.  
3) API는 `SELECT ... FROM knob_reco ORDER BY created_at DESC LIMIT 1` 결과를 반환.  
4) 운영 반영은 외부 승인/수동 절차.

---
## 1B) Index 추천 조회 API
튜닝 실행은 배치/스케줄러가 미리 수행하고 `index_reco`에 저장해 둔다. API는 최신 저장값만 반환한다.

### GET /api/reco/index
```json
{
  "recommended_indexes": [
    { "table": "orders", "columns": ["customer_id"], "name": "idx_orders_customer_id" }
  ],
  "ddl_snippet": [
    "CREATE INDEX idx_orders_customer_id ON orders(customer_id);"
  ],
  "metric": { "latency_ms": 123.4, "throughput_qps": 45.6 },
  "updated_at": "...",
  "source": "latest_successful_run"
}
```

### 내부 운용 (index)
1) 배치/스케줄러가 내장 DB 대상으로 index 튜닝 실행(`index_extend`, recommend-only).  
2) 추천 인덱스/DDL/메트릭을 `PilotScopeUserData.index_reco`에 `DataManager.save_data()`로 저장.  
3) API는 `SELECT ... FROM index_reco ORDER BY created_at DESC LIMIT 1` 결과를 반환.  
4) 운영 반영은 외부 승인/수동 절차.

---
## 2) 쿼리 가속 API (Lero/MSCN, Embedded Execution)
목적: 단일 쿼리를 내장 DB에서 실행해 결과/지표 반환. 알고리즘/데이터셋 선택은 서버 내부 정책.

### POST /api/query
```json
{
  "sql": "SELECT ...",
  "collect": false,                  // true면 이번 실행을 데이터 수집에 활용
  "timeout_ms": 5000,
  "options": {
    "allow_fallback_native": true    // 모델 부재 시 기본 플랜으로 실행 후 수집 트리거 허용
  }
}
```
Response 200 (모델 사용 시):
```json
{
  "rows": [...],
  "row_count": 123,
  "latency_ms": 42.5,
  "plan": "EXPLAIN ...",
  "model_used": "lero_run_17",
  "notes": "cold_start=fallback_native"
}
```
Response 202 (모델 없음 + fallback 차단): `{ "status": "training_triggered", "run_id": "model-job-1" }`

### 내부 운용 (쿼리)
1) 요청 검증 → 내장 DB로 `PilotConfig` 생성(호스트/데이터셋은 서버 설정에서 로드).  
2) 내부 알고리즘 라우팅(예: Lero 우선, 없으면 MSCN). `dataset_name`도 내부 매핑 사용.  
3) 모델 준비: 캐시/MLflow 로드, 없으면 옵션에 따라 (a) native 실행+수집 트리거, (b) 학습 잡 트리거 후 202 반환.  
4) `scheduler.execute(sql)` → 모델 기반 플랜/카디널리티 적용 → 내장 DB 실행.  
5) 결과(rows/latency/plan) 응답. 필요 시 수집 데이터/로그 저장.  
6) 타임아웃/에러는 명시적 코드/메시지로 반환.

---
## 3) 학습/수집 API (옵션)
### POST /api/models/{algo}/train
클라이언트는 최소 옵션만 제공, dataset/workload 선택은 서버 내부 설정 사용.
```json
{ "enable_collection": true, "num_collection": 200, "num_training": 500, "num_epoch": 50 }
```
Response 202: `{ "run_id": "model-job-1", "status": "queued" }`

### GET /api/models/{run_id}
학습 상태/메트릭/로그 조회. 학습 완료 시 `model_id` 제공.

---
## Best Config/Recommendation 저장 플로우 요약
1) 배치가 `unified_test.py`로 튜닝 실행(knob/index recommend-only).  
2) 결과(`best_conf`/추천 인덱스/메트릭)를 `PilotScopeUserData` 테이블에 `DataManager.save_data()`로 저장.  
3) API 조회: `GET /api/reco/knob` / `GET /api/reco/index`는 저장된 최신 레코드만 반환.  
4) 운영 반영: 외부 승인/수동 절차에서 스니펫을 사용해 선택적 적용. PilotScope는 관여하지 않는다.
