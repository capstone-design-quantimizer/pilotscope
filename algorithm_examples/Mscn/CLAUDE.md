# MSCN (Multi-Set Convolutional Network)

카디널리티 추정. PostgreSQL 옵티마이저의 카디널리티 추정을 AI 모델로 대체.

## 작동 방식

1. SQL → 피처 벡터 변환 (테이블, JOIN, WHERE 조건)
2. MSCN 모델 → 카디널리티 예측
3. 예측값 PostgreSQL에 힌트 주입
4. PostgreSQL이 AI 예측으로 플랜 생성

## 전체 파이프라인

### 학습 단계

```
1. 쿼리 로딩
   load_training_sql() → Dataset.read_train_sql()
   └→ stats_train_time2int.txt 파일에서 SQL 리스트 로드

2. 데이터 수집
   MscnPretrainingModelEvent.iterative_data_collection()
   ├→ For each SQL:
   │   ├─ pull_subquery_card() → 서브쿼리 카디널리티 수집
   │   ├─ execute(sql) → PostgreSQL 실행
   │   └─ {"query": sql, "card": 실제값} 저장
   └→ DataManager.save_data_batch() → DB에 저장

3. 쿼리 파싱
   parse_queries() → mscnQueryMeta
   ├→ 테이블 추출: ['badges', 'posts']
   ├→ JOIN 조건: ['b.userid = p.owneruserid']
   └→ WHERE 조건: [['p.score', '>=', '0'], ...]

4. 피처 인코딩
   Feature.fit()
   ├→ 테이블 원-핫: table2vec
   ├→ 컬럼+연산자+값: column2vec + op2vec + 정규화
   ├→ JOIN 원-핫: join2vec
   └→ 패딩 + 마스킹 (가변 길이 처리)

5. 모델 학습
   MscnModel.fit()
   ├→ SetConv 신경망 (3-way MLP)
   ├→ Q-Error 손실 함수
   ├→ Adam 옵티마이저 (lr=0.001)
   └→ 100 epoch (기본값)

6. 모델 저장
   ExampleData/Mscn/Model/mscn_{timestamp}/
   ├─ nn_weights (PyTorch 가중치)
   ├─ feature_generator (인코딩 정보)
   └─ input_feature_dim (차원 정보)
```

### 예측 단계

```
1. 쿼리 실행 인터셉트
   MscnCardPushHandler.acquire_injected_data()
   └→ 실행 전 SQL 쿼리 획득

2. 서브쿼리 추출
   pull_subquery_card() → PostgreSQL EXPLAIN 분석
   └→ 모든 서브쿼리 리스트 생성

3. 피처 변환
   Feature.transform()
   ├→ 저장된 인코딩 사용 (table2vec, column2vec 등)
   └→ 동일한 형식으로 벡터화

4. 카디널리티 예측
   MscnModel.predict()
   ├→ SetConv 순전파
   ├→ 정규화된 출력 [0,1]
   └→ 역정규화 → 실제 카디널리티

5. PostgreSQL 힌트 주입
   inject_data()
   └→ Anchor 패치된 PostgreSQL에 카디널리티 전달
```

## 적합 워크로드

- 복잡한 JOIN 쿼리
- 옵티마이저 카디널리티 추정 부정확
- OLAP (분석 쿼리)

## ML 모델 아키텍처

### SetConv 신경망 구조

```
입력: SQL 쿼리
  ├─→ 테이블 벡터 [batch x max_tables x table_feats]
  ├─→ WHERE 조건 벡터 [batch x max_predicates x predicate_feats]
  └─→ JOIN 조건 벡터 [batch x max_joins x join_feats]

SetConv 신경망:
  ├─ 테이블 MLP
  │   ├─ Linear(table_feats → hid_units=256) + ReLU
  │   ├─ Linear(256 → 256) + ReLU
  │   └─ Masking + Sum + Average → [batch x 256]
  │
  ├─ WHERE MLP
  │   ├─ Linear(predicate_feats → 256) + ReLU
  │   ├─ Linear(256 → 256) + ReLU
  │   └─ Masking + Sum + Average → [batch x 256]
  │
  └─ JOIN MLP
      ├─ Linear(join_feats → 256) + ReLU
      ├─ Linear(256 → 256) + ReLU
      └─ Masking + Sum + Average → [batch x 256]

  [Concatenate: 256 + 256 + 256 = 768]
      ↓
  Linear(768 → 256) + ReLU
      ↓
  Linear(256 → 1) + Sigmoid
      ↓
  출력: 카디널리티 (정규화된 값 [0,1])
```

### 입력 피처 인코딩

#### 1. 테이블 피처
- **형식**: 원-핫 인코딩
- **예시**:
  ```
  table2vec = {'badges': [1,0,0], 'posts': [0,1,0], 'users': [0,0,1]}
  쿼리: "FROM badges, posts" → [[1,0,0], [0,1,0]]
  ```

#### 2. WHERE 조건 피처
- **형식**: [컬럼 원-핫 | 연산자 원-핫 | 정규화된 값]
- **예시**:
  ```
  조건: "posts.score >= 10"

  컬럼 인코딩: column2vec['posts.score'] = [0,1,0,0,...]
  연산자 인코딩: op2vec['>='] = [0,1,0,0,0]
  값 정규화: (10 - min_score) / (max_score - min_score) = 0.15

  최종 벡터: [0,1,0,0,..., 0,1,0,0,0, 0.15]
              └─ 컬럼      └─ 연산자  └─값
  ```

#### 3. JOIN 조건 피처
- **형식**: 원-핫 인코딩
- **예시**:
  ```
  join2vec = {
    'badges.userid = posts.owneruserid': [1,0,0],
    'posts.id = comments.postid': [0,1,0]
  }
  ```

### 출력 및 정규화

**학습 시**:
```python
실제 카디널리티: 1000
  ↓ 로그 변환
log(1000) = 6.91
  ↓ Min-Max 정규화
(6.91 - min) / (max - min) = 0.50
  ↓ 신경망 학습 타겟
[0, 1] 범위의 값
```

**예측 시**:
```python
신경망 출력: 0.50
  ↓ 역정규화
0.50 * (max - min) + min = 6.91
  ↓ 지수 변환
exp(6.91) ≈ 1000
  ↓ PostgreSQL 힌트
카디널리티 = 1000
```

**중요**: MSCN은 0 카디널리티를 처리 못하므로 학습 시 `+1`, 예측 시 `-1` 적용

### 손실 함수: Q-Error

```python
Q-Error = max(pred / true, true / pred)

예시:
  pred=100, true=50  → Q-Error = 100/50 = 2.0
  pred=50,  true=100 → Q-Error = 100/50 = 2.0
```

**평가 지표**: Median Q-Error, 90/95/99 percentile, Max, Mean

## 파일 구조

```
Mscn/
├── MscnPresetScheduler.py          # 팩토리 (진입점)
├── MscnPilotModel.py               # 모델 래퍼
├── MscnParadigmCardAnchorHandler.py # 카디널리티 힌트 주입
├── EventImplement.py               # 학습/수집 이벤트
└── source/
    ├── mscn_model.py               # SetConv 신경망
    ├── mscn_utils.py               # 쿼리 파싱, 피처 인코딩
    ├── data.py                     # 데이터셋 생성, 패딩/마스킹
    └── mscn_train.py               # 학습/평가 스크립트
```

## 사용

```python
from algorithm_examples.Mscn.MscnPresetScheduler import get_mscn_preset_scheduler

scheduler, tracker = get_mscn_preset_scheduler(
    config,
    enable_collection=True,
    enable_training=True,
    num_epoch=100,
    dataset_name="your_db"
)
```

## 주요 컴포넌트

### 1. MscnPretrainingModelEvent (EventImplement.py)

**데이터 수집**: `iterative_data_collection()`
```python
# 트레이닝 쿼리 로딩
sqls = load_training_sql(dataset_name)
  ↓
# 각 쿼리 실행 및 카디널리티 수집
for sql in sqls:
    pull_subquery_card()           # 서브쿼리 분해
    data = execute(sql)             # PostgreSQL 실행
    for sub_sql in data.subquery_2_card.keys():
        pull_record()               # COUNT(*) 결과 가져오기
        record = execute(sub_sql)   # 실제 카디널리티
        save {"query": sub_sql, "card": count}
  ↓
# DB 테이블에 저장
테이블: mscn_pretraining_{dataset_name}
컬럼: id (PK), query (TEXT), card (BIGINT)
```

**모델 학습**: `custom_model_training()`
```python
# DB에서 수집된 데이터 로드
data = data_manager.read_all(table_name)
  ↓
# 쿼리 파싱
tables, joins, predicates = parse_queries(data["query"])
  ├─ mscnQueryMeta: sqlglot으로 SQL 파싱
  ├─ 테이블 추출: ['badges', 'posts']
  ├─ JOIN 추출: ['b.userid = p.owneruserid']
  └─ WHERE 추출: [['p.score', '>=', '0'], ...]
  ↓
# 스키마 로드
schema = load_schema(db_controller)
  └─ 각 컬럼의 min/max 값 (정규화용)
  ↓
# 모델 학습
model = MscnModel()
model.fit(
    (tables, joins, predicates),
    data["card"] + 1,              # +1: 0 처리 불가
    schema,
    num_epochs=100,
    batch_size=2048,
    hid_units=256
)
```

### 2. MscnCardPushHandler (MscnParadigmCardAnchorHandler.py)

**카디널리티 예측 및 주입**:
```python
def acquire_injected_data(sql):
    # 1. 서브쿼리 추출
    pull_subquery_card()
    data = execute(sql)
    subqueries = data.subquery_2_card.keys()

    # 2. MSCN 예측
    _, preds, _ = mscn_model.predict(subqueries)

    # 3. PostgreSQL 힌트 생성
    new_subquery_2_card = {
        sub_sql: str(max(0.0, pred - 1))  # -1: 학습 시 +1 보정
        for sub_sql, pred in zip(subqueries, preds)
    }

    return new_subquery_2_card

def inject_data(sql, new_subquery_2_card):
    # Anchor 패치된 PostgreSQL에 카디널리티 힌트 전달
    # PostgreSQL은 MSCN 예측값으로 플랜 생성
```

### 3. Feature 클래스 (mscn_utils.py)

**어휘(Vocabulary) 구축**:
```python
# 모든 쿼리에서 고유 요소 추출
table2vec, _ = get_set_encoding(['badges', 'posts', 'users', ...])
  → {'badges': [1,0,0], 'posts': [0,1,0], 'users': [0,0,1], ...}

column2vec, _ = get_set_encoding(['badges.userid', 'posts.score', ...])
  → {'badges.userid': [1,0,0,...], 'posts.score': [0,1,0,...], ...}

op2vec, _ = get_set_encoding(['=', '>', '<', '>=', '<='])
  → {'=': [1,0,0,0,0], '>': [0,1,0,0,0], ...}

join2vec, _ = get_set_encoding(['b.userid = p.owneruserid', ...])
  → {'b.userid = p.owneruserid': [1,0,0], ...}
```

**패딩 및 마스킹** (data.py):
```python
# 가변 길이 쿼리를 고정 길이로 변환
max_num_predicates = 14  # 최대 WHERE 조건 수
max_num_joins = 7        # 최대 JOIN 수

# 패딩: 부족한 부분을 0으로 채움
sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')

# 마스킹: 실제 데이터와 패딩 구분
sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
  → [1, 1, 0, 0, ...]  # 1: 실제 데이터, 0: 패딩
```

### 4. SetConv 신경망 (mscn_model.py)

**순전파 (Forward Pass)**:
```python
def forward(samples, predicates, joins, sample_mask, predicate_mask, join_mask):
    # 테이블 처리
    hid_sample = F.relu(self.sample_mlp1(samples))
    hid_sample = F.relu(self.sample_mlp2(hid_sample))
    hid_sample = hid_sample * sample_mask       # 패딩 제거
    hid_sample = torch.sum(hid_sample, dim=1)   # 합산
    hid_sample = hid_sample / sample_mask.sum(1)  # 평균

    # WHERE, JOIN도 동일 방식 처리
    hid_predicate = ...
    hid_join = ...

    # 3개 임베딩 연결
    hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)

    # 최종 출력
    hid = F.relu(self.out_mlp1(hid))
    out = torch.sigmoid(self.out_mlp2(hid))
    return out
```

**특이사항**:
- Lero와 달리 데이터 수집 시 기존 데이터 자동 삭제 안 함 (누적)
- 모델 저장: `ExampleData/Mscn/Model/mscn_{timestamp}`
- GPU 지원: `CUDA` 환경 변수로 자동 감지

## 수정 시 주의

**피처 추출 변경**: 새 피처 형식은 별도 `model_name` 사용 (기존 모델 비호환)

**하이퍼파라미터**: 하드코딩 대신 `**kwargs`로 받기

**데이터 테이블 변경**: 기존 데이터 마이그레이션 필요

## 성능 튜닝

**학습 속도**: `num_collection`, `num_training`, `num_epoch` 감소

**정확도**: 더 많은 데이터, 더 많은 epoch

## 데이터 흐름 종합

```
┌──────────────────────────────────────────────────────────────────┐
│                          학습 Phase                               │
└──────────────────────────────────────────────────────────────────┘

1. 쿼리 파일 (stats_train_time2int.txt)
   ↓
2. load_training_sql() → SQL 리스트 로드
   ↓
3. For each SQL:
   ├─ PilotDataInteractor.pull_subquery_card()
   ├─ PostgreSQL 실행 → PilotTransData
   │   └─ subquery_2_card = {"sub1": 123, "sub2": 456, ...}
   └─ DataManager.save_data({"query": sql, "card": 123})
   ↓
4. DB 테이블: mscn_pretraining_{dataset}
   +----+──────────────────────────────+------+
   | id | query                        | card |
   +----+──────────────────────────────+------+
   | 1  | SELECT COUNT(*) FROM ...     | 1234 |
   | 2  | SELECT COUNT(*) FROM ...     | 567  |
   +----+──────────────────────────────+------+
   ↓
5. data_manager.read_all() → DataFrame
   ↓
6. parse_queries() → mscnQueryMeta
   ├─ tables: [['badges', 'posts'], ['users'], ...]
   ├─ joins: [['b.userid=p.owneruserid'], [], ...]
   └─ predicates: [[['p.score','>=','0']], ...]
   ↓
7. Feature.fit()
   ├─ get_set_encoding() → table2vec, column2vec, op2vec, join2vec
   ├─ encode_tables() → 원-핫 벡터
   ├─ encode_conditions() → [컬럼|연산자|값] 벡터
   ├─ normalize_labels() → log + min-max
   └─ make_dataset() → 패딩 + 마스킹
   ↓
8. PyTorch DataLoader
   ├─ samples: [batch x max_tables x table_feats]
   ├─ predicates: [batch x max_predicates x predicate_feats]
   ├─ joins: [batch x max_joins x join_feats]
   ├─ labels: [batch x 1] (정규화된 카디널리티)
   └─ masks: 패딩 구분
   ↓
9. SetConv 신경망 학습
   ├─ 100 epochs
   ├─ Adam optimizer (lr=0.001)
   ├─ Q-Error 손실
   └─ Batch size=2048
   ↓
10. 모델 저장
    ExampleData/Mscn/Model/mscn_20241019_103000/
    ├─ nn_weights (PyTorch state_dict)
    ├─ feature_generator (인코딩 정보)
    └─ input_feature_dim (차원)

┌──────────────────────────────────────────────────────────────────┐
│                          예측 Phase                               │
└──────────────────────────────────────────────────────────────────┘

1. 사용자 SQL 실행 요청
   ↓
2. MscnCardPushHandler.acquire_injected_data()
   ├─ PilotDataInteractor.pull_subquery_card()
   ├─ EXPLAIN으로 서브쿼리 추출
   └─ subqueries = ["sub1", "sub2", ...]
   ↓
3. Feature.transform()
   ├─ 저장된 table2vec, column2vec 사용
   ├─ 동일한 인코딩 적용
   └─ 패딩 + 마스킹
   ↓
4. SetConv.forward()
   ├─ GPU 로드 (if CUDA)
   ├─ 3-way MLP 처리
   └─ Sigmoid 출력 [0,1]
   ↓
5. unnormalize_labels()
   ├─ 역정규화: output * (max-min) + min
   ├─ 지수 변환: exp(...)
   └─ -1 보정: pred - 1
   ↓
6. new_subquery_2_card
   {"sub1": "999", "sub2": "456", ...}
   ↓
7. MscnCardPushHandler.inject_data()
   └─ PostgreSQL Anchor에 카디널리티 힌트 전달
   ↓
8. PostgreSQL 옵티마이저
   ├─ MSCN 예측값으로 플랜 생성
   └─ 최적화된 실행 계획 사용
```

## ML 모델 입출력 요약

**입력 (학습/예측 공통)**:
- **테이블 벡터**: `[batch x max_tables x len(table2vec)]`
  - 예시: `[[1,0,0], [0,1,0]]` (badges, posts)
- **WHERE 벡터**: `[batch x max_predicates x (len(column2vec) + len(op2vec) + 1)]`
  - 예시: `[[0,1,0,..., 0,1,0, 0.15]]` (posts.score >= 10)
- **JOIN 벡터**: `[batch x max_joins x len(join2vec)]`
  - 예시: `[[1,0,0]]` (badges.userid = posts.owneruserid)

**학습 타겟**:
- **정규화된 카디널리티**: `[0, 1]` 범위
- **변환**: `log(card + 1)` → min-max 정규화

**예측 출력**:
- **정규화된 값**: `[0, 1]`
- **역변환**: 역정규화 → `exp()` → `-1` → 카디널리티

**학습 내용**:
- SQL 쿼리 구조 (테이블, JOIN, WHERE 조건) → 카디널리티 매핑 학습
- 복잡한 JOIN과 필터 조건의 선택도(selectivity) 학습

**추정 대상**:
- 쿼리 결과의 행 수 (카디널리티)
- PostgreSQL 옵티마이저가 사용하는 핵심 메트릭

## 문제 해결

- **학습 느림** → 데이터 크기 감소 (`num_collection`, `num_training`), 복잡한 쿼리 제외
- **예측 부정확** → 데이터 부족 (500개 이상 권장), 학습/테스트 분포 유사하게
- **Baseline보다 나쁨** → MSCN은 복잡한 JOIN에 효과적, 단순 쿼리는 오버헤드
- **OOM (메모리 부족)** → `batch_size` 감소 (2048 → 1024 → 512)
- **GPU 사용 안됨** → `export CUDA=1` 설정 확인
- **학습 데이터 누적** → 기존 데이터 삭제: `DROP TABLE mscn_pretraining_{dataset}`

## 핵심 파일 위치

| 파일 | 라인 | 역할 |
|------|------|------|
| `EventImplement.py` | 32-52 | 데이터 수집 (iterative_data_collection) |
| `EventImplement.py` | 54-96 | 모델 학습 (custom_model_training) |
| `mscn_utils.py` | 88-109 | 쿼리 파싱 (mscnQueryMeta) |
| `mscn_utils.py` | 142-185 | 피처 인코딩 (Feature.fit) |
| `mscn_model.py` | 16-57 | SetConv 신경망 |
| `mscn_model.py` | 108-166 | 모델 학습 (fit) |
| `mscn_model.py` | 167-192 | 예측 (predict) |
| `data.py` | 270-321 | 패딩/마스킹 (make_dataset) |
| `MscnParadigmCardAnchorHandler.py` | 전체 | 카디널리티 예측 및 주입 |
