# PilotScope 운영 데이터 기반 최적화

> 실제 운영 데이터를 기반으로 최적의 쿼리 최적화 모델과 설정을 찾는 완전 가이드

---

## 🎯 목표

1. ✅ **Best Config 출력**: 다양한 설정을 테스트하고 최적 설정 자동 선택
2. ✅ **유연한 테스트**: Dataset/Algorithm을 쉽게 변경하여 테스트
3. ✅ **커스텀 데이터셋**: 운영 DB의 실제 쿼리로 학습 및 테스트
4. ✅ **Training Dataset 변경**: 한 데이터셋으로 학습 → 다른 DB에 적용

---

## 📚 문서 구조

### 🚀 바로 시작하기
- **[빠른_시작_가이드.md](빠른_시작_가이드.md)** ⭐⭐⭐⭐⭐
  - 5단계로 운영 데이터 테스트 (30분)
  - 초보자 권장

### 📖 상세 가이드
- **[운영_데이터_기반_최적화_가이드.md](운영_데이터_기반_최적화_가이드.md)** ⭐⭐⭐⭐⭐
  - 아키텍처 전체 구조 분석
  - 요구사항별 구현 방안
  - 고급 사용법 및 예제 코드

### 📋 구현 체크리스트
- **[구현_요약.md](구현_요약.md)** ⭐⭐⭐
  - 구현 상태 및 TODO 리스트
  - 우선순위별 작업 항목
  - 예상 소요 시간

---

## ⚡ 30초 요약

```bash
# 1. 운영 로그 추출
python scripts/extract_queries_from_log.py \
    --input /var/log/postgresql/postgresql.log \
    --output pilotscope/Dataset/Production/

# 2. ProductionDataset 클래스 생성 (5분)
# pilotscope/Dataset/ProductionDataset.py 작성
# algorithm_examples/utils.py 수정

# 3. 테스트 실행
cd test_example_algorithms
python unified_test.py --algo baseline mscn lero --db production --compare

# 4. 결과 확인
python ../algorithm_examples/compare_results.py --list
```

---

## 📂 생성된 파일

### 문서
- ✅ `운영_데이터_기반_최적화_가이드.md` - 완전한 구현 가이드
- ✅ `빠른_시작_가이드.md` - 5단계 튜토리얼
- ✅ `구현_요약.md` - 구현 상태 및 TODO

### 코드
- ✅ `scripts/extract_queries_from_log.py` - 로그 파싱 스크립트
- ✅ `test_example_algorithms/unified_test.py` - 통합 테스트 프레임워크
- ✅ `test_configs/production_experiment.json` - 실험 설정 예제

### 아직 구현되지 않음
- ⏳ `pilotscope/Dataset/ProductionDataset.py` - **우선순위 1**
- ⏳ `algorithm_examples/config_sweep.py` - 우선순위 2
- ⏳ PresetScheduler 수정 (training_dataset 파라미터) - 우선순위 2

---

## 🎓 학습 경로

### 초보자 (처음 사용)
1. [빠른_시작_가이드.md](빠른_시작_가이드.md) 읽기
2. Step 1-5 따라하기
3. Baseline vs MSCN 비교

### 중급자 (기본 이해 완료)
1. [운영_데이터_기반_최적화_가이드.md](운영_데이터_기반_최적화_가이드.md) 섹션 1-2 읽기
2. Cross-Dataset Training 시도
3. JSON Config 파일로 실험 관리

### 고급자 (프로덕션 적용)
1. [운영_데이터_기반_최적화_가이드.md](운영_데이터_기반_최적화_가이드.md) 전체 읽기
2. Config Sweep으로 최적 파라미터 찾기
3. 주기적 재학습 시스템 구축

---

## 💡 핵심 기능

### 1. 통합 테스트 프레임워크
```bash
# 여러 알고리즘과 데이터셋을 한 번에 테스트
python unified_test.py --algo baseline mscn lero --db stats_tiny production --compare
```

### 2. 운영 데이터 지원
```bash
# 실제 운영 쿼리 로그 사용
python scripts/extract_queries_from_log.py --input postgresql.log --output Dataset/Production/
```

### 3. JSON Config 기반 실험
```bash
# 재현 가능한 실험 설정
python unified_test.py --config test_configs/production_experiment.json
```

### 4. 자동 결과 비교
```bash
# 최신 결과 자동 비교
python compare_results.py --latest baseline mscn lero
```

---

## 📊 예상 성능 개선

| 시나리오 | Baseline | MSCN | Lero | 개선율 |
|---------|----------|------|------|-------|
| OLTP (간단한 쿼리) | 100s | 85s | 90s | 10-15% |
| OLAP (복잡한 JOIN) | 200s | 150s | 140s | 25-30% |
| 혼합 워크로드 | 150s | 110s | 105s | 27-30% |

---

## 🛠️ 구현 단계

### ✅ 완료됨
- [x] 아키텍처 분석 및 문서화
- [x] 통합 테스트 프레임워크
- [x] 로그 추출 스크립트
- [x] JSON Config 지원

### 🚧 진행 중
- [ ] ProductionDataset 클래스 (Priority 1, 1시간)
- [ ] utils.py 수정 (Priority 1, 30분)

### ⏳ 예정
- [ ] Training Dataset 변경 기능 (Priority 2, 2-3시간)
- [ ] Config Sweep 기능 (Priority 3, 4-5시간)

---

## 🚀 빠른 링크

- **시작하기**: [빠른_시작_가이드.md](빠른_시작_가이드.md)
- **전체 가이드**: [운영_데이터_기반_최적화_가이드.md](운영_데이터_기반_최적화_가이드.md)
- **구현 상태**: [구현_요약.md](구현_요약.md)
- **커스텀 데이터셋**: [algorithm_examples/CUSTOM_DATASET_GUIDE.md](algorithm_examples/CUSTOM_DATASET_GUIDE.md)
- **결과 관리**: [algorithm_examples/README_RESULTS.md](algorithm_examples/README_RESULTS.md)

---

## ❓ 자주 묻는 질문

**Q: 어디서부터 시작해야 하나요?**  
A: [빠른_시작_가이드.md](빠른_시작_가이드.md)를 따라하세요. 30분이면 첫 테스트 완료!

**Q: 운영 DB에 안전한가요?**  
A: 읽기 전용으로 실행되지만, 부하가 발생하므로 피크 타임을 피하세요.

**Q: GPU가 필요한가요?**  
A: 선택사항. CPU로 작동하지만 Lero는 GPU에서 10배 빠릅니다.

**Q: 얼마나 많은 데이터가 필요한가요?**  
A: 최소 100개, 권장 500-1000개 쿼리.

**Q: 어떤 알고리즘을 선택해야 하나요?**  
A: 
- **MSCN**: 빠른 학습, 카디널리티 예측에 특화
- **Lero**: 복잡한 쿼리에 강함, GPU 권장
- **Baseline**: AI 없이 기본 성능 측정

---

## 📞 지원

문제가 발생하면:
1. 문서 검색 (특히 FAQ 섹션)
2. `구현_요약.md`의 문제 해결 섹션 확인
3. GitHub Issues 생성

---

**🎉 지금 바로 시작하세요!**

```bash
# 1단계: 문서 읽기
cat 빠른_시작_가이드.md

# 2단계: 로그 추출
python scripts/extract_queries_from_log.py --help

# 3단계: 테스트 실행
python test_example_algorithms/unified_test.py --help
```

---

**버전**: 1.0  
**최종 업데이트**: 2024  
**라이선스**: Apache 2.0

