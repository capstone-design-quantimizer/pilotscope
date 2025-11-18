#!/usr/bin/env python3
"""
Representative 폴더의 쿼리 파일을 파싱하여 pilotscope 형식으로 변환하는 스크립트

Representative 쿼리 형식:
    -- tag | template=... | date=...
    WITH base AS ( ... ) SELECT * FROM scored ...;

PilotScope 형식:
    SELECT ...;

사용법:
    python parse_representative_queries.py
"""

import os
import re
from pathlib import Path


def parse_queries_from_file(file_path):
    """
    Representative 폴더의 쿼리 파일을 파싱하여 쿼리 리스트로 반환

    Args:
        file_path: 쿼리 파일 경로

    Returns:
        list: 파싱된 쿼리 문자열 리스트
    """
    queries = []
    current_query_lines = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # 빈 줄은 건너뛰기
            if not line:
                continue

            # 주석 라인은 건너뛰기
            if line.startswith('--'):
                continue

            # 쿼리 라인 수집
            current_query_lines.append(line)

            # 세미콜론으로 끝나면 쿼리 완성
            if line.endswith(';'):
                # 여러 줄을 하나의 문자열로 합치고 공백 정리
                query = ' '.join(current_query_lines)
                # 다중 공백을 단일 공백으로 변환
                query = re.sub(r'\s+', ' ', query).strip()
                queries.append(query)
                current_query_lines = []

    return queries


def split_train_test(queries, train_ratio=0.8):
    """
    쿼리 리스트를 train/test로 분할

    Args:
        queries: 쿼리 리스트
        train_ratio: 학습 데이터 비율 (기본값: 0.8)

    Returns:
        tuple: (train_queries, test_queries)
    """
    split_idx = int(len(queries) * train_ratio)
    return queries[:split_idx], queries[split_idx:]


def write_pilotscope_format(queries, output_path):
    """
    쿼리 리스트를 pilotscope 형식으로 파일에 작성

    Args:
        queries: 쿼리 리스트
        output_path: 출력 파일 경로
    """
    # newline='\n'을 명시하여 LF 라인 엔딩 사용 (CRLF 방지)
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        for query in queries:
            # pilotscope 형식: 쿼리만 작성 (줄번호 없이)
            f.write(f"{query}\n")


def process_representative_file(input_file, output_prefix):
    """
    Representative 쿼리 파일을 처리하여 train/test 파일 생성

    Args:
        input_file: 입력 파일 경로
        output_prefix: 출력 파일 prefix (예: "stock_strategy_representative_momentum")

    Returns:
        tuple: (train_count, test_count)
    """
    print(f"\n처리 중: {input_file}")

    # 쿼리 파싱
    queries = parse_queries_from_file(input_file)
    print(f"  파싱된 쿼리 수: {len(queries)}")

    # train/test 분할
    train_queries, test_queries = split_train_test(queries, train_ratio=0.8)
    print(f"  Train: {len(train_queries)}, Test: {len(test_queries)}")

    # 출력 디렉토리 (현재 스크립트와 같은 디렉토리)
    output_dir = Path(__file__).parent

    # 파일 작성
    train_file = output_dir / f"{output_prefix}_train.txt"
    test_file = output_dir / f"{output_prefix}_test.txt"

    write_pilotscope_format(train_queries, train_file)
    write_pilotscope_format(test_queries, test_file)

    print(f"  생성됨: {train_file.name}")
    print(f"  생성됨: {test_file.name}")

    return len(train_queries), len(test_queries)


def main():
    """메인 함수"""
    print("="*70)
    print("Representative 쿼리 파일 파싱 스크립트")
    print("="*70)

    # 현재 스크립트 디렉토리
    current_dir = Path(__file__).parent
    representative_dir = current_dir / "Representative"

    # 처리할 파일 매핑
    files_to_process = [
        {
            "input": representative_dir / "momentum_queries.txt",
            "output_prefix": "stock_strategy_representative_momentum",
            "workload_name": "representative_momentum"
        },
        {
            "input": representative_dir / "smallcap_turnover_queries.txt",
            "output_prefix": "stock_strategy_representative_smallcap_turnover",
            "workload_name": "representative_smallcap_turnover"
        },
        {
            "input": representative_dir / "value_quality_queries.txt",
            "output_prefix": "stock_strategy_representative_value_quality",
            "workload_name": "representative_value_quality"
        }
    ]

    # 모든 파일 처리
    total_train = 0
    total_test = 0

    for file_info in files_to_process:
        if not file_info["input"].exists():
            print(f"\n⚠️  파일을 찾을 수 없습니다: {file_info['input']}")
            continue

        train_count, test_count = process_representative_file(
            file_info["input"],
            file_info["output_prefix"]
        )
        total_train += train_count
        total_test += test_count

    print("\n" + "="*70)
    print("완료!")
    print(f"  총 Train 쿼리: {total_train}")
    print(f"  총 Test 쿼리: {total_test}")
    print("="*70)

    print("\n다음 단계:")
    print("1. StockStrategyDataset.py에 새로운 Dataset 클래스 추가")
    print("2. algorithm_examples/utils.py에 워크로드 등록")
    print("3. unified_test.py로 테스트:")
    print("   python unified_test.py --algo baseline --db stock_strategy_representative_momentum")
    print()


if __name__ == "__main__":
    main()
