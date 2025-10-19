#!/usr/bin/env python3
"""
PostgreSQL 로그 파일에서 SQL 쿼리를 추출하고 train/test 분할

사용법:
    python extract_queries_from_log.py \
        --input /var/log/postgresql/postgresql.log \
        --output pilotscope/Dataset/Production/ \
        --train-ratio 0.8
"""

import argparse
import re
import os
from collections import Counter
from pathlib import Path


def parse_postgresql_log(log_file_path):
    """
    PostgreSQL 로그 파일을 파싱하여 SQL 쿼리 추출
    
    Args:
        log_file_path: 로그 파일 경로
    
    Returns:
        List[str]: 추출된 SQL 쿼리 리스트
    """
    queries = []
    
    # PostgreSQL 로그 패턴
    # 예: 2024-01-01 10:00:00.000 UTC [12345] LOG:  statement: SELECT * FROM ...
    statement_pattern = re.compile(r'(?:LOG|STATEMENT):\s+(?:statement:|execute\s+\w+:)\s*(.+)', re.IGNORECASE)
    
    current_query = None
    
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 새로운 쿼리 시작
            match = statement_pattern.search(line)
            if match:
                if current_query:
                    queries.append(current_query.strip())
                current_query = match.group(1)
            elif current_query and not line.strip().startswith('['):
                # 멀티라인 쿼리 계속
                current_query += ' ' + line.strip()
        
        # 마지막 쿼리 추가
        if current_query:
            queries.append(current_query.strip())
    
    return queries


def filter_queries(queries, query_types=['SELECT']):
    """
    특정 타입의 쿼리만 필터링
    
    Args:
        queries: 전체 쿼리 리스트
        query_types: 포함할 쿼리 타입 (예: ['SELECT', 'INSERT'])
    
    Returns:
        List[str]: 필터링된 쿼리 리스트
    """
    filtered = []
    
    for query in queries:
        query_upper = query.upper().strip()
        
        # 쿼리 타입 확인
        is_target_type = any(query_upper.startswith(qtype) for qtype in query_types)
        
        # 시스템 쿼리 제외
        is_system_query = any([
            'pg_catalog' in query_upper,
            'information_schema' in query_upper,
            'pg_stat' in query_upper,
            'pg_class' in query_upper,
            'pilotscope' in query_upper.lower(),  # PilotScope 내부 쿼리 제외
        ])
        
        if is_target_type and not is_system_query:
            filtered.append(query)
    
    return filtered


def normalize_query(query):
    """
    쿼리 정규화 (파라미터 제거, 공백 정리)
    
    Args:
        query: 원본 쿼리
    
    Returns:
        str: 정규화된 쿼리
    """
    # 여러 공백을 하나로
    query = re.sub(r'\s+', ' ', query)
    
    # 주석 제거
    query = re.sub(r'--.*?$', '', query, flags=re.MULTILINE)
    query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
    
    # 세미콜론 제거 (나중에 다시 추가할 것)
    query = query.rstrip(';')
    
    return query.strip()


def deduplicate_queries(queries, max_occurrences=10):
    """
    중복 쿼리 제거 및 빈도 제한
    
    Args:
        queries: 전체 쿼리 리스트
        max_occurrences: 각 쿼리의 최대 출현 횟수
    
    Returns:
        List[str]: 중복 제거된 쿼리 리스트
    """
    normalized = [normalize_query(q) for q in queries]
    
    # 빈도 계산
    query_counts = Counter(normalized)
    
    # 중복 제거하되, 인기 있는 쿼리는 max_occurrences까지 유지
    result = []
    query_occurrence = {}
    
    for query in normalized:
        if query not in query_occurrence:
            query_occurrence[query] = 0
        
        if query_occurrence[query] < max_occurrences:
            result.append(query)
            query_occurrence[query] += 1
    
    return result


def split_train_test(queries, train_ratio=0.8, shuffle=True):
    """
    쿼리를 train/test로 분할
    
    Args:
        queries: 전체 쿼리 리스트
        train_ratio: 학습 데이터 비율
        shuffle: 무작위 섞기 여부
    
    Returns:
        Tuple[List[str], List[str]]: (train_queries, test_queries)
    """
    import random
    
    queries_copy = queries.copy()
    
    if shuffle:
        random.shuffle(queries_copy)
    
    split_idx = int(len(queries_copy) * train_ratio)
    
    train_queries = queries_copy[:split_idx]
    test_queries = queries_copy[split_idx:]
    
    return train_queries, test_queries


def save_queries_to_file(queries, output_file):
    """
    쿼리를 파일로 저장 (세미콜론으로 구분)
    
    Args:
        queries: 쿼리 리스트
        output_file: 출력 파일 경로
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for query in queries:
            # 각 쿼리는 세미콜론으로 종료
            f.write(query + ';\n')
    
    print(f"✅ Saved {len(queries)} queries to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Extract SQL queries from PostgreSQL log')
    parser.add_argument('--input', required=True, help='Input PostgreSQL log file path')
    parser.add_argument('--output', required=True, help='Output directory for train/test files')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/test split ratio (default: 0.8)')
    parser.add_argument('--query-types', nargs='+', default=['SELECT'], 
                       help='Query types to extract (default: SELECT)')
    parser.add_argument('--max-occurrences', type=int, default=10, 
                       help='Max occurrences for each unique query (default: 10)')
    parser.add_argument('--no-shuffle', action='store_true', help='Do not shuffle queries')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PostgreSQL Query Log Extractor")
    print("=" * 60)
    
    # Step 1: 로그 파싱
    print(f"\n📂 Reading log file: {args.input}")
    queries = parse_postgresql_log(args.input)
    print(f"   Found {len(queries)} total statements")
    
    # Step 2: 쿼리 필터링
    print(f"\n🔍 Filtering {args.query_types} queries...")
    filtered = filter_queries(queries, args.query_types)
    print(f"   Filtered to {len(filtered)} queries")
    
    # Step 3: 중복 제거
    print(f"\n🗑️  Deduplicating (max {args.max_occurrences} occurrences per query)...")
    deduplicated = deduplicate_queries(filtered, args.max_occurrences)
    print(f"   Deduplicated to {len(deduplicated)} queries")
    
    if len(deduplicated) == 0:
        print("\n❌ No queries found. Please check your log file.")
        return
    
    # Step 4: Train/Test 분할
    print(f"\n✂️  Splitting train/test (ratio: {args.train_ratio})...")
    train_queries, test_queries = split_train_test(
        deduplicated, 
        train_ratio=args.train_ratio, 
        shuffle=not args.no_shuffle
    )
    print(f"   Train: {len(train_queries)} queries")
    print(f"   Test:  {len(test_queries)} queries")
    
    # Step 5: 파일 저장
    print(f"\n💾 Saving to {args.output}")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / "production_train.txt"
    test_file = output_dir / "production_test.txt"
    
    save_queries_to_file(train_queries, train_file)
    save_queries_to_file(test_queries, test_file)
    
    print("\n" + "=" * 60)
    print("✨ Extraction complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Create ProductionDataset class in pilotscope/Dataset/ProductionDataset.py")
    print(f"2. Add 'production' to load_test_sql() in algorithm_examples/utils.py")
    print(f"3. Run tests with config.db = 'production_db'")
    print("=" * 60)


if __name__ == '__main__':
    main()

