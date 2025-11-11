#!/usr/bin/env python3
"""
Split workload files by strategy_template into separate train/test SQL files

Usage:
    python split_workload_by_template.py --input workload_02 --output StockStrategy --train-ratio 0.8
"""
import json
import os
import argparse
import random
from pathlib import Path
from collections import defaultdict


def parse_workloads_by_template(workload_dir):
    """
    Parse all workload batch files and group SQL queries by strategy_template

    Args:
        workload_dir: Directory containing pilotscope_batch_*.txt files

    Returns:
        dict: {strategy_template: [sql_queries]}
    """
    template_queries = defaultdict(list)
    batch_files = sorted(Path(workload_dir).glob("pilotscope_batch_*.txt"))

    print(f"Found {len(batch_files)} batch files")

    for batch_file in batch_files:
        print(f"Processing {batch_file.name}...")

        with open(batch_file, 'r', encoding='utf-8') as f:
            json_buffer = ""

            for line in f:
                stripped = line.strip()

                if stripped == '---':
                    # Parse accumulated JSON
                    if json_buffer.strip():
                        try:
                            data = json.loads(json_buffer)
                            if 'metadata' in data and 'sql_query' in data:
                                template = data['metadata'].get('strategy_template', 'unknown')
                                sql_query = data['sql_query']
                                template_queries[template].append(sql_query)
                        except json.JSONDecodeError as e:
                            print(f"  Warning: Failed to parse JSON: {e}")

                    json_buffer = ""
                else:
                    json_buffer += line

            # Parse last document
            if json_buffer.strip():
                try:
                    data = json.loads(json_buffer)
                    if 'metadata' in data and 'sql_query' in data:
                        template = data['metadata'].get('strategy_template', 'unknown')
                        sql_query = data['sql_query']
                        template_queries[template].append(sql_query)
                except json.JSONDecodeError as e:
                    print(f"  Warning: Failed to parse last JSON: {e}")

    return template_queries


def normalize_template_name(template):
    """
    Convert strategy template name to valid filename

    Examples:
        'Value Investing Style' -> 'value_investing'
        'Momentum Investing Style' -> 'momentum_investing'
        'ML Hybrid Style' -> 'ml_hybrid'
    """
    # Remove 'Style' suffix
    name = template.replace(' Style', '').strip()
    # Convert to lowercase and replace spaces with underscores
    name = name.lower().replace(' ', '_')
    return name


def split_and_save_queries(template_queries, output_dir, train_ratio=0.8, shuffle=True):
    """
    Split queries by template into train/test files

    Args:
        template_queries: dict {strategy_template: [sql_queries]}
        output_dir: Output directory
        train_ratio: Train/test split ratio
        shuffle: Whether to shuffle queries before splitting
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = []

    for template, queries in template_queries.items():
        print(f"\nProcessing template: {template}")
        print(f"  Total queries: {len(queries)}")

        # Shuffle if requested
        if shuffle:
            queries_copy = queries.copy()
            random.shuffle(queries_copy)
        else:
            queries_copy = queries

        # Split train/test
        split_idx = int(len(queries_copy) * train_ratio)
        train_queries = queries_copy[:split_idx]
        test_queries = queries_copy[split_idx:]

        print(f"  Train: {len(train_queries)} queries")
        print(f"  Test:  {len(test_queries)} queries")

        # Generate filename prefix
        template_name = normalize_template_name(template)

        # Save train file
        train_file = output_path / f"stock_strategy_{template_name}_train.txt"
        with open(train_file, 'w', encoding='utf-8') as f:
            for query in train_queries:
                # Ensure semicolon at end
                query = query.rstrip(';')
                f.write(query + ';\n')
        print(f"  [+] Saved {train_file}")

        # Save test file
        test_file = output_path / f"stock_strategy_{template_name}_test.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            for query in test_queries:
                query = query.rstrip(';')
                f.write(query + ';\n')
        print(f"  [+] Saved {test_file}")

        summary.append({
            'template': template,
            'normalized_name': template_name,
            'train_count': len(train_queries),
            'test_count': len(test_queries),
            'train_file': str(train_file),
            'test_file': str(test_file)
        })

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Split workload files by strategy_template into train/test SQL files'
    )
    parser.add_argument('--input', required=True,
                       help='Input directory containing workload batch files')
    parser.add_argument('--output', required=True,
                       help='Output directory for SQL files')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Train/test split ratio (default: 0.8)')
    parser.add_argument('--no-shuffle', action='store_true',
                       help='Do not shuffle queries before splitting')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for shuffling (default: 42)')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print("=" * 60)
    print("Workload Splitter by Strategy Template")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Shuffle: {not args.no_shuffle}")
    print("=" * 60)

    # Step 1: Parse workloads
    print("\n[*] Parsing workload files...")
    template_queries = parse_workloads_by_template(args.input)

    print(f"\n[+] Found {len(template_queries)} strategy templates:")
    for template, queries in template_queries.items():
        print(f"   - {template}: {len(queries)} queries")

    # Step 2: Split and save
    print("\n[*] Splitting and saving queries...")
    summary = split_and_save_queries(
        template_queries,
        args.output,
        train_ratio=args.train_ratio,
        shuffle=not args.no_shuffle
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for item in summary:
        print(f"\n[*] {item['template']}")
        print(f"   Normalized name: {item['normalized_name']}")
        print(f"   Train: {item['train_count']} queries")
        print(f"   Test:  {item['test_count']} queries")

    print("\n" + "=" * 60)
    print("Complete! Next steps:")
    print("=" * 60)
    print(f"1. Create Dataset classes in pilotscope/Dataset/StockStrategyDataset.py")
    print(f"2. Register datasets in algorithm_examples/utils.py")
    print(f"3. Test with: python unified_test.py --algo mscn --db stock_strategy_value_investing")
    print("=" * 60)


if __name__ == '__main__':
    main()