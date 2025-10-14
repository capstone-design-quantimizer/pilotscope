"""
Algorithm Performance Comparison Tool
======================================
Compare multiple test results from different algorithms.

Usage Examples:

1. Compare specific test runs:
   python compare_results.py results/baseline_*.json results/mscn_*.json

2. Compare latest runs for each algorithm:
   python compare_results.py --latest baseline mscn lero

3. List all saved results:
   python compare_results.py --list
"""

import sys
sys.path.append("../")

import argparse
from pathlib import Path
from algorithm_examples.utils import compare_algorithms, list_saved_results, load_test_results


def get_latest_result(algo_name, db_name="stats_tiny", results_dir="results"):
    """Get the most recent result file for a given algorithm and database."""
    results_path = Path(results_dir)
    pattern = f"{algo_name}_{db_name}_*.json"
    
    matching_files = sorted(results_path.glob(pattern), reverse=True)
    if matching_files:
        return matching_files[0]
    else:
        print(f"⚠️  No results found for {algo_name} on {db_name}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Compare algorithm performance test results")
    parser.add_argument('files', nargs='*', help='JSON result files to compare')
    parser.add_argument('--latest', nargs='+', help='Compare latest results for specified algorithms')
    parser.add_argument('--list', action='store_true', help='List all saved results')
    parser.add_argument('--metric', default='total_time', 
                       choices=['total_time', 'average_time', 'query_count'],
                       help='Metric to compare (default: total_time)')
    parser.add_argument('--db', default='stats_tiny', help='Database name (default: stats_tiny)')
    parser.add_argument('--output', help='Output path for comparison chart (without extension)')
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        list_saved_results()
        return
    
    # Collect files to compare
    result_files = []
    
    if args.latest:
        # Compare latest results for specified algorithms
        for algo in args.latest:
            latest_file = get_latest_result(algo, args.db)
            if latest_file:
                result_files.append(str(latest_file))
    elif args.files:
        # Use explicitly specified files
        result_files = args.files
    else:
        parser.print_help()
        return
    
    if len(result_files) < 2:
        print("❌ Need at least 2 result files to compare")
        return
    
    # Perform comparison
    print(f"\n🔍 Comparing {len(result_files)} results...")
    for f in result_files:
        print(f"  • {Path(f).name}")
    
    compare_algorithms(
        result_files,
        metric=args.metric,
        output_path=args.output
    )


if __name__ == '__main__':
    # If run without arguments, show example usage
    if len(sys.argv) == 1:
        print(__doc__)
        print("\n" + "="*60)
        print("Quick Examples:")
        print("="*60)
        print("\n1. List all saved results:")
        print("   python compare_results.py --list")
        print("\n2. Compare latest baseline vs mscn:")
        print("   python compare_results.py --latest baseline mscn")
        print("\n3. Compare latest baseline vs mscn vs lero:")
        print("   python compare_results.py --latest baseline mscn lero")
        print("\n4. Compare specific files:")
        print("   python compare_results.py results/baseline_stats_tiny_20231014_120000.json results/mscn_stats_tiny_20231014_130000.json")
        print("="*60 + "\n")
    else:
        main()
