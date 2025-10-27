#!/usr/bin/env python3
"""
MLflow Query Tool - Query and compare experiment results
Usage:
    python scripts/mlflow_query.py list                          # List all experiments
    python scripts/mlflow_query.py runs mscn_stats_tiny          # List runs in experiment
    python scripts/mlflow_query.py best mscn_stats_tiny          # Find best run
    python scripts/mlflow_query.py compare run1 run2            # Compare two runs
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pilotscope.Common.MLflowTracker import MLflowTracker
import mlflow
from tabulate import tabulate


def list_experiments():
    """List all MLflow experiments"""
    experiments = mlflow.search_experiments()

    if not experiments:
        print("No experiments found.")
        return

    table_data = []
    for exp in experiments:
        table_data.append([
            exp.experiment_id,
            exp.name,
            exp.lifecycle_stage,
            mlflow.search_runs(experiment_ids=[exp.experiment_id]).shape[0]
        ])

    print("\n" + "="*80)
    print("MLflow Experiments")
    print("="*80)
    print(tabulate(table_data, headers=["ID", "Name", "Stage", "# Runs"], tablefmt="grid"))


def list_runs(experiment_name, limit=10):
    """List runs in an experiment"""
    runs = MLflowTracker.list_runs(experiment_name, limit=limit)

    if not runs:
        print(f"No runs found in experiment: {experiment_name}")
        return

    table_data = []
    for run in runs:
        table_data.append([
            run.get('run_id', '')[:8],
            run.get('tags.mlflow.runName', '')[:30],
            run.get('status', ''),
            f"{run.get('metrics.test_total_time', 0):.3f}s" if 'metrics.test_total_time' in run else 'N/A',
            f"{run.get('metrics.test_average_time', 0):.4f}s" if 'metrics.test_average_time' in run else 'N/A',
            run.get('params.algorithm', ''),
            run.get('params.dataset', '')
        ])

    print("\n" + "="*120)
    print(f"Runs in experiment: {experiment_name} (showing {len(runs)} most recent)")
    print("="*120)
    print(tabulate(table_data,
                   headers=["Run ID", "Name", "Status", "Total Time", "Avg Time", "Algorithm", "Dataset"],
                   tablefmt="grid"))


def get_best_run(experiment_name, metric="test_total_time", ascending=True):
    """Find the best run in an experiment"""
    best = MLflowTracker.get_best_run(experiment_name, metric=metric, ascending=ascending)

    if not best:
        print(f"No runs found in experiment: {experiment_name}")
        return

    print("\n" + "="*80)
    print(f"Best run in {experiment_name} (by {metric})")
    print("="*80)
    print(f"Run ID: {best['run_id']}")
    print(f"Run Name: {best['run_name']}")
    print(f"{metric}: {best['metric_value']:.6f}")
    print("\nParameters:")
    for key, value in best['params'].items():
        print(f"  {key}: {value}")
    print(f"\nArtifact URI: {best['artifact_uri']}")


def compare_runs(run_id1, run_id2):
    """Compare two runs"""
    run1 = mlflow.get_run(run_id1)
    run2 = mlflow.get_run(run_id2)

    print("\n" + "="*120)
    print(f"Comparing runs")
    print("="*120)

    # Basic info
    table_data = [
        ["Run ID", run1.info.run_id[:8], run2.info.run_id[:8]],
        ["Run Name", run1.data.tags.get('mlflow.runName', 'N/A'), run2.data.tags.get('mlflow.runName', 'N/A')],
        ["Status", run1.info.status, run2.info.status],
    ]
    print(tabulate(table_data, headers=["", "Run 1", "Run 2"], tablefmt="grid"))

    # Metrics
    print("\nMetrics:")
    all_metrics = set(run1.data.metrics.keys()) | set(run2.data.metrics.keys())
    metric_data = []
    for metric in sorted(all_metrics):
        val1 = run1.data.metrics.get(metric, float('nan'))
        val2 = run2.data.metrics.get(metric, float('nan'))
        diff = val2 - val1 if not (val1 != val1 or val2 != val2) else float('nan')  # NaN check

        # Format with improvement indicator
        if diff != diff:  # NaN
            diff_str = "N/A"
        elif diff > 0:
            diff_str = f"+{diff:.4f} ⬆️"
        elif diff < 0:
            diff_str = f"{diff:.4f} ⬇️"
        else:
            diff_str = "0.0000 ➡️"

        metric_data.append([metric, f"{val1:.4f}", f"{val2:.4f}", diff_str])

    print(tabulate(metric_data, headers=["Metric", "Run 1", "Run 2", "Diff (2-1)"], tablefmt="grid"))

    # Parameters
    print("\nParameters:")
    all_params = set(run1.data.params.keys()) | set(run2.data.params.keys())
    param_data = []
    for param in sorted(all_params):
        val1 = run1.data.params.get(param, 'N/A')
        val2 = run2.data.params.get(param, 'N/A')
        same = "✓" if val1 == val2 else "✗"
        param_data.append([param, val1, val2, same])

    print(tabulate(param_data, headers=["Parameter", "Run 1", "Run 2", "Same"], tablefmt="grid"))


def main():
    parser = argparse.ArgumentParser(description="Query MLflow experiments and runs")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # List experiments
    subparsers.add_parser('list', help='List all experiments')

    # List runs
    runs_parser = subparsers.add_parser('runs', help='List runs in an experiment')
    runs_parser.add_argument('experiment', help='Experiment name')
    runs_parser.add_argument('--limit', type=int, default=10, help='Max number of runs to show')

    # Find best run
    best_parser = subparsers.add_parser('best', help='Find best run in experiment')
    best_parser.add_argument('experiment', help='Experiment name')
    best_parser.add_argument('--metric', default='test_total_time', help='Metric to optimize')
    best_parser.add_argument('--maximize', action='store_true', help='Maximize metric (default: minimize)')

    # Compare runs
    compare_parser = subparsers.add_parser('compare', help='Compare two runs')
    compare_parser.add_argument('run1', help='First run ID')
    compare_parser.add_argument('run2', help='Second run ID')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Set MLflow tracking URI to project root
    project_root = Path(__file__).parent.parent.resolve()
    mlruns_path = project_root / 'mlruns'
    mlflow.set_tracking_uri(str(mlruns_path.absolute()))

    if args.command == 'list':
        list_experiments()
    elif args.command == 'runs':
        list_runs(args.experiment, limit=args.limit)
    elif args.command == 'best':
        get_best_run(args.experiment, metric=args.metric, ascending=not args.maximize)
    elif args.command == 'compare':
        compare_runs(args.run1, args.run2)


if __name__ == '__main__':
    main()
