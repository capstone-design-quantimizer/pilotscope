#!/usr/bin/env python3
"""
Check for failed runs in MLflow
"""
import mlflow
from pathlib import Path

# MLflow 설정
project_root = Path(__file__).parent.parent.resolve()
mlflow_tracking_uri = str((project_root / "mlruns").absolute())
mlflow.set_tracking_uri(mlflow_tracking_uri)

# 모든 실험 가져오기
exps = mlflow.search_experiments()
print(f"Total experiments: {len(exps)}")

total_failed = 0
for exp in exps:
    if exp.lifecycle_stage == "deleted":
        continue

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FAILED'"
    )

    if len(runs) > 0:
        print(f"\n{exp.name}: {len(runs)} FAILED runs")
        for _, run in runs.iterrows():
            print(f"  - {run.run_id}: {run.get('tags.mlflow.runName', 'N/A')}")
            total_failed += 1

print(f"\n총 FAILED runs: {total_failed}개")
