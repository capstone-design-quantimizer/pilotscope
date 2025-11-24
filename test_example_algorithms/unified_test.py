#!/usr/bin/env python3
"""
통합 테스트 프레임워크 - 여러 알고리즘과 데이터셋을 쉽게 조합하여 테스트
모든 결과는 MLflow에 자동으로 로깅됩니다.

사용법:
    # 단일 알고리즘 테스트
    python unified_test.py --algo mscn --db stats_tiny

    # JSON config 파일로 실행
    python unified_test.py --config test_configs/production_experiment.json

    # 파라미터 조정
    python unified_test.py --algo mscn --db stats_tiny --epochs 100 --training-size 500

    # 기존 모델 로드 (학습 없이)
    python unified_test.py --algo mscn --db stats_tiny --no-training

    # MLflow UI에서 결과 확인 (브라우저: http://localhost:54321)
"""

import sys
sys.path.append("../")

# Force reload of PilotScheduler module to avoid bytecode cache issues
if 'pilotscope.PilotScheduler' in sys.modules:
    del sys.modules['pilotscope.PilotScheduler']
if 'pilotscope.DBController.BaseDBController' in sys.modules:
    del sys.modules['pilotscope.DBController.BaseDBController']

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from pilotscope.Common.Util import pilotscope_exit
from pilotscope.Common.TimeStatistic import TimeStatistic
from pilotscope.PilotConfig import PilotConfig, PostgreSQLConfig

from algorithm_examples.utils import load_test_sql

# Algorithm Registry
from algorithm_examples.Baseline.BaselinePresetScheduler import get_baseline_preset_scheduler
from algorithm_examples.Mscn.MscnPresetScheduler import get_mscn_preset_scheduler
from algorithm_examples.Lero.LeroPresetScheduler import get_lero_preset_scheduler
from algorithm_examples.KnobTuning.KnobPresetScheduler import get_knob_preset_scheduler
from algorithm_examples.Index.IndexPresetScheduler import get_index_preset_scheduler


# ============================================================================
# Algorithm Registry
# ============================================================================

ALGORITHM_REGISTRY = {
    "baseline": {
        "name": "Baseline (No AI)",
        "factory": get_baseline_preset_scheduler,
        "default_params": {}
    },
    "mscn": {
        "name": "MSCN (Cardinality Estimation)",
        "factory": get_mscn_preset_scheduler,
        "default_params": {
            "enable_collection": True,  # 기본값: 항상 collection 수행 (--no-collection으로 비활성화)
            "enable_training": True,
            "num_collection": 100,  # 기본 collection 크기 (빠른 테스트용, -1로 전체 수집 가능)
            "num_training": -1, # collection 중 일부를 활용 (-1: 전체 사용)
            "num_epoch": 100
        }
    },
    "lero": {
        "name": "Lero (Learned Optimizer)",
        "factory": get_lero_preset_scheduler,
        "default_params": {
            "enable_collection": True,  # 기본값: 항상 collection 수행 (--no-collection으로 비활성화)
            "enable_training": True,
            "num_collection": 100,  # 기본 collection 크기 (빠른 테스트용, -1로 전체 수집 가능)
            "num_training": 500,  # Plan 개수 (training SQL 보다 많음)
            "num_epoch": 100
        }
    },
    "knob": {
        "name": "KnobTuning (Configuration Optimization)",
        "factory": get_knob_preset_scheduler,
        "default_params": {}
    },
    "index": {
        "name": "Index Selection (Extend Algorithm)",
        "factory": get_index_preset_scheduler,
        "default_params": {}
    }
}


# ============================================================================
# Test Execution
# ============================================================================

def run_single_test(config: PilotConfig, algo_name: str, db_name: str,
                   workload_name: str = None, algo_params: Dict = None, use_mlflow: bool = True) -> Dict:
    """
    단일 알고리즘 + 데이터베이스 + 워크로드 조합 테스트 실행

    Args:
        config: PilotConfig 인스턴스
        algo_name: 알고리즘 이름 ('mscn', 'lero', 'baseline' 등)
        db_name: 데이터베이스 이름 ('stats_tiny', 'imdb' 등) 또는 데이터셋 식별자 ('stock_strategy_value_investing')
        workload_name: 워크로드 이름 (None이면 db_name과 동일, 'custom'이면 '{db_name}_custom')
        algo_params: 알고리즘별 추가 파라미터
        use_mlflow: MLflow 사용 여부 (기본값: True)

    Returns:
        Dict: 테스트 결과 정보
    """
    # Determine actual workload identifier for loading queries
    if workload_name is None or workload_name == "default":
        dataset_name = db_name  # Use default workload
    else:
        dataset_name = f"{db_name}_{workload_name}"  # e.g., "stats_tiny_custom"

    # Set actual PostgreSQL database name
    # For StockStrategy datasets, all workloads share the same 'stock_strategy' DB
    if dataset_name.startswith("stock_strategy"):
        config.db = "stock_strategy"
        actual_db_display = "stock_strategy"
    else:
        config.db = dataset_name
        actual_db_display = dataset_name

    print("\n" + "=" * 60)
    print(f"Testing: {algo_name.upper()} on {dataset_name}")
    if actual_db_display != dataset_name:
        print(f"  Dataset: {dataset_name}")
        print(f"  Actual DB: {actual_db_display}")
    print("=" * 60)

    # Knob Tuning의 경우 deep control 활성화 (DB 재시작 권한 필요)
    if algo_name == "knob":
        from algorithm_examples.ExampleConfig import example_pg_bin, example_pgdata
        config.enable_deep_control_local(example_pg_bin, example_pgdata)
        print(f"🔧 Knob Tuning deep control 활성화")
        print(f"   PostgreSQL bin: {example_pg_bin}")
        print(f"   PostgreSQL data: {example_pgdata}")

    # Lero의 경우 타임아웃을 늘림 (Lero는 쿼리당 여러 번 DB 호출)
    if algo_name == "lero":
        original_timeout = config.once_request_timeout
        config.once_request_timeout = 900  # 15분으로 증가
        print(f"⏱️  Lero 타임아웃 설정: {original_timeout}초 → {config.once_request_timeout}초")
        print(f"   (Lero는 쿼리당 여러 실행 계획을 생성하므로 시간이 오래 걸립니다)")

    # TimeStatistic 초기화
    TimeStatistic.clear()

    # 알고리즘 정보 가져오기
    if algo_name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    algo_info = ALGORITHM_REGISTRY[algo_name]

    # 파라미터 병합 (default + user provided)
    params = algo_info.get("default_params", {}).copy()
    if algo_params:
        params.update(algo_params)

    # Add use_mlflow parameter and dataset info
    params['use_mlflow'] = use_mlflow
    params['experiment_name'] = f"{algo_name}_{dataset_name}"  # Full dataset name including workload
    params['dataset_name'] = dataset_name  # For MLflow logging

    # 스케줄러 생성 (MLflow 지원 알고리즘은 (scheduler, tracker) 튜플 반환)
    factory = algo_info["factory"]
    mlflow_tracker = None

    try:
        result = factory(config, **params)

        # Handle return value (tuple for MLflow-enabled algorithms, scheduler only for others)
        if isinstance(result, tuple):
            scheduler, mlflow_tracker = result
        else:
            scheduler = result
            mlflow_tracker = None

    except Exception as e:
        print(f"\n❌ Error during scheduler initialization (collection/training phase):")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()

        # End MLflow run if it was started
        if mlflow_tracker:
            print(f"\n📊 Marking MLflow run as FAILED due to collection/training error")
            mlflow_tracker.end_run(status="FAILED")

        # Return early - skip test phase
        print(f"\n⏭️  Skipping test phase due to collection/training failure")
        return {
            "algorithm": algo_name,
            "database": db_name,
            "workload": workload_name or "default",
            "dataset": dataset_name,
            "elapsed_time": 0,
            "params": algo_params,
            "mlflow_run_id": None,
            "skipped": "collection_failed",
            "error": str(e)
        }

    # Backup initial DB state for all algorithms (enables safe cleanup later)
    # Even if algorithm doesn't modify config/indexes, backup is harmless
    try:
        if hasattr(scheduler.db_controller, 'backup_config'):
            scheduler.db_controller.backup_config()
    except Exception:
        pass  # Deep control not enabled, skip

    # # 중요: scheduler가 어느 파일에서 로드되었는지 확인
    # import inspect
    # print(f"\n🔍 Scheduler module file: {inspect.getfile(scheduler.__class__)}")
    # print(f"   Scheduler execute method file: {inspect.getfile(scheduler.execute)}")
    
    # # execute 메서드의 소스 코드 첫 줄 확인
    # source_lines = inspect.getsourcelines(scheduler.execute)[0][:20]
    # print(f"   First 5 lines of execute():")
    # for line in source_lines:
    #     print(f"     {line.rstrip()}")
    
    # Check if we should skip testing (when only collecting data)
    if algo_params and not algo_params.get('enable_training', True):
        print(f"\n⏭️  Skipping test phase (--no-training specified)")
        print(f"   ✅ Data collection completed!")
        print(f"\n💡 Next step: Train with collected data:")
        print(f"   python unified_test.py --algo {algo_name} --db {db_name}", end="")
        if workload_name:
            print(f" --workload {workload_name}", end="")
        print(f" \\\n       --no-collection --training-size 500 --epochs 100")

        # End MLflow run if exists
        run_id = None
        if mlflow_tracker:
            run_id = mlflow_tracker.run_id
            mlflow_tracker.end_run(status="FINISHED")
            print(f"\n   ✓ MLflow run completed: {run_id}")

        return {
            "algorithm": algo_name,
            "database": db_name,
            "workload": workload_name or "default",
            "dataset": dataset_name,
            "elapsed_time": 0,
            "params": algo_params,
            "mlflow_run_id": run_id,
            "skipped": "no_training"
        }

    # 테스트 SQL 로드
    print(f"\n📂 Loading test queries from {dataset_name}...")
    try:
        test_sqls = load_test_sql(dataset_name)
        print(f"   Loaded {len(test_sqls)} queries")
    except Exception as e:
        print(f"❌ Failed to load test queries: {e}")
        return None

    # 쿼리 실행 통계 초기화
    query_stats = {
        'total': len(test_sqls),
        'success': 0,
        'failed': 0,
        'db_execution_times': []
    }

    # 쿼리 실행
    print(f"\n{'='*60}")
    print(f"🧪 Test Phase: Running {len(test_sqls)} test queries")
    print(f"{'='*60}")
    print(f"📊 Algorithm: {algo_name.upper()}")
    print(f"📊 Dataset: {dataset_name}")
    if test_sqls:
        sql_preview = test_sqls[0][:100] if len(test_sqls[0]) > 100 else test_sqls[0]
        print(f"📝 First query preview: {sql_preview}...")
    print(f"{'='*60}\n")

    start_time = time.time()

    # 실제 DB 쿼리 실행 시간 추적 (AI 모델 추론 시간 제외)
    total_db_execution_time = 0.0

    try:
        for i, sql in enumerate(test_sqls):
            query_num = i + 1
            if query_num == 1 or query_num % 10 == 0 or query_num == len(test_sqls):
                print(f"⏱️  Executing query {query_num}/{len(test_sqls)}...")

            try:
                # Python 레벨 시간 측정 (전체 오버헤드 포함)
                TimeStatistic.start(algo_name.capitalize())
                result = scheduler.execute(sql)
                TimeStatistic.end(algo_name.capitalize())

                # 실제 DB 실행 시간 추출 (pull_execution_time=True로 수집된 데이터)
                exec_time = None
                if hasattr(scheduler, 'last_execution_data'):
                    exec_data = scheduler.last_execution_data
                    if exec_data is not None and hasattr(exec_data, 'execution_time') and exec_data.execution_time is not None:
                        # execution_time이 milliseconds인 경우도 있으므로 확인
                        exec_time = float(exec_data.execution_time)
                        # DB에 따라 milliseconds로 저장될 수 있음 (PostgreSQL은 보통 milliseconds)
                        if exec_time > 1000:  # 1000ms 이상이면 밀리초로 간주
                            exec_time = exec_time / 1000.0
                        total_db_execution_time += exec_time
                        query_stats['db_execution_times'].append(exec_time)

                query_stats['success'] += 1

                if query_num == 1 or query_num % 10 == 0 or query_num == len(test_sqls):
                    if exec_time is not None:
                        print(f"   ✅ Query {query_num} completed: {exec_time:.3f}s")
                    else:
                        print(f"   ✅ Query {query_num} completed (time not tracked)")

            except Exception as e:
                query_stats['failed'] += 1
                print(f"   ❌ Query {query_num} failed: {str(e)[:100]}")
    finally:
        # 스케줄러/인터랙터 리소스 정리
        print(f"\n🧹 Cleaning up resources for {algo_name}...")
        if hasattr(scheduler, 'data_interactor'):
            try:
                # Reset data_interactor to remove all registered anchors
                scheduler.data_interactor.reset()
                print(f"   ✓ data_interactor reset")
            except Exception as cleanup_err:
                print(f"   ⚠️  Warning: Failed to reset data_interactor: {cleanup_err}")
        if hasattr(scheduler, 'db_controller'):
            try:
                # Cleanup DB controller connections
                scheduler.db_controller.cleanup()
                print(f"   ✓ db_controller cleanup")
            except Exception as cleanup_err:
                print(f"   ⚠️  Warning: Failed to cleanup db_controller: {cleanup_err}")
    
    end_time = time.time()
    elapsed = end_time - start_time

    # 통계 출력
    print(f"\n{'='*60}")
    print(f"📈 Test Execution Statistics")
    print(f"{'='*60}")
    print(f"✅ Successful: {query_stats['success']}/{query_stats['total']}")
    print(f"❌ Failed: {query_stats['failed']}/{query_stats['total']}")

    if query_stats['db_execution_times']:
        avg_time = sum(query_stats['db_execution_times']) / len(query_stats['db_execution_times'])
        min_time = min(query_stats['db_execution_times'])
        max_time = max(query_stats['db_execution_times'])
        print(f"\n⏱️  DB Execution Time Statistics:")
        print(f"   Average: {avg_time:.3f}s")
        print(f"   Min/Max: {min_time:.3f}s / {max_time:.3f}s")
        print(f"   Total DB time: {total_db_execution_time:.3f}s")

    print(f"\n⏱️  Wall Time Statistics:")
    print(f"   Total wall time: {elapsed:.3f}s (Python + DB + AI)")
    if total_db_execution_time > 0:
        overhead = elapsed - total_db_execution_time
        print(f"   Overhead: {overhead:.3f}s (AI 추론 + 데이터 수집)")
        print(f"   DB execution %: {(total_db_execution_time/elapsed)*100:.1f}%")
    else:
        print(f"   ⚠️  DB execution time not tracked")

    if query_stats['success'] > 0:
        print(f"\n📊 Throughput: {query_stats['success'] / elapsed:.2f} queries/sec")
    print(f"{'='*60}\n")

    # 결과 저장
    print(f"💾 Saving results...")
    name_2_value = TimeStatistic.get_sum_data()

    # Prepare metrics
    test_metrics = {
        "total_time": elapsed,
        "average_time": elapsed / len(test_sqls) if test_sqls else 0,
        "query_count": len(test_sqls),
        "db_execution_time": total_db_execution_time,
        "overhead_time": elapsed - total_db_execution_time if total_db_execution_time > 0 else 0
    }

    # Log to MLflow if tracker exists
    if mlflow_tracker:
        print(f"📊 Logging test results to MLflow...")
        mlflow_tracker.log_test_results(test_metrics, test_dataset=dataset_name, num_test_queries=len(test_sqls))

        # Update model with test results (save_model() will handle MLflow upload)
        if algo_name != "baseline" and hasattr(scheduler, 'pilot_model'):
            model = scheduler.pilot_model
            if hasattr(model, 'add_test_result'):
                performance = {
                    "total_time": elapsed,
                    "average_time": elapsed / len(test_sqls) if test_sqls else 0,
                    "num_queries": len(test_sqls),
                    "db_execution_time": total_db_execution_time
                }
                model.add_test_result(dataset_name, len(test_sqls), performance)
                # save_model() now automatically uploads to MLflow via mlflow_tracker
                model.save_model()
                print(f"   ✓ Model updated with test results")

        # End MLflow run (save run_id before it's cleared)
        run_id = mlflow_tracker.run_id

        # Determine run status based on success rate
        success_rate = query_stats['success'] / query_stats['total'] if query_stats['total'] > 0 else 0
        if success_rate < 0.5:
            mlflow_status = "FAILED"
            print(f"   ⚠️  Low success rate: {success_rate:.1%} - marking run as FAILED")
        else:
            mlflow_status = "FINISHED"

        mlflow_tracker.end_run(status=mlflow_status)
        print(f"   ✓ MLflow run completed: {run_id} (status: {mlflow_status})")

    print(f"\n✅ Test completed in {elapsed:.2f}s")

    # Cleanup: Always restore DB to initial state
    print(f"\n🧹 Restoring database to initial state...")
    try:
        # Drop all user-created indexes (safe even if none were created)
        scheduler.db_controller.drop_all_indexes()
        print("   ✓ Indexes cleaned")
    except Exception as e:
        # Expected for algorithms without deep control or if no indexes exist
        pass

    try:
        # Recover original config and restart (safe because we backed up earlier)
        if hasattr(scheduler.db_controller, 'recover_config'):
            scheduler.db_controller.recover_config()
            scheduler.db_controller.restart()
            print("   ✓ PostgreSQL configuration restored")
    except Exception as e:
        # Expected if deep control not enabled
        pass

    return {
        "algorithm": algo_name,
        "database": db_name,
        "workload": workload_name or "default",
        "dataset": dataset_name,  # For backward compatibility
        "elapsed_time": elapsed,
        "params": params,
        "mlflow_run_id": mlflow_tracker.run_id if mlflow_tracker else None
    }


def run_multiple_tests(config: PilotConfig, algorithms: List[str],
                      databases: List[str], workload: str = None, algo_params: Dict = None) -> List[Dict]:
    """
    여러 알고리즘 + 데이터베이스 + 워크로드 조합을 순차적으로 테스트

    Note: 이 함수는 이제 deprecated되었습니다. 간소화를 위해 run_single_test를 직접 사용하세요.

    Args:
        config: PilotConfig 인스턴스
        algorithms: 테스트할 알고리즘 리스트
        databases: 테스트할 데이터베이스 리스트
        workload: 워크로드 이름 (None이면 default, 'custom' 등)
        algo_params: 알고리즘별 파라미터 (algo_name을 키로 하는 dict)

    Returns:
        List[Dict]: 각 테스트의 결과 정보
    """
    results = []

    for db_name in databases:
        config.db = db_name  # Set actual database name

        for algo in algorithms:
            # 알고리즘별 파라미터 가져오기
            params = algo_params.get(algo, {}) if algo_params else {}

            try:
                result = run_single_test(config, algo, db_name, workload, params)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"\n❌ Test failed for {algo} on {db_name} (workload: {workload}): {e}")
                import traceback
                traceback.print_exc()
            finally:
                # 리소스 정리 - 더 철저하게
                print(f"\n🧹 Cleaning up resources for {algo} on {db_name}...")

                try:
                    pilotscope_exit()
                except Exception as e:
                    print(f"⚠️  Error during pilotscope_exit: {e}")

                # 강제로 가비지 컬렉션 실행
                import gc
                gc.collect()

                # 짧은 대기 시간 (DB 연결 완전히 닫히도록)
                import time
                time.sleep(2)

    return results


# ============================================================================
# JSON Config File Support
# ============================================================================

def load_config_file(config_file: str) -> Dict:
    """JSON config 파일 로드"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_from_config_file(config_file: str):
    """
    JSON config 파일에서 실험 설정을 읽어 실행
    
    Config 파일 형식:
    {
      "db_config": {...},
      "experiments": [
        {
          "name": "exp1",
          "algorithm": "mscn",
          "dataset": "stats_tiny",
          "params": {...}
        }
      ],
      "comparison": {...}
    }
    """
    print(f"\n📄 Loading config from: {config_file}")
    config_data = load_config_file(config_file)
    
    # DB Config 생성
    db_config_data = config_data.get("db_config", {})
    config = PostgreSQLConfig(**db_config_data)
    
    # 실험 실행
    experiments = config_data.get("experiments", [])
    results = []
    
    for exp in experiments:
        exp_name = exp.get("name", "unnamed")
        algo = exp.get("algorithm")
        db_name = exp.get("database", exp.get("dataset"))  # Support both 'database' and legacy 'dataset'
        workload = exp.get("workload", None)
        params = exp.get("params", {})

        print(f"\n🧪 Running experiment: {exp_name}")
        config.db = db_name

        result = run_single_test(config, algo, db_name, workload, params)
        if result:
            result["experiment_name"] = exp_name
            results.append(result)
        
        pilotscope_exit()
    
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified test framework for multiple algorithms and datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single algorithm on single database
  python unified_test.py --algo mscn --db stats_tiny

  # Test with custom workload on stats_tiny database
  python unified_test.py --algo mscn --db stats_tiny --workload custom --epochs 100

  # Test with custom parameters
  python unified_test.py --algo mscn --db stats_tiny --epochs 50 --training-size 500

  # Use JSON config file
  python unified_test.py --config test_configs/production_experiment.json
        """
    )
    
    # Mode selection
    parser.add_argument('--config', help='JSON config file path')
    
    # Algorithm & Dataset selection
    parser.add_argument('--algo',
                       choices=list(ALGORITHM_REGISTRY.keys()),
                       help='Algorithm to test')
    parser.add_argument('--db',
                       help='Database to test (e.g., stats_tiny, imdb, stock_strategy_value_investing)')
    parser.add_argument('--workload',
                       help='Workload to use (default: same as db, custom: use custom workload)')
    
    # Algorithm parameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--training-size', type=int,
                       help='Number of queries to use for training (-1 for all)')
    parser.add_argument('--collection-size', type=int,
                       help='Number of queries to collect for training (default: 100, -1 for all)')
    parser.add_argument('--no-collection', action='store_true',
                       help='Skip data collection (use existing collected data)')
    parser.add_argument('--no-training', action='store_true',
                       help='Skip model training (only collect data)')

    # DB Config
    parser.add_argument('--db-host', help='Database host')
    parser.add_argument('--db-port', help='Database port')
    parser.add_argument('--db-user', help='Database user')
    parser.add_argument('--db-pwd', help='Database password')
    parser.add_argument('--timeout', type=int, 
                       help='Request timeout in seconds (default: 600, recommend 900+ for Lero)')
    
    args = parser.parse_args()
    
    # JSON Config 파일 모드
    if args.config:
        run_from_config_file(args.config)
        return
    
    # CLI 모드
    if not args.algo or not args.db:
        parser.print_help()
        print("\n❌ Error: --algo and --db are required (or use --config)")
        return
    
    # DB Config 생성
    config = PostgreSQLConfig()
    if args.db_host:
        config.db_host = args.db_host
    if args.db_port:
        config.db_port = args.db_port
    if args.db_user:
        config.db_user = args.db_user
    if args.db_pwd:
        config.db_user_pwd = args.db_pwd
    if args.timeout:
        config.once_request_timeout = args.timeout
        config.sql_execution_timeout = args.timeout
        print(f"⏱️  타임아웃 설정: {args.timeout}초")
    
    # 알고리즘 파라미터 설정
    params = {}
    if args.epochs is not None:
        params['num_epoch'] = args.epochs
    if args.training_size is not None:
        params['num_training'] = args.training_size
    if args.collection_size is not None:
        params['num_collection'] = args.collection_size
    # Collection은 기본값으로 활성화되어 있음 (--no-collection으로 비활성화)
    if args.no_collection:
        params['enable_collection'] = False
    if args.no_training:
        params['enable_training'] = False

    # 테스트 실행
    try:
        result = run_single_test(config, args.algo, args.db, args.workload, params)

        if result:
            print("\n" + "=" * 60)
            print("✨ Test completed!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("⚠️  Test completed with issues")
            print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pilotscope_exit()


if __name__ == '__main__':
    main()

