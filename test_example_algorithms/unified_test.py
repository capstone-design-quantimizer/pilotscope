#!/usr/bin/env python3
"""
통합 테스트 프레임워크 - 여러 알고리즘과 데이터셋을 쉽게 조합하여 테스트

사용법:
    # 여러 알고리즘과 데이터셋 조합 테스트
    python unified_test.py --algo mscn lero baseline --db stats_tiny production --compare

    # JSON config 파일로 실행
    python unified_test.py --config test_configs/production_experiment.json

    # 특정 조합만 테스트
    python unified_test.py --algo mscn --db production --epochs 100 --training-size 500
"""

import sys
sys.path.append("../")

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from pilotscope.Common.Util import pilotscope_exit
from pilotscope.Common.TimeStatistic import TimeStatistic
from pilotscope.Common.Drawer import Drawer
from pilotscope.PilotConfig import PilotConfig, PostgreSQLConfig
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor

from algorithm_examples.utils import load_test_sql, save_test_result, compare_algorithms
from algorithm_examples.ExampleConfig import get_time_statistic_img_path

# Algorithm Registry
from algorithm_examples.Baseline.BaselinePresetScheduler import get_baseline_preset_scheduler
from algorithm_examples.Mscn.MscnPresetScheduler import get_mscn_preset_scheduler
from algorithm_examples.Lero.LeroPresetScheduler import get_lero_preset_scheduler
from algorithm_examples.KnobTuning.KnobPresetScheduler import get_knob_preset_scheduler


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
            "enable_collection": True,
            "enable_training": True,
            "num_collection": -1,
            "num_training": -1,
            "num_epoch": 100
        }
    },
    "lero": {
        "name": "Lero (Learned Optimizer)",
        "factory": get_lero_preset_scheduler,
        "default_params": {
            "enable_collection": True,
            "enable_training": True,
            "num_collection": -1,
            "num_training": -1,
            "num_epoch": 100
        }
    },
    "knob": {
        "name": "KnobTuning (Configuration Optimization)",
        "factory": get_knob_preset_scheduler,
        "default_params": {}
    }
}


# ============================================================================
# Test Execution
# ============================================================================

def run_single_test(config: PilotConfig, algo_name: str, dataset_name: str, 
                   algo_params: Dict = None) -> Dict:
    """
    단일 알고리즘 + 데이터셋 조합 테스트 실행
    
    Args:
        config: PilotConfig 인스턴스
        algo_name: 알고리즘 이름 ('mscn', 'lero', 'baseline' 등)
        dataset_name: 데이터셋 이름 ('stats_tiny', 'production' 등)
        algo_params: 알고리즘별 추가 파라미터
    
    Returns:
        Dict: 테스트 결과 정보
    """
    print("\n" + "=" * 60)
    print(f"Testing: {algo_name.upper()} on {dataset_name}")
    print("=" * 60)
    
    # TimeStatistic 초기화
    TimeStatistic.reset()
    
    # 알고리즘 정보 가져오기
    if algo_name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    algo_info = ALGORITHM_REGISTRY[algo_name]
    
    # 파라미터 병합 (default + user provided)
    params = algo_info.get("default_params", {}).copy()
    if algo_params:
        params.update(algo_params)
    
    # 스케줄러 생성 (모든 알고리즘 동일한 패턴)
    factory = algo_info["factory"]
    scheduler_or_interactor = factory(config, **params)
    
    # 테스트 SQL 로드
    print(f"\n📂 Loading test queries from {dataset_name}...")
    try:
        test_sqls = load_test_sql(dataset_name)
        print(f"   Loaded {len(test_sqls)} queries")
    except Exception as e:
        print(f"❌ Failed to load test queries: {e}")
        return None
    
    # 쿼리 실행
    print(f"\n🚀 Running {len(test_sqls)} queries...")
    start_time = time.time()
    
    for i, sql in enumerate(test_sqls):
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(test_sqls)} queries")
        
        try:
            TimeStatistic.start(algo_name.capitalize())
            scheduler_or_interactor.execute(sql)
            TimeStatistic.end(algo_name.capitalize())
        except Exception as e:
            print(f"⚠️  Query {i} failed: {e}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # 결과 저장
    print(f"\n💾 Saving results...")
    name_2_value = TimeStatistic.get_sum_data()
    
    result_file = save_test_result(algo_name, dataset_name, extra_info={
        "params": params,
        "num_queries": len(test_sqls),
        "wall_time": elapsed
    })
    
    # 시각화
    img_path = get_time_statistic_img_path(algo_name, dataset_name)
    Drawer.draw_bar(name_2_value, img_path, is_rotation=False)
    
    # 모델 메타데이터 업데이트 (AI 알고리즘인 경우)
    if algo_name != "baseline" and hasattr(scheduler_or_interactor, 'pilot_model'):
        from pilotscope.ModelRegistry import ModelRegistry
        
        model = scheduler_or_interactor.pilot_model
        
        # 성능 메트릭 계산
        performance = {
            "total_time": elapsed,
            "average_time": elapsed / len(test_sqls) if test_sqls else 0,
            "num_queries": len(test_sqls)
        }
        
        # 테스트 결과 추가
        model.add_test_result(dataset_name, len(test_sqls), performance)
        
        # 모델 저장 (메타데이터 업데이트)
        model.save_model()
        
        # 레지스트리에 등록
        registry = ModelRegistry()
        registry.register_model(model.metadata)
        
        print(f"✅ Model metadata saved: {model.model_id}")
    
    print(f"\n✅ Test completed in {elapsed:.2f}s")
    print(f"   Result: {result_file}")
    print(f"   Chart:  img/{img_path}.png")
    
    return {
        "algorithm": algo_name,
        "dataset": dataset_name,
        "result_file": str(result_file),
        "elapsed_time": elapsed,
        "params": params
    }


def run_multiple_tests(config: PilotConfig, algorithms: List[str], 
                      datasets: List[str], algo_params: Dict = None) -> List[Dict]:
    """
    여러 알고리즘 + 데이터셋 조합을 순차적으로 테스트
    
    Args:
        config: PilotConfig 인스턴스
        algorithms: 테스트할 알고리즘 리스트
        datasets: 테스트할 데이터셋 리스트
        algo_params: 알고리즘별 파라미터 (algo_name을 키로 하는 dict)
    
    Returns:
        List[Dict]: 각 테스트의 결과 정보
    """
    results = []
    
    for dataset in datasets:
        config.db = dataset
        
        for algo in algorithms:
            # 알고리즘별 파라미터 가져오기
            params = algo_params.get(algo, {}) if algo_params else {}
            
            try:
                result = run_single_test(config, algo, dataset, params)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"\n❌ Test failed for {algo} on {dataset}: {e}")
            finally:
                # 리소스 정리
                try:
                    pilotscope_exit()
                except:
                    pass
    
    return results


# ============================================================================
# Comparison & Report
# ============================================================================

def compare_results(results: List[Dict], output_dir: str = "results"):
    """
    여러 테스트 결과를 비교
    
    Args:
        results: run_multiple_tests()의 반환값
        output_dir: 비교 차트 저장 디렉토리
    """
    if len(results) < 2:
        print("\n⚠️  Need at least 2 results to compare")
        return
    
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    
    # 결과 파일 리스트 추출
    result_files = [r["result_file"] for r in results]
    
    # 비교 실행
    output_path = f"{output_dir}/comparison_all"
    compare_algorithms(result_files, metric='total_time', output_path=output_path)
    
    # 요약 출력
    print("\n📊 Test Results:")
    print("-" * 60)
    for result in sorted(results, key=lambda x: x["elapsed_time"]):
        print(f"  {result['algorithm']:10s} on {result['dataset']:15s}: {result['elapsed_time']:8.2f}s")
    print("-" * 60)


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
        dataset = exp.get("dataset")
        params = exp.get("params", {})
        
        print(f"\n🧪 Running experiment: {exp_name}")
        config.db = dataset
        
        result = run_single_test(config, algo, dataset, params)
        if result:
            result["experiment_name"] = exp_name
            results.append(result)
        
        pilotscope_exit()
    
    # 비교
    comparison_config = config_data.get("comparison", {})
    if comparison_config.get("enabled", True):
        compare_results(results, 
                       output_dir=comparison_config.get("output_dir", "results"))
    
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
  # Test multiple algorithms on multiple datasets
  python unified_test.py --algo mscn lero baseline --db stats_tiny production --compare
  
  # Test with custom parameters
  python unified_test.py --algo mscn --db production --epochs 50 --training-size 500
  
  # Use JSON config file
  python unified_test.py --config test_configs/production_experiment.json
        """
    )
    
    # Mode selection
    parser.add_argument('--config', help='JSON config file path')
    
    # Algorithm & Dataset selection
    parser.add_argument('--algo', nargs='+', 
                       choices=list(ALGORITHM_REGISTRY.keys()),
                       help='Algorithms to test')
    parser.add_argument('--db', nargs='+', 
                       help='Datasets to test (e.g., stats_tiny, imdb, production)')
    
    # Algorithm parameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--training-size', type=int, 
                       help='Number of queries to use for training (-1 for all)')
    parser.add_argument('--collection-size', type=int,
                       help='Number of queries to collect for training (-1 for all)')
    parser.add_argument('--no-collection', action='store_true',
                       help='Disable data collection (use existing data)')
    parser.add_argument('--no-training', action='store_true',
                       help='Disable model training (use existing model)')
    
    # Output options
    parser.add_argument('--compare', action='store_true',
                       help='Compare results after all tests')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for results (default: results)')
    
    # DB Config
    parser.add_argument('--db-host', help='Database host')
    parser.add_argument('--db-port', help='Database port')
    parser.add_argument('--db-user', help='Database user')
    parser.add_argument('--db-pwd', help='Database password')
    
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
    
    # 알고리즘 파라미터 설정
    algo_params = {}
    for algo in args.algo:
        params = {}
        
        if args.epochs is not None:
            params['num_epoch'] = args.epochs
        if args.training_size is not None:
            params['num_training'] = args.training_size
        if args.collection_size is not None:
            params['num_collection'] = args.collection_size
        if args.no_collection:
            params['enable_collection'] = False
        if args.no_training:
            params['enable_training'] = False
        
        algo_params[algo] = params
    
    # 테스트 실행
    try:
        results = run_multiple_tests(config, args.algo, args.db, algo_params)
        
        # 비교
        if args.compare and len(results) > 1:
            compare_results(results, args.output_dir)
        
        print("\n" + "=" * 60)
        print("✨ All tests completed!")
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

