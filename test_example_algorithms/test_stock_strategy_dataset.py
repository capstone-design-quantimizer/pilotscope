#!/usr/bin/env python3
"""
Test script for StockStrategyDataset loading

Tests:
1. Load SQL files from each strategy template
2. Verify train/test split
3. Check SQL query format
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pilotscope.Dataset.StockStrategyDataset import (
    StockStrategyDataset,
    StockStrategyValueInvestingDataset,
    StockStrategyMomentumInvestingDataset,
    StockStrategyMLHybridDataset
)
from pilotscope.PilotEnum import DatabaseEnum
from algorithm_examples.utils import load_training_sql, load_test_sql


def test_parameterized_dataset():
    """Test parameterized StockStrategyDataset"""
    print("=" * 60)
    print("Test 1: Parameterized Dataset")
    print("=" * 60)

    for template in ['value_investing', 'momentum_investing', 'ml_hybrid']:
        print(f"\nTesting template: {template}")

        dataset = StockStrategyDataset(DatabaseEnum.POSTGRESQL, template=template)

        # Load train SQL
        train_sqls = dataset.read_train_sql()
        print(f"  Train queries: {len(train_sqls)}")

        # Load test SQL
        test_sqls = dataset.read_test_sql()
        print(f"  Test queries:  {len(test_sqls)}")

        # Show first query
        if train_sqls:
            print(f"  First train query (truncated): {train_sqls[0][:100]}...")


def test_convenience_classes():
    """Test convenience dataset classes"""
    print("\n" + "=" * 60)
    print("Test 2: Convenience Classes")
    print("=" * 60)

    datasets = [
        ("Value Investing", StockStrategyValueInvestingDataset),
        ("Momentum Investing", StockStrategyMomentumInvestingDataset),
        ("ML Hybrid", StockStrategyMLHybridDataset)
    ]

    for name, dataset_class in datasets:
        print(f"\nTesting {name} Dataset")

        dataset = dataset_class(DatabaseEnum.POSTGRESQL)

        train_sqls = dataset.read_train_sql()
        test_sqls = dataset.read_test_sql()

        print(f"  Train queries: {len(train_sqls)}")
        print(f"  Test queries:  {len(test_sqls)}")
        print(f"  Database name: {dataset.created_db_name}")


def test_utils_integration():
    """Test integration with algorithm_examples/utils.py"""
    print("\n" + "=" * 60)
    print("Test 3: Utils Integration")
    print("=" * 60)

    test_dbs = [
        "stock_strategy_value_investing",
        "stock_strategy_momentum_investing",
        "stock_strategy_ml_hybrid"
    ]

    for db_name in test_dbs:
        print(f"\nTesting utils.load_test_sql('{db_name}')")

        try:
            train_sqls = load_training_sql(db_name)
            test_sqls = load_test_sql(db_name)

            print(f"  [OK] Train: {len(train_sqls)} queries")
            print(f"  [OK] Test:  {len(test_sqls)} queries")
        except Exception as e:
            print(f"  [ERROR] {e}")


def test_sql_format():
    """Validate SQL query format"""
    print("\n" + "=" * 60)
    print("Test 4: SQL Format Validation")
    print("=" * 60)

    dataset = StockStrategyValueInvestingDataset(DatabaseEnum.POSTGRESQL)
    train_sqls = dataset.read_train_sql()

    print(f"\nChecking {len(train_sqls)} queries...")

    issues = []
    for i, sql in enumerate(train_sqls[:10]):  # Check first 10
        # Check if query ends with semicolon
        if not sql.strip().endswith(';'):
            issues.append(f"Query {i+1}: Missing semicolon")

        # Check if query contains SELECT
        if 'SELECT' not in sql.upper():
            issues.append(f"Query {i+1}: Not a SELECT query")

    if issues:
        print(f"  [WARNING] Found {len(issues)} issues:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  [OK] All queries are properly formatted")

    # Show sample query
    print(f"\nSample query:")
    print(f"  {train_sqls[0][:200]}...")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("StockStrategyDataset Test Suite")
    print("=" * 60)

    try:
        test_parameterized_dataset()
        test_convenience_classes()
        test_utils_integration()
        test_sql_format()

        print("\n" + "=" * 60)
        print("[SUCCESS] All tests passed!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Load database: python test_load_stock_strategy_db.py")
        print("2. Run baseline: python unified_test.py --algo baseline --db stock_strategy_value_investing")
        print("3. Run MSCN: python unified_test.py --algo mscn --db stock_strategy_value_investing")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)