#!/usr/bin/env python3
"""
Load StockStrategy database from dump file into PostgreSQL

Usage:
    python load_stock_strategy_db.py

Note:
- All 3 templates (value_investing, momentum_investing, ml_hybrid) share the SAME database.
- Only the workload (train/test SQL files) differs per template.
- You only need to load the database ONCE.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pilotscope.PilotConfig import PostgreSQLConfig
from pilotscope.Dataset.StockStrategyDataset import StockStrategyValueInvestingDataset
from pilotscope.PilotEnum import DatabaseEnum


def load_database():
    """
    Load StockStrategy database (shared by all templates)
    """
    print("=" * 60)
    print("StockStrategy Database Loader")
    print("=" * 60)

    # Use any template - they all share the same database
    dataset = StockStrategyValueInvestingDataset(DatabaseEnum.POSTGRESQL)

    db_name = dataset.created_db_name
    print(f"\nDatabase name: {db_name}")
    print(f"Dump file: {dataset.data_file}")
    print("\nAvailable workload templates:")
    print("  - value_investing:    Value metrics (P/E, P/B, dividend)")
    print("  - momentum_investing: Momentum indicators (RSI, MA)")
    print("  - ml_hybrid:          ML predictions + metrics")

    # Create config with deep control enabled
    config = PostgreSQLConfig(db=db_name)
    config.enable_deep_control_local()  # Enable PostgreSQL binary access for pg_dump

    print(f"\n[1/3] Checking database existence...")
    try:
        # Try to connect to see if DB exists
        from pilotscope.Factory.DBControllerFectory import DBControllerFactory
        test_controller = DBControllerFactory.get_db_controller(config)
        test_controller.execute("SELECT 1")
        test_controller._disconnect()

        print(f"    Database '{db_name}' already exists!")

        user_input = input("\n    Overwrite? (y/N): ").strip().lower()
        if user_input != 'y':
            print("\n    Skipping database load.")
            print("=" * 60)
            return

        # Drop existing database
        print(f"    Dropping existing database '{db_name}'...")
        import subprocess
        subprocess.run([
            "psql", "-U", "postgres", "-h", "localhost", "-p", "5432",
            "-c", f"DROP DATABASE IF EXISTS {db_name};"
        ], check=True)
        print("    Database dropped.")

    except Exception:
        print(f"    Database '{db_name}' does not exist. Will create new.")

    print(f"\n[2/3] Loading database from dump file...")
    try:
        dataset.load_to_db(config)
        print("    Database loaded successfully!")
    except Exception as e:
        print(f"    ERROR: Failed to load database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n[3/3] Verifying database...")
    try:
        from pilotscope.Factory.DBControllerFectory import DBControllerFactory
        db_controller = DBControllerFactory.get_db_controller(config)

        # Count tables
        result = db_controller.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """)
        table_count = result[0][0]

        print(f"    Tables loaded: {table_count}")

        # Show table names and row counts
        tables = db_controller.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)

        print("\n    Table details:")
        for (table_name,) in tables:
            try:
                row_count_result = db_controller.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = row_count_result[0][0]
                print(f"      - {table_name:30s}: {row_count:,} rows")
            except Exception as e:
                print(f"      - {table_name:30s}: ERROR ({e})")

        db_controller._disconnect()

    except Exception as e:
        print(f"    WARNING: Verification failed: {e}")

    print("\n" + "=" * 60)
    print("Database loaded successfully!")
    print("=" * 60)
    print("\nNow you can test with different workloads:")
    print(f"  python unified_test.py --algo baseline --db {db_name}")
    print(f"  # Default workload: value_investing\n")
    print(f"  python unified_test.py --algo baseline --db {db_name} --workload momentum_investing")
    print(f"  python unified_test.py --algo baseline --db {db_name} --workload ml_hybrid")
    print(f"  # Custom workloads on the SAME database\n")
    print(f"\nIMPORTANT: All workloads use the same database '{db_name}'.")
    print("Only the query files differ per workload.")
    print("\nDo NOT use: --db stock_strategy_value_investing")
    print("(That would try to connect to a non-existent database!)")
    print("=" * 60)


def main():
    try:
        load_database()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()