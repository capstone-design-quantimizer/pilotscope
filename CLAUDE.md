# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PilotScope is a middleware for deploying AI4DB (Artificial Intelligence for Databases) algorithms into actual database systems. It bridges the gap between ML researchers and DB systems, allowing AI models to steer database query optimization through a unified interface.

**Key concept**: PilotScope acts as a "middleware" that intercepts database queries, applies AI-driven optimizations (like cardinality estimation, query plan hints), and returns optimized results - all without modifying the core database engine.

## Development Environment

### Docker-Based Development (Recommended)

This project uses Docker with **volume mounting** for development:

```bash
# Start development environment
docker-compose up -d

# Enter container
docker-compose exec pilotscope-dev bash

# Inside container
conda activate pilotscope
cd test_example_algorithms
python test_mscn_example.py
```

**Important**: Code changes on the host are immediately reflected in the container (no rebuild needed). Only rebuild when:
- Modifying `requirements.txt`
- Changing `Dockerfile.dev`
- Changing PostgreSQL/Spark configuration

### Installation (Non-Docker)

```bash
# Install PilotScope Core
pip install -e .

# Install with development dependencies
pip install -e '.[dev]'
```

**Requirements**: Python 3.8, PostgreSQL 13.1 (for PG algorithms), Spark 3.3.2 (for Spark algorithms)

## Core Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────┐
│  Algorithm Layer (AI4DB Algorithms)        │
│  - MSCN, Lero, KnobTuning, Index Selection │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│  PilotScope Core (Middleware)              │
│  - PilotScheduler: Orchestrates execution  │
│  - PilotDataInteractor: Push/Pull data     │
│  - AnchorHandlers: Intercept at key points │
│  - Events: Trigger training/collection     │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│  Database Layer (PostgreSQL/Spark)         │
│  - Modified with PilotScope "anchors"      │
└─────────────────────────────────────────────┘
```

### Key Components

**PilotScheduler** (`pilotscope/PilotScheduler.py`):
- Central orchestrator for query execution
- Manages AI model lifecycle (training, inference)
- Registers "handlers" (AI algorithms) and "events" (triggers)
- Call sequence: `init()` → `execute(sql)` → triggers handlers → returns results

**PilotDataInteractor** (`pilotscope/DBInteractor/PilotDataInteractor.py`):
- Low-level interface to database
- **Push operators**: Inject data into DB (e.g., hint cards, plan hints)
- **Pull operators**: Collect data from DB (e.g., execution time, cardinality, plans)
- Example: `push_card()` to override optimizer's cardinality estimates

**AnchorHandlers** (`pilotscope/Anchor/`):
- Intercept points in query execution pipeline
- `BasePushHandler`: Modify query behavior (inject hints, override costs)
- `BasePullHandler`: Collect execution metrics
- Custom handlers implement AI algorithms (e.g., `MscnCardPushHandler` predicts cardinalities)

**PresetSchedulers** (`algorithm_examples/*/PresetScheduler.py`):
- Factory functions that wire up AI algorithms with minimal boilerplate
- Example: `get_mscn_preset_scheduler()` configures MSCN for cardinality estimation
- Parameters: `enable_collection`, `enable_training`, `num_epoch`, etc.

**Events** (`pilotscope/PilotEvent.py`):
- `PretrainingModelEvent`: Triggered during `scheduler.init()` to train models
- `PeriodicModelUpdateEvent`: Retrain models periodically
- Implement `iterative_data_collection()` and `custom_model_training()`

### Data Flow Example (MSCN Cardinality Estimation)

```python
# 1. Setup
config = PostgreSQLConfig(db="stats_tiny")
scheduler = get_mscn_preset_scheduler(
    config,
    enable_collection=True,  # Collect training data
    enable_training=True,    # Train model
    num_epoch=100
)

# 2. Init triggers PretrainingModelEvent
scheduler.init()
# → Collects training queries
# → Trains MSCN model
# → Saves model to ExampleData/Mscn/Model/

# 3. Execute query
scheduler.execute("SELECT * FROM users WHERE age > 30")
# → MscnCardPushHandler.predict() estimates cardinalities
# → push_card() injects estimates into PostgreSQL
# → PostgreSQL uses AI-predicted cards for query planning
# → execute() runs optimized query
# → pull_execution_time() collects metrics
```

## Running Tests

### Algorithm Tests

```bash
# Run specific algorithm example
cd test_example_algorithms
python test_mscn_example.py          # MSCN cardinality estimation
python test_lero_example.py          # Lero query optimizer
python test_knob_example.py          # Knob tuning

# Baseline (no AI)
python test_baseline_performance.py
```

### Unified Testing Framework

```bash
# Compare multiple algorithms
python unified_test.py --algo baseline mscn lero --db stats_tiny --compare

# With custom parameters
python unified_test.py --algo mscn --db production \
    --epochs 100 --training-size 500 --collection-size 500

# Load existing model
python unified_test.py --algo mscn --db production \
    --no-training --load-model mscn_20241019_103000
```

### Unit Tests

```bash
# PostgreSQL integration tests
cd test_pilotscope/test_pg
python test_data_interactor.py
python test_scheduler_and_event.py

# Specific functionality
python -m pytest test_pilotscope/test_pg/test_pg_push_card.py -v
```

## Working with Production Data

### Extract Queries from Logs

```bash
# Extract SQL from PostgreSQL logs
python scripts/extract_queries_from_log.py \
    --input /var/log/postgresql/postgresql.log \
    --output pilotscope/Dataset/Production/ \
    --train-ratio 0.8
```

### Create Custom Dataset

1. Create dataset class in `pilotscope/Dataset/`:

```python
from pilotscope.Dataset.BaseDataset import BaseDataset
from pilotscope.PilotEnum import DatabaseEnum

class ProductionDataset(BaseDataset):
    sub_dir = "Production"
    train_sql_file = "production_train.txt"
    test_sql_file = "production_test.txt"
    file_db_type = DatabaseEnum.POSTGRESQL

    def __init__(self, use_db_type, created_db_name="production_db"):
        super().__init__(use_db_type, created_db_name)
        self.download_urls = None
```

2. Register in `algorithm_examples/utils.py`:

```python
from pilotscope.Dataset.ProductionDataset import ProductionDataset

def load_test_sql(db):
    if "production" == db.lower():
        return ProductionDataset(DatabaseEnum.POSTGRESQL).read_test_sql()
    # ... existing code
```

## Model Management

### Model Storage (Timestamp-Based)

Models are saved with timestamps to avoid conflicts:

```
ExampleData/Mscn/Model/
  ├── mscn_20241019_103000          # Model file
  ├── mscn_20241019_103000.json     # Metadata (hyperparams, performance)
  └── ...
```

### Model Registry CLI

```bash
# List all models
python scripts/model_manager.py list --algo mscn

# Find best model
python scripts/model_manager.py best --algo mscn --dataset production

# Compare models
python scripts/model_manager.py compare mscn_20241019_103000 mscn_20241019_110000

# Cleanup old models (keep top 5)
python scripts/model_manager.py cleanup --algo mscn --keep 5

# Tag models
python scripts/model_manager.py tag mscn_20241019_103000 production best
```

### Model Metadata Structure

Metadata stored in `{model_id}.json` tracks:
- **training**: Dataset, hyperparameters, training time
- **testing**: Array of test results on different datasets (allows cross-dataset evaluation)
- **tags**: User-defined labels (e.g., "production", "best")

Key insight: One model can be tested on multiple datasets, with results accumulated in metadata.

## Database Configuration

### PostgreSQL

Configuration in `pilotscope/pilotscope_conf.json`:

```json
{
  "PostgreSQLConfig": {
    "db_host": "localhost",
    "db_port": "5432",
    "db_user": "postgres",
    "db_user_pwd": "postgres"
  }
}
```

In Docker, PostgreSQL runs at:
- **Internal**: `localhost:5432` (inside container)
- **External**: `localhost:54323` (from host)

### Spark

Spark configuration for distributed query optimization (less commonly used).

## Common Development Patterns

### Adding a New AI Algorithm

1. Create algorithm directory in `algorithm_examples/YourAlgorithm/`
2. Implement:
   - `YourAlgorithmPilotModel.py` (extends `EnhancedPilotModel`)
   - `YourAlgorithmHandler.py` (extends `BasePushHandler` or `BasePullHandler`)
   - `EventImplement.py` (training logic)
   - `YourAlgorithmPresetScheduler.py` (factory function)
3. Add to `unified_test.py` ALGORITHM_REGISTRY

### Debugging Query Execution

Enable execution time debugging:

```bash
export DEBUG_EXECUTION_TIME=1
python your_test.py
```

Check `PilotTransData` attributes after execution:
- `execution_time`: Query runtime
- `estimated_cost`: Optimizer's cost estimate
- `subquery_2_card`: Cardinality estimates per subquery
- `physical_plan`: Actual execution plan

### Performance Optimization

For training:
- Use `num_collection` and `num_training` to limit dataset size
- Reduce `num_epoch` for faster iterations
- GPU acceleration for Lero (PyTorch-based)

For testing:
- Disable collection/training: `enable_collection=False, enable_training=False`
- Load pre-trained models: `--load-model <model_id>`

## File Structure Notes

- `pilotscope/`: Core middleware library
- `algorithm_examples/`: AI algorithm implementations (MSCN, Lero, etc.)
- `test_example_algorithms/`: High-level test scripts
- `test_pilotscope/`: Unit tests for core components
- `ExampleData/`: Trained models and training data
- `docs/`: Comprehensive guides (Docker, production optimization, model management)

## Important Constraints

- **PostgreSQL Version**: Must use modified PostgreSQL 13.1 from `pilotscope-postgresql` branch
- **Python 3.8**: Required for compatibility with dependencies (torch, numpy versions)
- **Conda Environment**: Recommended in Docker due to complex dependency tree
- **Anchor Modifications**: Database must be patched with PilotScope "anchors" (interception points)

## Documentation

See `docs/` for detailed guides:
- `DOCKER_GUIDE.md`: Development environment setup
- `PRODUCTION_OPTIMIZATION.md`: Using real production data
- `MODEL_MANAGEMENT.md`: Version control for trained models
