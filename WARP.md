# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

PilotScope is an AI4DB (Artificial Intelligence for Databases) middleware framework that bridges AI algorithms with database systems. It provides a unified interface for deploying machine learning algorithms on databases like PostgreSQL and Apache Spark, focusing on query optimization tasks such as:

- Knob tuning (parameter optimization)
- Index recommendation 
- Cardinality estimation
- End-to-end query optimization

The framework acts as a middleware layer that abstracts database-specific details, allowing AI algorithms to work across different database systems without modification.

## Development Commands

### Environment Setup
```powershell
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# For full development features (includes torch, scikit-learn, etc.)
pip install -e ".[dev]"
```

### Running Tests
```powershell
# Run all algorithm examples (requires database setup)
python -m unittest discover test_example_algorithms/ -v

# Run core framework tests
python -m unittest discover test_pilotscope/ -v

# Run individual test examples
python test_example_algorithms/test_knob_example.py
python test_example_algorithms/test_index_example.py
```

### Database Setup Required
Most tests require a PostgreSQL or Spark database to be running. Check `pilotscope/pilotscope_conf.json` for default connection settings:
- PostgreSQL: localhost:5432, user: pilotscope, password: pilotscope, db: stats_tiny
- Spark: local[*] mode

## Architecture Overview

### Core Components

**Database Components** (Yellow in framework diagram):
- **DBController**: Database-specific implementations (`PostgreSQLController`, `SparkSQLController`)
- **DBInteractor**: Handles communication between ML algorithms and databases
- **DataManager**: Manages data collection and storage

**Deployment Components** (Green in framework diagram):  
- **PilotScheduler**: Orchestrates AI algorithm execution and data collection
- **Anchor System**: Push/Pull mechanism for injecting AI predictions and collecting data
- **Factory Pattern**: Creates appropriate components based on database type

### Key Abstractions

**PilotConfig**: Central configuration object
- `PostgreSQLConfig`: For PostgreSQL connections, supports local/remote deep control
- `SparkConfig`: For Spark connections with datasource configuration

**PilotDataInteractor**: Main interface for data exchange
- `push_*()` methods: Inject AI predictions (cardinality, cost estimates, etc.)  
- `pull_*()` methods: Collect database metrics (execution time, plans, buffer cache)
- `execute()`: Run SQL with registered push/pull operations

**PilotScheduler**: High-level orchestration
- `register_custom_handlers()`: Add AI algorithm implementations
- `register_required_data()`: Specify data collection requirements
- `register_events()`: Set up training/update events
- `execute()`: Execute SQL with AI algorithms applied

**Anchor System**: Push/Pull mechanism
- `BasePushHandler`: Inject predictions into database (subclass for custom algorithms)
- `BasePullHandler`: Extract data from database execution
- Database-specific implementations in `Anchor/PostgreSQL/` and `Anchor/Spark/`

### Directory Structure

```
pilotscope/
├── Anchor/           # Push/Pull data exchange mechanism
├── Common/           # Utility functions, metrics, drawing tools  
├── DBController/     # Database-specific control logic
├── DBInteractor/     # Database communication layer
├── DataManager/      # Data collection and storage
├── Dataset/          # Benchmark dataset loaders (IMDB, Stats, TPC-DS)
├── Exception/        # Custom exceptions
├── Factory/          # Factory pattern implementations
└── Core files:       # PilotConfig, PilotScheduler, PilotModel, etc.

algorithm_examples/   # Sample AI4DB algorithm implementations
test_example_algorithms/  # Integration tests with real databases
```

### AI Algorithm Integration Pattern

1. **Inherit from `BasePushHandler`** for algorithms that make predictions
2. **Implement required methods**:
   - `_push_data()`: Inject predictions into database
   - `_update_injected_data()`: Update predictions based on SQL context
3. **Use `PilotScheduler`** to orchestrate execution:
   - Register your custom handler
   - Register required data collection  
   - Execute SQL queries with AI assistance

### Data Flow

1. **Configuration**: Set up database connection via `PilotConfig`
2. **Registration**: Register AI handlers and data requirements via `PilotScheduler` 
3. **Execution**: SQL execution triggers:
   - Push operations inject AI predictions
   - Database executes with AI guidance
   - Pull operations collect execution metrics
   - Results stored via `DataManager`
4. **Events**: Trigger model updates, retraining based on collected data

## Important Implementation Notes

- **Database Compatibility**: Framework supports PostgreSQL 13.1+ and Spark 3.3.2+
- **Configuration Management**: Uses `pilotscope_conf.json` for default settings, override via code
- **Deep Control**: PostgreSQL supports advanced features like config changes, database restart (requires `enable_deep_control_local/remote`)
- **Thread Safety**: Uses APScheduler for background tasks and periodic model updates
- **Data Exchange**: All data flows through `PilotTransData` objects for consistent typing
- **Error Handling**: Custom exceptions in `Exception/Exception.py`, use `pilotscope_exit()` for cleanup

## Database Requirements

- **PostgreSQL**: Requires custom build with PilotScope modifications for full functionality
- **Spark**: Uses standard Spark with custom SQL extensions for AI integration
- **Datasets**: Framework includes loaders for IMDB, Stats, and TPC-DS benchmarks in various sizes (tiny, full)

The framework is designed for research and experimentation with AI4DB algorithms, providing the infrastructure to focus on algorithm development rather than database integration details.