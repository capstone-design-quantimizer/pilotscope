# PilotScope AI4DB Framework - Copilot Instructions

## Overview
PilotScope is a middleware for deploying AI4DB (Artificial Intelligence for Databases) algorithms into production database systems. It bridges ML research and database deployment through a unified push-pull architecture.

## Core Architecture

### Configuration Pattern
- **PostgreSQLConfig**: Use for PostgreSQL connections
  - Enable deep control with `enable_deep_control_local()` for DB management
  - Set database via `config.db = "database_name"`
- **SparkConfig**: Use for Spark connections  
  - Enable cardinality estimation with `enable_cardinality_estimation()`
  - Connect to PostgreSQL datasource via `use_postgresql_datasource()`

### Push-Pull System (Critical Pattern)
The core interaction model uses **Anchor Handlers**:

#### Push Handlers (`BasePushHandler`)
Inject AI predictions into the database:
- `CardPushHandler`: Override cardinality estimates with ML predictions
- `IndexPushHandler`: Manage index recommendations (workload-level)
- `KnobPushHandler`: Tune database parameters
- `HintPushHandler`: Inject query hints
- **Trigger Levels**: `QUERY` (per-query) vs `WORKLOAD` (session-wide)

#### Pull Handlers (`BasePullHandler`) 
Extract data from database execution:
- `RecordPullHandler`: Collect query results
- `PhysicalPlanPullHandler`: Extract execution plans
- `ExecutionTimePullHandler`: Measure query performance
- **Fetch Methods**: `INNER` (during execution) vs `OUTER` (separate query)

### Data Flow Pattern
```python
# Standard workflow
data_interactor = PilotDataInteractor(config)
data_interactor.push_card(subquery_2_cardinality)  # Inject ML predictions
data_interactor.pull_execution_time()              # Collect metrics
data = data_interactor.execute(sql)                # Execute with AI
```

## Key Implementation Patterns

### Custom Algorithm Integration
Extend anchor handlers for new AI algorithms:
```python
class CustomCardPushHandler(CardPushHandler):
    def acquire_injected_data(self, sql):
        # Return ML model predictions for this SQL
        return self.ml_model.predict(sql)
```

### Scheduler-Based Deployment  
Use `PilotScheduler` for production scenarios:
- Register custom handlers with `register_custom_handlers()`
- Define data collection requirements with `register_required_data()`
- Set up retraining events with `register_events()`

### Database Controller Extensions
Located in `pilotscope/DBController/`:
- `PostgreSQLController`: Handles PostgreSQL-specific operations
- `SparkSQLController`: Manages Spark SQL interactions
- Support for hypothetical indexes, deep control, SSH connections

## Configuration Management

### Database Connection
Always configure via concrete config classes:
```python
config = PostgreSQLConfig(host="localhost", port="5432", 
                         user="postgres", pwd="password")
config.db = "target_database"
```

### Environment Setup
- Python 3.8+ required
- PostgreSQL 13.1, Spark 3.3.2 supported
- Install with `pip install -e .` from project root
- Dependencies include PyTorch, SQLAlchemy, psycopg2-binary

## Testing Patterns

### Algorithm Examples
Located in `algorithm_examples/`:
- Each algorithm has dedicated test files in `test_example_algorithms/`
- Follow pattern: `get_{algorithm}_preset_scheduler()` factory functions
- Use `TimeStatistic` for performance measurement
- Generate visualizations with `Drawer.draw_bar()`

### Configuration 
Use `ExampleConfig.py` for test database paths and result storage

## Common Integration Points

### Multi-Database Support
- Factory pattern: `DBControllerFactory.get_db_controller(config)`
- Database-specific anchor implementations in `pilotscope/Anchor/{PostgreSQL,Spark}/`

### Data Exchange Protocol
- `PilotTransData`: Unified data container across push-pull operations
- JSON-based parameter transmission via SQL comments
- HTTP-based data fetching for complex metrics

### Error Handling
- Database timeouts: `DBStatementTimeoutException`
- Connection issues: `DatabaseCrashException` 
- Configuration errors: `PilotScopeInternalError`

## Development Guidelines

- Extend `BasePushHandler`/`BasePullHandler` for new AI capabilities
- Use factory patterns for database-agnostic components  
- Follow anchor-based architecture for data injection/collection
- Implement proper cleanup in `_roll_back()` methods
- Test with both PostgreSQL and Spark configurations when applicable