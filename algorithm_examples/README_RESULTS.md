# Test Results Management Guide

## Overview

PilotScope now includes utilities to save, load, and compare algorithm performance test results in JSON format with visualization charts.

## Quick Start

### 1. Running Tests (Automatic Saving)

All test files now automatically save results in JSON format:

```bash
# Run baseline test
python test_example_algorithms/test_baseline_performance.py

# Run MSCN test
python test_example_algorithms/test_mscn_example.py

# Run Lero test
python test_example_algorithms/test_lero_example.py
```

**Output:**
- `results/{algo}_{db}_{timestamp}.json` - JSON format with all metrics
- `img/{algo}_{db}.png` - Individual visualization chart

### 2. Comparing Results

#### Option A: Compare Latest Results

```bash
# Compare latest baseline vs MSCN
python algorithm_examples/compare_results.py --latest baseline mscn

# Compare baseline vs MSCN vs Lero
python algorithm_examples/compare_results.py --latest baseline mscn lero
```

#### Option B: Compare Specific Test Runs

```bash
python algorithm_examples/compare_results.py \
    results/baseline_stats_tiny_20231014_120000.json \
    results/mscn_stats_tiny_20231014_130000.json \
    results/lero_stats_tiny_20231014_140000.json
```

#### Option C: List All Saved Results

```bash
python algorithm_examples/compare_results.py --list
```

### 3. Using the Utility Functions in Your Code

```python
from algorithm_examples.utils import (
    save_test_result,
    compare_algorithms,
    list_saved_results,
    load_test_results
)

# Save results after your test
save_test_result('my_algorithm', 'stats_tiny', extra_info={'note': 'test run'})

# Compare multiple results
compare_algorithms([
    'results/baseline_stats_tiny_20231014_120000.json',
    'results/mscn_stats_tiny_20231014_130000.json'
], metric='total_time', output_path='results/my_comparison')

# List all saved results
grouped = list_saved_results()

# Load specific results for custom processing
results = load_test_results([
    'results/baseline_stats_tiny_20231014_120000.json',
    'results/mscn_stats_tiny_20231014_130000.json'
])
```

## File Structure

```
algorithm_examples/
├── compare_results.py          # Comparison tool
├── utils.py                    # Utility functions (save/load/compare)
└── test_*.py                   # Test files (auto-save results)

results/                        # Auto-created directory
├── baseline_stats_tiny_20231014_120000.json
├── mscn_stats_tiny_20231014_130000.json
├── lero_stats_tiny_20231014_140000.json
└── comparison_total_time_20231014_150000.png

img/                           # Individual algorithm charts
├── baseline_stats_tiny.png
├── mscn_stats_tiny.png
└── lero_stats_tiny.png
```

## Available Metrics

- **total_time**: Total execution time for all queries
- **average_time**: Average execution time per query
- **query_count**: Number of queries executed

## API Reference

### `save_test_result(algo_name, db_name, extra_info=None)`

Save current TimeStatistic data in JSON format.

**Parameters:**
- `algo_name` (str): Algorithm name
- `db_name` (str): Database name
- `extra_info` (dict, optional): Additional metadata

**Returns:** Path to saved JSON file

### `compare_algorithms(result_files, metric='total_time', output_path=None)`

Compare multiple test results and generate a comparison chart.

**Parameters:**
- `result_files` (list): List of JSON file paths
- `metric` (str): Metric to compare ('total_time', 'average_time', 'query_count')
- `output_path` (str, optional): Output path for chart (without extension)

**Returns:** Dictionary of comparison data

### `list_saved_results(results_dir='results')`

List all saved test results grouped by algorithm and database.

**Returns:** Dictionary mapping (algorithm, database) to list of result files

### `load_test_results(result_files)`

Load multiple test result JSON files.

**Parameters:**
- `result_files` (list): List of JSON file paths

**Returns:** Dictionary mapping algorithm names to their metrics

## Example Workflow

```bash
# 1. Run baseline test
python test_example_algorithms/test_baseline_performance.py

# 2. Run MSCN test
python test_example_algorithms/test_mscn_example.py

# 3. List all results
python algorithm_examples/compare_results.py --list

# 4. Compare latest results
python algorithm_examples/compare_results.py --latest baseline mscn

# Output:
# 📊 Comparison chart saved: results/comparison_total_time_20231014_150000.png
# 
# ============================================================
# Comparison Results (total_time):
# ============================================================
#   mscn           :    45.2341s
#   baseline       :    67.8912s
# ============================================================
```

## Notes

- All test files automatically save results in JSON format when using `save_test_result()`
- JSON format enables easy programmatic comparison and custom analysis
- Comparison charts are saved as PNG images
- Timestamps ensure unique filenames and enable result history tracking
