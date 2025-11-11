#!/usr/bin/env python3
"""
Analyze workload files to extract unique strategy templates
"""
import json
import os
from collections import Counter
from pathlib import Path


def analyze_workload_templates(workload_dir):
    """
    Parse all workload batch files and extract strategy_template statistics

    Args:
        workload_dir: Directory containing pilotscope_batch_*.txt files

    Returns:
        Counter: strategy_template frequencies
    """
    templates = []
    batch_files = sorted(Path(workload_dir).glob("pilotscope_batch_*.txt"))

    print(f"Found {len(batch_files)} batch files")

    for batch_file in batch_files:
        print(f"Processing {batch_file.name}...")

        with open(batch_file, 'r', encoding='utf-8') as f:
            json_buffer = ""

            for line in f:
                stripped = line.strip()

                if stripped == '---':
                    # Parse accumulated JSON
                    if json_buffer.strip():
                        try:
                            data = json.loads(json_buffer)
                            if 'metadata' in data and 'strategy_template' in data['metadata']:
                                template = data['metadata']['strategy_template']
                                templates.append(template)
                        except json.JSONDecodeError as e:
                            print(f"  Warning: Failed to parse JSON: {e}")

                    json_buffer = ""
                else:
                    json_buffer += line

            # Parse last document
            if json_buffer.strip():
                try:
                    data = json.loads(json_buffer)
                    if 'metadata' in data and 'strategy_template' in data['metadata']:
                        template = data['metadata']['strategy_template']
                        templates.append(template)
                except json.JSONDecodeError as e:
                    print(f"  Warning: Failed to parse last JSON: {e}")

    return Counter(templates)


if __name__ == "__main__":
    workload_dir = os.path.join(
        os.path.dirname(__file__),
        "..", "pilotscope", "Dataset", "StockStrategy", "workload_02"
    )

    print("=" * 60)
    print("Workload Strategy Template Analyzer")
    print("=" * 60)

    template_counts = analyze_workload_templates(workload_dir)

    print("\n" + "=" * 60)
    print("Strategy Template Statistics")
    print("=" * 60)

    total = sum(template_counts.values())

    for template, count in template_counts.most_common():
        percentage = (count / total) * 100
        print(f"{template:30s}: {count:5d} queries ({percentage:5.2f}%)")

    print("-" * 60)
    print(f"{'Total':30s}: {total:5d} queries")
    print("=" * 60)