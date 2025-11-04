"""
Script to parse stats_pilotscope_input.txt and split into train/test SQL files
for StatsTinyCustomDataset.
"""
import json
import os


def parse_and_split_queries(input_file, output_dir, train_ratio=0.8):
    """
    Parse JSON documents from input file (separated by ---), extract SQL queries,
    and split into train/test sets.

    Args:
        input_file: Path to stats_pilotscope_input.txt
        output_dir: Directory to save train/test files
        train_ratio: Ratio of training data (default 0.8)
    """
    queries = []

    print(f"Reading {input_file}...")

    # Read and parse JSON documents separated by ---
    with open(input_file, 'r', encoding='utf-8') as f:
        json_buffer = ""

        for line in f:
            stripped = line.strip()

            # Skip YAML document separator
            if stripped == '---':
                # Try to parse accumulated JSON
                if json_buffer.strip():
                    try:
                        data = json.loads(json_buffer)
                        if 'sql_query' in data:
                            queries.append(data['sql_query'])
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON: {e}")
                        print(f"Buffer start: {json_buffer[:200]}...")

                    # Progress indicator
                    if len(queries) % 10000 == 0:
                        print(f"Parsed {len(queries)} queries...")

                json_buffer = ""
            else:
                json_buffer += line

        # Parse last document if any
        if json_buffer.strip():
            try:
                data = json.loads(json_buffer)
                if 'sql_query' in data:
                    queries.append(data['sql_query'])
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse last JSON: {e}")

    print(f"Total queries parsed: {len(queries)}")

    # Split into train/test
    split_idx = int(len(queries) * train_ratio)
    train_queries = queries[:split_idx]
    test_queries = queries[split_idx:]

    print(f"Train queries: {len(train_queries)}")
    print(f"Test queries: {len(test_queries)}")

    # Write train file
    train_file = os.path.join(output_dir, "stats_custom_train.txt")
    with open(train_file, 'w', encoding='utf-8') as f:
        for query in train_queries:
            # Remove semicolon if exists and add it back for consistency
            query = query.rstrip(';')
            f.write(query + ';\n')

    print(f"Saved train queries to {train_file}")

    # Write test file
    test_file = os.path.join(output_dir, "stats_custom_test.txt")
    with open(test_file, 'w', encoding='utf-8') as f:
        for query in test_queries:
            query = query.rstrip(';')
            f.write(query + ';\n')

    print(f"Saved test queries to {test_file}")

    return len(train_queries), len(test_queries)


if __name__ == "__main__":
    # File paths
    input_file = os.path.join(
        os.path.dirname(__file__),
        "..", "pilotscope", "Dataset", "StatsTiny", "stats_pilotscope_input.txt"
    )
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "..", "pilotscope", "Dataset", "StatsTiny"
    )

    # Parse and split
    train_count, test_count = parse_and_split_queries(input_file, output_dir, train_ratio=0.8)

    print("\n=== Summary ===")
    print(f"Training queries: {train_count}")
    print(f"Test queries: {test_count}")
    print(f"Total: {train_count + test_count}")
