#!/usr/bin/env python3
"""
Convert StockStrategy representative_* workload queries from CTE form
to equivalent derived-table form (no WITH) to avoid PostgreSQL Anchor CTE bugs.

Example transformation (single line):
    WITH base AS (<BASE_SQL>), scored AS (<SCORED_SQL>) SELECT * FROM scored ORDER BY ...;
    =>
    SELECT * FROM (<SCORED_SQL with 'FROM base' replaced by '( <BASE_SQL> ) base'>) scored ORDER BY ...;

Usage:
    python scripts/rewrite_representative_no_cte.py \
        --root pilotscope/Dataset/StockStrategy \
        --output-suffix _nocte
"""

import argparse
import re
from pathlib import Path
from typing import Optional


CTE_PATTERN = re.compile(
    r"""
    WITH\s+base\s+AS\s*\(\s*(?P<base>.+?)\s*\)
    \s*,\s*
    scored\s+AS\s*\(\s*(?P<scored>.+?)\s*\)
    \s*SELECT\s+\*\s+FROM\s+scored\s+(?P<tail>.+)
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def rewrite_cte_query(query: str) -> Optional[str]:
    """
    Rewrite a single-line CTE query to a derived-table query.
    Returns rewritten SQL or None if pattern does not match.
    """
    match = CTE_PATTERN.match(query.strip().rstrip(";"))
    if not match:
        return None

    base_sql = match.group("base").strip()
    scored_sql = match.group("scored").strip()
    tail = match.group("tail").strip()

    # Replace the first occurrence of "FROM base" in the scored SQL
    def _replace_from_base(match_obj):
        return f"FROM ({base_sql}) base"

    scored_rewritten = re.sub(
        r"\bFROM\s+base\b", _replace_from_base, scored_sql, count=1, flags=re.IGNORECASE
    )

    rewritten = f"SELECT * FROM ({scored_rewritten}) scored {tail};"
    return rewritten


def process_file(path: Path, output_suffix: str) -> Path:
    """
    Rewrite all lines in a file; lines that do not match the CTE pattern are copied as-is.
    Returns the output path.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    rewritten_lines = []

    for line in lines:
        new_line = rewrite_cte_query(line)
        rewritten_lines.append(new_line if new_line else line)

    output_path = path.with_name(path.stem + output_suffix + path.suffix)
    output_path.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite representative_* StockStrategy queries to non-CTE form."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("pilotscope/Dataset/StockStrategy"),
        help="Directory containing stock_strategy_representative_* files.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_nocte",
        help="Suffix appended to output filenames (before extension).",
    )
    args = parser.parse_args()

    target_files = sorted(args.root.glob("stock_strategy_representative_*_*.txt"))
    if not target_files:
        print(f"No representative_* files found under {args.root}")
        return

    for file_path in target_files:
        out_path = process_file(file_path, args.output_suffix)
        print(f"Rewrote: {file_path.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
