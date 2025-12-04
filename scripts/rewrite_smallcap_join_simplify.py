#!/usr/bin/env python3
"""
Simplify representative_smallcap_turnover queries to reduce complex JOINs that Anchor deparser breaks.

Original pattern (no CTE, already flattened):
SELECT * FROM (
  SELECT base.*, window_calc AS final_score
  FROM (
    SELECT ... FROM stocks_daily_info sdi
      LEFT JOIN fundamentals_daily fd
        ON fd.ticker = sdi.ticker AND fd.event_date = sdi.event_date
    WHERE sdi.event_date = DATE '...'
      AND sdi.market_cap >= ...
  ) base
) scored ORDER BY ... LIMIT ...;

Rewritten pattern:
SELECT * FROM (
  SELECT sdi_cols, fd_cols, window_calc AS final_score
  FROM stocks_daily_info sdi
  LEFT JOIN (
    SELECT ticker, event_date, per, pbr, eps, bps, dividend_yield
    FROM fundamentals_daily
    WHERE event_date = DATE '...'
  ) fd ON fd.ticker = sdi.ticker AND fd.event_date = sdi.event_date
  WHERE sdi.event_date = DATE '...' AND sdi.market_cap >= ...
) scored ORDER BY ... LIMIT ...;

This keeps semantics (left join on matching ticker/date) but avoids multiple joins/aliases that Anchor mangles.
"""

import re
from pathlib import Path


def rewrite_line(line: str) -> str:
    line = line.strip().rstrip(";")
    # Extract date and market_cap
    m_date = re.search(r"sdi\.event_date\s*=\s*DATE\s*'([^']+)'", line, re.IGNORECASE)
    m_cap = re.search(r"sdi\.market_cap\s*>=\s*([0-9]+)", line, re.IGNORECASE)
    if not m_date or not m_cap:
        return line + ";"
    event_date = m_date.group(1)
    market_cap = m_cap.group(1)

    # Columns to select (keep as-is to preserve schema)
    sdi_cols = "sdi.ticker, sdi.company_name, sdi.market, sdi.market_cap, sdi.open_price, sdi.low_price, sdi.close_price, sdi.pct_change, sdi.rsi_14, sdi.ma_20d, sdi.momentum_3m, sdi.momentum_12m, sdi.volatility_20d, sdi.event_date"
    fd_cols = "fd.per, fd.pbr, fd.eps, fd.bps, fd.dividend_yield"

    rewritten = f"""
SELECT * FROM (
  SELECT
    {sdi_cols},
    {fd_cols},
    (
      (sdi.momentum_3m - AVG(sdi.momentum_3m) OVER (PARTITION BY sdi.event_date)) / NULLIF(STDDEV_SAMP(sdi.momentum_3m) OVER (PARTITION BY sdi.event_date), 0)
      + -((fd.per - AVG(fd.per) OVER (PARTITION BY sdi.event_date)) / NULLIF(STDDEV_SAMP(fd.per) OVER (PARTITION BY sdi.event_date), 0))
    ) AS final_score
  FROM stocks_daily_info sdi
  LEFT JOIN (
    SELECT ticker, event_date, per, pbr, eps, bps, dividend_yield
    FROM fundamentals_daily
    WHERE event_date = DATE '{event_date}'
  ) fd
    ON fd.ticker = sdi.ticker AND fd.event_date = sdi.event_date
  WHERE sdi.event_date = DATE '{event_date}'
    AND sdi.market_cap >= {market_cap}
) scored
ORDER BY event_date ASC, final_score DESC
LIMIT 200;
""".strip()
    return rewritten


def process_file(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    out_lines = [rewrite_line(l) for l in lines]
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Rewrote {path.name}")


def main():
    root = Path("pilotscope/Dataset/StockStrategy")
    targets = [
        root / "stock_strategy_representative_smallcap_turnover_train.txt",
        root / "stock_strategy_representative_smallcap_turnover_test.txt",
    ]
    for p in targets:
        if not p.exists():
            print(f"Missing {p}, skip")
            continue
        process_file(p)


if __name__ == "__main__":
    main()
