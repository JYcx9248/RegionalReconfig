#!/usr/bin/env python3
"""Load the controller's metrics.jsonl into SQLite with Koala's table layout.

Koala (coordinator/metricCollectorService.go) stores
    metrics(operator_id text, timestamp datetime, metric_type text, metric_value float)
so Koala's plotting scripts (scripts/plot/plotgraphs.py) can read rtier runs; operator_id is
the reporting node ("name:id", or "controller" for reconfiguration events).

    python3 scripts/metrics_to_sqlite.py results/run1/metrics.jsonl results/run1/metricCollector.db
"""

import json
import sqlite3
import sys


def main(src: str, dst: str) -> None:
    db = sqlite3.connect(dst)
    db.execute(
        "CREATE TABLE IF NOT EXISTS metrics "
        "(operator_id text, timestamp datetime, metric_type text, metric_value float)"
    )
    rows = []
    with open(src) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append((r["node"], r["ts"], r["type"], r["value"]))
    db.executemany("INSERT INTO metrics VALUES (?, ?, ?, ?)", rows)
    db.commit()
    db.close()
    print(f"{len(rows)} rows -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: metrics_to_sqlite.py metrics.jsonl out.db")
    main(sys.argv[1], sys.argv[2])
