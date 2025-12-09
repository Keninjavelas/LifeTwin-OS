"""
Simple exporter to extract training data from a local Room/SQLite database file.

Usage:
  python export_from_sqlite.py --db /path/to/lifetwin_db --out ml/data

This script reads `app_events` and `daily_summaries` tables and writes JSONL files
`events.jsonl` and `summaries.jsonl` suitable for downstream training pipelines.
"""
import argparse
import sqlite3
import json
import os
from datetime import datetime


def row_to_dict(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def export(db_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Export events
    events_out = os.path.join(out_dir, "events.jsonl")
    with open(events_out, "w", encoding="utf-8") as f:
        for row in cur.execute("SELECT * FROM app_events ORDER BY timestamp ASC"): 
            d = dict(row)
            # normalize timestamp to ISO
            if d.get("timestamp") is not None:
                d["timestamp_iso"] = datetime.utcfromtimestamp(d["timestamp"]/1000.0).isoformat() + "Z"
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Export summaries
    summaries_out = os.path.join(out_dir, "summaries.jsonl")
    with open(summaries_out, "w", encoding="utf-8") as f:
        try:
            for row in cur.execute("SELECT * FROM daily_summaries ORDER BY date ASC"):
                d = dict(row)
                # Convert date (if stored as integer) into ISO if possible
                if isinstance(d.get("date"), int):
                    d["date_iso"] = datetime.utcfromtimestamp(d["date"]/1000.0).date().isoformat()
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        except sqlite3.OperationalError:
            # table might not exist yet
            pass

    conn.close()
    print(f"Exported events -> {events_out}")
    print(f"Exported summaries -> {summaries_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export training data from a Room/SQLite DB file")
    parser.add_argument("--db", required=True, help="Path to the SQLite DB file (lifetwin_db)")
    parser.add_argument("--out", default="ml/data", help="Output directory for exported files")
    args = parser.parse_args()
    export(args.db, args.out)
