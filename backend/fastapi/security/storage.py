"""Simple SQLite-backed storage for encrypted summaries.

This module provides a minimal persistence layer that stores ciphertext blobs and
associated metadata. It's intentionally dependency-free (uses stdlib sqlite3).

Schema (table `encrypted_summaries`):
 - id: integer primary key
 - device_id: text
 - created_at: timestamp (ISO)
 - algorithm: text
 - wrapped_dek: blob (hex)
 - ciphertext: blob (hex)
 - metadata: text (json)
"""

from __future__ import annotations

import sqlite3
import os
from datetime import datetime
import json
from typing import Optional, Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "encrypted_store.sqlite")


def ensure_db_dir():
    dirpath = os.path.dirname(DB_PATH)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS encrypted_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT,
            created_at TEXT,
            algorithm TEXT,
            wrapped_dek TEXT,
            ciphertext TEXT,
            metadata TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def store_encrypted_summary(device_id: str, algorithm: str, wrapped_dek_hex: str, ciphertext_hex: str, metadata: Optional[Dict[str, Any]] = None) -> int:
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat() + "Z"
    meta_json = json.dumps(metadata or {})
    cur.execute(
        "INSERT INTO encrypted_summaries (device_id, created_at, algorithm, wrapped_dek, ciphertext, metadata) VALUES (?, ?, ?, ?, ?, ?)",
        (device_id, now, algorithm, wrapped_dek_hex, ciphertext_hex, meta_json),
    )
    conn.commit()
    rowid = cur.lastrowid
    conn.close()
    return rowid


def list_encrypted_summaries(limit: int = 100) -> List[Dict[str, Any]]:
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, device_id, created_at, algorithm, wrapped_dek, ciphertext, metadata FROM encrypted_summaries ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "device_id": r["device_id"],
            "created_at": r["created_at"],
            "algorithm": r["algorithm"],
            "wrapped_dek": r["wrapped_dek"],
            "ciphertext": r["ciphertext"],
            "metadata": json.loads(r["metadata"] or "{}"),
        })
    return out


if __name__ == "__main__":
    print("Initializing encrypted store at:", DB_PATH)
    init_db()
    print("Done")
