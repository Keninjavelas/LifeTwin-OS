import sqlite3
import tempfile
import os
from ml.exporters.export_from_sqlite import export


def create_sample_db(path):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE app_events (id INTEGER PRIMARY KEY, timestamp INTEGER, type TEXT, packageName TEXT)''')
    cur.execute('''CREATE TABLE daily_summaries (id INTEGER PRIMARY KEY, deviceId TEXT, date INTEGER, totalScreenTime INTEGER, topApps TEXT, mostCommonHour INTEGER, notificationCount INTEGER)''')
    # insert a couple of events
    cur.execute("INSERT INTO app_events (timestamp, type, packageName) VALUES (?, ?, ?)", (1609459200000, 'screen', 'com.example.app'))
    cur.execute("INSERT INTO app_events (timestamp, type,packageName) VALUES (?, ?, ?)", (1609462800000, 'screen_off', None))
    conn.commit()
    conn.close()


def test_exporter_creates_files(tmp_path):
    db_file = tmp_path / "test.db"
    create_sample_db(db_file.as_posix())
    out_dir = tmp_path / "out"
    export(db_file.as_posix(), out_dir.as_posix())
    assert (out_dir / "events.jsonl").exists()
    assert (out_dir / "summaries.jsonl").exists()
