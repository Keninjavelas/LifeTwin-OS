Exporters
=========

This folder contains small utilities to export data from a local Room/SQLite database file into JSONL files suitable for training.

Usage example:

```bash
python ml/exporters/export_from_sqlite.py --db /path/to/lifetwin_db --out ml/data
```

Notes:
- On Android you can pull the app database using `adb` and point this script at the pulled file.
- The script is intentionally dependency-free (uses Python stdlib).
