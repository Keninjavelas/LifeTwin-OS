from pathlib import Path
import json
from typing import Dict


def load_exported_summaries() -> Dict:
    export_path = Path(__file__).resolve().parents[2] / "data" / "summaries_export.json"
    if not export_path.exists():
        return {}
    try:
        return json.loads(export_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
