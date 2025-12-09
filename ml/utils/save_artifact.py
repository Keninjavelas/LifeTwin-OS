import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any):
    try:
        ensure_dir(path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def save_meta(out_dir: Path, name: str, meta: Dict[str, Any]):
    p = out_dir / f"{name}.json"
    return save_json(p, meta)


def save_metrics(out_dir: Path, name: str, metrics: Dict[str, Any]):
    p = out_dir / f"{name}.metrics.json"
    return save_json(p, metrics)


def save_vocab(out_dir: Path, vocab: Dict[str, int], filename: str = 'vocab.json'):
    p = out_dir / filename
    return save_json(p, vocab)
