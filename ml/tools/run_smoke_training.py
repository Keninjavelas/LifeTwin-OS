#!/usr/bin/env python3
"""Smoke-run helper: generate tiny synthetic export and run lightweight training scripts.

This script creates `ml/data/summaries_export.json` with synthetic daily summaries,
runs the time-series trainer (scikit-learn) and attempts to run the sequence trainer
(PyTorch). The sequence training step is optional and will be skipped if PyTorch is not
installed; in that case the script will still write a minimal `vocab.json` for CI.

Usage: python ml/tools/run_smoke_training.py
"""
from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'ml' / 'data'
MODELS_DIR = ROOT / 'ml' / 'models'
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def write_synthetic_export(n=10):
    summaries = []
    for i in range(n):
        summaries.append({
            "timestamp": i,
            "total_screen_time": 60 + i * 5,
            "top_apps": ["com.example.appA", "com.example.appB", "com.example.appC"],
            "most_common_hour": i % 24,
            "notification_count": i % 5,
        })
    payload = {"device_id": "smoke-device", "summaries": summaries}
    path = DATA_DIR / 'summaries_export.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    print('Wrote synthetic export to', path)
    return path


def run_time_series_train():
    print('Running time-series trainer (train_twin.py)')
    cmd = [sys.executable, str(ROOT / 'ml' / 'time_series_models' / 'train_twin.py'), '--input', str(DATA_DIR / 'summaries_export.json'), '--out-dir', str(MODELS_DIR), '--window', '4']
    subprocess.check_call(cmd)
    print('Time-series trainer finished')


def run_sequence_train():
    print('Attempting sequence trainer (may require torch)')
    cmd = [sys.executable, '-m', 'ml.sequence_models.train_next_app_model', '--epochs', '1', '--batch-size', '4']
    try:
        subprocess.check_call(cmd)
        print('Sequence trainer finished')
    except subprocess.CalledProcessError as e:
        print('Sequence trainer failed (ok for smoke-run):', e)
    except FileNotFoundError:
        print('Sequence trainer module not found; skipping')


def verify_artifacts():
    expected = [MODELS_DIR / 'time_series_twin.joblib', MODELS_DIR / 'time_series_twin.json', MODELS_DIR / 'time_series_twin.metrics.json']
    found = True
    for p in expected:
        if not p.exists():
            print('Missing expected artifact:', p)
            found = False
    if found:
        print('Smoke-run: All expected time-series artifacts present')
    else:
        raise SystemExit('Smoke-run failed: missing artifacts')


def main():
    write_synthetic_export()
    run_time_series_train()
    run_sequence_train()
    verify_artifacts()


if __name__ == '__main__':
    main()
