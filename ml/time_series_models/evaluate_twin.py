"""
Evaluate a trained time-series twin model.

Loads `ml/models/time_series_twin.joblib` and computes MAE on the provided export.
"""
import argparse
import os
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error


def load_export(path):
    if not os.path.exists(path):
        return []
    if path.endswith('summaries_export.json'):
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
            return payload.get('summaries', [])
    # jsonl fallback
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def build_series(summaries, window):
    series = [int(s.get('total_screen_time', 0)) for s in summaries]
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X, dtype=float), np.array(y, dtype=float)


def evaluate(args):
    model_path = Path(args.model)
    if not model_path.exists():
        print('Model not found:', model_path)
        return 1
    data = load_export(args.input)
    if not data:
        print('No data at', args.input)
        return 1

    obj = joblib.load(model_path)
    model = obj.get('model')
    window = obj.get('window', args.window)

    X, y = build_series(data, window)
    if X.size == 0:
        print('Not enough data')
        return 1

    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    print(f'Evaluation MAE={mae:.2f}')
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='ml/models/time_series_twin.joblib')
    p.add_argument('--input', default='ml/data/summaries_export.json')
    p.add_argument('--window', type=int, default=4)
    args = p.parse_args()
    raise SystemExit(evaluate(args))
