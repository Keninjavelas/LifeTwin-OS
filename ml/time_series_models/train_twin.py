"""
Train a simple time-series forecasting model for the twin.

This script expects exported daily summaries in `ml/data/summaries.jsonl` or
`ml/data/summaries_export.json` produced by the backend exporter.

It trains a small scikit-learn regression (RandomForest) to predict next-day
`total_screen_time` from the last N days of `total_screen_time`.

Outputs a joblib model at `ml/models/time_series_twin.joblib`.
"""
import argparse
import os
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib


def load_summaries_from_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                out.append(obj)
            except Exception:
                continue
    return out


def load_exported(path):
    # support the backend's `summaries_export.json` format
    if path.endswith('summaries_export.json') and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
            return payload.get('summaries', [])
    # otherwise try jsonl
    return load_summaries_from_jsonl(path)


def build_dataset(summaries, window=4):
    series = [int(s.get('total_screen_time', 0)) for s in summaries]
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    if not X:
        return None, None
    return np.array(X, dtype=float), np.array(y, dtype=float)


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    # prefer backend export, else ml/data/summaries.jsonl
    export_path = args.input
    summaries = load_exported(export_path)
    if not summaries:
        print('No summaries found at', export_path)
        return 1

    X, y = build_dataset(summaries, window=args.window)
    if X is None:
        print('Not enough data to build dataset')
        return 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f'Trained model MAE={mae:.2f}')

    model_path = Path(args.out_dir) / 'time_series_twin.joblib'
    model_obj = {'model': model, 'window': args.window}
    joblib.dump(model_obj, model_path)

    # save metadata + metrics via helper
    try:
        from ml.utils.save_artifact import save_meta, save_metrics

        meta = {
            'name': 'time_series_twin',
            'window': int(args.window),
            'framework': 'scikit-learn',
            'saved_at': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
        }
        save_meta(Path(args.out_dir), 'time_series_twin', meta)
        metrics = {"mae": float(mae)}
        save_metrics(Path(args.out_dir), 'time_series_twin', metrics)
    except Exception:
        # best-effort; do not fail training for save helpers
        pass

    print('Saved model to', model_path, 'metadata to', meta_path, 'metrics to', metrics_path)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='ml/data/summaries_export.json')
    p.add_argument('--out-dir', default='ml/models')
    p.add_argument('--window', type=int, default=4)
    args = p.parse_args()
    raise SystemExit(train(args))
"""Train a simple time-series model (placeholder) and save it.

Uses RandomForestRegressor on the exported `total_screen_time` series as a demo.
"""
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import joblib
from ml.utils.preprocess import build_time_series_dataset


def main():
    X, y = build_time_series_dataset(window=4)
    if X.size == 0:
        print("No time-series data available. Run backend export to create ml/data/summaries_export.json")
        return
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    model_dir = Path(__file__).resolve().parents[2] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "time_series_twin.joblib"
    # save a consistent dict so loader code can expect {'model', 'window'}
    model_obj = {'model': model, 'window': 4}
    joblib.dump(model_obj, model_path)
    meta_path = model_dir / "time_series_twin.json"
    try:
        import json as _json

        _json.dump({
            'name': 'time_series_twin',
            'window': 4,
            'framework': 'scikit-learn',
            'saved_at': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
        }, open(meta_path, 'w', encoding='utf-8'))
    except Exception:
        pass
    print("Saved time-series model to", model_path, 'and metadata to', meta_path)


if __name__ == "__main__":
    main()
