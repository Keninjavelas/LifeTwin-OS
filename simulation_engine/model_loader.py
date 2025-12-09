"""
Small helper to load a time-series twin model for simulation.

This looks for `ml/models/time_series_twin.joblib` under the project root and
returns the loaded joblib object or None.
"""
from pathlib import Path
import joblib


def load_time_series_model():
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / 'ml' / 'models' / 'time_series_twin.joblib'
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None
