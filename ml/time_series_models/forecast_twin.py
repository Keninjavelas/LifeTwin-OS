"""Stub for digital twin time-series forecasting.

Predicts energy/mood/focus for the next few hours based on past state.
"""

from typing import Dict
from ml.utils.load_export import load_exported_summaries
from pathlib import Path
import joblib


def forecast_next_hours(history: Dict, model_obj: object = None) -> Dict:
    """Placeholder forecasting function.

    history: dictionary containing recent aggregated metrics (screen time, sessions, etc.).
    Returns a dict with predicted curves for the next N hours.
    """

    # If a trained time-series model object is provided, use it.
    model_obj_used = model_obj
    if model_obj_used is None:
        model_path = Path(__file__).resolve().parents[2] / "models" / "time_series_twin.joblib"
        if model_path.exists():
            try:
                model_obj_used = joblib.load(model_path)
            except Exception:
                model_obj_used = None

    if model_obj_used is not None:
        try:
            payload = load_exported_summaries()
            summaries = payload.get("summaries", [])
            if summaries:
                # build a simple feature vector from the last N total_screen_time values
                series = [s.get("total_screen_time", 0) for s in summaries]
                window = int(model_obj_used.get('window', 4)) if isinstance(model_obj_used, dict) else 4
                series = series[-window:]
                if len(series) < window:
                    series = ([0] * (window - len(series))) + series
                # predict next value(s) by iteratively predicting one step ahead
                hours = list(range(1, window+1))
                preds = []
                cur = list(series)
                model = model_obj_used.get('model') if isinstance(model_obj_used, dict) else getattr(model_obj_used, 'model', None) or model_obj_used
                for _ in hours:
                    x = [[float(v) for v in cur[-window:]]]
                    next_val = float(model.predict(x)[0])
                    preds.append(next_val)
                    cur.append(next_val)
                # Normalize predictions into 0..1 energy/focus/mood proxies
                max_val = max(preds) if max(preds) > 0 else 1.0
                energy = [round(max(0.0, 1.0 - (p / (max_val * 2))), 3) for p in preds]
                focus = [round(max(0.0, 1.0 - (p / (max_val * 1.8))), 3) for p in preds]
                mood = [round(max(0.0, 1.0 - (p / (max_val * 2.2))), 3) for p in preds]
                return {"hours_ahead": hours, "energy": energy, "focus": focus, "mood": mood}
        except Exception:
            pass

    # Fallback heuristic when no model
    payload = load_exported_summaries()
    summaries = payload.get("summaries", [])
    if summaries:
        avg_screen = sum(s.get("total_screen_time", 0) for s in summaries) / max(1, len(summaries))
        base = max(0.1, 1.0 - (avg_screen / 500.0))
        hours = [1, 2, 3, 4]
        energy = [round(max(0.0, base - 0.05 * h), 3) for h in hours]
        focus = [round(max(0.0, base - 0.06 * h), 3) for h in hours]
        mood = [round(max(0.0, base - 0.04 * h), 3) for h in hours]
        return {"hours_ahead": hours, "energy": energy, "focus": focus, "mood": mood}

    # Default fallback when no exported data is present
    return {
        "hours_ahead": [1, 2, 3, 4],
        "energy": [0.8, 0.7, 0.6, 0.5],
        "focus": [0.7, 0.65, 0.6, 0.55],
        "mood": [0.75, 0.7, 0.68, 0.65],
    }
