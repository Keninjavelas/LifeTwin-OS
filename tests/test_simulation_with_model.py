import os
import json
from pathlib import Path
import joblib

from simulation_engine.engine import run_scenario


def make_simple_dummy_model(out_path: Path, window: int = 3):
    # create a small sklearn model trained on synthetic data so it's picklable
    from sklearn.linear_model import LinearRegression
    import numpy as _np

    # simple dataset: inputs are sliding windows producing predictable next value
    X = []
    y = []
    for i in range(10, 30):
        seq = [float(i + j) for j in range(window)]
        X.append(seq)
        y.append(float(i + window))
    X = _np.array(X)
    y = _np.array(y)
    model = LinearRegression()
    model.fit(X, y)

    model_obj = {'model': model, 'window': window}
    out_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_obj, out_path / 'time_series_twin.joblib')
    with open(out_path / 'time_series_twin.json', 'w', encoding='utf-8') as f:
        json.dump({'name': 'time_series_twin', 'window': window}, f)


def test_simulation_uses_trained_model(tmp_path, monkeypatch):
    # prepare a synthetic model under ml/models
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / 'ml' / 'models'
    # use a temp directory to avoid altering repo state during tests
    tmp_model_dir = tmp_path / 'models'
    make_simple_dummy_model(tmp_model_dir, window=3)

    # monkeypatch the simulation loader to look in tmp_model_dir
    import simulation_engine.model_loader as ml_loader


    def _load_time_series_model():
        return joblib.load(tmp_model_dir / 'time_series_twin.joblib')


    monkeypatch.setattr(ml_loader, 'load_time_series_model', _load_time_series_model)

    # base history with summaries required by forecast_next_hours
    base_history = {'dummy': True}
    result = run_scenario(base_history, bedtime_shift_hours=1, social_usage_delta_min=5)
    assert 'base' in result and 'simulated' in result
    # both outputs should contain predicted hours and numeric arrays
    for key in ('base', 'simulated'):
        out = result[key]
        assert 'hours_ahead' in out and isinstance(out['hours_ahead'], list)
        assert 'energy' in out and isinstance(out['energy'], list)