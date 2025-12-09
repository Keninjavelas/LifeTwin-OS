Time-series Twin Training
=========================

This folder contains simple training and evaluation scripts for a time-series "twin" that
predicts next-day `total_screen_time` from the previous N days.

Quick start:

1. Export summaries from the backend to `ml/data/summaries_export.json` using the backend `/admin/export-summaries` endpoint.
2. Train a model:

```bash
python ml/time_series_models/train_twin.py --input ml/data/summaries_export.json --out-dir ml/models --window 4
```

3. Evaluate the saved model:

```bash
python ml/time_series_models/evaluate_twin.py --model ml/models/time_series_twin.joblib --input ml/data/summaries_export.json
```

Notes:
- These scripts are intentionally small and use `scikit-learn` + `joblib` so they work easily in CPU CI.
- For production-quality forecasting, replace the RandomForest with a proper time-series model (ARIMA, Prophet, LSTM, etc.) and add cross-validation.
