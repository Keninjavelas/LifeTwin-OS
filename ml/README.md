# ML Pipeline

This folder contains helper scripts for training and exporting small demo models using exported summaries.

## Setup

Recommended in a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Demo Pipeline

Populate demo summaries and export:

```bash
python backend/fastapi/scripts/demo_export.py
```

Run sequence trainer:

```bash
python -m ml.sequence_models.train_next_app_model --epochs 5 --batch-size 8
```

Export and evaluate ONNX (optional):

```bash
python ml/sequence_models/export_and_eval.py \
  --model-path ml/models/next_app_model.pt \
  --onnx-path ml/models/next_app_model.onnx \
  --quantize
```

Train time-series twin model:

```bash
python ml/time_series_models/train_twin.py
```

## Smoke Training

To run a small smoke training that validates artifact generation (model, metadata, metrics):

```bash
# from repository root
PYTHONPATH="$PWD" python ml/tools/run_smoke_training.py
```

The script will write artifacts under `ml/models/` including `time_series_twin.joblib`, `time_series_twin.json`, and `time_series_twin.metrics.json`.

## ML Artifact Writers

- `ml/exporters/model_writer.py` — saves trained models (joblib, PyTorch, ONNX).
- `ml/exporters/metadata_writer.py` — writes model metadata JSON (version, timestamp, params).
- `ml/exporters/metrics_writer.py` — saves evaluation metrics (accuracy, loss, etc.).
- `ml/exporters/vocab_writer.py` — exports vocabulary mappings for sequence models.
- `ml/exporters/onnx_helpers.py` — ONNX export utilities (quantization, validation).

## Notes

- The time-series training uses RandomForest as a placeholder and saves `ml/models/time_series_twin.joblib`.
- The sequence trainer uses PyTorch and saves `ml/models/next_app_model.pt`.
- These scripts are intentionally minimal to be a starting point for experimentation.

