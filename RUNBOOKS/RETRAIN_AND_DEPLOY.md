# Retrain & Deploy Runbook

This runbook documents steps to retrain models, export artifacts, and push models to devices.

1. Export summaries from backend

   - POST `/admin/export-summaries` with `{"token":"demo-token","device_id":"<id>"}`
   - The file will be written to `ml/data/summaries_export.json`.

2. Train time-series twin

   - Run:
     ```bash
     PYTHONPATH="$PWD" python ml/time_series_models/train_twin.py --input ml/data/summaries_export.json --out-dir ml/models
     ```
   - Artifacts written:
     - `ml/models/time_series_twin.joblib`
     - `ml/models/time_series_twin.json` (metadata)
     - `ml/models/time_series_twin.metrics.json`

3. Train next-app sequence model (optional: requires PyTorch)

   - Run:
     ```bash
     PYTHONPATH="$PWD" python ml/sequence_models/train_next_app_model.py --epochs 5 --batch-size 8
     ```
   - Artifacts written:
     - `ml/models/next_app_model.pt` (state_dict)
     - `ml/models/vocab.json`
     - `ml/models/next_app_model.metrics.json`
   - To export ONNX (if PyTorch installed):
     ```bash
     python ml/sequence_models/export_to_onnx.py --ckpt ml/models/next_app_model.pt --out ml/models/next_app_model.onnx
     ```

4. Push model to device (debug)

   - Copy `ml/models/next_app_model.onnx` and `ml/models/vocab.json` into the app's files directory on-device, or use the provided `mobile/native-modules/inference/README_PUSH.md` helper.

5. Verify on device

   - In the app, open Settings -> Open Model Debug and press `Reload Model` then `Get Status` to confirm `modelPresent` and `onnxRuntimeAvailable` flags.

6. CI

   - CI runs smoke training and tests in `.github/workflows/ci-ml.yml` (Python) and `.github/workflows/mobile-js-tests.yml` (mobile JS).
