# QA Checklist — Final QA & Cleanup

Follow this checklist when preparing a release or validating end-to-end behavior.

- [ ] Run unit & integration tests:
  - `PYTHONPATH="$PWD" pytest -q`
  - `./run_tests.sh` to run full matrix (includes RL/LLM smoke and mobile JS tests when available)
- [ ] Validate model training artifacts exist:
  - `ml/models/time_series_twin.joblib`
  - `ml/models/time_series_twin.json`
  - `ml/models/time_series_twin.metrics.json`
  - `ml/models/next_app_model.pt` and `ml/models/vocab.json` (if trained)
- [ ] Verify backend export endpoint:
  - POST `/admin/export-summaries` and confirm `ml/data/summaries_export.json` is written and readable
- [ ] Mobile smoke checks (device/emulator):
  - Install app on emulator/device
  - Place `next_app_model.onnx` and `vocab.json` in app files dir or push via helper
  - Open Settings -> Open Model Debug -> Reload Model, Get Status
  - Open Settings -> Automation -> Start/Stop and confirm no crashes
- [ ] CI checks:
  - Confirm `.github/workflows/ci-ml.yml` and `mobile-js-tests.yml` run on PRs
- [ ] Documentation & runbooks:
  - Update `RUNBOOKS/RETRAIN_AND_DEPLOY.md` with any steps you changed
  - Ensure README sections reference the new scripts
