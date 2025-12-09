# LifeTwin OS

LifeTwin OS is a privacy‑first, on‑device‑first "personal digital twin". It learns from device behavior,
builds predictive models, and surfaces insights via a mobile app and web dashboard, with a lightweight backend for sync.

## Current state (this iteration)

- React Native mobile app skeleton with debug screens for model status, automation, and keystore.
- FastAPI backend with simulation routes and summary export; JSON export fixed for datetime serialization.
- Python ML pipeline with artifact writers (model/meta/metrics/vocab), ONNX helpers, RL/LLM smoke stubs, and tests.
- Android native stubs for inference (reflective ONNX detection), automation, and keystore; JS wrappers + Jest smoke tests.
- CI workflows for Python + mobile JS; manual Android build workflow (gated) and runbooks/QA checklist.

## Latest improvements

- Added reflective ONNX session handling with session store, unload, and heuristic run path in `NativeModelLoader` and exposed via RN module.
- Added Android Keystore envelope helpers plus JS E2EE wrapper and Jest smoke test.
- Added debug ONNX asset and asset-to-filesDir copy to simplify local Android debug builds.
- Added manual Android build workflow (workflow_dispatch) with optional AAR fetch.
- Strengthened README quick-start and test guidance; cleaned pycache artifacts.

## What remains (high level)

- Ship real on‑device runtime (ONNX/Torch) and wire native inference end‑to‑end.
- Device/emulator validation for automation + E2EE flows; add instrumentation tests.
- Advanced ML (production RL/LLM, quantization) and full release QA/packaging.

---

## High‑Level Architecture

At a high level, LifeTwin OS is composed of:

1. **Mobile app (Android‑first, React Native + Kotlin)**
   - Collects usage events (app opens, notifications, screen on/off) via native modules.
   - Stores events locally and computes daily summaries.
   - Runs a simple next‑app prediction (Markov model today; ML later).
   - Shows insights and basic analytics to the user.

2. **Backend (FastAPI)**
   - Issues a simple token (MLP stage) to the device.
   - Accepts encrypted‑ready daily summaries from the phone.
   - Serves historical summaries to the dashboard.

3. **Web dashboard (Next.js)**
   - Displays today’s summary and a history list of daily summaries.
   - Provides placeholder pages for trends and heatmaps for future analytics.

4. **ML & Simulation (Python, scaffolding)**
   - Sequence model stubs for next‑app prediction.
   - Time‑series model stubs for digital twin forecasting (energy/focus/mood).
   - Simulation engine stubs for "what‑if" scenarios.
   - RL environment stubs for learning automation policies.
   - LLM summarization interface stubs for natural‑language daily summaries.

---

## Repository Structure (Current)

- `mobile/`
  - `app/`
    - React Native + TypeScript app skeleton:
      - Navigation stack with Home, App Usage Summary, Insights, Settings, and Permissions screens.
      - Zustand state store holding events, daily summary, and a Markov next‑app predictor.
      - TS service layer:
        - `nativeCollectors` mock to simulate native data collection and syncing.
        - `automationEngine` for rule‑based automation decisions (skeleton).
        - Utility modules for app categories and basic validation.
  - `native-modules/`
    - Kotlin Android stubs for future deep integration:
      - `usage-stats/UsageStatsCollector.kt` — UsageStatsManager polling for app events.
      - `notification-service/NotificationLogger.kt` — NotificationListenerService stub.
      - `screen-events/ScreenEventReceiver.kt` — screen on/off receiver.
      - `accessibility/InteractionAccessibilityService.kt` — touch/scroll gesture collection stub.
      - `sensors/SensorFusionManager.kt` — sensor fusion stub (accelerometer + others).
      - `summaries/DailySummaryWorker.kt` — WorkManager stub for daily/weekly aggregation.
      - `automation/AutomationManager.kt` — hooks to toggle DND, suggest breaks, and block apps.

- `backend/`
  - `fastapi/`
    - `main.py` — FastAPI app with:
      - `POST /auth/login` — returns a demo token for a device.
      - `POST /sync/upload-summary` — stores in‑memory daily summaries keyed by device.
      - `POST /sync/get-summary` — returns stored summaries for a device.
    - `auth/auth_notes.md` — notes on future auth (JWT, device registration).
    - `security/crypto_notes.md` — notes on future encryption and E2EE summary storage.
    - `sync/` — reserved for future sync‑specific modules.

- `web-dashboard/`
  - `app/`
    - Next.js + React skeleton:
      - `pages/index.tsx` — loads summaries from FastAPI and shows today’s view + history.
      - `pages/timeline.tsx` — placeholder for event timeline.
      - `pages/settings.tsx` — placeholder for device/settings management.
      - `pages/analytics/trends.tsx` — placeholder for trends analytics.
      - `pages/analytics/heatmap.tsx` — placeholder for usage heatmaps and behavior clusters.
    - `components/` — reserved for future charts and shared UI.

- `ml/`
  - `sequence-models/`
    - `train_next_app_model.py` — PyTorch LSTM stub for next‑app prediction + training pipeline skeleton.
    - `evaluate_next_app_model.py` — stub for computing top‑1/top‑3 accuracy.
  - `time-series-models/`
    - `forecast_twin.py` — stub forecasting function for energy/focus/mood curves.
  - `rl-agent/`
    - `env.py` — Gym‑style `LifeTwinEnv` stub for RL automation.
    - `train_policy.py` — stub for PPO/DQN training.
  - `summaries.py` — simple template‑based summary function intended to be replaced by an on‑device LLM call.

- `simulation-engine/`
  - `engine.py` — core `run_scenario(...)` stub combining history with what‑if parameters.
  - `api/simulation_api.py` — FastAPI router exposing `POST /simulate/what-if` for simulations.

- `docs/`
  - `specs/lifetwin-mlp-extension-spec.md` — stub spec for evolving beyond the MLP.
  - `demo/demo_plan.md` — stubbed demo script for showing the system.

- `tests/`
  - `test_placeholders.md` — notes on where and how to add tests for each layer.

---

## Current Capabilities

**MLP‑level (implemented):**

- Mobile app UI skeleton with:
  - Home dashboard showing basic daily summary info and predicted next app (Markov‑based).
  - App Usage Summary screen listing total screen time, sessions, and top apps.
  - Insights screen with simple textual insights (most used app, average session length, peak hour).
  - Settings screen with a sync toggle and project description.
  - Permissions onboarding screen explaining required permissions (logic currently mocked).
- Local logic:
  - Zustand store for events and daily summaries.
  - Simple Markov‑chain next‑app predictor based on observed `app_open` sequences.
- Backend:
  - Minimal FastAPI API for login and daily summary upload/retrieval.
- Web dashboard:
  - Fetches daily summaries from backend and renders today’s view + historical list.

**Beyond MLP (scaffolding):**

- Richer data collection: AccessibilityService, sensor fusion, WorkManager summarization, automation hooks.
- ML: sequence model, time‑series forecasting, RL environment and trainer stubs.
- Simulation engine and API integration.
- LLM summarization interfaces.
- Security/E2EE and auth design notes.
- Docs and test scaffolding for future expansion.

---

## How to Use This Repo

Because this is primarily a **design + scaffolding** repository, some components are not fully wired into runnable,
production projects yet. The intended usage is:

1. **Study the architecture** via this README and the `docs/` folder.
2. **Turn the mobile skeleton into a runnable RN app** by initializing a React Native project and copying the
   `mobile/app` source + config, then implementing the referenced Kotlin native modules.
3. **Run the FastAPI backend** locally as a simple summary sync service.
4. **Run the Next.js dashboard** to visualize summaries from the backend.
5. **Iteratively replace stubs** in `ml/`, `simulation-engine/`, `mobile/native-modules/`, and `backend/fastapi` with
   real models, collectors, security features, and automation logic.

See `completed.md` and `tasks_left.md` for a concise view of what is already in place and what remains to be built.

## Quick start (local)

### Backend (FastAPI)

```bash
cd backend/fastapi
python -m venv .venv
source .venv/Scripts/activate  # Windows Bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### ML utilities (optional)

```bash
cd ml
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
PYTHONPATH="$PWD" pytest -q
```

### Mobile app (JS layer)

- The repo contains the RN app source under `mobile/app/`; initialize a React Native project and copy the source or wire the native modules accordingly.
- Jest smoke tests live in `mobile/app/__tests__/`; run with `npm test` or `yarn test` inside your RN project.

### Android native modules

- Optional ONNX/Torch AARs are not bundled. See `mobile/native-modules/inference/README.md` and `ONNX_GRADLE_HINT.md` for guidance.
- A manual Android CI workflow is provided at `.github/workflows/android-build.yml` (gated; requires AAR URL/inputs).

### Web dashboard (Next.js)

```bash
cd web-dashboard/app
npm install
npm run dev
```

## Tests

- Python: `PYTHONPATH="$PWD" pytest -q` from repo root (already passing in this iteration: 5 passed, 1 skipped).
- JS (mobile app): run Jest inside your RN project; smoke tests for NativeInference, Automation, Keystore, and E2EE wrappers are included.
- Android: no automated device tests are run here; use the manual workflow or Android Studio for instrumentation.

## Documentation and runbooks

- QA checklist: `RUNBOOKS/QA_CHECKLIST.md`
- Retrain/deploy: `RUNBOOKS/RETRAIN_AND_DEPLOY.md`
- Simulation/API notes: `simulation_engine/api/simulation_api.py`
- ML export helpers: `ml/utils/save_artifact.py` and training scripts under `ml/`
