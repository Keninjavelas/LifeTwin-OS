# LifeTwin OS — Task Tracker (current iteration)

This file summarizes what’s done in this iteration and what remains to ship a production-ready release.

## Completed in this iteration (highlights)

- CI: Python + mobile JS workflows; manual Android build workflow (gated, optional AAR fetch).
- Backend: simulation routes mounted; export summaries JSON serialization fixed (datetimes → ISO); tests green.
- ML: artifact writers (model/meta/metrics/vocab), ONNX export helpers, RL/LLM smoke stubs; pytest passing.
- Native (Android): reflective ONNX detection/session store + unload + best-effort run; automation and keystore stubs; debug ONNX asset copy helper.
- RN app: debug screens for Model status, Automation, Keystore; JS wrappers + Jest smoke tests (NativeInference/Automation/Keystore/E2EE).

## 1. Data Collection & Local Intelligence

- Wire up **real Android collectors**:
  - Implement full `UsageStatsCollector` queries and persistence of app usage events.
  - Implement `NotificationLogger` to store notification posts, opens, and dismissals.
  - Implement `ScreenEventReceiver` logging session start/end with durations.
  - Implement `InteractionAccessibilityService` for gesture/touch patterns (respecting Play Store policies).
  - Implement `SensorFusionManager` with actual accelerometer and context features.
- Implement a **local database layer** (e.g., Room + SQLite/SQLCipher) for events and summaries.
- Implement **DailySummaryWorker** logic to aggregate raw events into daily/weekly summaries.

## 2. ML Models (Sequence + Time‑Series)

- Build data export pipeline from local DB to `ml/` training code.
- Implement full training loop in `train_next_app_model.py` with real datasets.
- Train a first **next‑app sequence model** and export it to ONNX/TFLite.
- Integrate the exported model into Kotlin (on‑device inference) and expose predictions to React Native.
- Implement and train a **time‑series twin model** in `forecast_twin.py` (or a companion script) for
  energy/focus/mood prediction.
- Evaluate models with proper metrics and store results.

## 3. Simulation Engine & Dashboard Integration

- Replace the placeholder logic in `simulation-engine/engine.py` with real calls to a trained forecasting model.
- Extend FastAPI backend to mount the simulation router.
- Add dashboard UI for simulation:
  - Controls for bedtime, app usage levels, and other knobs.
  - Side‑by‑side plots of baseline vs simulated energy/focus curves.

## 4. Automation Layer (Rule‑Based and RL)

- Finish rule‑based automation:
  - Compute social/category usage from event data and app category mapping.
  - Persist and surface automation logs to the user.
  - Wire `AutomationManager` methods into Android APIs (DND, notifications, optional app blocking).
  - Add rich UI in the mobile app for automation toggles and logs.
- RL policy:
  - Flesh out observation/action spaces and reward functions in `LifeTwinEnv`.
  - Integrate PPO/DQN library (e.g., stable‑baselines3) into `train_policy.py`.
  - Run experiments, evaluate policies vs rule‑based baselines, and export a compact policy model.
  - Integrate the trained policy into Kotlin for on‑device inference, with a rule‑based safety wrapper.

## 5. LLM‑Based Summaries

- Choose or fine‑tune a **small LLM** (1–3B params) for summarization.
- Quantize the model (e.g., ONNX/MLC‑LLM/GGUF) suitable for on‑device or edge deployment.
- Implement an inference wrapper that replaces the template logic in `ml/summaries.py`.
- Surface daily/weekly natural‑language summaries in the mobile app and web dashboard.

## 6. Security, Privacy & E2EE

- Implement full **end‑to‑end encryption** for summary sync:
  - Key generation and management per user/device.
  - Encrypt summaries client‑side before upload; store only ciphertext on the server.
  - Add integrity protection (MAC/signatures).
- Migrate local storage to **encrypted DB** (SQLCipher or equivalent).
- Add **biometric / passcode locking** for sensitive views in the mobile app.
- Improve permission flows with clear explanations and granular controls.
- Implement data **export & deletion** flows (local + server).

## 7. Production Polish, Testing & Performance

- Implement proper test suites:
  - Mobile: component tests and store logic tests.
  - Backend: unit + integration tests for FastAPI routes.
  - ML: tests for data loaders, model IO, and evaluation.
  - Simulation engine: unit tests for scenario transformations.
- Add loading/skeleton states and error handling to mobile and web UIs.
- Optimize battery and performance:
  - Batch data writes.
  - Schedule heavy work for charge/Wi‑Fi.
  - Profile model inference latency and memory.
- Set up CI (e.g., GitHub Actions) for linting, tests, and basic build steps.

---

This is a living list; as models and features mature, update this file to track progress from
conceptual scaffolding to a production‑grade LifeTwin OS implementation.
