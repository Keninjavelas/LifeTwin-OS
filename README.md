# LifeTwin OS

LifeTwin OS is a privacy‑first, on‑device‑first "personal digital twin" system. It learns from your device behavior,
builds simple predictive models, and presents insights through a mobile app and a web dashboard, with a lightweight
backend for sync.

This repository currently implements:

- A **Minimum Lovable Product (MLP)** mobile app skeleton (React Native + TypeScript).
- A minimal **FastAPI backend** for daily summary sync.
- A minimal **Next.js web dashboard** for viewing summaries.
- **Scaffolding** for future features: richer data collection, ML models (sequence + time‑series), simulation engine,
  rule‑based and RL automation, on‑device LLM summaries, and security/E2EE.

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
