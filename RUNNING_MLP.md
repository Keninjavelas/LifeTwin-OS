# LifeTwin OS — Running the MLP

This guide explains how to run the current Minimum Lovable Product (MLP) pieces of LifeTwin OS:

1. FastAPI backend (summary sync API)
2. Next.js web dashboard
3. (Optional) Mobile MLP scaffold inside a real React Native project

---

## 1. Run the FastAPI backend

From the repo root, go to the backend folder:

```bash
cd backend/fastapi
```

Install dependencies (optionally inside a virtualenv):

```bash
pip install -r requirements.txt
```

Start the backend server:

```bash
uvicorn main:app --reload --port 8000
```

The backend exposes:

- `POST /auth/login` — returns a simple demo token
- `POST /sync/upload-summary` — accepts a daily summary
- `POST /sync/get-summary` — returns all summaries for a device

---

## 2. Run the web dashboard

In a new terminal, from the repo root:

```bash
cd web-dashboard/app
npm install
npm run dev
```

Then open:

- `http://localhost:3000`

The dashboard will:

- Call `POST /auth/login` on `http://localhost:8000`
- Call `POST /sync/get-summary` for the demo device `demo-device`
- Show today's summary if any, or a "No summaries" message otherwise

> Make sure the FastAPI backend is running before you start the dashboard.

---

## 3. Seed a demo daily summary (optional but recommended)

With the FastAPI backend running, from `backend/fastapi` you can quickly seed a summary:

```bash
cd backend/fastapi
python -c "from fastapi.testclient import TestClient; from main import app; from datetime import datetime; c = TestClient(app); r = c.post('/auth/login', json={'device_id':'demo-device'}); token = r.json()['token']; summary = {'timestamp': datetime.utcnow().isoformat() + 'Z', 'total_screen_time': 60, 'top_apps': ['com.example.mail','com.example.chat'], 'most_common_hour': 20, 'predicted_next_app': 'com.example.chat', 'notification_count': 10}; c.post('/sync/upload-summary', json={'token': token, 'device_id':'demo-device', 'summary': summary})"
```

Reload `http://localhost:3000` and you should see the seeded summary.

---

## 4. Mobile app — how to turn the scaffold into a runnable app

The `mobile/app` directory contains **TypeScript source and configuration**, not a fully initialized React Native
project. To run it on an Android device/emulator:

1. **Create a new React Native project** (once):

   ```bash
   npx react-native init LifeTwinOSMLP --template react-native-template-typescript
   ```

2. **Copy the LifeTwin OS mobile scaffold into that project**:

   - From this repo, copy `mobile/app/src` into the new RN project (e.g., into its own `src` folder or replacing
     the default `App.tsx` and related files).
   - Merge `mobile/app/package.json` dependencies into the RN project’s `package.json` (add `react-navigation`,
     `zustand`, etc. if they are not present).
   - Replace or merge the RN project’s `tsconfig.json` with `mobile/app/tsconfig.json`.

3. **Keep using the mock native collectors at first**:

   - The file `src/services/nativeCollectors.ts` currently returns mock data.
   - This lets you run the UI and see sample daily summaries without Android native modules implemented.

4. **Implement the native modules later** (optional, advanced):

   - Port the Kotlin stubs in `mobile/native-modules/` into the RN project’s `android` codebase and wire them as
     React Native native modules:
     - UsageStats collector
     - Notification listener
     - Screen on/off receiver
     - Accessibility service
     - Sensor fusion
     - Automation manager, etc.

5. **Run the RN app** from the new RN project (not this repo):

   ```bash
   npx react-native start
   npx react-native run-android
   ```

At this stage, the LifeTwin OS repository serves as the **source of truth and scaffold** for the mobile, backend, and
web code. The actual mobile binary comes from a standard RN project that imports this code.

---

## 5. ML Pipeline & Device Integration (Advanced)

### Prerequisites

- Python 3.10+, `pip install -r ml/requirements.txt` (torch optional)
- Java/Android SDK and Android Studio for Android builds
- `adb` on PATH for pushing model files to device/emulator

### Training and Export (Local)

Export summaries from backend or produce an `ml/data/summaries_export.json` file:

```bash
curl -X POST "http://localhost:8000/admin/export-summaries" \
  -H "Content-Type: application/json" \
  -d '{"token":"demo-token","device_id":"demo"}'
```

Run time-series training (saves joblib + metadata + metrics):

```bash
PYTHONPATH="$PWD" python ml/time_series_models/train_twin.py \
  --input ml/data/summaries_export.json \
  --out-dir ml/models \
  --window 4
```

Run sequence model training (requires PyTorch; writes `next_app_model.pt`, `vocab.json`, and metrics):

```bash
PYTHONPATH="$PWD" python -m ml.sequence_models.train_next_app_model \
  --epochs 5 \
  --batch-size 8
```

Smoke-run locally (quick artifact validation):

```bash
PYTHONPATH="$PWD" python ml/tools/run_smoke_training.py
```

### Push Model + Vocab to Emulator/Device

Build and install app debug build in Android Studio (or via `./gradlew :app:installDebug`), then push files using the helper script:

```bash
./mobile/native-modules/inference/push_model_to_device.sh \
  com.your.app \
  ml/models/next_app_model.onnx \
  ml/models/vocab.json
```

If `run-as` is not available the script attempts root copy; prefer emulator/debuggable app.

### Run Simulation Client

```bash
PYTHONPATH="$PWD" python ml/tools/run_simulation_client.py \
  --url http://localhost:8000 \
  --out sim.json
```

### Android ONNX Runtime Integration

To enable ONNX Runtime Mobile, add the recommended AAR/dependency to your app module's `build.gradle`:

```gradle
// Example (replace with latest recommended coord)
implementation "com.microsoft.onnxruntime:onnxruntime-android:1.15.1"
```

The project includes a reflection-based loader in `mobile/native-modules/inference/NativeModelLoader.kt` that safely detects the runtime and attempts to create a session at runtime without requiring the dependency at compile time. After adding the dependency, rebuild the app and verify logs for `ONNX Runtime classes detected` and `ONNX Runtime session created successfully`.

---

## 6. Testing

### Python Tests

From repo root:

```bash
PYTHONPATH="$PWD" pytest -q
```

Current status: 5 passed, 1 skipped.

### Mobile JS Tests (Jest)

From `mobile/app`:

```bash
npm test
```

Smoke tests for `NativeInference`, `Automation`, `Keystore`, and `E2EE` wrappers.

---

## 7. CI/CD & Manual Workflows

- **Python tests**: `.github/workflows/python-tests.yml` — runs pytest on push/PR.
- **Mobile JS tests**: `.github/workflows/mobile-js-tests.yml` — runs Jest on push/PR.
- **Manual Android build**: `.github/workflows/android-build-manual.yml` — workflow_dispatch with optional AAR fetch.

---

## Support

If you need Android Gradle edits to include ONNX/Torch artifacts and implement a full inference pathway, the reflection-backed session code in `NativeModelLoader.kt` is ready; you'll need to run the Android build in your environment or CI.
