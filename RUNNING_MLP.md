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
- Show today’s summary if any, or a "No summaries" message otherwise

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
