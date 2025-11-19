# LifeTwin OS Web Dashboard (MLP)

This is the Next.js + TypeScript web dashboard for the LifeTwin OS Minimum Lovable Product.

## Features

- Logs in to the FastAPI backend using `POST /auth/login`.
- Fetches daily summaries via `POST /sync/get-summary` for a demo device (`demo-device`).
- Shows:
  - Today’s total screen time
  - Top apps
  - Notification count
  - Peak usage hour
  - Predicted next app (if provided)
- Provides placeholder pages for:
  - Timeline
  - Settings
  - Analytics (trends, heatmaps)

## Running locally

From the repo root:

```bash
cd web-dashboard/app
npm install
npm run dev
```

Then open `http://localhost:3000` in your browser. Make sure the FastAPI backend is running on
`http://localhost:8000`.
