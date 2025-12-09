# LifeTwin OS Backend (FastAPI)

FastAPI backend for the LifeTwin OS MLP, handling device auth, summary sync, exports, and simulation.

## Endpoints

### Authentication

- `POST /auth/login` — accepts `{ "device_id": string }`, returns `{ "token": string }`.

### Sync

- `POST /sync/upload-summary` — accepts `{ token, device_id, summary }` to store a daily summary.
- `POST /sync/get-summary` — accepts `{ token, device_id }`, returns `{ summaries: [...] }`.

### Admin & Export

- `POST /admin/export-summaries` — exports all summaries for a device as JSON (with datetime fields in ISO format).

### Simulation

- `POST /simulate/what-if` — runs simulation scenarios (mounted from `simulation_engine/api/simulation_api.py`).

## Running Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Tests

From repo root:

```bash
PYTHONPATH="$PWD" pytest -q
```

Current status: 5 passed, 1 skipped.

## Notes

- JSON serialization for datetime fields fixed (ISO format).
- Simulation routes mounted and working.
- See `backend/fastapi/security/crypto_notes.md` for E2EE plan.
- See `backend/fastapi/auth/auth_notes.md` for future JWT/device registration.
