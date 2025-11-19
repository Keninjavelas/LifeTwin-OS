# LifeTwin OS MLP Backend (FastAPI)

Minimal backend used by the MLP to login a device and sync daily summaries.

## Endpoints

- `POST /auth/login` — accepts `{ "device_id": string }`, returns `{ "token": string }`.
- `POST /sync/upload-summary` — accepts `{ token, device_id, summary }` to store a daily summary.
- `POST /sync/get-summary` — accepts `{ token, device_id }`, returns `{ summaries: [...] }`.

## Running locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
