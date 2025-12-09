from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Optional
import json
import sys
from pathlib import Path
from pydantic import Field

# Ensure project root is on sys.path so we can import simulation_engine when running from backend/fastapi
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from simulation_engine.api.simulation_api import router as simulation_router

# Prefer SQLAlchemy-backed storage when available; fall back to simple sqlite3 store.
try:
    from backend.fastapi.security import storage_sqlalchemy as enc_storage  # type: ignore
except Exception:
    from backend.fastapi.security import storage as enc_storage  # type: ignore

app = FastAPI(title="LifeTwin OS MLP Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount simulation engine routes under /simulate
app.include_router(simulation_router)


@app.get("/health")
async def health():
    return {"status": "ok"}

# In-memory storage for the MLP stage
USERS: Dict[str, str] = {"demo": "demo-token"}
SUMMARIES: Dict[str, List[Dict]] = {}


class LoginRequest(BaseModel):
    device_id: str


class LoginResponse(BaseModel):
    token: str


class DailySummary(BaseModel):
    timestamp: datetime
    total_screen_time: int
    top_apps: List[str]
    most_common_hour: int
    predicted_next_app: Optional[str] = None
    notification_count: int


class UploadSummaryRequest(BaseModel):
    token: str
    device_id: str
    summary: DailySummary


class GetSummariesRequest(BaseModel):
    token: str
    device_id: str


class EncryptedSummaryUpload(BaseModel):
    token: str
    device_id: str
    algorithm: str = Field(..., example="AES-GCM")
    wrapped_dek_hex: str
    ciphertext_hex: str
    metadata: Optional[Dict] = None


@app.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    # For MLP: issue a static token per device
    token = USERS.get("demo")
    return LoginResponse(token=token)


def verify_token(token: str) -> None:
    if token not in USERS.values():
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/sync/upload-summary")
async def upload_summary(req: UploadSummaryRequest):
    verify_token(req.token)
    device_summaries = SUMMARIES.setdefault(req.device_id, [])
    device_summaries.append(req.summary.dict())
    return {"status": "ok"}


@app.post("/sync/get-summary")
async def get_summary(req: GetSummariesRequest):
    verify_token(req.token)
    return {"summaries": SUMMARIES.get(req.device_id, [])}



@app.post("/admin/export-summaries")
async def export_summaries(req: GetSummariesRequest):
    """Export stored summaries to ml/data/summaries_export.json under project root.

    This is a lightweight helper for the ML pipeline to pick up demo data.
    """
    verify_token(req.token)
    device_id = req.device_id
    data = SUMMARIES.get(device_id, [])

    # Ensure datetime objects are converted to ISO strings for JSON serialization.
    def _serialize(obj):
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        try:
            # datetime instances
            import datetime as _dt

            if isinstance(obj, _dt.datetime):
                return obj.isoformat()
        except Exception:
            pass
        return obj

    serializable = _serialize(data)

    export_dir = PROJECT_ROOT / "ml" / "data"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "summaries_export.json"

    with open(export_path, "w", encoding="utf-8") as f:
        json.dump({"device_id": device_id, "summaries": serializable}, f, indent=2)

    return {"status": "ok", "export_path": str(export_path), "count": len(data)}


@app.post("/admin/store-encrypted-summary")
async def store_encrypted_summary(req: EncryptedSummaryUpload):
    """Store an encrypted summary blob (ciphertext + wrapped DEK) in a SQLite store.

    This endpoint is intended for POC/demo flows where the mobile client encrypts
    the summary and uploads ciphertext (server stores ciphertext only).
    """
    verify_token(req.token)
    # Store and return row id
    rowid = enc_storage.store_encrypted_summary(
        device_id=req.device_id,
        algorithm=req.algorithm,
        wrapped_dek_hex=req.wrapped_dek_hex,
        ciphertext_hex=req.ciphertext_hex,
        metadata=req.metadata,
    )
    return {"status": "ok", "rowid": rowid}


@app.get("/admin/list-encrypted-summaries")
async def list_encrypted_summaries(limit: int = 100):
    """List recent encrypted summaries stored on the server."""
    items = enc_storage.list_encrypted_summaries(limit=limit)
    return {"status": "ok", "items": items}
