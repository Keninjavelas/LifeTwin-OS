from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Optional

app = FastAPI(title="LifeTwin OS MLP Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
