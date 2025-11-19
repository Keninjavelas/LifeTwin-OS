from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict

from simulation_engine.engine import run_scenario

router = APIRouter(prefix="/simulate", tags=["simulate"])


class SimulationRequest(BaseModel):
    base_history: Dict
    bedtime_shift_hours: int = 0
    social_usage_delta_min: int = 0


@router.post("/what-if")
async def simulate(req: SimulationRequest):
    return run_scenario(req.base_history, req.bedtime_shift_hours, req.social_usage_delta_min)
