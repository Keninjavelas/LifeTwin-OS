from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
import logging

from simulation_engine.engine import run_scenario, run_comprehensive_scenario

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulate", tags=["simulate"])


class SimulationRequest(BaseModel):
    """Legacy simulation request for backward compatibility."""
    base_history: Dict
    bedtime_shift_hours: int = 0
    social_usage_delta_min: int = 0


class ComprehensiveSimulationRequest(BaseModel):
    """Enhanced simulation request with comprehensive scenario parameters."""
    base_history: Dict = Field(..., description="Historical behavioral data")
    
    # Behavioral modification parameters
    bedtime_shift_hours: Optional[int] = Field(0, description="Change in bedtime (positive = later)", ge=-6, le=6)
    social_usage_delta_min: Optional[int] = Field(0, description="Change in social media usage (minutes)", ge=-300, le=300)
    work_app_delta_min: Optional[int] = Field(0, description="Change in productivity app usage (minutes)", ge=-480, le=480)
    exercise_delta_min: Optional[int] = Field(0, description="Change in fitness/health app usage (minutes)", ge=-120, le=120)
    notification_delta: Optional[int] = Field(0, description="Change in notification frequency", ge=-100, le=100)
    screen_break_frequency: Optional[int] = Field(0, description="Screen breaks per hour", ge=0, le=12)
    sleep_quality_modifier: Optional[float] = Field(0.0, description="Sleep quality adjustment", ge=-1.0, le=1.0)
    
    class Config:
        schema_extra = {
            "example": {
                "base_history": {
                    "total_screen_time": 360,
                    "social_screen_time": 120,
                    "work_screen_time": 240,
                    "notification_count": 45,
                    "energy_level": 0.7,
                    "focus_level": 0.6,
                    "mood_level": 0.75
                },
                "bedtime_shift_hours": -1,
                "social_usage_delta_min": -30,
                "exercise_delta_min": 20,
                "screen_break_frequency": 2
            }
        }


class ScenarioPreset(BaseModel):
    """Predefined scenario preset."""
    name: str
    description: str
    parameters: Dict


class SimulationResponse(BaseModel):
    """Simulation response with baseline and simulated predictions."""
    baseline: Dict
    simulated: Dict
    impact_analysis: Optional[Dict] = None
    scenario_params: Dict
    model_info: Dict


@router.post("/what-if", response_model=Dict)
async def simulate_legacy(req: SimulationRequest):
    """Legacy simulation endpoint for backward compatibility."""
    try:
        result = run_scenario(req.base_history, req.bedtime_shift_hours, req.social_usage_delta_min)
        return result
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@router.post("/comprehensive", response_model=SimulationResponse)
async def simulate_comprehensive(req: ComprehensiveSimulationRequest):
    """Run a comprehensive behavioral scenario with multiple parameters."""
    try:
        scenario_params = {
            'bedtime_shift_hours': req.bedtime_shift_hours,
            'social_usage_delta_min': req.social_usage_delta_min,
            'work_app_delta_min': req.work_app_delta_min,
            'exercise_delta_min': req.exercise_delta_min,
            'notification_delta': req.notification_delta,
            'screen_break_frequency': req.screen_break_frequency,
            'sleep_quality_modifier': req.sleep_quality_modifier
        }
        
        result = run_comprehensive_scenario(req.base_history, scenario_params)
        return SimulationResponse(**result)
        
    except Exception as e:
        logger.error(f"Comprehensive simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@router.get("/presets", response_model=List[ScenarioPreset])
async def get_scenario_presets():
    """Get predefined scenario presets for common use cases."""
    presets = [
        ScenarioPreset(
            name="Digital Detox",
            description="Reduce social media and increase breaks for better focus",
            parameters={
                "social_usage_delta_min": -60,
                "screen_break_frequency": 3,
                "notification_delta": -20
            }
        ),
        ScenarioPreset(
            name="Productivity Boost",
            description="Optimize work habits and sleep schedule",
            parameters={
                "bedtime_shift_hours": -1,
                "work_app_delta_min": 30,
                "social_usage_delta_min": -30,
                "screen_break_frequency": 2
            }
        ),
        ScenarioPreset(
            name="Wellness Focus",
            description="Prioritize exercise and sleep quality",
            parameters={
                "exercise_delta_min": 30,
                "bedtime_shift_hours": -1,
                "sleep_quality_modifier": 0.2,
                "social_usage_delta_min": -45
            }
        ),
        ScenarioPreset(
            name="Mindful Usage",
            description="Balanced approach with regular breaks and reduced notifications",
            parameters={
                "screen_break_frequency": 2,
                "notification_delta": -15,
                "social_usage_delta_min": -20,
                "work_app_delta_min": -15
            }
        ),
        ScenarioPreset(
            name="Early Bird",
            description="Earlier bedtime with morning productivity focus",
            parameters={
                "bedtime_shift_hours": -2,
                "work_app_delta_min": 45,
                "exercise_delta_min": 20,
                "sleep_quality_modifier": 0.3
            }
        )
    ]
    
    return presets


@router.post("/preset/{preset_name}", response_model=SimulationResponse)
async def simulate_preset(preset_name: str, base_history: Dict):
    """Run a simulation using a predefined preset."""
    try:
        # Get preset parameters
        presets = await get_scenario_presets()
        preset = next((p for p in presets if p.name.lower().replace(" ", "_") == preset_name.lower()), None)
        
        if not preset:
            raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
        
        result = run_comprehensive_scenario(base_history, preset.parameters)
        return SimulationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preset simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@router.get("/health")
async def simulation_health():
    """Check simulation engine health and model availability."""
    try:
        from simulation_engine.engine import _simulation_engine
        
        model_available = _simulation_engine.model_obj is not None
        confidence = _simulation_engine._get_prediction_confidence()
        
        return {
            "status": "healthy",
            "model_available": model_available,
            "prediction_confidence": confidence,
            "model_metadata": _simulation_engine.model_metadata.get('model_name', 'unknown') if _simulation_engine.model_metadata else None
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "model_available": False
        }
