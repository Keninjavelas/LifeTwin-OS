"""Core simulation engine stub for LifeTwin OS.

Supports simple scenarios like changing bedtime or social app usage to see effect on energy curve.
"""

from typing import Dict

from ml.time_series_models.forecast_twin import forecast_next_hours  # type: ignore


def run_scenario(base_history: Dict, bedtime_shift_hours: int = 0, social_usage_delta_min: int = 0) -> Dict:
    """Run a simple what-if scenario.

    This is a pure function stub that adjusts inputs and calls the forecasting model.
    """

    adjusted = dict(base_history)
    adjusted["bedtime_shift_hours"] = bedtime_shift_hours
    adjusted["social_usage_delta_min"] = social_usage_delta_min

    forecast = forecast_next_hours(adjusted)
    return {
        "base": forecast_next_hours(base_history),
        "simulated": forecast,
    }
