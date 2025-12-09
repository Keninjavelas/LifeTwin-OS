from simulation_engine.engine import run_scenario


def test_run_scenario_returns_expected_keys():
    base_history = {"screen_time": 120}
    result = run_scenario(base_history, bedtime_shift_hours=1, social_usage_delta_min=10)
    assert isinstance(result, dict)
    assert "base" in result and "simulated" in result
    assert isinstance(result["base"], dict)
    assert isinstance(result["simulated"], dict)
