from fastapi.testclient import TestClient
from backend.fastapi.main import app


def test_simulation_endpoint_returns_expected_shape():
    client = TestClient(app)
    payload = {
        "base_history": {"some": "data"},
        "bedtime_shift_hours": 1,
        "social_usage_delta_min": 10,
    }
    resp = client.post("/simulate/what-if", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "base" in data and "simulated" in data
    for key in ("base", "simulated"):
        assert isinstance(data[key], dict)
        assert "hours_ahead" in data[key]
        assert "energy" in data[key]