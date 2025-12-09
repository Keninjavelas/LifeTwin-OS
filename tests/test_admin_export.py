from fastapi.testclient import TestClient
from backend.fastapi.main import app
from pathlib import Path
import json


def test_admin_export_writes_file(tmp_path, monkeypatch):
    client = TestClient(app)

    # upload a summary for device 'dev1'
    payload = {
        "token": "demo-token",
        "device_id": "dev1",
        "summary": {
            "timestamp": "2020-01-01T00:00:00Z",
            "total_screen_time": 120,
            "top_apps": ["com.example.app"],
            "most_common_hour": 14,
            "notification_count": 3
        }
    }
    r = client.post("/sync/upload-summary", json=payload)
    assert r.status_code == 200

    # call export; backend writes to ml/data/summaries_export.json under repo root
    r2 = client.post("/admin/export-summaries", json={"token": "demo-token", "device_id": "dev1"})
    assert r2.status_code == 200
    data = r2.json()
    export_path = Path(data.get("export_path"))
    assert export_path.exists()

    content = json.loads(export_path.read_text(encoding="utf-8"))
    assert content.get("device_id") == "dev1"
    assert isinstance(content.get("summaries"), list)
    # response returned a count; ensure it matches written summaries length
    assert data.get("count") == len(content.get("summaries"))
