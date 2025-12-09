"""Populate backend SUMMARIES with demo data and call the export endpoint.

Run from the repository root with:

python backend/fastapi/scripts/demo_export.py

This script imports the FastAPI module objects directly and invokes the async export function.
"""
import asyncio
from datetime import datetime
from pathlib import Path

from backend.fastapi import main as api_main


async def run():
    # Prepare demo device and token (match main.USERS)
    token = api_main.USERS.get("demo")
    device_id = "demo-device"

    # Create a few demo daily summaries matching DailySummary schema in main.py
    demo_summaries = []
    now = datetime.utcnow()
    for i in range(3):
        demo_summaries.append(
            {
                "timestamp": (now).isoformat(),
                "total_screen_time": 60 + i * 30,
                "top_apps": ["com.example.app{}".format(i % 3)],
                "most_common_hour": 20,
                "predicted_next_app": None,
                "notification_count": 5 + i,
            }
        )

    # Inject into the running module's in-memory store
    api_main.SUMMARIES[device_id] = demo_summaries

    # Construct request model and call export
    req = api_main.GetSummariesRequest(token=token, device_id=device_id)
    result = await api_main.export_summaries(req)
    print("Export result:", result)

    # Show where the file was written
    export_path = Path(result["export_path"])
    print("Export file exists:", export_path.exists())
    if export_path.exists():
        print(export_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    asyncio.run(run())
