"""Demo script: ensure exported summaries exist (by calling backend demo), then run sequence model training."""
import asyncio
import sys
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))


async def ensure_export():
    # Import and run the backend demo exporter
    try:
        from backend.fastapi.scripts.demo_export import run as demo_run
    except Exception:
        # Fallback to importing the module differently
        from backend.fastapi.scripts.demo_export import run as demo_run
    # run demo_export
    await demo_run()


def main():
    # Run the async exporter
    asyncio.run(ensure_export())
    # Now run training script
    from ml.sequence_models.train_next_app_model import main as train_main

    train_main()


if __name__ == "__main__":
    main()
