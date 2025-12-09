import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# ensure repo root on sys.path so imports like backend.fastapi.scripts.demo_export work
sys.path.append(str(REPO_ROOT))


async def do_export() -> bool:
    try:
        from backend.fastapi.scripts.demo_export import run as demo_run
    except Exception as e:
        print("Failed to import demo_export:", e)
        return False

    try:
        await demo_run()
        return True
    except Exception as e:
        print("demo_run failed:", e)
        return False


def run_training() -> bool:
    try:
        # Import trainer lazily so missing heavy deps (torch) are handled
        import importlib
        trainer = importlib.import_module("ml.sequence_models.train_next_app_model")
        if hasattr(trainer, "main"):
            trainer.main()
            return True
        else:
            print("Trainer module has no main()")
            return False
    except Exception as e:
        print("Training skipped or failed:", e)
        return False


def main() -> None:
    ok = asyncio.run(do_export())
    if not ok:
        print("Export failed; aborting pipeline.")
        return

    export_path = Path(__file__).resolve().parents[1] / "data" / "summaries_export.json"
    print("Export path:", export_path)
    if not export_path.exists():
        print("Export file not found after export; aborting.")
        return

    print("Export file created. Attempting to run trainer (if available)...")
    train_ok = run_training()
    if train_ok:
        print("Training finished successfully.")
    else:
        print("Training was skipped or failed. See messages above.")


if __name__ == "__main__":
    main()
