#!/usr/bin/env python3
"""Example client for simulation API.

Sends a small what-if request to the backend simulation endpoint and saves JSON output.

Usage:
  PYTHONPATH="$PWD" python ml/tools/run_simulation_client.py --url http://localhost:8000
"""
import argparse
import json
import requests
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--out", default="simulation_result.json")
    args = p.parse_args()

    payload = {
        "base_history": {"dummy": True},
        "bedtime_shift_hours": 1,
        "social_usage_delta_min": 10,
    }
    resp = requests.post(f"{args.url}/simulate/what-if", json=payload)
    resp.raise_for_status()
    data = resp.json()
    Path(args.out).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("Wrote simulation output to", args.out)


if __name__ == "__main__":
    main()
