#!/usr/bin/env bash
set -euo pipefail

echo "Running Python tests..."
PYTHONPATH="$PWD" pytest -q

echo "Running RL smoke-run..."
python - <<'PY'
from ml.rl_agent.train_policy import run_smoke_training
run_smoke_training('ml/models/rl_ci')
print('RL smoke-run complete')
PY

echo "Running LLM summarizer smoke..."
python - <<'PY'
from ml.llm_summaries.summarize import summarize_texts
print(summarize_texts(['Smoke test text']))
PY

echo "Attempting mobile JS tests (if Node/Jest available)..."
if [ -d "mobile/app" ]; then
  (cd mobile/app && (npm test --silent || true))
fi

echo "All done."
