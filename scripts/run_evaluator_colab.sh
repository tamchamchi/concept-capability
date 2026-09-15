#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/content/concept-capability-data}"
ARCHIVE_PATH="${2:-}"

python -m pip install -q numpy pillow pandas pyarrow matplotlib

ARGS=(
  --data-root "$DATA_ROOT"
  --device cuda
)

if [[ -n "$ARCHIVE_PATH" ]]; then
  ARGS+=(--archive "$ARCHIVE_PATH")
fi

python scripts/run_evaluator_pipeline.py "${ARGS[@]}"
