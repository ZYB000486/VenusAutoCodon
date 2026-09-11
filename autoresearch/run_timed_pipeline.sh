#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TOTAL_HOURS="${TOTAL_HOURS:-16}"

TIMEOUT_BIN="timeout"
if ! command -v "$TIMEOUT_BIN" >/dev/null 2>&1; then
  if command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_BIN="gtimeout"
  else
    echo "Neither timeout nor gtimeout is available on PATH." >&2
    exit 1
  fi
fi

exec "$TIMEOUT_BIN" "${TOTAL_HOURS}h" "$PYTHON_BIN" "$ROOT_DIR/run_pipeline.py" "$@"
