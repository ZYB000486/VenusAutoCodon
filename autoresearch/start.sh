#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SPECIES="${SPECIES:-saccharomyces_cerevisiae}"
MODE="${MODE:-auto}"
DEVICE="${DEVICE:-cuda}"
RM_STAGE_SECONDS="${RM_STAGE_SECONDS:-28800}"
RL_STAGE_SECONDS="${RL_STAGE_SECONDS:-14400}"
RUN_NAME="${RUN_NAME:-scer_autoresearch_8h_4h_$(date +%Y%m%d_%H%M%S)}"

exec "$PYTHON_BIN" -u "$ROOT_DIR/run_pipeline.py" \
  --species "$SPECIES" \
  --mode "$MODE" \
  --run-name "$RUN_NAME" \
  --device "$DEVICE" \
  --rm-stage-seconds "$RM_STAGE_SECONDS" \
  --rl-stage-seconds "$RL_STAGE_SECONDS" \
  "$@"
