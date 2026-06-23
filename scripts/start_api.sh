#!/usr/bin/env bash
set -euo pipefail

export WIFI_ANALYZER_WORKSPACE="${WIFI_ANALYZER_WORKSPACE:-/workspace/wifi-log-analyzer}"
export WIFI_ANALYZER_JOB_WORKERS="${WIFI_ANALYZER_JOB_WORKERS:-1}"

cd "$WIFI_ANALYZER_WORKSPACE"
exec uvicorn src.api:app --host 0.0.0.0 --port "${PORT:-8000}"

