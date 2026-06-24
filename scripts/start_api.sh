#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${WIFI_ANALYZER_APP_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

export WIFI_ANALYZER_WORKSPACE="${WIFI_ANALYZER_WORKSPACE:-$APP_DIR}"
export WIFI_ANALYZER_JOB_WORKERS="${WIFI_ANALYZER_JOB_WORKERS:-1}"

cd "$APP_DIR"
exec uvicorn src.api:app --host 0.0.0.0 --port "${PORT:-8000}"
