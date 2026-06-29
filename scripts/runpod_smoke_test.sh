#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${WIFI_ANALYZER_APP_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON:-$(command -v python || command -v python3)}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${TMPDIR:-/tmp}/wifi-analyzer-pycache}"

cd "$APP_DIR"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements.txt

# Compile each importable module before pytest so deployment errors are obvious.
"$PYTHON_BIN" -m py_compile \
  src/api.py \
  src/finetuning.py \
  src/groq_diagnosis.py \
  src/inference_flan_t5.py \
  src/pcap_analysis.py \
  scripts/inference_trt_llm.py \
  scripts/merge_flan_t5_lora.py

"$PYTHON_BIN" -m pytest -q
