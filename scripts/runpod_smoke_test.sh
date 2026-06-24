#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${WIFI_ANALYZER_APP_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

cd "$APP_DIR"

python -m pip install --upgrade pip
pip install -r requirements.txt

python -m py_compile \
  src/api.py \
  src/finetuning.py \
  src/groq_diagnosis.py \
  src/inference_flan_t5.py \
  src/local_llm_diagnosis.py \
  src/pcap_analysis.py

pytest -q
