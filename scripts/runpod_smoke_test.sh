#!/usr/bin/env bash
set -euo pipefail

cd "${WIFI_ANALYZER_WORKSPACE:-/workspace/wifi-log-analyzer}"

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
