#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-$(command -v python || command -v python3)}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${TMPDIR:-/tmp}/wifi-analyzer-pycache}"

# Compile each importable module before pytest so syntax/import errors fail fast.
"$PYTHON_BIN" -m py_compile \
  src/api.py \
  src/finetuning.py \
  src/groq_diagnosis.py \
  src/inference_flan_t5.py \
  src/pcap_analysis.py \
  scripts/inference_trt_llm.py \
  scripts/merge_flan_t5_lora.py

"$PYTHON_BIN" -m pytest -q
