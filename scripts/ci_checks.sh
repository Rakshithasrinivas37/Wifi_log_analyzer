#!/usr/bin/env bash
set -euo pipefail

python -m py_compile \
  src/api.py \
  src/finetuning.py \
  src/groq_diagnosis.py \
  src/inference_flan_t5.py \
  src/local_llm_diagnosis.py \
  src/pcap_analysis.py

pytest -q

