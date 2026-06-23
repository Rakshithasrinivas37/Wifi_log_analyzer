# WiFi Log Analyzer

WiFi Log Analyzer is a Python project for detecting WiFi client failures from
logs, correlating those failures with 802.11 PCAP evidence, and generating
root-cause recommendations with an LLM.

The core pipeline is:

1. Fine-tune FLAN-T5 on labeled WiFi logs.
2. Run FLAN-T5 inference on timestamped log text.
3. Keep only error logs with timestamps and MAC addresses.
4. Correlate error logs with PCAP deauthentication/disassociation packets.
5. Send diagnosis evidence to Groq or another LLM for recommended actions.

## Project Layout

```text
src/
  finetuning.py          FLAN-T5 LoRA fine-tuning service functions
  inference_flan_t5.py   FLAN-T5 inference service functions
  pcap_analysis.py       PCAP/log correlation service functions
  groq_diagnosis.py      Groq LLM diagnosis service functions
  api.py                 FastAPI app for RunPod/API deployment

data/
  datasets/              Training and validation datasets
  samples/               Sample logs and PCAP files

models/
  flan-t5-log-lora-model/  Fine-tuned LoRA adapter artifacts

tests/                   RunPod-safe smoke/unit tests
RUNPOD_DEPLOYMENT.md     RunPod deployment notes
CI_CD.md                 GitHub Actions and Docker image notes
requirements.txt         Python dependencies
```

## Install

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU inference, confirm CUDA is visible:

```bash
nvidia-smi
```

## Run Tests

```bash
pytest -q
```

The tests avoid real Groq API calls and avoid loading the full FLAN-T5 model,
so they are safe to run during RunPod deployment.

## Python Service Usage

The main modules are importable service-style code, not CLI entrypoints.

### Fine-Tuning

```python
from pathlib import Path
from src.finetuning import FineTuningConfig, fine_tune_flan_t5

summary = fine_tune_flan_t5(
    FineTuningConfig(
        train_csv=Path("data/datasets/training.csv"),
        validation_csv=Path("data/datasets/validation.csv"),
        output_dir=Path("models/flan-t5-log-lora-model"),
    )
)
```

### Inference

```python
from pathlib import Path
from src.inference_flan_t5 import InferenceConfig, run_flan_t5_inference

result = run_flan_t5_inference(
    InferenceConfig(
        logfile=Path("data/samples/wifi_events_3600.txt"),
        model_dir=Path("models/flan-t5-log-lora-model"),
        output=Path("output.jsonl"),
        device="cuda",
        dtype="fp16",
        batch_size=16,
    )
)

print(result.elapsed_seconds)
print(result.generation_seconds)
```

`elapsed_seconds` is total wall-clock time including model load, inference, and
output writing. `generation_seconds` is only the log classification loop.

### PCAP Analysis

```python
from pathlib import Path
from src.pcap_analysis import PcapAnalysisConfig, run_pcap_analysis

records = run_pcap_analysis(
    PcapAnalysisConfig(
        errors_jsonl=Path("output.jsonl"),
        pcap=Path("data/samples/wifi_events_3600.pcap"),
        output=Path("diagnosis.jsonl"),
        window_seconds=3.0,
    )
)
```

The output evidence contains the client MAC, correlated error logs, teardown
events, packet type/subtype, reason codes, and reason-code hints.

### Groq Diagnosis

Set your Groq key first:

```bash
export GROQ_API_KEY="your_groq_api_key"
```

Then call the diagnosis service:

```python
from pathlib import Path
from src.groq_diagnosis import GroqDiagnosisConfig, run_groq_diagnosis

rows = run_groq_diagnosis(
    GroqDiagnosisConfig(
        input=Path("diagnosis.jsonl"),
        output=Path("groq_diagnosis.jsonl"),
        model="llama-3.1-8b-instant",
        max_tokens=600,
        max_record_chars=1500,
        max_error_logs=4,
        max_teardown_events=2,
    )
)
```

Groq diagnosis runs sequentially. If you hit token or rate limits, reduce
`max_tokens`, `max_error_logs`, or `max_teardown_events`, or increase
`sleep_seconds`.

## FastAPI / RunPod

Start the API server:

```bash
export WIFI_ANALYZER_WORKSPACE=/workspace/wifi-log-analyzer
export WIFI_ANALYZER_JOB_WORKERS=1
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open:

```text
https://<your-runpod-url>/docs
```

See [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) for the full RunPod setup.
See [CI_CD.md](CI_CD.md) for GitHub Actions and Docker image publishing.

For long-running work, use the background job endpoints:

```text
POST /jobs/inference/flan-t5
POST /jobs/pcap/analyze
POST /jobs/diagnosis/groq
POST /jobs/diagnosis/local-llm
POST /jobs/pipeline/groq
GET  /jobs/{job_id}
```

Set `WIFI_ANALYZER_JOB_WORKERS=1` on a single GPU to avoid running multiple
model jobs at the same time.

## Data Formats

### Training CSV

Expected columns:

```csv
label,input
normal,hostapd: wlan0: STA 3c:22:fb:10:24:38 authenticated
error,hostapd: wlan0: STA 3c:22:fb:10:24:38 WPA: EAPOL-Key timeout
```

Labels must be:

```text
normal
error
```

### Inference Log Text

Each useful log line should start with an ISO timestamp:

```text
2026-06-22T10:30:00.000000Z hostapd: wlan0: STA 3c:22:fb:10:24:38 authenticated
```

Blank lines and lines with length `<= 10` are ignored.

### Inference Output JSONL

Only predicted error rows are written:

```json
{"log":"hostapd: wlan0: STA 3c:22:fb:10:24:38 WPA: EAPOL-Key timeout","mac_addresses":["3c:22:fb:10:24:38"],"prediction":"error","timestamp":"1782095400.100000"}
```

### Diagnosis Evidence JSONL

Created by PCAP analysis:

```json
{"mac":"3c:22:fb:10:24:38","timestamp":"1782095400.500000","error_log_count":1,"error_logs":["EAPOL timeout"],"pcap_session":{"teardown_events":[{"kind":"deauth","packet_type":0,"packet_subtype":12,"reason_code":15}]}}
```

### Groq Diagnosis JSONL

Created by LLM diagnosis:

```json
{"mac":"3c:22:fb:10:24:38","diagnosis":{"root_cause":"4-way handshake timeout","confidence":0.9,"why":["reason code 15"],"recommended_action":{"summary":"Check PSK/security settings."}}}
```

## Notes

- Use a persistent RunPod volume for models, uploaded PCAP files, and outputs.
- Run one local LLM diagnosis request at a time on a single T4 because each
  request loads the model into memory.
- Groq billing is separate from RunPod billing.
- This project is intended for lab/testing workflows; validate findings against
  real AP/controller logs before operational changes.
