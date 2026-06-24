# RunPod Deployment

This project is easiest to run on a RunPod GPU Pod with the FastAPI server.
Use Serverless later only if you convert the long-running endpoints into a
RunPod serverless handler.

## Recommended Pod

- Use a PyTorch/Jupyter template for quick setup.
- Pick at least a T4/A10 for FLAN-T5 inference and testing.
- Pick A10/A40/A100 if you want to run larger local LLM diagnosis models.
- Use a persistent volume mounted at `/workspace` so outputs and model files
  survive Pod restarts.

## Upload Or Clone The Project

Open a terminal in the Pod and place the project under `/workspace`:

```bash
cd /workspace
git clone <your_repo_url> wifi-log-analyzer
cd wifi-log-analyzer
```

If you are not using Git, upload the project folder through JupyterLab or SCP.

## Optional: Use The CI/CD Docker Image

If you push this project to GitHub, the `Build RunPod Image` workflow publishes
a Docker image to GitHub Container Registry:

```text
ghcr.io/<owner>/<repo>:latest
```

Use that image in a RunPod custom template and expose port `8000`.

The image does not include `data/`, `models/`, or generated outputs. Mount a
persistent volume at `/workspace` and upload/download those files there.

In the Docker image, project code lives in:

```text
/app
```

Runtime data lives in:

```text
/workspace/wifi-log-analyzer
```

This prevents the `/workspace` network volume from hiding the application code.

See [CI_CD.md](CI_CD.md) for the full CI/CD flow.

## Runtime Secrets

Set Groq credentials on the RunPod Pod/template, not inside the Docker image.
The API reads the key from this environment variable:

```text
GROQ_API_KEY
```

GitHub Actions secrets do not automatically appear inside RunPod. If you add
`GROQ_API_KEY` to GitHub Actions, that only makes it available to GitHub
workflows. You still need to add `GROQ_API_KEY` in RunPod as a Pod/template
environment variable or RunPod secret.

## Install Dependencies

```bash
cd /workspace/wifi-log-analyzer
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check GPU visibility:

```bash
nvidia-smi
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## Run Smoke Tests

After installing dependencies, run the test suite before starting the API:

```bash
cd /workspace/wifi-log-analyzer
pytest -q
```

These tests avoid real Groq API calls and avoid loading the FLAN-T5 model, so
they are safe to run as deployment checks.

## Start The FastAPI Server

Expose HTTP port `8000` in your RunPod Pod/template, then start:

```bash
cd /workspace/wifi-log-analyzer
export WIFI_ANALYZER_WORKSPACE=/workspace/wifi-log-analyzer
export WIFI_ANALYZER_JOB_WORKERS=1
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

If you are using the custom Docker image, the API starts automatically from
`/app`. You should not need to run `uvicorn` manually. If you open a shell in
that container, run code checks from `/app`, while keeping inputs, models, and
outputs under `/workspace/wifi-log-analyzer`.

Open the RunPod HTTP service URL and visit:

```text
/docs
```

For example:

```text
https://<your-runpod-proxy-url>/docs
```

Health check:

```bash
curl https://<your-runpod-proxy-url>/health
```

## FastAPI Endpoints

- `POST /inference/flan-t5`: classify log lines and write `output.jsonl`.
- `POST /pcap/analyze`: correlate error logs with PCAP teardown packets.
- `POST /diagnosis/groq`: run Groq diagnosis.
- `POST /diagnosis/local-llm`: run local open-source LLM diagnosis.
- `POST /pipeline/groq`: run FLAN-T5 inference, PCAP analysis, and Groq diagnosis.
- `POST /jobs/...`: run the same long-running operations in the background.
- `GET /jobs/{job_id}`: poll a background job.
- `GET /files?path=...`: download an output file from the workspace.

Example Groq pipeline request:

```bash
curl -X POST https://<your-runpod-proxy-url>/pipeline/groq \
  -H "Content-Type: application/json" \
  -d '{
    "logfile": "data/samples/wifi_events_3600.txt",
    "pcap": "data/samples/wifi_events_3600.pcap",
    "model_dir": "models/flan-t5-log-lora-model",
    "output_dir": "outputs/run-001",
    "groq": {
      "model": "llama-3.1-8b-instant",
      "max_tokens": 600,
      "max_record_chars": 1500,
      "max_error_logs": 4,
      "max_teardown_events": 2
    }
  }'
```

Download result:

```bash
curl -L "https://<your-runpod-proxy-url>/files?path=outputs/run-001/groq_diagnosis.jsonl" \
  -o groq_diagnosis.jsonl
```

## Background Jobs

For RunPod, prefer job endpoints for long-running work so the HTTP connection
does not need to stay open.

Submit the full Groq pipeline as a job:

```bash
curl -X POST https://<your-runpod-proxy-url>/jobs/pipeline/groq \
  -H "Content-Type: application/json" \
  -d '{
    "logfile": "data/samples/wifi_events_3600.txt",
    "pcap": "data/samples/wifi_events_3600.pcap",
    "model_dir": "models/flan-t5-log-lora-model",
    "output_dir": "outputs/run-001",
    "inference": {
      "device": "cuda",
      "dtype": "fp16",
      "batch_size": 16
    },
    "groq": {
      "model": "llama-3.1-8b-instant",
      "max_tokens": 600
    }
  }'
```

Poll the job:

```bash
curl https://<your-runpod-proxy-url>/jobs/<job_id>
```

Set `WIFI_ANALYZER_JOB_WORKERS=1` on a single GPU if you want to avoid running
multiple model jobs at the same time.

## Run FLAN-T5 Log Inference Through The API

```bash
curl -X POST https://<your-runpod-proxy-url>/inference/flan-t5 \
  -H "Content-Type: application/json" \
  -d '{
    "logfile": "data/samples/wifi_events_3600.txt",
    "model_dir": "models/flan-t5-log-lora-model",
    "output": "output.jsonl",
    "device": "cuda",
    "dtype": "fp16",
    "batch_size": 16,
    "max_source_length": 128,
    "max_new_tokens": 2
  }'
```

If CUDA memory is low, reduce `batch_size` to `8`, then `4`, then `1`.

## Run PCAP Correlation Through The API

```bash
curl -X POST https://<your-runpod-proxy-url>/pcap/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "errors_jsonl": "output.jsonl",
    "pcap": "data/samples/wifi_events_3600.pcap",
    "output": "diagnosis.jsonl"
  }'
```

## Run Groq Diagnosis Through The API

Store `GROQ_API_KEY` as a RunPod secret or environment variable.

```bash
export GROQ_API_KEY="your_groq_api_key"

curl -X POST https://<your-runpod-proxy-url>/diagnosis/groq \
  -H "Content-Type: application/json" \
  -d '{
    "input": "diagnosis.jsonl",
    "output": "groq_diagnosis.jsonl",
    "model": "llama-3.1-8b-instant",
    "max_tokens": 600,
    "max_record_chars": 1500,
    "max_error_logs": 4,
    "max_teardown_events": 2,
    "retries": 5,
    "retry_sleep_seconds": 20
  }'
```

If Groq returns token/rate-limit errors, lower `max_tokens`, reduce
`max_error_logs`/`max_teardown_events`, or set `sleep_seconds`.

## Run Local Open-Source LLM Diagnosis

On a single T4, run one local LLM request at a time because each request loads
the model into memory.

```bash
curl -X POST https://<your-runpod-proxy-url>/diagnosis/local-llm \
  -H "Content-Type: application/json" \
  -d '{
    "input": "diagnosis.jsonl",
    "output": "local_llm_diagnosis.jsonl",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "load_in_4bit": true,
    "max_new_tokens": 512
  }'
```

For larger models, use a larger GPU or keep diagnosis on Groq.

## Keep Outputs

Write important outputs under `/workspace`, for example:

```bash
mkdir -p /workspace/wifi-log-analyzer/outputs
```

RunPod Pod storage outside the mounted workspace/volume may be temporary.

## End-To-End Example

```bash
curl -X POST https://<your-runpod-proxy-url>/pipeline/groq \
  -H "Content-Type: application/json" \
  -d '{
    "logfile": "data/samples/wifi_events_3600.txt",
    "pcap": "data/samples/wifi_events_3600.pcap",
    "model_dir": "models/flan-t5-log-lora-model",
    "output_dir": "outputs/run-001",
    "inference": {
      "device": "cuda",
      "dtype": "fp16",
      "batch_size": 16,
      "max_source_length": 128,
      "max_new_tokens": 2
    },
    "groq": {
      "model": "llama-3.1-8b-instant",
      "max_tokens": 600,
      "max_record_chars": 1500,
      "max_error_logs": 4,
      "max_teardown_events": 2
    }
  }'
```
