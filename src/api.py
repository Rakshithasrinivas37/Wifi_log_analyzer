"""FastAPI service for the WiFi log analyzer pipeline.

The API calls project service functions directly. It does not use subprocesses
or ``python -m`` CLI commands.

Run locally or on RunPod:

```
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import time
from typing import Any, Callable, Literal, TypeVar

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


T = TypeVar("T")
APP_ROOT = Path(os.environ.get("WIFI_ANALYZER_APP_DIR", Path.cwd())).resolve()
WORKSPACE_ROOT = Path(os.environ.get("WIFI_ANALYZER_WORKSPACE", Path.cwd())).resolve()
JOB_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.environ.get("WIFI_ANALYZER_JOB_WORKERS", "2"))
)
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = Lock()

app = FastAPI(
    title="WiFi Log Analyzer API",
    description="Run FLAN-T5 inference, PCAP correlation, and LLM diagnosis.",
    version="1.0.0",
)


class InferenceRequest(BaseModel):
    """Request body for FLAN-T5 log inference."""

    logfile: str
    model_dir: str
    output: str = "output.jsonl"
    base_model: str = "google/flan-t5-small"
    max_source_length: int = 128
    max_new_tokens: int = 2
    batch_size: int = 16
    device: Literal["auto", "cuda", "cpu"] = "auto"
    dtype: Literal["auto", "fp16", "fp32"] = "auto"


class InferenceOptions(BaseModel):
    """Optional FLAN-T5 settings for an end-to-end pipeline request."""

    base_model: str = "google/flan-t5-small"
    max_source_length: int = 128
    max_new_tokens: int = 2
    batch_size: int = 16
    device: Literal["auto", "cuda", "cpu"] = "auto"
    dtype: Literal["auto", "fp16", "fp32"] = "auto"


class FineTuningRequest(BaseModel):
    """Request body for FLAN-T5 LoRA fine-tuning."""

    train_csv: str = "data/datasets/training.csv"
    validation_csv: str | None = "data/datasets/validation.csv"
    model: str = "google/flan-t5-small"
    output_dir: str = "models/flan-t5-log-lora-model"
    text_field: str = "input"
    label_field: str = "label"
    validation_ratio: float = 0.15
    max_source_length: int = 256
    max_target_length: int = 4
    epochs: float = 5.0
    learning_rate: float = 2e-4
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q", "v"])
    eval_strategy: Literal["epoch", "steps"] = "epoch"
    logging_steps: int = 20
    save_steps: int = 200
    eval_steps: int = 200
    device: Literal["auto", "cuda", "cpu"] = "auto"
    fp16: bool | None = None


class PcapAnalysisRequest(BaseModel):
    """Request body for PCAP/log correlation."""

    errors_jsonl: str = "output.jsonl"
    pcap: str
    output: str = "diagnosis.jsonl"
    window_seconds: float = 3.0


class GroqDiagnosisRequest(BaseModel):
    """Request body for Groq diagnosis."""

    input: str = "diagnosis.jsonl"
    output: str = "groq_diagnosis.jsonl"
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.0
    max_tokens: int = 600
    max_record_chars: int = 1500
    max_error_logs: int = 4
    max_teardown_events: int = 2
    retries: int = 3
    retry_sleep_seconds: float = 15.0
    sleep_seconds: float = 0.0
    limit: int | None = None


class LocalLlmDiagnosisRequest(BaseModel):
    """Request body for local open-source LLM diagnosis."""

    input: str = "diagnosis.jsonl"
    output: str = "local_llm_diagnosis.jsonl"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    model_type: Literal["causal", "seq2seq"] = "causal"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    device_map: str = "auto"
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    max_input_tokens: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.0
    max_record_chars: int = 12000
    limit: int | None = None


class GroqPipelineRequest(BaseModel):
    """Request body for an end-to-end pipeline using Groq diagnosis."""

    logfile: str
    pcap: str
    model_dir: str
    output_dir: str = "outputs"
    inference: InferenceOptions = Field(default_factory=InferenceOptions)
    pcap_window_seconds: float = 3.0
    groq: GroqDiagnosisRequest = Field(default_factory=GroqDiagnosisRequest)


class InferenceResponse(BaseModel):
    """Response from FLAN-T5 inference."""

    output: str
    row_count: int
    elapsed_seconds: float
    generation_seconds: float
    memory_summary: str
    preview: list[dict[str, Any]]


class FineTuningResponse(BaseModel):
    """Response from FLAN-T5 fine-tuning."""

    output_dir: str
    train_rows: int
    validation_rows: int
    metrics: dict[str, Any]
    prediction_samples: list[dict[str, Any]] = Field(default_factory=list)


class JsonlResponse(BaseModel):
    """Generic JSONL-producing endpoint response."""

    output: str
    row_count: int
    preview: list[dict[str, Any]]


class JobSubmitResponse(BaseModel):
    """Response returned when a background job is accepted."""

    job_id: str
    status: str
    status_url: str


class JobStatusResponse(BaseModel):
    """Current status for a background job."""

    job_id: str
    name: str
    status: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: Any | None = None
    error: dict[str, Any] | None = None


def resolve_path(path_text: str) -> Path:
    """Resolve a path and ensure it stays inside the workspace."""

    path = Path(path_text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    resolved = path.resolve()
    if resolved != WORKSPACE_ROOT and WORKSPACE_ROOT not in resolved.parents:
        raise HTTPException(
            status_code=400,
            detail=f"path must stay inside workspace root: {WORKSPACE_ROOT}",
        )
    return resolved


def resolve_read_path(
    path_text: str,
    label: str,
    want_dir: bool | None = None,
) -> Path:
    """Resolve readable inputs from workspace first, then app directory."""

    path = Path(path_text)
    candidates = [path.resolve()] if path.is_absolute() else [
        (WORKSPACE_ROOT / path).resolve(),
        (APP_ROOT / path).resolve(),
    ]
    allowed_roots = (WORKSPACE_ROOT, APP_ROOT)

    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if not any(candidate == root or root in candidate.parents for root in allowed_roots):
            continue
        if not candidate.exists():
            continue
        return require_existing_path(candidate, label, want_dir=want_dir)

    raise HTTPException(
        status_code=400,
        detail=(
            f"{label} does not exist or is outside allowed roots. "
            f"Checked: {', '.join(checked)}"
        ),
    )


def resolve_model_dir(path_text: str) -> Path:
    """Resolve model_dir from workspace first, then the app directory."""

    path = Path(path_text)
    candidates = [path.resolve()] if path.is_absolute() else [
        (WORKSPACE_ROOT / path).resolve(),
        (APP_ROOT / path).resolve(),
    ]
    allowed_roots = (WORKSPACE_ROOT, APP_ROOT)

    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if not any(candidate == root or root in candidate.parents for root in allowed_roots):
            continue
        if candidate.is_dir():
            return candidate

    raise HTTPException(
        status_code=400,
        detail=(
            "model_dir does not exist or is outside allowed roots. "
            f"Checked: {', '.join(checked)}"
        ),
    )


def require_existing_path(path: Path, label: str, want_dir: bool | None = None) -> Path:
    """Return path when it exists, otherwise raise a clear request error."""

    if not path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"{label} does not exist: {path}",
        )
    if want_dir is True and not path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be a directory: {path}",
        )
    if want_dir is False and not path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be a file: {path}",
        )
    return path


def save_upload_file(upload: UploadFile, subdir: str) -> str:
    """Save an uploaded file under the workspace and return a relative path."""

    original_name = Path(upload.filename or "uploaded_file").name
    if not original_name or original_name in {".", ".."}:
        original_name = "uploaded_file"
    relative_path = Path("uploads") / subdir / f"{uuid.uuid4().hex}_{original_name}"
    destination = resolve_path(str(relative_path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    upload.file.seek(0)
    with destination.open("wb") as output_file:
        shutil.copyfileobj(upload.file, output_file)
    return str(relative_path)


def run_service(action: Callable[[], T]) -> T:
    """Run service code and convert unexpected exceptions to HTTP 500."""

    try:
        return action()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error_type": type(exc).__name__, "message": str(exc)},
        ) from exc


def jsonable(value: Any) -> Any:
    """Convert service results into JSON-serializable values."""

    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def submit_job(name: str, action: Callable[[], Any]) -> JobSubmitResponse:
    """Submit a background job and return its ID."""

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "name": name,
            "status": "queued",
            "created_at": time(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }

    def runner() -> None:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "running"
            JOBS[job_id]["started_at"] = time()
        try:
            result = action()
        except Exception as exc:
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["finished_at"] = time()
                JOBS[job_id]["error"] = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            return
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "succeeded"
            JOBS[job_id]["finished_at"] = time()
            JOBS[job_id]["result"] = jsonable(result)

    JOB_EXECUTOR.submit(runner)
    return JobSubmitResponse(
        job_id=job_id,
        status="queued",
        status_url=f"/jobs/{job_id}",
    )


def jsonl_preview(path: Path, limit: int = 5) -> list[dict[str, Any]]:
    """Read a small JSONL preview."""

    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def execute_flan_t5_inference(request: InferenceRequest) -> InferenceResponse:
    """Execute FLAN-T5 inference from a request object."""

    logfile = resolve_read_path(request.logfile, "logfile", want_dir=False)
    model_dir = resolve_model_dir(request.model_dir)
    output = resolve_path(request.output)

    from src.inference_flan_t5 import (
        InferenceConfig,
        run_flan_t5_inference as run_flan_t5_inference_service,
    )

    result = run_flan_t5_inference_service(
        InferenceConfig(
            logfile=logfile,
            model_dir=model_dir,
            output=output,
            base_model=request.base_model,
            max_source_length=request.max_source_length,
            max_new_tokens=request.max_new_tokens,
            batch_size=request.batch_size,
            device=request.device,
            dtype=request.dtype,
        )
    )
    return InferenceResponse(
        output=str(output),
        row_count=len(result.rows),
        elapsed_seconds=result.elapsed_seconds,
        generation_seconds=result.generation_seconds,
        memory_summary=result.memory_summary,
        preview=result.rows[:5],
    )


def execute_flan_t5_finetuning(request: FineTuningRequest) -> FineTuningResponse:
    """Execute FLAN-T5 LoRA fine-tuning from a request object."""

    train_csv = resolve_read_path(request.train_csv, "train_csv", want_dir=False)
    validation_csv = (
        resolve_read_path(request.validation_csv, "validation_csv", want_dir=False)
        if request.validation_csv
        else None
    )
    output_dir = resolve_path(request.output_dir)

    from src.finetuning import (
        FineTuningConfig,
        fine_tune_flan_t5 as fine_tune_flan_t5_service,
    )

    result = fine_tune_flan_t5_service(
        FineTuningConfig(
            train_csv=train_csv,
            validation_csv=validation_csv,
            model=request.model,
            output_dir=output_dir,
            text_field=request.text_field,
            label_field=request.label_field,
            validation_ratio=request.validation_ratio,
            max_source_length=request.max_source_length,
            max_target_length=request.max_target_length,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            train_batch_size=request.train_batch_size,
            eval_batch_size=request.eval_batch_size,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            weight_decay=request.weight_decay,
            warmup_ratio=request.warmup_ratio,
            seed=request.seed,
            lora_r=request.lora_r,
            lora_alpha=request.lora_alpha,
            lora_dropout=request.lora_dropout,
            target_modules=request.target_modules,
            eval_strategy=request.eval_strategy,
            logging_steps=request.logging_steps,
            save_steps=request.save_steps,
            eval_steps=request.eval_steps,
            device=request.device,
            fp16=request.fp16,
        )
    )
    return FineTuningResponse(**result)


def execute_pcap_analysis(request: PcapAnalysisRequest) -> JsonlResponse:
    """Execute PCAP analysis from a request object."""

    from src.pcap_analysis import (
        PcapAnalysisConfig,
        run_pcap_analysis as run_pcap_analysis_service,
    )

    output = resolve_path(request.output)
    records = run_pcap_analysis_service(
        PcapAnalysisConfig(
            errors_jsonl=resolve_read_path(request.errors_jsonl, "errors_jsonl", want_dir=False),
            pcap=resolve_read_path(request.pcap, "pcap", want_dir=False),
            output=output,
            window_seconds=request.window_seconds,
        )
    )
    return JsonlResponse(
        output=str(output),
        row_count=len(records),
        preview=records[:5],
    )


def execute_groq_diagnosis(request: GroqDiagnosisRequest) -> JsonlResponse:
    """Execute Groq diagnosis from a request object."""

    from src.groq_diagnosis import (
        GroqDiagnosisConfig,
        run_groq_diagnosis as run_groq_diagnosis_service,
    )

    output = resolve_path(request.output)
    rows = run_groq_diagnosis_service(
        GroqDiagnosisConfig(
            input=resolve_read_path(request.input, "input", want_dir=False),
            output=output,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            limit=request.limit,
            sleep_seconds=request.sleep_seconds,
            max_record_chars=request.max_record_chars,
            max_error_logs=request.max_error_logs,
            max_teardown_events=request.max_teardown_events,
            retries=request.retries,
            retry_sleep_seconds=request.retry_sleep_seconds,
        )
    )
    return JsonlResponse(
        output=str(output),
        row_count=len(rows),
        preview=rows[:5],
    )


def execute_local_llm_diagnosis(request: LocalLlmDiagnosisRequest) -> JsonlResponse:
    """Execute local LLM diagnosis from a request object."""

    from src.local_llm_diagnosis import (
        LocalLlmDiagnosisConfig,
        run_local_llm_diagnosis as run_local_llm_diagnosis_service,
    )

    output = resolve_path(request.output)
    rows = run_local_llm_diagnosis_service(
        LocalLlmDiagnosisConfig(
            input=resolve_read_path(request.input, "input", want_dir=False),
            output=output,
            model=request.model,
            model_type=request.model_type,
            load_in_4bit=request.load_in_4bit,
            load_in_8bit=request.load_in_8bit,
            device_map=request.device_map,
            torch_dtype=request.torch_dtype,
            max_input_tokens=request.max_input_tokens,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            max_record_chars=request.max_record_chars,
            limit=request.limit,
        )
    )
    return JsonlResponse(
        output=str(output),
        row_count=len(rows),
        preview=rows[:5],
    )


def execute_groq_pipeline(request: GroqPipelineRequest) -> dict[str, Any]:
    """Execute the end-to-end Groq pipeline."""

    output_dir = resolve_path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_output = output_dir / "output.jsonl"
    diagnosis_output = output_dir / "diagnosis.jsonl"
    groq_output = output_dir / "groq_diagnosis.jsonl"

    inference_result = execute_flan_t5_inference(
        InferenceRequest(
            logfile=request.logfile,
            model_dir=request.model_dir,
            output=str(inference_output),
            base_model=request.inference.base_model,
            max_source_length=request.inference.max_source_length,
            max_new_tokens=request.inference.max_new_tokens,
            batch_size=request.inference.batch_size,
            device=request.inference.device,
            dtype=request.inference.dtype,
        )
    )
    pcap_result = execute_pcap_analysis(
        PcapAnalysisRequest(
            errors_jsonl=str(inference_output),
            pcap=request.pcap,
            output=str(diagnosis_output),
            window_seconds=request.pcap_window_seconds,
        )
    )
    groq_result = execute_groq_diagnosis(
        request.groq.model_copy(
            update={
                "input": str(diagnosis_output),
                "output": str(groq_output),
            }
        )
    )

    return {
        "outputs": {
            "inference": str(inference_output),
            "diagnosis": str(diagnosis_output),
            "groq_diagnosis": str(groq_output),
        },
        "steps": {
            "inference": inference_result.model_dump(),
            "pcap": pcap_result.model_dump(),
            "groq": groq_result.model_dump(),
        },
        "preview": jsonl_preview(groq_output),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Return basic service status."""

    return {
        "status": "ok",
        "workspace_root": str(WORKSPACE_ROOT),
        "python": sys.version,
    }


@app.post("/inference/flan-t5", response_model=InferenceResponse)
def run_flan_t5_inference(request: InferenceRequest) -> InferenceResponse:
    """Run FLAN-T5 inference and write error predictions JSONL."""

    return run_service(lambda: execute_flan_t5_inference(request))


@app.post("/finetune/flan-t5", response_model=FineTuningResponse)
def run_flan_t5_finetuning(request: FineTuningRequest) -> FineTuningResponse:
    """Fine-tune FLAN-T5 with LoRA and save an adapter."""

    return run_service(lambda: execute_flan_t5_finetuning(request))


@app.post("/pcap/analyze", response_model=JsonlResponse)
def run_pcap_analysis(request: PcapAnalysisRequest) -> JsonlResponse:
    """Correlate inference error rows with PCAP teardown evidence."""

    return run_service(lambda: execute_pcap_analysis(request))


@app.post("/diagnosis/groq", response_model=JsonlResponse)
def run_groq_diagnosis(request: GroqDiagnosisRequest) -> JsonlResponse:
    """Run Groq diagnosis over diagnosis JSONL."""

    return run_service(lambda: execute_groq_diagnosis(request))


@app.post("/diagnosis/local-llm", response_model=JsonlResponse)
def run_local_llm_diagnosis(request: LocalLlmDiagnosisRequest) -> JsonlResponse:
    """Run local open-source LLM diagnosis over diagnosis JSONL."""

    return run_service(lambda: execute_local_llm_diagnosis(request))


@app.post("/pipeline/groq")
def run_groq_pipeline(request: GroqPipelineRequest) -> dict[str, Any]:
    """Run inference, PCAP analysis, and Groq diagnosis in sequence."""

    return run_service(lambda: execute_groq_pipeline(request))


@app.post("/jobs/inference/flan-t5", response_model=JobSubmitResponse)
def submit_flan_t5_inference_job(request: InferenceRequest) -> JobSubmitResponse:
    """Submit FLAN-T5 inference as a background job."""

    return submit_job("inference/flan-t5", lambda: execute_flan_t5_inference(request))


@app.post("/jobs/inference/flan-t5/upload", response_model=JobSubmitResponse)
def submit_flan_t5_inference_upload_job(
    logfile: UploadFile = File(...),
    model_dir: str = Form("models/flan-t5-log-lora-model"),
    output: str = Form("outputs/output.jsonl"),
    base_model: str = Form("google/flan-t5-small"),
    max_source_length: int = Form(128),
    max_new_tokens: int = Form(2),
    batch_size: int = Form(16),
    device: Literal["auto", "cuda", "cpu"] = Form("auto"),
    dtype: Literal["auto", "fp16", "fp32"] = Form("auto"),
) -> JobSubmitResponse:
    """Upload a log file and submit FLAN-T5 inference as a background job."""

    logfile_path = save_upload_file(logfile, "inference")
    request = InferenceRequest(
        logfile=logfile_path,
        model_dir=model_dir,
        output=output,
        base_model=base_model,
        max_source_length=max_source_length,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    return submit_job("inference/flan-t5/upload", lambda: execute_flan_t5_inference(request))


@app.post("/jobs/finetune/flan-t5", response_model=JobSubmitResponse)
def submit_flan_t5_finetuning_job(request: FineTuningRequest) -> JobSubmitResponse:
    """Submit FLAN-T5 fine-tuning as a background job."""

    return submit_job("finetune/flan-t5", lambda: execute_flan_t5_finetuning(request))


@app.post("/jobs/pcap/analyze", response_model=JobSubmitResponse)
def submit_pcap_analysis_job(request: PcapAnalysisRequest) -> JobSubmitResponse:
    """Submit PCAP analysis as a background job."""

    return submit_job("pcap/analyze", lambda: execute_pcap_analysis(request))


@app.post("/jobs/pcap/analyze/upload", response_model=JobSubmitResponse)
def submit_pcap_analysis_upload_job(
    errors_jsonl: UploadFile = File(...),
    pcap: UploadFile = File(...),
    output: str = Form("outputs/diagnosis.jsonl"),
    window_seconds: float = Form(3.0),
) -> JobSubmitResponse:
    """Upload inference JSONL and PCAP files, then submit PCAP analysis."""

    errors_jsonl_path = save_upload_file(errors_jsonl, "pcap")
    pcap_path = save_upload_file(pcap, "pcap")
    request = PcapAnalysisRequest(
        errors_jsonl=errors_jsonl_path,
        pcap=pcap_path,
        output=output,
        window_seconds=window_seconds,
    )
    return submit_job("pcap/analyze/upload", lambda: execute_pcap_analysis(request))


@app.post("/jobs/diagnosis/groq", response_model=JobSubmitResponse)
def submit_groq_diagnosis_job(request: GroqDiagnosisRequest) -> JobSubmitResponse:
    """Submit Groq diagnosis as a background job."""

    return submit_job("diagnosis/groq", lambda: execute_groq_diagnosis(request))


@app.post("/jobs/diagnosis/local-llm", response_model=JobSubmitResponse)
def submit_local_llm_diagnosis_job(
    request: LocalLlmDiagnosisRequest,
) -> JobSubmitResponse:
    """Submit local LLM diagnosis as a background job."""

    return submit_job(
        "diagnosis/local-llm",
        lambda: execute_local_llm_diagnosis(request),
    )


@app.post("/jobs/pipeline/groq", response_model=JobSubmitResponse)
def submit_groq_pipeline_job(request: GroqPipelineRequest) -> JobSubmitResponse:
    """Submit the end-to-end Groq pipeline as a background job."""

    return submit_job("pipeline/groq", lambda: execute_groq_pipeline(request))


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Return background job status and result/error when finished."""

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
        return JobStatusResponse(**job)


@app.get("/files")
def download_file(path: str) -> FileResponse:
    """Download a file from the workspace."""

    resolved = resolve_path(path)
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"file not found: {resolved}")
    return FileResponse(str(resolved), filename=resolved.name)
