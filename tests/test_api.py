"""Tests for FastAPI routing, path resolution, uploads, and job polling."""

from __future__ import annotations

from pathlib import Path
from time import sleep

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from src import api


def test_health_endpoint() -> None:
    """Health endpoint should report that the service is running."""

    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_resolve_model_dir_uses_app_root_when_workspace_model_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    app = tmp_path / "app"
    model = app / "models" / "flan-t5-log-lora-model"
    workspace.mkdir()
    model.mkdir(parents=True)

    monkeypatch.setattr(api, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(api, "APP_ROOT", app)

    assert api.resolve_model_dir("models/flan-t5-log-lora-model") == model.resolve()


def test_resolve_read_path_uses_app_root_when_workspace_file_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    app = tmp_path / "app"
    logfile = app / "data" / "inputs" / "wifi_events_3600.txt"
    workspace.mkdir()
    logfile.parent.mkdir(parents=True)
    logfile.write_text("2026-06-22T10:30:00+08:00 hostapd: ok\n", encoding="utf-8")

    monkeypatch.setattr(api, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(api, "APP_ROOT", app)

    assert api.resolve_read_path("data/inputs/wifi_events_3600.txt", "logfile") == logfile.resolve()


def test_resolve_trt_engine_dir_allows_workspace_sibling(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace" / "wifi-log-analyzer"
    app = tmp_path / "app"
    engine = tmp_path / "workspace" / "trt_engine" / "t5-small"
    workspace.mkdir(parents=True)
    app.mkdir()
    engine.mkdir(parents=True)

    monkeypatch.setattr(api, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(api, "APP_ROOT", app)

    assert api.resolve_trt_engine_dir(str(engine)) == engine.resolve()


def test_background_job_endpoint(monkeypatch) -> None:
    """Background jobs should be submitted, executed, and returned by status API."""

    def fake_execute_groq_diagnosis(request):
        return api.JsonlResponse(
            output="groq_diagnosis.jsonl",
            row_count=1,
            preview=[{"mac": "3c:22:fb:10:24:38"}],
        )

    monkeypatch.setattr(api, "execute_groq_diagnosis", fake_execute_groq_diagnosis)
    client = TestClient(api.app)

    submit_response = client.post(
        "/jobs/diagnosis/groq",
        json={"input": "diagnosis.jsonl", "output": "groq_diagnosis.jsonl"},
    )

    assert submit_response.status_code == 200
    job_id = submit_response.json()["job_id"]

    for _ in range(100):
        status_response = client.get(f"/jobs/{job_id}")
        assert status_response.status_code == 200
        status = status_response.json()
        if status["status"] == "succeeded":
            break
        sleep(0.01)

    assert status["status"] == "succeeded"
    assert status["result"]["row_count"] == 1


def test_inference_upload_job_endpoint(monkeypatch, tmp_path: Path) -> None:
    """Uploaded log files should be saved before FLAN-T5 inference jobs run."""

    monkeypatch.setattr(api, "WORKSPACE_ROOT", tmp_path)

    def fake_execute_flan_t5_inference(request):
        uploaded_log = tmp_path / request.logfile
        assert uploaded_log.is_file()
        assert request.model_dir == "models/flan-t5-log-lora-model"
        return api.InferenceResponse(
            output="outputs/output.jsonl",
            row_count=1,
            elapsed_seconds=0.1,
            generation_seconds=0.05,
            memory_summary="ok",
            preview=[{"prediction": "error"}],
        )

    monkeypatch.setattr(
        api,
        "execute_flan_t5_inference",
        fake_execute_flan_t5_inference,
    )
    client = TestClient(api.app)

    submit_response = client.post(
        "/jobs/inference/flan-t5/upload",
        data={
            "model_dir": "models/flan-t5-log-lora-model",
            "output": "outputs/output.jsonl",
            "device": "cpu",
        },
        files={
            "logfile": (
                "wifi_logs.txt",
                b"2026-06-22T10:30:00+08:00 hostapd: test\n",
                "text/plain",
            )
        },
    )

    assert submit_response.status_code == 200
    job_id = submit_response.json()["job_id"]

    for _ in range(100):
        status_response = client.get(f"/jobs/{job_id}")
        assert status_response.status_code == 200
        status = status_response.json()
        if status["status"] == "succeeded":
            break
        sleep(0.01)

    assert status["status"] == "succeeded"
    assert status["result"]["row_count"] == 1


def test_trt_inference_endpoint(monkeypatch) -> None:
    """TRT inference endpoint should pass request fields to the executor."""

    def fake_execute_trt_llm_inference(request):
        assert request.engine_dir == "/workspace/trt_engine/t5-small"
        assert request.batch_size == 16
        return api.TrtInferenceResponse(
            output="outputs/trt_output.jsonl",
            row_count=1,
            elapsed_seconds=0.1,
            engine_metadata={"is_enc_dec_model": True},
            tokenizer="google/flan-t5-small",
            preview=[{"prediction": "error"}],
        )

    monkeypatch.setattr(api, "execute_trt_llm_inference", fake_execute_trt_llm_inference)
    client = TestClient(api.app)

    response = client.post(
        "/inference/trt-llm",
        json={
            "logfile": "data/inputs/wifi_logs.txt",
            "engine_dir": "/workspace/trt_engine/t5-small",
            "batch_size": 16,
        },
    )

    assert response.status_code == 200
    assert response.json()["row_count"] == 1


def test_trt_inference_upload_job_endpoint(monkeypatch, tmp_path: Path) -> None:
    """Uploaded log files should be saved before TRT inference jobs run."""

    monkeypatch.setattr(api, "WORKSPACE_ROOT", tmp_path)

    def fake_execute_trt_llm_inference(request):
        uploaded_log = tmp_path / request.logfile
        assert uploaded_log.is_file()
        assert request.engine_dir == "/workspace/trt_engine/t5-small"
        assert request.include_all_predictions is True
        return api.TrtInferenceResponse(
            output="outputs/trt_output.jsonl",
            row_count=1,
            elapsed_seconds=0.1,
            engine_metadata={"is_enc_dec_model": True},
            tokenizer="google/flan-t5-small",
            preview=[{"prediction": "error"}],
        )

    monkeypatch.setattr(api, "execute_trt_llm_inference", fake_execute_trt_llm_inference)
    client = TestClient(api.app)

    submit_response = client.post(
        "/jobs/inference/trt-llm/upload",
        data={
            "engine_dir": "/workspace/trt_engine/t5-small",
            "output": "outputs/trt_output.jsonl",
            "include_all_predictions": "true",
        },
        files={
            "logfile": (
                "wifi_logs.txt",
                b"2026-06-22T10:30:00+08:00 hostapd: test\n",
                "text/plain",
            )
        },
    )

    assert submit_response.status_code == 200
    job_id = submit_response.json()["job_id"]

    for _ in range(100):
        status_response = client.get(f"/jobs/{job_id}")
        assert status_response.status_code == 200
        status = status_response.json()
        if status["status"] == "succeeded":
            break
        sleep(0.01)

    assert status["status"] == "succeeded"
    assert status["result"]["row_count"] == 1


def test_pcap_upload_job_endpoint(monkeypatch, tmp_path: Path) -> None:
    """Uploaded JSONL and PCAP files should be saved before PCAP jobs run."""

    monkeypatch.setattr(api, "WORKSPACE_ROOT", tmp_path)

    def fake_execute_pcap_analysis(request):
        uploaded_errors = tmp_path / request.errors_jsonl
        uploaded_pcap = tmp_path / request.pcap
        assert uploaded_errors.is_file()
        assert uploaded_pcap.is_file()
        return api.JsonlResponse(
            output="outputs/diagnosis.jsonl",
            row_count=1,
            preview=[{"mac": "3c:22:fb:10:24:38"}],
        )

    monkeypatch.setattr(api, "execute_pcap_analysis", fake_execute_pcap_analysis)
    client = TestClient(api.app)

    submit_response = client.post(
        "/jobs/pcap/analyze/upload",
        data={"output": "outputs/diagnosis.jsonl", "window_seconds": "3.0"},
        files={
            "errors_jsonl": (
                "output.jsonl",
                b'{"timestamp":"1.000000","prediction":"error"}\n',
                "application/json",
            ),
            "pcap": ("capture.pcap", b"\xd4\xc3\xb2\xa1", "application/vnd.tcpdump.pcap"),
        },
    )

    assert submit_response.status_code == 200
    job_id = submit_response.json()["job_id"]

    for _ in range(100):
        status_response = client.get(f"/jobs/{job_id}")
        assert status_response.status_code == 200
        status = status_response.json()
        if status["status"] == "succeeded":
            break
        sleep(0.01)

    assert status["status"] == "succeeded"
    assert status["result"]["row_count"] == 1


def test_finetuning_job_endpoint(monkeypatch) -> None:
    """Fine-tuning jobs should return serialized training summaries."""

    def fake_execute_flan_t5_finetuning(request):
        return api.FineTuningResponse(
            output_dir="models/flan-t5-log-lora-model",
            train_rows=10,
            validation_rows=2,
            metrics={"eval_accuracy": 1.0},
        )

    monkeypatch.setattr(
        api,
        "execute_flan_t5_finetuning",
        fake_execute_flan_t5_finetuning,
    )
    client = TestClient(api.app)

    submit_response = client.post(
        "/jobs/finetune/flan-t5",
        json={
            "train_csv": "data/datasets/training.csv",
            "validation_csv": "data/datasets/validation.csv",
            "output_dir": "models/flan-t5-log-lora-model",
        },
    )

    assert submit_response.status_code == 200
    job_id = submit_response.json()["job_id"]

    for _ in range(100):
        status_response = client.get(f"/jobs/{job_id}")
        assert status_response.status_code == 200
        status = status_response.json()
        if status["status"] == "succeeded":
            break
        sleep(0.01)

    assert status["status"] == "succeeded"
    assert status["result"]["train_rows"] == 10
