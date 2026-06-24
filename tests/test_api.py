from __future__ import annotations

from pathlib import Path
from time import sleep

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from src import api


def test_health_endpoint() -> None:
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


def test_background_job_endpoint(monkeypatch) -> None:
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


def test_finetuning_job_endpoint(monkeypatch) -> None:
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
