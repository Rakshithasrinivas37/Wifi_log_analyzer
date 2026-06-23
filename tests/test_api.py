from __future__ import annotations

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
