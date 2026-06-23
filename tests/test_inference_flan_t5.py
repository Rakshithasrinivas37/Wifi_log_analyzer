from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from src import inference_flan_t5 as inference


def test_iso_to_epoch_formats_microseconds() -> None:
    assert inference.iso_to_epoch("2026-06-22T10:37:58.400000Z") == "1782095878.400000"


def test_iter_log_rows_skips_blank_and_short_lines() -> None:
    lines = [
        "\n",
        "tiny\n",
        "2026-06-22T10:30:00Z hostapd: wlan0: STA 3c:22:fb:10:24:38 authenticated\n",
    ]

    rows = list(inference.iter_log_rows(lines))

    assert rows == [
        {
            "timestamp": "1782095400.000000",
            "log": "hostapd: wlan0: STA 3c:22:fb:10:24:38 authenticated",
        }
    ]


def test_extract_mac_addresses_returns_unique_ordered_values() -> None:
    text = (
        "STA 3c:22:fb:10:24:38 failed with AP 00:25:9c:7a:10:01; "
        "retry from 3C:22:FB:10:24:38"
    )

    assert inference.extract_mac_addresses(text) == [
        "3c:22:fb:10:24:38",
        "00:25:9c:7a:10:01",
    ]


def test_write_jsonl_writes_sorted_json_lines(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    rows = [{"prediction": "error", "timestamp": "1.000000", "log": "x"}]

    inference.write_jsonl(rows, output)

    loaded = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert loaded == rows


def test_total_inference_time_decorator_updates_result() -> None:
    @inference.measure_total_inference_time
    def fake_run() -> inference.InferenceResult:
        return inference.InferenceResult(
            rows=[],
            elapsed_seconds=0.0,
            generation_seconds=0.0,
            output=None,
            memory_summary="ok",
        )

    result = fake_run()

    assert result.elapsed_seconds >= 0
