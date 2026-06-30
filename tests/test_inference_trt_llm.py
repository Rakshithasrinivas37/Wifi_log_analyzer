"""Tests for TensorRT-LLM inference helpers that do not require a TRT engine."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import inference_trt_llm as inference


class FakeGenerator:
    """Small stand-in that returns predetermined labels for log batches."""

    def classify_batch(self, log_bodies: list[str]) -> list[str]:
        """Return one normal and one error prediction for filtering tests."""

        return ["normal", "error"]


def test_iter_trt_log_rows_keeps_only_errors_by_default(tmp_path: Path) -> None:
    """TRT inference output should contain only error rows unless opted into all rows."""

    logfile = tmp_path / "wifi_logs.txt"
    logfile.write_text(
        "\n".join(
            [
                "2026-06-22T10:30:00+08:00 hostapd: wlan0: STA 00:11:22:33:44:55 authenticated",
                "2026-06-22T10:30:01+08:00 hostapd: wlan0: STA aa:bb:cc:dd:ee:ff WPA timeout",
            ]
        ),
        encoding="utf-8",
    )

    rows = list(
        inference.iter_trt_log_rows(
            logfile=logfile,
            generator=FakeGenerator(),
            batch_size=2,
            include_all_predictions=False,
        )
    )

    assert rows == [
        {
            "timestamp": "1782095401.000000",
            "log": "hostapd: wlan0: STA aa:bb:cc:dd:ee:ff WPA timeout",
            "prediction": "error",
            "mac_addresses": ["aa:bb:cc:dd:ee:ff"],
        }
    ]


def test_write_jsonl_writes_only_filtered_trt_rows(tmp_path: Path) -> None:
    """Writing filtered TRT rows should produce an error-only JSONL file."""

    output = tmp_path / "output.jsonl"
    rows = [{"timestamp": "1.000000", "log": "x", "prediction": "error"}]

    inference.write_jsonl(rows, output)

    loaded = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert loaded == rows
