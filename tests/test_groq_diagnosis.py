"""Tests for Groq prompt trimming, JSON parsing, and diagnosis output."""

from __future__ import annotations

import json
from pathlib import Path

from src import groq_diagnosis as groq


def evidence_record() -> dict[str, object]:
    """Build one representative PCAP/log evidence record for tests."""

    return {
        "mac": "3c:22:fb:10:24:38",
        "timestamp": "1782095400.100000",
        "error_logs": [
            "EAPOL invalid MIC",
            "EAPOL timeout",
            "EAPOL timeout again",
        ],
        "pcap_session": {
            "first_seen": 1782095400.0,
            "last_seen": 1782095401.0,
            "packet_count": 3,
            "reason_code_hints": [{"reason_code": 15}],
            "teardown_events": [
                {"kind": "deauth", "reason_code": 15, "ts": 1782095400.5},
                {"kind": "deauth", "reason_code": 15, "ts": 1782095400.6},
            ],
        },
    }


def test_trim_record_limits_logs_and_teardown_events() -> None:
    """Large evidence records should be trimmed before being sent to Groq."""

    trimmed = groq.trim_record(
        evidence_record(),
        max_chars=10_000,
        max_error_logs=1,
        max_teardown_events=1,
    )

    assert trimmed["error_logs"] == ["EAPOL invalid MIC"]
    assert trimmed["error_logs_truncated"] == 2
    assert len(trimmed["pcap_session"]["teardown_events"]) == 1
    assert trimmed["pcap_session"]["teardown_events_truncated"] == 1


def test_parse_llm_json_handles_fenced_json() -> None:
    """Groq JSON parsing should tolerate markdown fenced JSON blocks."""

    parsed = groq.parse_llm_json(
        '```json\n{"root_cause": "4-way handshake timeout", "confidence": 0.9}\n```'
    )

    assert parsed["root_cause"] == "4-way handshake timeout"
    assert parsed["confidence"] == 0.9


def test_diagnose_records_uses_mocked_groq_call(tmp_path: Path, monkeypatch) -> None:
    """Groq diagnosis should write parsed JSONL rows when the provider succeeds."""

    input_path = tmp_path / "diagnosis.jsonl"
    output_path = tmp_path / "groq.jsonl"
    input_path.write_text(json.dumps(evidence_record()) + "\n", encoding="utf-8")

    def fake_call_groq(*args, **kwargs) -> tuple[str, str]:
        return (
            json.dumps(
                {
                    "mac": "3c:22:fb:10:24:38",
                    "root_cause": "4-way handshake timeout",
                    "confidence": 0.95,
                    "why": ["reason code 15"],
                    "recommended_action": {
                        "summary": "Check PSK/security settings.",
                        "immediate_steps": ["Verify PSK"],
                        "validation_steps": ["Reconnect client"],
                        "if_still_failing": ["Capture fresh PCAP"],
                        "data_to_collect": ["AP logs"],
                    },
                }
            ),
            "stop",
        )

    monkeypatch.setattr(groq, "call_groq", fake_call_groq)

    rows = groq.run_groq_diagnosis(
        groq.GroqDiagnosisConfig(input=input_path, output=output_path)
    )

    assert rows[0]["diagnosis"]["root_cause"] == "4-way handshake timeout"
    assert json.loads(output_path.read_text(encoding="utf-8"))["mac"] == "3c:22:fb:10:24:38"


def test_invalid_llm_json_returns_parse_error_fallback(monkeypatch) -> None:
    """Invalid Groq output should become a structured fallback diagnosis."""

    def fake_call_groq(*args, **kwargs) -> tuple[str, str]:
        return ('{"root_cause": "unfinished', "length")

    monkeypatch.setattr(groq, "call_groq", fake_call_groq)

    row = groq.diagnose_record(
        evidence_record(),
        groq.GroqDiagnosisConfig(input=Path("unused.jsonl")),
    )

    assert row["diagnosis"]["root_cause"] == "LLM response was not valid JSON"
    assert row["diagnosis"]["finish_reason"] == "length"
