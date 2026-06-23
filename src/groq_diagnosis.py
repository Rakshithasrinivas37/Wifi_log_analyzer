"""Use Groq-hosted LLMs to turn PCAP/log evidence into action plans.

This module exposes callable diagnosis functions for service use. It does not
parse CLI arguments or execute work at import time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a senior WiFi troubleshooting assistant.

You receive structured evidence extracted from infrastructure logs and an
802.11 PCAP. Use only the provided evidence. Do not invent packet fields,
logs, vendor details, credentials, topology, or measurements that are not in
the input.

Return only valid JSON with this shape:
{
  "mac": "client mac",
  "root_cause": "short diagnosis",
  "confidence": 0.0,
  "why": ["evidence-backed reason"],
  "recommended_action": {
    "summary": "one sentence",
    "immediate_steps": ["specific action"],
    "validation_steps": ["how to confirm the fix"],
    "if_still_failing": ["next escalation step"],
    "data_to_collect": ["extra evidence to capture if unresolved"]
  }
}

Make recommended_action detailed and operational. Prefer concrete WiFi actions:
PSK/security-mode checks, PMF compatibility, RADIUS checks, DHCP pool/VLAN/relay
checks, RF checks, firmware/driver checks, or AP capacity checks when supported
by evidence.
"""


@dataclass
class GroqDiagnosisConfig:
    """Settings for Groq diagnosis."""

    input: Path
    output: Path | None = None
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.0
    max_tokens: int = 600
    limit: int | None = None
    sleep_seconds: float = 0.0
    max_record_chars: int = 1500
    max_error_logs: int = 4
    max_teardown_events: int = 2
    retries: int = 3
    retry_sleep_seconds: float = 15.0


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load JSONL evidence records."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
            if limit is not None and len(records) >= limit:
                break
    return records


def compact_session(session: Any, max_teardown_events: int) -> Any:
    """Keep the most diagnostic PCAP fields when a record is too large."""

    if not isinstance(session, dict):
        return session
    return {
        "first_seen": session.get("first_seen"),
        "last_seen": session.get("last_seen"),
        "packet_count": session.get("packet_count"),
        "assoc_status_code": session.get("assoc_status_code"),
        "eapol_frames": session.get("eapol_frames"),
        "dhcp": session.get("dhcp"),
        "reason_codes": session.get("reason_codes"),
        "reason_code_hints": session.get("reason_code_hints"),
        "teardown_events": session.get("teardown_events", [])[:max_teardown_events],
    }


def trim_record(
    record: dict[str, Any],
    max_chars: int,
    max_error_logs: int,
    max_teardown_events: int,
) -> dict[str, Any]:
    """Trim very large records before sending to Groq."""

    trimmed = dict(record)
    logs = list(trimmed.get("error_logs", []))
    if max_error_logs >= 0 and len(logs) > max_error_logs:
        trimmed["error_logs"] = logs[:max_error_logs]
        trimmed["error_logs_truncated"] = len(logs) - max_error_logs

    session = trimmed.get("pcap_session")
    if isinstance(session, dict):
        trimmed_session = dict(session)
        teardown_events = list(trimmed_session.get("teardown_events", []))
        if max_teardown_events >= 0 and len(teardown_events) > max_teardown_events:
            trimmed_session["teardown_events"] = teardown_events[:max_teardown_events]
            trimmed_session["teardown_events_truncated"] = (
                len(teardown_events) - max_teardown_events
            )
        trimmed["pcap_session"] = trimmed_session

    text = json.dumps(trimmed, sort_keys=True)
    if len(text) <= max_chars:
        return trimmed

    if max_error_logs > 4:
        logs = list(trimmed.get("error_logs", []))
        trimmed["error_logs"] = logs[:4]
        trimmed["error_logs_truncated"] = max(
            trimmed.get("error_logs_truncated", 0),
            len(logs) - 4,
        )

    text = json.dumps(trimmed, sort_keys=True)
    if len(text) <= max_chars:
        return trimmed

    trimmed["pcap_session"] = compact_session(
        trimmed.get("pcap_session"),
        max_teardown_events,
    )
    return trimmed


def build_user_prompt(record: dict[str, Any]) -> str:
    """Create the Groq user prompt for one evidence record."""

    return (
        "Analyze this WiFi failure evidence and produce the requested JSON.\n\n"
        f"Evidence:\n{json.dumps(record, indent=2, sort_keys=True)}"
    )


def call_groq(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_sleep_seconds: float,
) -> tuple[str, str | None]:
    """Call Groq chat completions and return response text plus finish reason."""

    try:
        from groq import Groq
    except ImportError as exc:
        raise ImportError("Install the Groq SDK first: pip install groq") from exc

    client = Groq()
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            return choice.message.content or "", finish_reason
        except Exception as exc:
            message = str(exc).lower()
            retryable = any(
                token in message
                for token in (
                    "rate limit",
                    "rate_limit",
                    "tokens per minute",
                    "too many requests",
                    "429",
                )
            )
            if not retryable or attempt >= retries:
                raise
            time.sleep(retry_sleep_seconds * (attempt + 1))

    raise RuntimeError("Groq request failed after retries")


def parse_llm_json(text: str) -> dict[str, Any]:
    """Parse model JSON, tolerating accidental fenced output."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def parse_error_diagnosis(
    record: dict[str, Any],
    raw_text: str,
    error: json.JSONDecodeError,
    finish_reason: str | None,
) -> dict[str, Any]:
    """Return a structured fallback when the LLM response is invalid JSON."""

    return {
        "mac": record.get("mac"),
        "root_cause": "LLM response was not valid JSON",
        "confidence": 0.0,
        "why": [
            "Groq returned text that Python could not parse as JSON.",
            f"JSON parser error: {error.msg}",
            f"Groq finish_reason: {finish_reason or 'unknown'}",
        ],
        "recommended_action": {
            "summary": "Rerun this record with a larger output token limit or a smaller prompt.",
            "immediate_steps": [
                "Increase max_tokens, for example 600 or 800.",
                "Lower max_error_logs and max_teardown_events if rate limits occur.",
                "Keep temperature at 0.0 for more stable JSON output.",
            ],
            "validation_steps": [
                "Confirm the output row has a parsed diagnosis object.",
                "Check whether finish_reason is length, which indicates truncation.",
            ],
            "if_still_failing": [
                "Run one record and inspect raw_llm_response.",
            ],
            "data_to_collect": [
                "The raw_llm_response field from this output row.",
            ],
        },
        "parse_error": str(error),
        "finish_reason": finish_reason,
        "raw_llm_response": raw_text,
    }


def diagnose_record(
    record: dict[str, Any],
    config: GroqDiagnosisConfig,
) -> dict[str, Any]:
    """Ask Groq to diagnose one evidence record."""

    trimmed = trim_record(
        record,
        config.max_record_chars,
        config.max_error_logs,
        config.max_teardown_events,
    )
    raw_text, finish_reason = call_groq(
        model=config.model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(trimmed),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        retries=config.retries,
        retry_sleep_seconds=config.retry_sleep_seconds,
    )
    try:
        diagnosis = parse_llm_json(raw_text)
    except json.JSONDecodeError as error:
        diagnosis = parse_error_diagnosis(record, raw_text, error, finish_reason)

    return {
        "mac": record.get("mac"),
        "diagnosis": diagnosis,
        "source_evidence": trimmed,
    }


def diagnose_records(
    records: list[dict[str, Any]],
    config: GroqDiagnosisConfig,
) -> list[dict[str, Any]]:
    """Diagnose records sequentially."""

    output_rows: list[dict[str, Any]] = []
    for record in records[:25]:
        output_rows.append(diagnose_record(record, config))
        if config.sleep_seconds:
            time.sleep(config.sleep_seconds)
    return output_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def build_first_prompt(config: GroqDiagnosisConfig) -> str:
    """Build the first prompt for inspection without calling Groq."""

    records = load_jsonl(config.input, limit=1)
    if not records:
        raise ValueError(f"no records found in {config.input}")
    trimmed = trim_record(
        records[0],
        config.max_record_chars,
        config.max_error_logs,
        config.max_teardown_events,
    )
    return build_user_prompt(trimmed)


def run_groq_diagnosis(config: GroqDiagnosisConfig) -> list[dict[str, Any]]:
    """Load evidence JSONL, call Groq, optionally write JSONL, and return rows."""

    records = load_jsonl(config.input, config.limit)
    if not records:
        raise ValueError(f"no records found in {config.input}")
    rows = diagnose_records(records, config)
    if config.output is not None:
        write_jsonl(config.output, rows)
    return rows
