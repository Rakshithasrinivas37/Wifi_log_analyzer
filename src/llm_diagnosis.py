"""Use an LLM to turn PCAP/log evidence JSONL into detailed remediation plans.

Input rows should come from ``src.pcap_analysis``. Each input line is one JSON
dictionary containing a MAC address, correlated error logs, and PCAP session
evidence such as EAPOL counts, DHCP packets, teardown reason codes, and
reason-code hints.

Example:

```
python -m src.llm_diagnosis \
  --input diagnosis.jsonl \
  --output llm_diagnosis.jsonl \
  --model gpt-4.1-mini
```
"""

from __future__ import annotations

import argparse
import json
import time
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


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("diagnosis.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("llm_diagnosis.jsonl"))
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of input records to process.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional pause between API calls for rate-limit friendliness.",
    )
    parser.add_argument(
        "--max-record-chars",
        type=int,
        default=12000,
        help="Trim very large evidence records before sending to the LLM.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first prompt payload and exit without calling the API.",
    )
    return parser.parse_args()


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
                raise SystemExit(f"{path}:{line_no} is not valid JSON") from exc
            if limit is not None and len(records) >= limit:
                break
    return records


def trim_record(record: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Trim evidence records to keep prompts bounded."""

    text = json.dumps(record, sort_keys=True)
    if len(text) <= max_chars:
        return record

    trimmed = dict(record)
    logs = list(trimmed.get("error_logs", []))
    if len(logs) > 8:
        trimmed["error_logs"] = logs[:8]
        trimmed["error_logs_truncated"] = len(logs) - 8

    text = json.dumps(trimmed, sort_keys=True)
    if len(text) <= max_chars:
        return trimmed

    trimmed["pcap_session"] = compact_session(trimmed.get("pcap_session"))
    return trimmed


def compact_session(session: Any) -> Any:
    """Keep the most diagnostic PCAP fields when a record is too large."""

    if not isinstance(session, dict):
        return session
    return {
        "first_seen": session.get("first_seen"),
        "last_seen": session.get("last_seen"),
        "packet_count": session.get("packet_count"),
        "reason_codes": session.get("reason_codes"),
        "reason_code_hints": session.get("reason_code_hints"),
        "teardown_events": session.get("teardown_events", [])[:8],
    }


def build_user_prompt(record: dict[str, Any]) -> str:
    """Create the user prompt for one evidence record."""

    return (
        "Analyze this WiFi failure evidence and produce the requested JSON.\n\n"
        f"Evidence:\n{json.dumps(record, indent=2, sort_keys=True)}"
    )


def call_openai(model: str, system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI and return response text.

    Uses the Responses API when available in the installed SDK, with a
    Chat Completions fallback for older Colab environments.
    """

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Install the OpenAI SDK first: pip install openai"
        ) from exc

    client = OpenAI()
    if hasattr(client, "responses"):
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return response.output_text

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def parse_llm_json(text: str) -> dict[str, Any]:
    """Parse an LLM JSON response, allowing accidental fenced output."""

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


def enrich_with_llm(record: dict[str, Any], model: str, max_record_chars: int) -> dict[str, Any]:
    """Return one LLM diagnosis row for one evidence record."""

    trimmed = trim_record(record, max_record_chars)
    raw_text = call_openai(model, SYSTEM_PROMPT, build_user_prompt(trimmed))
    diagnosis = parse_llm_json(raw_text)
    return {
        "mac": record.get("mac"),
        "diagnosis": diagnosis,
        "source_evidence": trimmed,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    records = load_jsonl(args.input, args.limit)
    if not records:
        raise SystemExit(f"no records found in {args.input}")

    first_prompt = build_user_prompt(trim_record(records[0], args.max_record_chars))
    if args.dry_run:
        print(SYSTEM_PROMPT)
        print("\n--- USER PROMPT ---\n")
        print(first_prompt)
        return

    output_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        output_rows.append(enrich_with_llm(record, args.model, args.max_record_chars))
        print(f"processed {index}/{len(records)} mac={record.get('mac')}", flush=True)
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)

    write_jsonl(args.output, output_rows)
    print(f"wrote {len(output_rows)} LLM diagnoses to {args.output}")


if __name__ == "__main__":
    main()
