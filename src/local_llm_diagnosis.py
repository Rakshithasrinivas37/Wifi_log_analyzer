"""Use a local/open-source Hugging Face LLM for WiFi diagnosis reasoning.

This module exposes callable service functions for FastAPI/RunPod use. It does
not parse CLI arguments or execute work at import time.
"""

from __future__ import annotations

import json
import re
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
class LocalLlmDiagnosisConfig:
    """Settings for local/open-source LLM diagnosis."""

    input: Path
    output: Path | None = None
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    model_type: str = "causal"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    device_map: str = "auto"
    torch_dtype: str = "auto"
    max_input_tokens: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.0
    limit: int | None = None
    max_record_chars: int = 12000


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load JSONL records."""

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


def compact_session(session: Any) -> Any:
    """Keep the most useful PCAP fields when evidence is too large."""

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


def trim_record(record: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Trim large evidence records before prompting."""

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


def build_user_prompt(record: dict[str, Any]) -> str:
    """Build one diagnosis prompt."""

    return (
        "Analyze this WiFi failure evidence and produce the requested JSON.\n\n"
        f"Evidence:\n{json.dumps(record, indent=2, sort_keys=True)}"
    )


def resolve_torch_dtype(dtype_name: str) -> Any:
    """Resolve a torch dtype from config text."""

    import torch

    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "auto":
        return "auto"
    raise ValueError("torch_dtype must be one of: auto, float16, bfloat16, float32")


def quantization_config(load_in_4bit: bool, load_in_8bit: bool) -> Any:
    """Build optional bitsandbytes quantization config."""

    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose only one of load_in_4bit or load_in_8bit")
    if not load_in_4bit and not load_in_8bit:
        return None

    import torch
    from transformers import BitsAndBytesConfig

    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def load_model(config: LocalLlmDiagnosisConfig) -> tuple[Any, Any]:
    """Load tokenizer and local model."""

    if config.model_type not in {"causal", "seq2seq"}:
        raise ValueError("model_type must be either 'causal' or 'seq2seq'")

    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "device_map": config.device_map,
        "trust_remote_code": True,
        "torch_dtype": resolve_torch_dtype(config.torch_dtype),
    }
    qconfig = quantization_config(config.load_in_4bit, config.load_in_8bit)
    if qconfig is not None:
        model_kwargs["quantization_config"] = qconfig

    if config.model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)

    model.eval()
    return tokenizer, model


def build_model_input(
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    model_type: str,
) -> str:
    """Build either a chat-template prompt or a plain instruction prompt."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if model_type == "causal" and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{system_prompt}\n\n{user_prompt}\n\nJSON:"


def generate_text(
    tokenizer: Any,
    model: Any,
    prompt: str,
    model_type: str,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate text from a local model."""

    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    if not hasattr(model, "hf_device_map"):
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

    do_sample = temperature > 0
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )

    if model_type == "causal":
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def parse_llm_json(text: str) -> dict[str, Any]:
    """Parse model output as JSON, tolerating fenced text."""

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def diagnose_record(
    record: dict[str, Any],
    tokenizer: Any,
    model: Any,
    config: LocalLlmDiagnosisConfig,
) -> dict[str, Any]:
    """Run local LLM diagnosis for one evidence record."""

    trimmed = trim_record(record, config.max_record_chars)
    prompt = build_model_input(
        tokenizer,
        SYSTEM_PROMPT,
        build_user_prompt(trimmed),
        config.model_type,
    )
    text = generate_text(
        tokenizer,
        model,
        prompt,
        config.model_type,
        config.max_input_tokens,
        config.max_new_tokens,
        config.temperature,
    )
    return {
        "mac": record.get("mac"),
        "diagnosis": parse_llm_json(text),
        "source_evidence": trimmed,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def build_first_prompt(config: LocalLlmDiagnosisConfig) -> str:
    """Build the first prompt for inspection without loading the model."""

    records = load_jsonl(config.input, limit=1)
    if not records:
        raise ValueError(f"no records found in {config.input}")
    return build_user_prompt(trim_record(records[0], config.max_record_chars))


def run_local_llm_diagnosis(config: LocalLlmDiagnosisConfig) -> list[dict[str, Any]]:
    """Load evidence JSONL, run local LLM diagnosis, and optionally write JSONL."""

    records = load_jsonl(config.input, config.limit)
    if not records:
        raise ValueError(f"no records found in {config.input}")

    tokenizer, model = load_model(config)
    rows = [diagnose_record(record, tokenizer, model, config) for record in records]
    if config.output is not None:
        write_jsonl(config.output, rows)
    return rows
