#!/usr/bin/env python3
"""Run inference with TensorRT-LLM encoder-decoder engines.

The script supports two useful paths:

* classify a timestamped WiFi log file and write JSONL rows compatible with the
  rest of this project; or
* generate from one or more raw text prompts for quick engine smoke tests.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAC_RE = re.compile(r"\b[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}\b")


def make_prompt(log_body: str) -> str:
    """Build the prompt sent to FLAN-T5-style engines."""

    return (
        "Classify this WiFi log line as normal or error.\n"
        f"Log line: {log_body}\n"
        "Answer with exactly one label:"
    )


def normalize_label(text: str) -> str:
    """Normalize generated text to a binary label when possible."""

    normalized = text.strip().lower()
    if normalized.startswith("normal"):
        return "normal"
    if normalized.startswith("error"):
        return "error"
    return normalized


def extract_mac_addresses(text: str) -> list[str]:
    """Return unique MAC addresses from text in first-seen order."""

    seen: set[str] = set()
    macs: list[str] = []
    for match in MAC_RE.finditer(text):
        mac = match.group(0).lower()
        if mac not in seen:
            seen.add(mac)
            macs.append(mac)
    return macs


def split_timestamp_and_log(line: str) -> tuple[str, str]:
    """Split one timestamped log line into timestamp and log body."""

    stripped = line.strip()
    timestamp, separator, log_body = stripped.partition(" ")
    if not separator:
        return "", stripped
    return timestamp, log_body.strip()


def iso_to_epoch(timestamp: str) -> str:
    """Convert an ISO-8601 timestamp to seconds.microseconds epoch text."""

    normalized = timestamp.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    epoch = dt.astimezone(timezone.utc).timestamp()
    return f"{epoch:.6f}"


def iter_log_rows(lines: Iterable[str]) -> Iterator[dict[str, str]]:
    """Yield timestamp/log rows for non-empty lines long enough for inference."""

    for line in lines:
        stripped = line.strip()
        if len(stripped) <= 10:
            continue
        timestamp, log_body = split_timestamp_and_log(stripped)
        if not log_body:
            continue
        yield {
            "timestamp": iso_to_epoch(timestamp),
            "log": log_body,
        }


def batched(
    rows: Iterable[dict[str, str]],
    batch_size: int,
) -> Iterator[list[dict[str, str]]]:
    """Yield fixed-size row batches."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    batch: list[dict[str, str]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_jsonl(rows: Iterable[dict[str, object]], output: Path) -> None:
    """Write prediction rows as JSONL."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


@dataclass(frozen=True)
class EngineTokenIds:
    """Generation token IDs used by the TensorRT-LLM decoder."""

    decoder_start_token_id: int
    pad_token_id: int
    eos_token_id: int
    bos_token_id: int | None


@dataclass(frozen=True)
class EngineMetadata:
    """Basic metadata used to validate an encoder-decoder TRT-LLM engine."""

    engine_dir: str
    version: str | None
    encoder_architecture: str | None
    decoder_architecture: str | None
    encoder_model_type: str | None
    decoder_model_type: str | None

    @property
    def is_enc_dec_model(self) -> bool:
        return (
            self.encoder_architecture == "EncoderModel"
            and self.decoder_architecture == "DecoderModel"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "engine_dir": self.engine_dir,
            "version": self.version,
            "encoder_architecture": self.encoder_architecture,
            "decoder_architecture": self.decoder_architecture,
            "encoder_model_type": self.encoder_model_type,
            "decoder_model_type": self.decoder_model_type,
            "is_enc_dec_model": self.is_enc_dec_model,
        }


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def existing_engine_dir(path: Path) -> bool:
    """Return True when ``path`` contains encoder and decoder engines."""

    return (
        (path / "encoder" / "config.json").is_file()
        and (path / "encoder" / "rank0.engine").is_file()
        and (path / "decoder" / "config.json").is_file()
        and (path / "decoder" / "rank0.engine").is_file()
    )


def read_engine_metadata(engine_dir: Path) -> EngineMetadata:
    """Read encoder/decoder config metadata from a TensorRT-LLM engine."""

    encoder_config = read_json(engine_dir / "encoder" / "config.json")
    decoder_config = read_json(engine_dir / "decoder" / "config.json")
    encoder_pretrained = encoder_config.get("pretrained_config", {})
    decoder_pretrained = decoder_config.get("pretrained_config", {})

    return EngineMetadata(
        engine_dir=str(engine_dir),
        version=decoder_config.get("version") or encoder_config.get("version"),
        encoder_architecture=encoder_pretrained.get("architecture"),
        decoder_architecture=decoder_pretrained.get("architecture"),
        encoder_model_type=encoder_pretrained.get("model_type"),
        decoder_model_type=decoder_pretrained.get("model_type"),
    )


def validate_enc_dec_engine(engine_dir: Path) -> EngineMetadata:
    """Return engine metadata or raise when the engine is not encoder-decoder."""

    metadata = read_engine_metadata(engine_dir)
    if not metadata.is_enc_dec_model:
        raise ValueError(
            "The selected TensorRT-LLM engine is not an encoder-decoder engine: "
            f"{json.dumps(metadata.to_dict(), sort_keys=True)}"
        )
    return metadata


def engine_dir_candidates() -> list[Path]:
    """Return likely engine locations for local and RunPod usage."""

    candidates: list[Path] = []
    if os.environ.get("TRT_ENGINE_DIR"):
        candidates.append(Path(os.environ["TRT_ENGINE_DIR"]))
    candidates.extend(
        [
            Path.cwd() / "trt_engine" / "t5-small",
            Path.cwd().parent / "trt_engine" / "t5-small",
            PROJECT_ROOT / "trt_engine" / "t5-small",
            PROJECT_ROOT.parent / "trt_engine" / "t5-small",
            Path("/workspace/trt_engine/t5-small"),
            Path("/workspace/wifi-log-analyzer/trt_engine/t5-small"),
        ]
    )
    return candidates


def resolve_tokenizer(tokenizer: str | None) -> str:
    """Resolve the tokenizer path/name, preferring local project artifacts."""

    if tokenizer:
        return tokenizer

    candidates: list[Path] = []
    if os.environ.get("TRT_TOKENIZER"):
        return os.environ["TRT_TOKENIZER"]
    candidates.extend(
        [
            PROJECT_ROOT / "models" / "flan-t5-log-merged-model",
            PROJECT_ROOT / "models" / "flan-t5-log-lora-model",
            Path("/workspace/wifi-log-analyzer/models/flan-t5-log-merged-model"),
            Path("/workspace/wifi-log-analyzer/models/flan-t5-log-lora-model"),
        ]
    )
    for candidate in candidates:
        if (candidate / "tokenizer.json").is_file() or (
            candidate / "tokenizer_config.json"
        ).is_file():
            return str(candidate)
    return "google/flan-t5-small"


def resolve_engine_dir(engine_dir: Path | None) -> Path:
    """Resolve an engine directory that contains ``encoder/`` and ``decoder/``."""

    candidates = [engine_dir] if engine_dir is not None else engine_dir_candidates()
    checked: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        expanded = candidate.expanduser().resolve()
        checked.append(str(expanded))
        if existing_engine_dir(expanded):
            return expanded

        child_matches = (
            [
                child
                for child in expanded.iterdir()
                if child.is_dir() and existing_engine_dir(child)
            ]
            if expanded.is_dir()
            else []
        )
        if len(child_matches) == 1:
            return child_matches[0].resolve()

    checked_text = "\n  - ".join(checked) if checked else "(no paths checked)"
    raise FileNotFoundError(
        "Could not find a TensorRT-LLM encoder-decoder engine directory. "
        "Pass --engine-dir pointing at a folder that contains encoder/ and "
        f"decoder/.\nChecked:\n  - {checked_text}"
    )


def first_int(*values: Any, default: int | None = None) -> int:
    """Return the first non-None value as an int."""

    for value in values:
        if value is not None:
            return int(value)
    if default is None:
        raise ValueError("no token ID value is available")
    return default


def read_engine_token_ids(engine_dir: Path, tokenizer: Any) -> EngineTokenIds:
    """Read decoder generation IDs from the engine config, with tokenizer fallbacks."""

    config_path = engine_dir / "decoder" / "config.json"
    config = read_json(config_path)
    pretrained_config = config.get("pretrained_config", {})

    pad_token_id = first_int(
        pretrained_config.get("pad_token_id"),
        getattr(tokenizer, "pad_token_id", None),
        default=0,
    )
    eos_token_id = first_int(
        pretrained_config.get("eos_token_id"),
        getattr(tokenizer, "eos_token_id", None),
        default=1,
    )
    decoder_start_token_id = first_int(
        pretrained_config.get("decoder_start_token_id"),
        getattr(tokenizer, "decoder_start_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        default=pad_token_id,
    )
    bos_value = pretrained_config.get("bos_token_id")
    bos_token_id = (
        int(bos_value)
        if bos_value is not None
        else getattr(tokenizer, "bos_token_id", None)
    )

    return EngineTokenIds(
        decoder_start_token_id=decoder_start_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
    )


class TrtLlmEncDecGenerator:
    """Small wrapper around TensorRT-LLM's ``EncDecModelRunner``."""

    def __init__(
        self,
        engine_dir: Path,
        tokenizer_name_or_path: str,
        max_source_length: int,
        max_new_tokens: int,
        num_beams: int,
        log_level: str,
        debug_mode: bool,
    ) -> None:
        self.engine_dir = engine_dir
        self.max_source_length = max_source_length
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.debug_mode = debug_mode

        try:
            import torch
            from tensorrt_llm import logger
            from tensorrt_llm.runtime import EncDecModelRunner
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "TensorRT-LLM inference requires torch, transformers, and "
                "tensorrt_llm to be installed in this environment."
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT-LLM inference requires a CUDA GPU.")

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.token_ids = read_engine_token_ids(engine_dir, self.tokenizer)

        logger.set_level(log_level)
        self.runner = EncDecModelRunner.from_engine(
            engine_name=engine_dir.name,
            engine_dir=str(engine_dir),
            debug_mode=debug_mode,
        )

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate decoded text for a batch of prompts."""

        if not prompts:
            return []

        tokenized_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
        )
        input_ids = tokenized_inputs.input_ids.type(self.torch.IntTensor).to("cuda")
        attention_mask = tokenized_inputs.attention_mask
        decoder_input_ids = self.torch.IntTensor(
            [[self.token_ids.decoder_start_token_id]]
        ).to("cuda")
        decoder_input_ids = decoder_input_ids.repeat((input_ids.shape[0], 1))

        with self.torch.inference_mode():
            output = self.runner.generate(
                encoder_input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                bos_token_id=self.token_ids.bos_token_id,
                pad_token_id=self.token_ids.pad_token_id,
                eos_token_id=self.token_ids.eos_token_id,
                debug_mode=self.debug_mode,
                return_dict=True,
                attention_mask=attention_mask,
            )
        self.torch.cuda.synchronize()

        output_ids = output["output_ids"][:, 0, :]
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def classify_batch(self, log_bodies: list[str]) -> list[str]:
        """Generate normalized labels for log bodies."""

        outputs = self.generate_batch(
            [make_prompt(log_body) for log_body in log_bodies]
        )
        return [normalize_label(text) for text in outputs]


def iter_trt_log_rows(
    logfile: Path,
    generator: TrtLlmEncDecGenerator,
    batch_size: int,
    include_all_predictions: bool,
) -> Iterable[dict[str, object]]:
    """Yield prediction rows for a timestamped WiFi log file."""

    with logfile.open("r", encoding="utf-8") as file:
        for row_batch in batched(iter_log_rows(file), batch_size):
            predictions = generator.classify_batch([row["log"] for row in row_batch])
            for row, prediction in zip(row_batch, predictions):
                if not include_all_predictions and prediction != "error":
                    continue
                yield {
                    "timestamp": row["timestamp"],
                    "log": row["log"],
                    "prediction": prediction,
                    "mac_addresses": extract_mac_addresses(row["log"]),
                }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with TensorRT-LLM encoder-decoder engines."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--inspect-engine",
        action="store_true",
        help="Only inspect engine metadata and report whether it is enc-dec.",
    )
    input_group.add_argument(
        "--logfile",
        type=Path,
        help="Timestamped WiFi log text file.",
    )
    input_group.add_argument(
        "--text",
        action="append",
        help="Raw prompt text for a quick generation test. Can be passed multiple times.",
    )
    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=None,
        help=(
            "Engine directory containing encoder/ and decoder/. Defaults to "
            "TRT_ENGINE_DIR or common RunPod paths."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Tokenizer name or local tokenizer path that matches the engine. "
            "Defaults to TRT_TOKENIZER, local project tokenizer artifacts, "
            "then google/flan-t5-small."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSONL output path for --logfile mode.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument(
        "--include-all-predictions",
        action="store_true",
        help="Write normal and error predictions instead of only error rows.",
    )
    parser.add_argument("--log-level", default="error")
    parser.add_argument("--debug-mode", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    engine_dir = resolve_engine_dir(args.engine_dir)
    engine_metadata = validate_enc_dec_engine(engine_dir)
    if args.inspect_engine:
        print(json.dumps(engine_metadata.to_dict(), sort_keys=True))
        return 0

    tokenizer = resolve_tokenizer(args.tokenizer)
    started_at = perf_counter()
    generator = TrtLlmEncDecGenerator(
        engine_dir=engine_dir,
        tokenizer_name_or_path=tokenizer,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        log_level=args.log_level,
        debug_mode=args.debug_mode,
    )

    if args.text:
        outputs = generator.generate_batch(args.text)
        for prompt, output in zip(args.text, outputs):
            print(
                json.dumps(
                    {
                        "prompt": prompt,
                        "output": output,
                        "is_enc_dec_model": engine_metadata.is_enc_dec_model,
                    },
                    sort_keys=True,
                )
            )
        return 0

    rows = list(
        iter_trt_log_rows(
            logfile=args.logfile,
            generator=generator,
            batch_size=args.batch_size,
            include_all_predictions=args.include_all_predictions,
        )
    )
    if args.output:
        write_jsonl(rows, args.output)
    else:
        for row in rows:
            print(json.dumps(row, sort_keys=True))

    summary = {
        "engine_dir": str(engine_dir),
        "is_enc_dec_model": engine_metadata.is_enc_dec_model,
        "encoder_architecture": engine_metadata.encoder_architecture,
        "decoder_architecture": engine_metadata.decoder_architecture,
        "logfile": str(args.logfile),
        "output": str(args.output) if args.output else None,
        "rows": len(rows),
        "elapsed_seconds": perf_counter() - started_at,
    }
    print(json.dumps(summary, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
