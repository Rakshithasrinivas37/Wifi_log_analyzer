"""FLAN-T5 log classification inference.

This module exposes callable functions for service use. It does not parse CLI
arguments or execute work at import time.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from time import perf_counter, time
from typing import Any, TypeVar

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MAC_RE = re.compile(r"\b[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}\b")
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class InferenceConfig:
    """Settings for FLAN-T5 inference."""

    logfile: Path
    model_dir: Path
    output: Path | None = None
    base_model: str = "google/flan-t5-small"
    max_source_length: int = 128
    max_new_tokens: int = 2
    batch_size: int = 16
    device: str = "auto"
    dtype: str = "auto"
    print_memory_every: int = 0
    empty_cache_between_batches: bool = False


@dataclass
class InferenceResult:
    """Result returned by ``run_flan_t5_inference``."""

    rows: list[dict[str, object]]
    elapsed_seconds: float
    generation_seconds: float
    output: str | None
    memory_summary: str


def measure_total_inference_time(func: F) -> F:
    """Decorate an inference function and store total wall-clock seconds."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started_at = perf_counter()
        result = func(*args, **kwargs)
        result.elapsed_seconds = perf_counter() - started_at
        return result

    return wrapper  # type: ignore[return-value]


def count_candidate_log_lines(logfile: Path) -> int:
    """Count non-empty input lines long enough to be considered for inference."""

    with logfile.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if len(line.strip()) > 10)


def get_inference_config(args: tuple[Any, ...], kwargs: dict[str, Any]) -> InferenceConfig | None:
    """Find an InferenceConfig passed to a decorated inference function."""

    if args and isinstance(args[0], InferenceConfig):
        return args[0]
    config = kwargs.get("config")
    if isinstance(config, InferenceConfig):
        return config
    return None


def print_monitoring_event(event: dict[str, object]) -> None:
    """Emit one structured monitoring event for server/container logs."""

    print(json.dumps(event, sort_keys=True), file=sys.stderr)


def monitor_inference_run(func: F) -> F:
    """Decorate inference with structured start/success/failure monitoring."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        config = get_inference_config(args, kwargs)
        started_at = perf_counter()
        base_event: dict[str, object] = {
            "event": "flan_t5_inference",
            "status": "started",
            "timestamp": time(),
        }
        if config is not None:
            candidate_lines: int | str
            try:
                candidate_lines = count_candidate_log_lines(config.logfile)
            except OSError as error:
                candidate_lines = f"unavailable: {error}"
            base_event.update(
                {
                    "logfile": str(config.logfile),
                    "model_dir": str(config.model_dir),
                    "output": str(config.output) if config.output else None,
                    "base_model": config.base_model,
                    "batch_size": config.batch_size,
                    "device": config.device,
                    "dtype": config.dtype,
                    "candidate_lines": candidate_lines,
                }
            )
        print_monitoring_event(base_event)

        try:
            result = func(*args, **kwargs)
        except Exception as error:
            failed_event = dict(base_event)
            failed_event.update(
                {
                    "status": "failed",
                    "elapsed_seconds": perf_counter() - started_at,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if torch.cuda.is_available():
                failed_event["memory_summary"] = cuda_memory_summary("after failed inference")
            print_monitoring_event(failed_event)
            raise

        succeeded_event = dict(base_event)
        succeeded_event.update(
            {
                "status": "succeeded",
                "elapsed_seconds": getattr(result, "elapsed_seconds", perf_counter() - started_at),
                "generation_seconds": getattr(result, "generation_seconds", None),
                "error_rows": len(getattr(result, "rows", [])),
                "result_output": getattr(result, "output", None),
                "memory_summary": getattr(result, "memory_summary", None),
            }
        )
        print_monitoring_event(succeeded_event)
        return result

    return wrapper  # type: ignore[return-value]


def make_prompt(log_body: str) -> str:
    """Build the prompt sent to FLAN-T5."""

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
    """Yield timestamp/log rows for non-empty lines longer than 10 characters."""

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


def is_cuda_oom(error: BaseException) -> bool:
    """Return True when an exception looks like CUDA out-of-memory."""

    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def format_mb(value: int | float) -> str:
    """Format bytes as MiB text."""

    return f"{value / 1024 / 1024:.1f} MiB"


def cuda_memory_summary(label: str) -> str:
    """Return a compact CUDA memory summary for debugging."""

    if not torch.cuda.is_available():
        return f"[memory] {label}: CUDA is not available"

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    free, total = torch.cuda.mem_get_info()
    return (
        f"[memory] {label}: allocated={format_mb(allocated)} "
        f"reserved={format_mb(reserved)} peak_allocated={format_mb(peak_allocated)} "
        f"peak_reserved={format_mb(peak_reserved)} free={format_mb(free)} "
        f"total={format_mb(total)}"
    )


def print_cuda_memory(label: str) -> None:
    """Print CUDA memory to stderr for notebook/server logs."""

    print(cuda_memory_summary(label), file=sys.stderr)


def debug_cuda_oom(step_mb: int = 512) -> None:
    """Intentionally trigger CUDA OOM so the failure can be inspected."""

    if not torch.cuda.is_available():
        raise RuntimeError("debug_cuda_oom requires a CUDA GPU")
    if step_mb < 1:
        raise ValueError("step_mb must be at least 1")

    print_cuda_memory("before debug OOM")
    tensors: list[torch.Tensor] = []
    elements = step_mb * 1024 * 1024 // 2
    try:
        while True:
            tensors.append(torch.empty(elements, dtype=torch.float16, device="cuda"))
            print_cuda_memory(f"allocated debug block {len(tensors)}")
    except RuntimeError as error:
        if not is_cuda_oom(error):
            raise
        print(str(error), file=sys.stderr)
        tensors.clear()
        torch.cuda.empty_cache()
        print_cuda_memory("after debug OOM cleanup")


class FlanT5LogClassifier:
    """Load a FLAN-T5 LoRA adapter and classify log bodies."""

    def __init__(
        self,
        model_dir: Path,
        base_model: str,
        max_source_length: int,
        max_new_tokens: int,
        device: str,
        dtype: str,
    ) -> None:
        self.torch = torch
        self.max_source_length = max_source_length
        self.max_new_tokens = max_new_tokens
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        self.torch_dtype = self._resolve_dtype(dtype)

        print(
            f"loading model on device={self.device} dtype={self.torch_dtype}",
            file=sys.stderr,
        )
        print_cuda_memory("before model load")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        base = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(base, model_dir).to(self.device)
        self.model.eval()
        print_cuda_memory("after model load")

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        """Pick a torch dtype for the current device."""

        if self.device != "cuda":
            return torch.float32
        if dtype in {"auto", "fp16"}:
            return torch.float16
        if dtype == "fp32":
            return torch.float32
        raise ValueError("dtype must be one of: auto, fp16, fp32")

    def classify(self, log_body: str) -> str:
        """Generate one label for a log body."""

        return self.classify_batch([log_body])[0]

    def classify_batch(self, log_bodies: list[str]) -> list[str]:
        """Generate one label per log body."""

        inputs = self.tokenizer(
            [make_prompt(log_body) for log_body in log_bodies],
            return_tensors="pt",
            padding=True,
            max_length=self.max_source_length,
            truncation=True,
        ).to(self.device)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [normalize_label(text) for text in decoded]


def iter_inference_rows(
    logfile: Path,
    classifier: FlanT5LogClassifier,
    batch_size: int,
    print_memory_every: int = 0,
    empty_cache_between_batches: bool = False,
) -> Iterator[dict[str, object]]:
    """Classify usable log lines and yield only predicted error rows."""

    processed_batches = 0
    with logfile.open("r", encoding="utf-8") as file:
        for row_batch in batched(iter_log_rows(file), batch_size):
            processed_batches += 1
            predictions = classifier.classify_batch([row["log"] for row in row_batch])
            for row, prediction in zip(row_batch, predictions):
                if prediction != "error":
                    continue
                yield {
                    "timestamp": row["timestamp"],
                    "log": row["log"],
                    "prediction": prediction,
                    "mac_addresses": extract_mac_addresses(row["log"]),
                }
            if empty_cache_between_batches and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if print_memory_every and processed_batches % print_memory_every == 0:
                print_cuda_memory(f"after batch {processed_batches}")


def collect_inference_rows(
    logfile: Path,
    classifier: FlanT5LogClassifier,
    batch_size: int,
    print_memory_every: int = 0,
    empty_cache_between_batches: bool = False,
) -> tuple[list[dict[str, object]], float]:
    """Run inference for the whole file and collect predicted error rows."""

    started_at = perf_counter()
    rows = list(
        iter_inference_rows(
            logfile,
            classifier,
            batch_size,
            print_memory_every,
            empty_cache_between_batches,
        )
    )
    return rows, perf_counter() - started_at


def write_jsonl(rows: Iterable[dict[str, object]], output: Path) -> None:
    """Write prediction rows as JSONL."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


@monitor_inference_run
@measure_total_inference_time
def run_flan_t5_inference(config: InferenceConfig) -> InferenceResult:
    """Load the model, run inference, optionally write JSONL, and return rows."""

    classifier = FlanT5LogClassifier(
        model_dir=config.model_dir,
        base_model=config.base_model,
        max_source_length=config.max_source_length,
        max_new_tokens=config.max_new_tokens,
        device=config.device,
        dtype=config.dtype,
    )
    try:
        rows, generation_seconds = collect_inference_rows(
            config.logfile,
            classifier,
            config.batch_size,
            config.print_memory_every,
            config.empty_cache_between_batches,
        )
    except RuntimeError as error:
        if torch.cuda.is_available():
            print_cuda_memory("at runtime error")
            torch.cuda.empty_cache()
        raise error

    if config.output is not None:
        write_jsonl(rows, config.output)
    return InferenceResult(
        rows=rows,
        elapsed_seconds=0.0,
        generation_seconds=generation_seconds,
        output=str(config.output) if config.output else None,
        memory_summary=cuda_memory_summary("after inference"),
    )
