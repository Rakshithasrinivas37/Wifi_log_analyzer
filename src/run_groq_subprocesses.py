"""Run Groq diagnosis in parallel using Python subprocesses.

This script splits a JSONL diagnosis file into chunks, starts multiple child
processes that each run ``src.groq_diagnosis``, then merges the child outputs
back into one JSONL file.

Example:

```
python -m src.run_groq_subprocesses \
  --input src/diagnosis.jsonl \
  --output groq_diagnosis.jsonl \
  --model llama-3.1-8b-instant \
  --workers 4 \
  --max-tokens 500
```
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the subprocess runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("diagnosis.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("groq_diagnosis.jsonl"))
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--max-record-chars", type=int, default=12000)
    parser.add_argument("--max-error-logs", type=int, default=8)
    parser.add_argument("--max-teardown-events", type=int, default=8)
    parser.add_argument("--child-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--start-delay-seconds", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=15.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary chunk files for debugging.",
    )
    return parser.parse_args()


def load_lines(path: Path, limit: int | None) -> list[str]:
    """Load non-empty JSONL lines from the input file."""

    lines: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            lines.append(line if line.endswith("\n") else line + "\n")
            if limit is not None and len(lines) >= limit:
                break
    return lines


def split_contiguous(lines: list[str], workers: int) -> list[list[str]]:
    """Split input lines into contiguous chunks."""

    if workers < 1:
        raise SystemExit("--workers must be at least 1")
    workers = min(workers, len(lines))
    chunk_size = (len(lines) + workers - 1) // workers
    return [lines[index : index + chunk_size] for index in range(0, len(lines), chunk_size)]


def write_chunk(path: Path, lines: list[str]) -> None:
    """Write one chunk JSONL file."""

    with path.open("w", encoding="utf-8") as file:
        file.writelines(lines)


def build_child_command(
    args: argparse.Namespace,
    chunk_input: Path,
    chunk_output: Path,
) -> list[str]:
    """Build the child process command."""

    return [
        sys.executable,
        "-m",
        "src.groq_diagnosis",
        "--input",
        str(chunk_input),
        "--output",
        str(chunk_output),
        "--model",
        args.model,
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--max-record-chars",
        str(args.max_record_chars),
        "--max-error-logs",
        str(args.max_error_logs),
        "--max-teardown-events",
        str(args.max_teardown_events),
        "--sleep-seconds",
        str(args.child_sleep_seconds),
        "--retries",
        str(args.retries),
        "--retry-sleep-seconds",
        str(args.retry_sleep_seconds),
        "--concurrency",
        "1",
    ]


def run_children(
    args: argparse.Namespace,
    chunk_inputs: list[Path],
    chunk_outputs: list[Path],
) -> None:
    """Start child processes and fail if any child fails."""

    env = os.environ.copy()
    processes: list[tuple[int, subprocess.Popen[str]]] = []
    for index, (chunk_input, chunk_output) in enumerate(
        zip(chunk_inputs, chunk_outputs),
        start=1,
    ):
        command = build_child_command(args, chunk_input, chunk_output)
        process = subprocess.Popen(
            command,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes.append((index, process))
        print(f"started worker {index}/{len(chunk_inputs)} pid={process.pid}")
        if args.start_delay_seconds:
            import time

            time.sleep(args.start_delay_seconds)

    failures: list[str] = []
    for index, process in processes:
        stdout, stderr = process.communicate()
        if stdout:
            print(f"\n--- worker {index} stdout ---\n{stdout.rstrip()}")
        if stderr:
            print(f"\n--- worker {index} stderr ---\n{stderr.rstrip()}", file=sys.stderr)
        if process.returncode != 0:
            failures.append(f"worker {index} failed with exit code {process.returncode}")

    if failures:
        raise SystemExit("\n".join(failures))


def merge_outputs(output: Path, chunk_outputs: list[Path]) -> int:
    """Merge child JSONL outputs in original chunk order."""

    output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output.open("w", encoding="utf-8") as merged:
        for chunk_output in chunk_outputs:
            with chunk_output.open("r", encoding="utf-8") as chunk:
                for line in chunk:
                    if not line.strip():
                        continue
                    merged.write(line if line.endswith("\n") else line + "\n")
                    row_count += 1
    return row_count


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    lines = load_lines(args.input, args.limit)
    if not lines:
        raise SystemExit(f"no records found in {args.input}")

    temp_context = None
    if args.keep_temp:
        temp_dir = Path(tempfile.mkdtemp(prefix="groq_chunks_"))
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="groq_chunks_")
        temp_dir = Path(temp_context.name)
    try:
        chunks = split_contiguous(lines, args.workers)
        chunk_inputs: list[Path] = []
        chunk_outputs: list[Path] = []
        for index, chunk_lines in enumerate(chunks):
            chunk_input = temp_dir / f"chunk_{index:03d}.jsonl"
            chunk_output = temp_dir / f"chunk_{index:03d}.out.jsonl"
            write_chunk(chunk_input, chunk_lines)
            chunk_inputs.append(chunk_input)
            chunk_outputs.append(chunk_output)

        run_children(args, chunk_inputs, chunk_outputs)
        row_count = merge_outputs(args.output, chunk_outputs)
        print(f"wrote {row_count} Groq diagnoses to {args.output}")
        if args.keep_temp:
            print(f"kept temp files in {temp_dir}")
    finally:
        if temp_context is not None:
            temp_context.cleanup()


if __name__ == "__main__":
    main()
