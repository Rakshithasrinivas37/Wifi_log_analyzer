#!/usr/bin/env python3
"""Run one WiFi analyzer pipeline shard inside a Kubernetes indexed Job.

Kubernetes sets ``JOB_COMPLETION_INDEX`` for Indexed Jobs. This script uses that
index as the node rank, so rank 0 processes ``wifi_logs.txt`` and rank 1
processes ``wifi_logs-1.txt`` by default.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag."""

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    """Read an integer environment value with a default."""

    value = os.environ.get(name)
    return default if value is None else int(value)


def resolve_path(value: str, base_dir: Path) -> Path:
    """Resolve absolute paths unchanged and relative paths under ``base_dir``."""

    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def split_csv_paths(value: str, base_dir: Path) -> list[Path]:
    """Parse comma-separated path lists from environment variables."""

    return [
        resolve_path(item.strip(), base_dir)
        for item in value.split(",")
        if item.strip()
    ]


def path_for_rank(paths: list[Path], rank: int, label: str) -> Path:
    """Return the path assigned to the current rank."""

    if rank < 0 or rank >= len(paths):
        raise ValueError(f"{label} does not define an entry for rank {rank}")
    return paths[rank]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows, ignoring blank lines."""

    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


APP_DIR = Path(os.environ.get("WIFI_ANALYZER_APP_DIR", "/app")).resolve()
WORKSPACE = Path(os.environ.get("WIFI_ANALYZER_WORKSPACE", "/workspace")).resolve()

# Prefer Kubernetes Indexed Job rank, while allowing manual NODE_RANK overrides.
NODE_RANK = env_int("NODE_RANK", env_int("JOB_COMPLETION_INDEX", 0))
WORLD_SIZE = env_int("WORLD_SIZE", 2)

LOG_FILES = split_csv_paths(
    os.environ.get(
        "CLUSTER_LOG_FILES",
        "data/inputs/wifi_logs.txt,data/inputs/wifi_logs-1.txt",
    ),
    APP_DIR,
)
PCAP_FILES = split_csv_paths(
    os.environ.get(
        "CLUSTER_PCAP_FILES",
        "data/inputs/wifi_logs.pcap,data/inputs/wifi_logs-1.pcap",
    ),
    APP_DIR,
)

INPUT_LOG = path_for_rank(LOG_FILES, NODE_RANK, "CLUSTER_LOG_FILES")
INPUT_PCAP = path_for_rank(PCAP_FILES, NODE_RANK, "CLUSTER_PCAP_FILES")
MODEL_DIR = resolve_path(
    os.environ.get("CLUSTER_MODEL_DIR", "models/flan-t5-log-lora-model"),
    APP_DIR,
)

RUN_PCAP = env_bool("CLUSTER_RUN_PCAP", True)
RUN_GROQ = env_bool("CLUSTER_RUN_GROQ", False)
MERGE_OUTPUTS = env_bool("CLUSTER_MERGE_OUTPUTS", False)
MERGE_KIND = os.environ.get("CLUSTER_MERGE_KIND", "last").strip().lower()

INFERENCE_DEVICE = os.environ.get("CLUSTER_INFERENCE_DEVICE", "cpu")
INFERENCE_DTYPE = os.environ.get("CLUSTER_INFERENCE_DTYPE", "fp32")
BATCH_SIZE = env_int("CLUSTER_BATCH_SIZE", 4)

NODE_DIR = WORKSPACE / f"node_{NODE_RANK}"
INFERENCE_OUT = NODE_DIR / "inference.jsonl"
DIAGNOSIS_OUT = NODE_DIR / "diagnosis.jsonl"
GROQ_OUT = NODE_DIR / "groq_diagnosis.jsonl"
DONE_FLAG = WORKSPACE / f"node_{NODE_RANK}_done.flag"
FINAL_OUT = WORKSPACE / "final_results.jsonl"


def validate_inputs() -> None:
    """Fail early when required project assets are missing."""

    missing = [
        path
        for path in [INPUT_LOG, MODEL_DIR]
        if not path.exists()
    ]
    if RUN_PCAP and not INPUT_PCAP.exists():
        missing.append(INPUT_PCAP)
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required cluster input(s):\n{formatted}")


def run_stage1() -> None:
    """Run FLAN-T5 inference for this node's assigned log file."""

    print(f"[Node {NODE_RANK}] Stage 1: FLAN-T5 inference")
    print(f"[Node {NODE_RANK}] Log file: {INPUT_LOG}")

    from src.inference_flan_t5 import InferenceConfig, run_flan_t5_inference

    result = run_flan_t5_inference(
        InferenceConfig(
            logfile=INPUT_LOG,
            model_dir=MODEL_DIR,
            output=INFERENCE_OUT,
            device=INFERENCE_DEVICE,
            dtype=INFERENCE_DTYPE,
            batch_size=BATCH_SIZE,
        )
    )
    print(
        f"[Node {NODE_RANK}] Stage 1 done: "
        f"{len(result.rows)} error rows in {result.elapsed_seconds:.1f}s"
    )


def run_stage2() -> None:
    """Correlate this node's error logs with its assigned PCAP file."""

    if not RUN_PCAP:
        print(f"[Node {NODE_RANK}] Stage 2 skipped: CLUSTER_RUN_PCAP=false")
        return

    print(f"[Node {NODE_RANK}] Stage 2: PCAP analysis")
    print(f"[Node {NODE_RANK}] PCAP file: {INPUT_PCAP}")

    from src.pcap_analysis import PcapAnalysisConfig, run_pcap_analysis

    records = run_pcap_analysis(
        PcapAnalysisConfig(
            errors_jsonl=INFERENCE_OUT,
            pcap=INPUT_PCAP,
            output=DIAGNOSIS_OUT,
            window_seconds=float(os.environ.get("CLUSTER_WINDOW_SECONDS", "3.0")),
        )
    )
    print(f"[Node {NODE_RANK}] Stage 2 done: {len(records)} diagnosis records")


def run_stage3() -> None:
    """Optionally run Groq diagnosis for this node's PCAP evidence."""

    if not RUN_GROQ:
        print(f"[Node {NODE_RANK}] Stage 3 skipped: CLUSTER_RUN_GROQ=false")
        return

    print(f"[Node {NODE_RANK}] Stage 3: Groq diagnosis")

    from src.groq_diagnosis import GroqDiagnosisConfig, run_groq_diagnosis

    rows = run_groq_diagnosis(
        GroqDiagnosisConfig(
            input=DIAGNOSIS_OUT,
            output=GROQ_OUT,
            model=os.environ.get("CLUSTER_GROQ_MODEL", "llama-3.1-8b-instant"),
            max_tokens=env_int("CLUSTER_GROQ_MAX_TOKENS", 600),
            retries=env_int("CLUSTER_GROQ_RETRIES", 3),
            retry_sleep_seconds=float(
                os.environ.get("CLUSTER_GROQ_RETRY_SLEEP_SECONDS", "15.0")
            ),
        )
    )
    print(f"[Node {NODE_RANK}] Stage 3 done: {len(rows)} Groq diagnoses")


def output_for_rank(rank: int) -> Path:
    """Return the output file that should be merged for a rank."""

    if MERGE_KIND == "groq":
        return WORKSPACE / f"node_{rank}" / "groq_diagnosis.jsonl"
    if MERGE_KIND == "diagnosis":
        return WORKSPACE / f"node_{rank}" / "diagnosis.jsonl"
    if MERGE_KIND == "inference":
        return WORKSPACE / f"node_{rank}" / "inference.jsonl"
    if RUN_GROQ:
        return WORKSPACE / f"node_{rank}" / "groq_diagnosis.jsonl"
    if RUN_PCAP:
        return WORKSPACE / f"node_{rank}" / "diagnosis.jsonl"
    return WORKSPACE / f"node_{rank}" / "inference.jsonl"


def mark_done() -> None:
    """Write this node's completion flag for the merge step."""

    DONE_FLAG.write_text("done\n", encoding="utf-8")
    print(f"[Node {NODE_RANK}] Done flag created: {DONE_FLAG}")


def print_node_output() -> None:
    """Print this node's final JSONL output so Kubernetes logs keep the result."""

    node_output = output_for_rank(NODE_RANK)
    rows = load_jsonl(node_output)
    print(f"[Node {NODE_RANK}] Final node output: {node_output}")
    print(f"[Node {NODE_RANK}] Final node row count: {len(rows)}")
    for row in rows[:5]:
        print(json.dumps(row, sort_keys=True))


def wait_and_merge() -> None:
    """Let rank 0 wait for all ranks and merge their final JSONL outputs."""

    print("[Node 0] Waiting for all nodes to finish...")
    for rank in range(1, WORLD_SIZE):
        flag = WORKSPACE / f"node_{rank}_done.flag"
        while not flag.exists():
            print(f"[Node 0] Waiting for Node {rank}: {flag}")
            time.sleep(env_int("CLUSTER_MERGE_POLL_SECONDS", 10))
        print(f"[Node 0] Node {rank} finished")

    all_rows: list[dict[str, Any]] = []
    for rank in range(WORLD_SIZE):
        node_output = output_for_rank(rank)
        rows = load_jsonl(node_output)
        print(f"[Node 0] Merging {len(rows)} rows from {node_output}")
        all_rows.extend(rows)

    write_jsonl(FINAL_OUT, all_rows)
    print(f"[Node 0] Merged {len(all_rows)} rows into {FINAL_OUT}")


def main() -> int:
    """Run this node's assigned pipeline shard."""

    sys.path.insert(0, str(APP_DIR))
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    NODE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"[Node {NODE_RANK}/{WORLD_SIZE}] WiFi analyzer cluster worker")
    print(f"App dir       : {APP_DIR}")
    print(f"Workspace     : {WORKSPACE}")
    print(f"Log file      : {INPUT_LOG}")
    print(f"PCAP file     : {INPUT_PCAP}")
    print(f"Model dir     : {MODEL_DIR}")
    print(f"Device/dtype  : {INFERENCE_DEVICE}/{INFERENCE_DTYPE}")
    print("=" * 60)

    validate_inputs()
    run_stage1()
    run_stage2()
    run_stage3()
    print_node_output()

    if MERGE_OUTPUTS:
        mark_done()
        if NODE_RANK == 0:
            wait_and_merge()
            print("[Node 0] All done. Download final_results.jsonl from the workspace.")
        else:
            print(f"[Node {NODE_RANK}] Done. Node 0 will merge shared outputs.")
    else:
        print(f"[Node {NODE_RANK}] Done. Merge disabled; read results from pod logs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
