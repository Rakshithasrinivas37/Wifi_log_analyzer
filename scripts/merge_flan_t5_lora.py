"""Merge a fine-tuned FLAN-T5 LoRA adapter into its base model.

Run this after fine-tuning when you want a standalone Hugging Face model folder
instead of loading the base model and LoRA adapter separately.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_BASE_MODEL = "google/flan-t5-small"
DEFAULT_ADAPTER_DIR = Path("models/flan-t5-log-lora-model")
DEFAULT_OUTPUT_DIR = Path("models/flan-t5-log-merged-model")
METADATA_FILES = ("label_set.json", "eval_metrics.json")


def read_base_model_from_adapter(adapter_dir: Path) -> str | None:
    """Read the base model name from adapter_config.json when available."""

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        return None

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    base_model = config.get("base_model_name_or_path")
    return str(base_model) if base_model else None


def resolve_torch_dtype(dtype: str, device: str) -> torch.dtype:
    """Resolve a user dtype string to a torch dtype."""

    if dtype == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    raise ValueError("dtype must be one of: auto, fp16, bf16, fp32")


def resolve_device(device: str) -> str:
    """Pick a runtime device."""

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available")
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, cpu")
    return device


def copy_metadata_files(adapter_dir: Path, output_dir: Path) -> None:
    """Copy project metadata files that are not saved by Hugging Face."""

    for filename in METADATA_FILES:
        source = adapter_dir / filename
        if source.is_file():
            shutil.copy2(source, output_dir / filename)


def merge_flan_t5_lora(
    adapter_dir: Path = DEFAULT_ADAPTER_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    base_model: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
) -> Path:
    """Merge a LoRA adapter into the base FLAN-T5 model and save it."""

    adapter_dir = adapter_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter_dir does not exist: {adapter_dir}")

    resolved_device = resolve_device(device)
    torch_dtype = resolve_torch_dtype(dtype, resolved_device)
    resolved_base_model = (
        base_model
        or read_base_model_from_adapter(adapter_dir)
        or DEFAULT_BASE_MODEL
    )

    print(f"adapter_dir={adapter_dir}")
    print(f"output_dir={output_dir}")
    print(f"base_model={resolved_base_model}")
    print(f"device={resolved_device}")
    print(f"dtype={torch_dtype}")

    tokenizer_source = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else resolved_base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    base = AutoModelForSeq2SeqLM.from_pretrained(
        resolved_base_model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    base = base.to(resolved_device)

    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    copy_metadata_files(adapter_dir, output_dir)

    print(f"merged model saved to: {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    """Parse command arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    return parser.parse_args()


def main() -> None:
    """Run model merging from command arguments."""

    args = parse_args()
    merge_flan_t5_lora(
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        device=args.device,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
