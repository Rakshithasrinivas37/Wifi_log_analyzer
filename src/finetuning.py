"""Fine-tune FLAN-T5-small on CSV WiFi log labels with LoRA.

This module is intentionally importable service code, not a CLI script. Call
``fine_tune_flan_t5`` from FastAPI, notebooks, or another orchestration layer.
"""

from __future__ import annotations

import csv
import inspect
import json
import random
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)


VALID_LABELS = {"normal", "error"}


@dataclass
class FineTuningConfig:
    """Settings for FLAN-T5 LoRA fine-tuning."""

    train_csv: Path
    validation_csv: Path | None = None
    model: str = "google/flan-t5-small"
    output_dir: Path = Path("models/flan-t5-small-csv-lora")
    text_field: str = "input"
    label_field: str = "label"
    validation_ratio: float = 0.15
    max_source_length: int = 256
    max_target_length: int = 4
    epochs: float = 5.0
    learning_rate: float = 1e-4
    train_batch_size: int = 32
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q", "v"])
    eval_strategy: str = "epoch"
    logging_steps: int = 20
    save_steps: int = 200
    eval_steps: int = 200
    device: str = "auto"
    fp16: bool | None = None


def make_prompt(log_line: str) -> str:
    """Build the instruction prompt used for training and inference."""

    return (
        "Classify this WiFi log line as normal or error.\n"
        f"Log line: {log_line}\n"
        "Answer with exactly one label:"
    )


def normalize_label(text: str) -> str:
    """Normalize generated text to one binary label when possible."""

    normalized = text.strip().lower()
    if normalized.startswith("normal"):
        return "normal"
    if normalized.startswith("error"):
        return "error"
    return normalized


def load_csv(path: Path, text_field: str, label_field: str) -> list[dict[str, str]]:
    """Load CSV rows and convert them into text-to-text training examples."""

    if not path.exists():
        raise FileNotFoundError(f"CSV data not found: {path}")

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        missing = {text_field, label_field} - set(reader.fieldnames)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{path} missing required column(s): {missing_text}")

        for line_no, row in enumerate(reader, start=2):
            text = (row.get(text_field) or "").strip()
            label = (row.get(label_field) or "").strip().lower()
            if not text:
                continue
            if label not in VALID_LABELS:
                valid = ", ".join(sorted(VALID_LABELS))
                raise ValueError(f"{path}:{line_no} label must be one of: {valid}")
            rows.append({"source": make_prompt(text), "target": label})

    if len(rows) < 2:
        raise ValueError(f"{path} needs at least two labeled rows")
    return rows


def split_train_validation(
    rows: list[dict[str, str]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split rows while keeping at least one validation example."""

    if not 0.0 < validation_ratio < 0.5:
        raise ValueError("validation_ratio must be greater than 0 and less than 0.5")

    shuffled = rows.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(len(shuffled) * validation_ratio))
    validation = shuffled[:validation_count]
    train = shuffled[validation_count:]
    if not train:
        raise ValueError("not enough rows left for training after validation split")
    return train, validation


def compute_metrics(eval_pred: Any, tokenizer: Any) -> dict[str, float]:
    """Compute accuracy and error-class precision/recall/F1."""

    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_predictions = [
        normalize_label(text)
        for text in tokenizer.batch_decode(predictions, skip_special_tokens=True)
    ]
    decoded_labels = [
        normalize_label(text)
        for text in tokenizer.batch_decode(labels, skip_special_tokens=True)
    ]

    total = len(decoded_labels)
    correct = sum(
        prediction == label
        for prediction, label in zip(decoded_predictions, decoded_labels)
    )
    true_positive = sum(
        prediction == "error" and label == "error"
        for prediction, label in zip(decoded_predictions, decoded_labels)
    )
    false_positive = sum(
        prediction == "error" and label != "error"
        for prediction, label in zip(decoded_predictions, decoded_labels)
    )
    false_negative = sum(
        prediction != "error" and label == "error"
        for prediction, label in zip(decoded_predictions, decoded_labels)
    )

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": correct / max(1, total),
        "error_precision": precision,
        "error_recall": recall,
        "error_f1": f1,
    }


def tokenize_batch(
    batch: dict[str, list[str]],
    tokenizer: Any,
    max_source_length: int,
    max_target_length: int,
) -> dict[str, Any]:
    """Tokenize source prompts and target labels."""

    model_inputs = tokenizer(
        batch["source"],
        max_length=max_source_length,
        truncation=True,
    )
    labels = tokenizer(
        text_target=batch["target"],
        max_length=max_target_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_datasets(
    config: FineTuningConfig,
    tokenizer: Any,
) -> tuple[Dataset, Dataset, int, int]:
    """Load CSV files and return tokenized train/validation datasets."""

    train_rows = load_csv(config.train_csv, config.text_field, config.label_field)
    if config.validation_csv:
        validation_rows = load_csv(
            config.validation_csv,
            config.text_field,
            config.label_field,
        )
    else:
        train_rows, validation_rows = split_train_validation(
            train_rows,
            config.validation_ratio,
            config.seed,
        )

    tokenizer_fn = partial(
        tokenize_batch,
        tokenizer=tokenizer,
        max_source_length=config.max_source_length,
        max_target_length=config.max_target_length,
    )
    train_dataset = Dataset.from_list(train_rows).map(tokenizer_fn, batched=True)
    validation_dataset = Dataset.from_list(validation_rows).map(
        tokenizer_fn,
        batched=True,
    )
    return train_dataset, validation_dataset, len(train_rows), len(validation_rows)


def resolve_device(device: str) -> str:
    """Choose the training device and fail clearly when CUDA is requested."""

    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for fine-tuning, but CUDA is not available")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_fp16(fp16: bool | None, device: str) -> bool:
    """Use fp16 by default on CUDA and never on CPU."""

    if device != "cuda":
        return False
    return True if fp16 is None else fp16


def build_training_args(
    config: FineTuningConfig,
    device: str,
    use_fp16: bool,
) -> Seq2SeqTrainingArguments:
    """Create Seq2Seq training arguments while handling Transformers versions."""

    if config.eval_strategy not in {"epoch", "steps"}:
        raise ValueError("eval_strategy must be either 'epoch' or 'steps'")

    training_kwargs: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "num_train_epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_strategy": config.eval_strategy,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "error_f1",
        "greater_is_better": True,
        "predict_with_generate": True,
        "generation_max_length": config.max_target_length,
        "report_to": "none",
        "seed": config.seed,
        "fp16": use_fp16,
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
    }
    training_arg_params = inspect.signature(Seq2SeqTrainingArguments).parameters
    if "use_cpu" in training_arg_params:
        training_kwargs["use_cpu"] = device == "cpu"
    elif "no_cuda" in training_arg_params:
        training_kwargs["no_cuda"] = device == "cpu"

    strategy_name = (
        "eval_strategy"
        if "eval_strategy" in training_arg_params
        else "evaluation_strategy"
    )
    training_kwargs[strategy_name] = config.eval_strategy
    if config.eval_strategy == "steps":
        training_kwargs["eval_steps"] = config.eval_steps
        training_kwargs["save_steps"] = config.save_steps
    return Seq2SeqTrainingArguments(**training_kwargs)


def fine_tune_flan_t5(config: FineTuningConfig) -> dict[str, Any]:
    """Train and save a FLAN-T5 LoRA adapter."""

    set_seed(config.seed)
    device = resolve_device(config.device)
    use_fp16 = resolve_fp16(config.fp16, device)

    if device == "cuda":
        print(f"fine-tuning device: cuda ({torch.cuda.get_device_name(0)})", flush=True)
        print(
            f"cuda total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
            flush=True,
        )
    else:
        print("fine-tuning device: cpu", flush=True)
    print(f"fp16 enabled: {use_fp16}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    train_dataset, validation_dataset, train_count, validation_count = build_datasets(
        config,
        tokenizer,
    )

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)

    print(f"model parameter device: {next(model.parameters()).device}", flush=True)
    if device == "cuda":
        print(
            f"cuda memory allocated after model load: {torch.cuda.memory_allocated() / 1024**2:.1f} MiB",
            flush=True,
        )

    training_args = build_training_args(config, device=device, use_fp16=use_fp16)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
    )
    trainer.train()
    metrics = trainer.evaluate()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(config.output_dir)
    (config.output_dir / "label_set.json").write_text(
        json.dumps({"labels": sorted(VALID_LABELS)}, indent=2),
        encoding="utf-8",
    )
    (config.output_dir / "eval_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "output_dir": str(config.output_dir),
        "train_rows": train_count,
        "validation_rows": validation_count,
        "metrics": metrics,
    }
