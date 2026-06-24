from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("datasets")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from src import finetuning


def test_load_csv_builds_prompt_and_targets(tmp_path: Path) -> None:
    csv_path = tmp_path / "training.csv"
    csv_path.write_text(
        "label,input\n"
        "normal,hostapd authenticated\n"
        "error,EAPOL timeout\n",
        encoding="utf-8",
    )

    rows = finetuning.load_csv(csv_path, text_field="input", label_field="label")

    assert rows[0]["target"] == "normal"
    assert rows[1]["target"] == "error"
    assert "Classify this WiFi log line" in rows[0]["source"]
    assert "EAPOL timeout" in rows[1]["source"]


def test_load_csv_rejects_unknown_label(tmp_path: Path) -> None:
    csv_path = tmp_path / "training.csv"
    csv_path.write_text("label,input\nbad,EAPOL timeout\nnormal,ok\n", encoding="utf-8")

    with pytest.raises(ValueError, match="label must be one of"):
        finetuning.load_csv(csv_path, text_field="input", label_field="label")


def test_split_train_validation_is_deterministic() -> None:
    rows = [{"source": f"log {index}", "target": "normal"} for index in range(10)]

    train_a, validation_a = finetuning.split_train_validation(rows, 0.2, seed=42)
    train_b, validation_b = finetuning.split_train_validation(rows, 0.2, seed=42)

    assert train_a == train_b
    assert validation_a == validation_b
    assert len(train_a) == 8
    assert len(validation_a) == 2


def test_normalize_label_maps_generated_prefixes() -> None:
    assert finetuning.normalize_label("normal connection") == "normal"
    assert finetuning.normalize_label("ERROR timeout") == "error"
    assert finetuning.normalize_label("unknown") == "unknown"


def test_resolve_device_rejects_cuda_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(finetuning.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        finetuning.resolve_device("cuda")


def test_resolve_fp16_only_enables_on_cuda() -> None:
    assert finetuning.resolve_fp16(None, "cuda") is True
    assert finetuning.resolve_fp16(None, "cpu") is False
    assert finetuning.resolve_fp16(False, "cuda") is False
