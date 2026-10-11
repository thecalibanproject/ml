"""On-disk layout of a trained Stage-2 intent head (the hand-off from train to export).

::

    <output_dir>/
        caliban_checkpoint.json    CheckpointMeta (this module): backbone, base model, pooling,
                                   heads (labels, activation, calibration), metrics, data card
        encoder/                   HF ``save_pretrained`` of the fine-tuned encoder + tokenizer
        heads.safetensors          ``<head>.weight`` (out, hidden) and ``<head>.bias`` (out,)
        splits.json                train / val / test texts and labels (for parity and audits)
        metrics.json               copy of ``CheckpointMeta.metrics``
        calibration.json           copy of the per-head calibrations

Both backbones (SetFit and the ModernBERT multi-head) are reduced to the same graph::

    encoder(input_ids, attention_mask) -> pool (mean | cls) -> [L2 normalise] -> Linear per head

so a single exporter handles both. Pure pydantic: readable without the ``[train]`` extra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from caliban_ml.artifacts import BaseModelRef, Calibration, DataCard

CHECKPOINT_VERSION = 1
META_FILENAME = "caliban_checkpoint.json"
ENCODER_DIR = "encoder"
HEADS_FILENAME = "heads.safetensors"
SPLITS_FILENAME = "splits.json"
PRIMARY_HEAD = "intent"


class CheckpointHead(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    activation: Literal["softmax", "sigmoid"]
    labels: list[str] = Field(min_length=1)
    calibration: Calibration

    @property
    def output(self) -> str:
        return f"{self.name}_logits"

    @property
    def width(self) -> int:
        return 1 if self.activation == "sigmoid" else len(self.labels)


class CheckpointMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint_version: Literal[1] = CHECKPOINT_VERSION
    backbone: Literal["setfit", "modernbert_multihead"]
    base_model: BaseModelRef
    pooling: Literal["mean", "cls"]
    normalize: bool = False
    max_length: int = Field(ge=1)
    hidden_size: int = Field(ge=1)
    heads: list[CheckpointHead] = Field(min_length=1)
    metrics: dict[str, float] = Field(default_factory=dict)
    data_card: DataCard
    train_config: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _primary(self) -> CheckpointMeta:
        names = [h.name for h in self.heads]
        if len(set(names)) != len(names):
            raise ValueError("duplicate head names")
        if names[0] != PRIMARY_HEAD or self.heads[0].activation != "softmax":
            raise ValueError(f"the first head must be the softmax {PRIMARY_HEAD!r} head")
        return self

    @property
    def primary(self) -> CheckpointHead:
        return self.heads[0]

    def write(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / META_FILENAME
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (out_dir / "metrics.json").write_text(json.dumps(self.metrics, indent=2) + "\n")
        (out_dir / "calibration.json").write_text(json.dumps(
            {h.name: h.calibration.model_dump(exclude_none=True) for h in self.heads},
            indent=2) + "\n")
        return path

    @classmethod
    def read(cls, ckpt_dir: Path) -> CheckpointMeta:
        return cls.model_validate_json((Path(ckpt_dir) / META_FILENAME).read_text("utf-8"))


def write_splits(out_dir: Path, splits: dict[str, tuple[list[str], list[str]]]) -> Path:
    path = Path(out_dir) / SPLITS_FILENAME
    path.write_text(json.dumps({k: {"texts": t, "labels": y} for k, (t, y) in splits.items()},
                               indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_splits(ckpt_dir: Path) -> dict[str, tuple[list[str], list[str]]]:
    path = Path(ckpt_dir) / SPLITS_FILENAME
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text("utf-8"))
    return {k: (v["texts"], v["labels"]) for k, v in raw.items()}
