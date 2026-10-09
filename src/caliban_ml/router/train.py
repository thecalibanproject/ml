"""Stage-2 intent head training (behind the ``[train]`` extra).

Two backbones are planned (see docs/research/02, "Staged pipeline", Stage 2):

* ``setfit``: contrastive fine-tune of a small sentence encoder + a logistic head.
  Best with 8-64 exemplars per intent (typical for a new tenant). Cheap to retrain nightly.
* ``modernbert_multihead``: ModernBERT-base encoder with several linear heads sharing one
  forward pass: ``intent`` (softmax), ``difficulty`` (softmax over buckets),
  ``oos`` (sigmoid), later ``reasoning``, ``jailbreak``, ``pii_presence``. One ONNX graph,
  one output tensor per head, described by ``OnnxSpec.heads`` in the manifest.

Everything heavy is imported lazily; the light package (and CI) never imports torch.
All base models must be loaded from a local, licence-checked path
(``local_files_only=True``): training may download a base model once into the model
cache on the training box, but the training code never reaches the network by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from caliban_ml._extras import require
from caliban_ml.router.dataset import IntentDataset


@dataclass
class IntentHeadConfig:
    base_model_path: Path  # local directory with weights + tokenizer.json (pinned revision)
    base_model_id: str  # e.g. "answerdotai/ModernBERT-base" (for the manifest + licence check)
    base_model_revision: str
    base_model_licence: str
    backbone: Literal["setfit", "modernbert_multihead"] = "setfit"
    heads: tuple[str, ...] = ("intent", "oos")
    difficulty_buckets: tuple[str, ...] = ("easy", "medium", "hard")
    max_length: int = 256
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 16
    seed: int = 13
    val_fraction: float = 0.2
    target_precision: float = 0.95
    output_dir: Path = field(default_factory=lambda: Path("runs/intent_head"))


def train_intent_head(dataset: IntentDataset, config: IntentHeadConfig) -> Path:
    """Train an intent head and save a HF-format checkpoint to ``config.output_dir``.

    Returns the checkpoint directory, which :func:`caliban_ml.router.export.export_onnx`
    turns into a signed-ready ONNX artifact.
    """
    from caliban_ml.artifacts.licenses import classify_licence

    decision = classify_licence(config.base_model_licence)
    if not decision.ok:
        raise ValueError(
            f"base model {config.base_model_id} licence {config.base_model_licence!r} is "
            f"{decision.verdict.value}: {decision.reason}"
        )
    require("torch", "train")
    require("transformers", "train")
    if config.backbone == "setfit":
        require("setfit", "train")
        # TODO(train/setfit):
        #   1. (train, val) = dataset.split(config.val_fraction, config.seed); drop OOS
        #      from train (OOS is a threshold decision, not a class).
        #   2. SetFitModel.from_pretrained(config.base_model_path, local_files_only=True)
        #   3. Trainer(model, args=TrainingArguments(num_epochs=..., batch_size=...,
        #      seed=config.seed), train_dataset=...) .train()
        #   4. Collect val logits -> temperature scaling + per-label thresholds with
        #      caliban_ml.router.knn.select_label_thresholds (same exit semantics as kNN);
        #      OOS threshold on max-probability via select_oos_threshold.
        #   5. Save checkpoint + calibration.json + metrics.json to config.output_dir.
        raise NotImplementedError("SetFit training: see TODO(train/setfit)")
    require("datasets", "train")
    # TODO(train/modernbert_multihead):
    #   1. AutoModel.from_pretrained(config.base_model_path, local_files_only=True)
    #      + one nn.Linear per head on the pooled (CLS / mean) representation.
    #   2. Loss = CE(intent) + CE(difficulty, where labelled) + BCE(oos), with OOS examples
    #      masked out of the intent loss. Difficulty labels come from probe results
    #      (quality gap between strong and weak model, Hybrid-LLM style).
    #   3. Early-stop on val macro-F1; calibrate each head (temperature) on val.
    #   4. Wrap forward() so the exported graph outputs one tensor per head named
    #      "<head>_logits"; save checkpoint + calibration.json + metrics.json.
    raise NotImplementedError("ModernBERT multi-head training: see TODO(train/modernbert)")
