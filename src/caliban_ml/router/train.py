"""Stage-2 intent head training (behind the ``[train]`` extra).

Where this sits (docs research note 02, "Recommended design for Caliban"): the router is
staged, rules -> Stage-1 embedding kNN over per-tenant exemplars (core, already shipped) ->
Stage-2 small encoder (this module) -> LLM fallback. Stage 2 only sees traffic Stage 1
abstained on, and exits when its calibrated max-probability clears tau_2. Training stays in
Python; the contract with core is ONNX plus JSON (:mod:`caliban_ml.router.export`).

Two backbones:

* ``setfit``: contrastive fine-tune of a sentence encoder + a logistic head
  (SetFit, https://arxiv.org/abs/2209.11055). Best with 8-64 exemplars per intent (typical
  for a new tenant), cheap to retrain nightly. OOS examples are not a class: the OOS gate is
  a threshold on calibrated max-probability ("small encoder first, LLM on uncertainty",
  https://arxiv.org/abs/2410.01627; https://arxiv.org/abs/2507.01541).
* ``modernbert_multihead``: a ModernBERT-architecture encoder with linear heads sharing one
  forward pass: ``intent`` (softmax) and an explicit ``oos`` head (sigmoid, trained on the
  dataset's OOS examples; note 02: "use an explicit OOS head and thresholds in the
  encoder"). Base models: ModernBERT-base (English, https://arxiv.org/abs/2412.13663) or
  mmBERT-base (multilingual, same architecture, https://arxiv.org/abs/2509.06888, the
  backbone vLLM Semantic Router v0.4 moved to). Stay at base size: scaling the encoder
  barely helps routing (https://arxiv.org/abs/2609.34326). Note 02's other heads
  (``difficulty`` / quality gap in the Hybrid-LLM style, ``reasoning``, ``jailbreak``,
  ``pii_presence``) need labels the intent YAML does not carry yet and are refused for now.
  One ONNX graph, one output tensor per head (``<head>_logits``), described by
  ``OnnxSpec.heads`` in the manifest.

Both backbones are saved in one checkpoint layout (:mod:`caliban_ml.router.checkpoint`):
``encoder -> pool -> [normalise] -> Linear per head``.

Evaluation follows note 02's warnings. Tuned kNN over prompt embeddings matches or beats
learned routers (https://arxiv.org/abs/2505.12601, https://arxiv.org/abs/2601.07206), so
every run also scores the Stage-1 kNN (the same code core mirrors) on the same splits, over
the base and the fine-tuned encoder's embeddings (``knn_base_*``, ``knn_tuned_*``). Gains
measured in-distribution vanish on held-out categories (https://arxiv.org/abs/2610.01535),
so ``held_out_intents`` removes whole intents from training and scores them as OOS, and
``extra_eval`` scores independently written sets. Splits are three-way: ``train`` fits
weights, ``val`` picks the epoch and fits calibration (temperature, OOS gate, per-label exit
thresholds: the same selection code as the Stage-1 kNN), ``test`` is held out.

Everything heavy is imported lazily; the light package (and CI) never imports torch.
All base models must be loaded from a local, licence-checked path
(``local_files_only=True``): training may download a base model once into the model
cache on the training box, but the training code never reaches the network by itself.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from caliban_ml._extras import require
from caliban_ml.artifacts import BaseModelRef, DataCard, DatasetRef
from caliban_ml.router.dataset import OOS_LABEL, IntentDataset, IntentSpec

Split = tuple[list[str], list[str]]
SUPPORTED_HEADS = ("intent", "oos")


@dataclass
class IntentHeadConfig:
    base_model_path: Path  # local directory with weights + tokenizer.json (pinned revision)
    base_model_id: str  # e.g. "answerdotai/ModernBERT-base" (for the manifest + licence check)
    base_model_revision: str
    base_model_licence: str
    backbone: Literal["setfit", "modernbert_multihead"] = "setfit"
    heads: tuple[str, ...] = ("intent", "oos")  # modernbert_multihead only
    difficulty_buckets: tuple[str, ...] = ("easy", "medium", "hard")
    pooling: Literal["mean", "cls"] = "mean"  # modernbert_multihead only (SetFit uses its body's)
    max_length: int = 256
    epochs: int = 3
    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-3  # multi-head: the freshly initialised linear heads
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    patience: int = 3  # multi-head early stopping on val macro-F1 (epochs without improvement)
    setfit_iterations: int = 20  # SetFit: contrastive pairs generated per training sample
    batch_size: int = 16
    seed: int = 13
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    oos_strategy: Literal["recall", "youden"] = "recall"
    target_in_scope_recall: float = 0.95
    target_precision: float = 0.95
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    # Data card for the exemplar set (licence-checked like the base model).
    dataset_name: str = "intents"
    dataset_source: str = ""
    dataset_revision: str | None = None
    dataset_licence: str = "caliban-internal"
    dataset_synthetic: bool = True
    languages: tuple[str, ...] = ("en",)
    # Intents removed from training entirely; their utterances are scored as out-of-scope
    # ("heldout_intents_*" metrics). Note 02 warns that in-distribution splits inflate
    # router evals; held-out categories are the honest OOS test.
    held_out_intents: tuple[str, ...] = ()
    # kNN baselines (note 02: tuned kNN often matches learned routers) on the same splits,
    # over the base encoder's and the fine-tuned encoder's pooled embeddings.
    knn_baseline: bool = True
    knn_k: int = 5
    output_dir: Path = field(default_factory=lambda: Path("runs/intent_head"))


# ---------------------------------------------------------------------- pure helpers


def dataset_from_examples(texts: list[str], labels: list[str],
                          like: IntentDataset | None = None) -> IntentDataset:
    """Rebuild an :class:`IntentDataset` from flat (text, label) lists (OOS = ``__oos__``)."""
    intents: dict[str, list[str]] = {}
    oos: list[str] = []
    for t, y in zip(texts, labels, strict=True):
        (oos if y == OOS_LABEL else intents.setdefault(y, [])).append(t)
    desc = {k: v.description for k, v in like.intents.items()} if like else {}
    return IntentDataset(version=1, oos=oos, intents={
        k: IntentSpec(description=desc.get(k, ""), utterances=v) for k, v in intents.items()})


def three_way_split(dataset: IntentDataset, *, val_fraction: float, test_fraction: float,
                    seed: int) -> dict[str, Split]:
    """Stratified train / val / test. Every intent keeps >= 1 train exemplar; with fewer
    than 4 OOS examples at a step, all of them go to the held-out side of that step."""
    if not (0 <= val_fraction < 1 and 0 <= test_fraction < 1):
        raise ValueError("val_fraction and test_fraction must be in [0, 1)")
    if test_fraction > 0:
        rest, test = dataset.split(test_fraction, seed)
        rest_ds = dataset_from_examples(*rest, like=dataset)
    else:
        rest_ds, test = dataset, ([], [])
    train, val = rest_ds.split(val_fraction, seed + 1)
    return {"train": train, "val": val, "test": test}


def encode_labels(labels: list[str], classes: list[str]) -> np.ndarray:
    """Class indices, ``-1`` for OOS. Unknown labels are an error."""
    index = {c: i for i, c in enumerate(classes)}
    unknown = sorted({y for y in labels if y not in index and y != OOS_LABEL})
    if unknown:
        raise ValueError(f"labels not in the label space: {unknown}")
    return np.array([index.get(y, -1) for y in labels], dtype=np.int64)


def build_data_card(config: IntentHeadConfig, splits: dict[str, Split]) -> DataCard:
    def ref(name: str, n: int) -> DatasetRef:
        return DatasetRef(name=name, source=config.dataset_source or config.dataset_name,
                          revision=config.dataset_revision, licence=config.dataset_licence,
                          n_examples=n, synthetic=config.dataset_synthetic)

    n_eval = len(splits["val"][0]) + len(splits["test"][0])
    return DataCard(
        training_datasets=[ref(config.dataset_name, len(splits["train"][0]))],
        eval_datasets=[ref(f"{config.dataset_name} (val+test split)", n_eval)],
        languages=list(config.languages),
        intended_use="Stage-2 intent classification for caliban/auto routing: the request "
                     "falls through to this head when Stage-1 kNN abstains.",
        limitations="Few-shot exemplars; the label space is fixed at training time. "
                    "Recalibrate and re-export whenever the exemplar set changes.",
    )


def check_licences(config: IntentHeadConfig) -> None:
    from caliban_ml.artifacts.licenses import classify_licence

    for subject, licence in ((config.base_model_id, config.base_model_licence),
                             (config.dataset_name, config.dataset_licence)):
        decision = classify_licence(licence)
        if not decision.ok:
            raise ValueError(f"{subject} licence {licence!r} is {decision.verdict.value}: "
                             f"{decision.reason}")


def check_heads(config: IntentHeadConfig) -> None:
    if not config.heads or config.heads[0] != "intent":
        raise ValueError("heads must start with 'intent'")
    unsupported = [h for h in config.heads if h not in SUPPORTED_HEADS]
    if unsupported:
        raise ValueError(f"heads {unsupported} need labels the intent dataset does not carry "
                         f"yet; supported: {SUPPORTED_HEADS}")


def hold_out_intents(dataset: IntentDataset, intents: tuple[str, ...] | list[str]
                     ) -> tuple[IntentDataset, Split]:
    """Remove ``intents`` from ``dataset``; return their utterances labelled ``__oos__``."""
    unknown = sorted(set(intents) - set(dataset.intents))
    if unknown:
        raise ValueError(f"held_out_intents not in the dataset: {unknown}")
    if not intents:
        return dataset, ([], [])
    keep = {k: v for k, v in dataset.intents.items() if k not in intents}
    if len(keep) < 2:
        raise ValueError("need at least two intents left after holding some out")
    texts = [u for k in sorted(intents) for u in dataset.intents[k].utterances]
    return (IntentDataset(version=1, intents=keep, oos=dataset.oos),
            (texts, [OOS_LABEL] * len(texts)))


def scored_eval_sets(splits: dict[str, Split], heldout: Split, extra: dict[str, Split],
                     classes: list[str]) -> dict[str, Split]:
    """All sets that are scored but never trained or calibrated on, by metric prefix."""
    sets: dict[str, Split] = {}
    if splits["test"][0]:
        sets["test"] = splits["test"]
    if heldout[0]:
        sets["heldout_intents"] = heldout
    for name, (texts, labels) in extra.items():
        if name in ("train", "val", "test", "heldout_intents") or not name.isidentifier():
            raise ValueError(f"invalid extra eval set name {name!r}")
        encode_labels(labels, classes)  # fail early on unknown labels
        sets[name] = (list(texts), list(labels))
    return sets


def knn_baseline_metrics(emb: dict[str, np.ndarray], splits: dict[str, Split],
                         eval_sets: dict[str, Split], *, k: int, prefix: str,
                         config: IntentHeadConfig) -> dict[str, float]:
    """Stage-1 kNN (the same code core mirrors) fitted on the train split's in-scope
    embeddings, calibrated on val, scored on every eval set. Pure numpy."""
    from caliban_ml.router.knn import evaluate, fit_and_calibrate

    tr_labels = splits["train"][1]
    keep = [i for i, y in enumerate(tr_labels) if y != OOS_LABEL]
    oos_strategy = config.oos_strategy
    if oos_strategy == "youden" and OOS_LABEL not in splits["val"][1]:
        oos_strategy = "recall"
    res = fit_and_calibrate(emb["train"][keep], [tr_labels[i] for i in keep], emb["val"],
                            splits["val"][1], k=k, oos_strategy=oos_strategy,
                            target_in_scope_recall=config.target_in_scope_recall,
                            target_precision=config.target_precision)
    out = {f"{prefix}{key}": v for key, v in res.metrics.items()}
    for name, (_, labels) in eval_sets.items():
        m = evaluate(res.classifier, res.calibration, emb[name], labels, prefix=f"{name}_")
        out.update({f"{prefix}{key}": v for key, v in m.items()})
    return out


def _embed_sets(T, model, tokenizer, splits: dict[str, Split], eval_sets: dict[str, Split],
                config: IntentHeadConfig, device: str) -> dict[str, np.ndarray]:
    import torch

    out: dict[str, np.ndarray] = {}
    model.eval()
    for name, (texts, _) in {**splits, **eval_sets}.items():
        if name in out or not texts:
            continue
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = tokenizer(texts[i:i + 64], padding=True, truncation=True,
                                max_length=config.max_length, return_tensors="pt").to(device)
                vecs.append(model.embed(enc["input_ids"], enc["attention_mask"])
                            .float().cpu().numpy())
        out[name] = np.concatenate(vecs)
    return out


# ---------------------------------------------------------------------- entry point


def train_intent_head(dataset: IntentDataset, config: IntentHeadConfig, *,
                      extra_eval: dict[str, Split] | None = None) -> Path:
    """Train an intent head and save a checkpoint to ``config.output_dir``.

    ``extra_eval`` maps a name to extra labelled (texts, labels) sets that are only scored
    (e.g. independently written paraphrases, other languages); labels must be intents of
    ``dataset`` or ``__oos__``. Every eval set is scored for the trained head and, with
    ``config.knn_baseline``, for the kNN baselines (``knn_base_*``, ``knn_tuned_*``).

    Returns the checkpoint directory, which :func:`caliban_ml.router.export.export_onnx`
    turns into a manifest-checked ONNX artifact.
    """
    check_licences(config)
    check_heads(config)
    if config.backbone not in ("setfit", "modernbert_multihead"):
        raise ValueError(f"unknown backbone {config.backbone!r}")
    torch = require("torch", "train")
    require("transformers", "train")
    require("safetensors", "train")
    if config.backbone == "setfit":
        require("setfit", "train")
        require("datasets", "train")

    from caliban_ml.router import _torch as T
    from caliban_ml.router.calibrate import (
        calibrate_logit_head,
        calibrate_oos_head,
        evaluate_logit_head,
        sigmoid,
    )
    from caliban_ml.router.checkpoint import (
        ENCODER_DIR,
        HEADS_FILENAME,
        CheckpointHead,
        CheckpointMeta,
        write_splits,
    )

    _seed_everything(config.seed)
    device = _device(torch, config.device)
    dataset, heldout = hold_out_intents(dataset, config.held_out_intents)
    classes = dataset.labels
    splits = three_way_split(dataset, val_fraction=config.val_fraction,
                             test_fraction=config.test_fraction, seed=config.seed)
    if not any(y != OOS_LABEL for y in splits["val"][1]):
        raise ValueError("validation split has no in-scope examples; add exemplars or raise "
                         "val_fraction")
    eval_sets = scored_eval_sets(splits, heldout, extra_eval or {}, classes)
    knn_metrics: dict[str, float] = {}
    if config.knn_baseline:
        base = T.PooledHeads(T.load_encoder(Path(config.base_model_path),
                                            attn_implementation="sdpa"), {},
                             pooling=config.pooling).to(device)
        from transformers import AutoTokenizer

        base_tok = AutoTokenizer.from_pretrained(str(config.base_model_path),
                                                 local_files_only=True)
        knn_metrics.update(knn_baseline_metrics(
            _embed_sets(T, base, base_tok, splits, eval_sets, config, device),
            splits, eval_sets, k=config.knn_k, prefix="knn_base_", config=config))
        del base
        if device == "cuda":
            torch.cuda.empty_cache()

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    if config.backbone == "setfit":
        model, tokenizer, train_metrics = _train_setfit(splits, classes, config, device)
    else:
        model, tokenizer, train_metrics = _train_multihead(splits, classes, config, device)
    train_seconds = time.perf_counter() - t0
    metrics: dict[str, float] = {"train_seconds": train_seconds, **train_metrics}
    if device == "cuda":
        metrics["train_peak_vram_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
        metrics["train_peak_vram_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20

    # Calibrate on val, report on test, in fp32 (the exported graph is fp32 before quant).
    model = model.float().to(device).eval()
    logits = {name: T.predict_logits(model, tokenizer, texts, max_length=config.max_length,
                                     device=device) if texts else None
              for name, (texts, _) in splits.items()}
    y = {name: encode_labels(labels, classes) for name, (_, labels) in splits.items()}
    cal = calibrate_logit_head(logits["val"]["intent"], y["val"], classes,
                               oos_strategy=config.oos_strategy,
                               target_in_scope_recall=config.target_in_scope_recall,
                               target_precision=config.target_precision)
    metrics.update(cal.metrics)
    if logits["test"] is not None:
        metrics.update(evaluate_logit_head(logits["test"]["intent"], y["test"], classes,
                                           cal.calibration, prefix="test_"))
    for name, (texts, labels) in eval_sets.items():
        if name == "test":
            continue
        lg = T.predict_logits(model, tokenizer, texts, max_length=config.max_length,
                              device=device)
        metrics.update(evaluate_logit_head(lg["intent"], encode_labels(labels, classes),
                                           classes, cal.calibration, prefix=f"{name}_"))
        logits[name] = lg
    if config.knn_baseline:
        knn_metrics.update(knn_baseline_metrics(
            _embed_sets(T, model, tokenizer, splits, eval_sets, config, device),
            splits, eval_sets, k=config.knn_k, prefix="knn_tuned_", config=config))
    metrics.update(knn_metrics)
    heads = [CheckpointHead(name="intent", activation="softmax", labels=classes,
                            calibration=cal.calibration)]
    if "oos" in model.head_names:
        oos_cal, oos_m = calibrate_oos_head(logits["val"]["oos"], y["val"] < 0,
                                            target_in_scope_recall=config.target_in_scope_recall)
        metrics.update(oos_m)
        thr = oos_cal.thresholds["oos"]
        for name, (_, labels) in eval_sets.items():
            if logits.get(name) is None:
                continue
            p = sigmoid(logits[name]["oos"].reshape(-1), oos_cal.temperature)
            is_oos = encode_labels(labels, classes) < 0
            if (~is_oos).any():
                metrics[f"{name}_oos_head_in_scope_acceptance"] = float(
                    (p[~is_oos] < thr).mean())
            if is_oos.any():
                metrics[f"{name}_oos_head_oos_rejection"] = float((p[is_oos] >= thr).mean())
        heads.append(CheckpointHead(name="oos", activation="sigmoid", labels=["oos"],
                                    calibration=oos_cal))

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(out / ENCODER_DIR))  # safetensors by default
    tokenizer.save_pretrained(str(out / ENCODER_DIR))
    from safetensors.torch import save_file

    save_file(model.head_state(), str(out / HEADS_FILENAME))
    write_splits(out, splits)
    train_config = {k: (str(v) if isinstance(v, Path) else v)
                    for k, v in asdict(config).items()}
    meta = CheckpointMeta(
        backbone=config.backbone,
        base_model=BaseModelRef(id=config.base_model_id, revision=config.base_model_revision,
                                licence=config.base_model_licence),
        pooling=model.pooling, normalize=model.normalize, max_length=config.max_length,
        hidden_size=int(model.encoder.config.hidden_size), heads=heads,
        metrics={k: float(v) for k, v in metrics.items()},
        data_card=build_data_card(config, splits), train_config=train_config,
    )
    meta.write(out)
    return out


# ---------------------------------------------------------------------- backbones


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(torch, wanted: str) -> str:
    if wanted == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if wanted == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False")
    return wanted


def _train_multihead(splits: dict[str, Split], classes: list[str], config: IntentHeadConfig,
                     device: str):
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from caliban_ml.router import _torch as T
    from caliban_ml.router.knn import _macro_f1

    path = Path(config.base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    encoder = T.load_encoder(path, attn_implementation="sdpa")
    texts, labels = splits["train"]
    y_all = encode_labels(labels, classes)
    want_oos = "oos" in config.heads and bool((y_all < 0).any())
    widths = {"intent": len(classes), **({"oos": 1} if want_oos else {})}
    model = T.PooledHeads(encoder, widths, pooling=config.pooling,
                          dropout=config.dropout).to(device)

    head_params = list(model.heads.parameters())
    head_ids = {id(p) for p in head_params}
    enc_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": config.learning_rate},
        {"params": head_params, "lr": config.head_learning_rate},
    ], weight_decay=config.weight_decay)
    steps_per_epoch = max(1, -(-len(texts) // config.batch_size))
    total = steps_per_epoch * config.epochs
    sched = get_linear_schedule_with_warmup(opt, int(config.warmup_ratio * total), total)
    ce = torch.nn.CrossEntropyLoss()
    bce = torch.nn.BCEWithLogitsLoss()
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    rng = np.random.default_rng(config.seed)

    val_texts, val_labels = splits["val"]
    y_val = encode_labels(val_labels, classes)
    best = (-1.0, -np.inf)
    best_state: dict | None = None
    best_epoch, bad_epochs, epochs_run = 0, 0, 0
    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(len(texts))
        for i in range(0, len(order), config.batch_size):
            idx = order[i:i + config.batch_size]
            enc = tokenizer([texts[j] for j in idx], padding=True, truncation=True,
                            max_length=config.max_length, return_tensors="pt").to(device)
            yb = torch.as_tensor(y_all[idx], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(enc["input_ids"], enc["attention_mask"])
            intent_logits = out[0].float()
            in_scope = yb >= 0
            loss = (ce(intent_logits[in_scope], yb[in_scope]) if in_scope.any()
                    else intent_logits.sum() * 0.0)
            if want_oos:
                loss = loss + bce(out[1].float().squeeze(-1), (~in_scope).float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        epochs_run = epoch + 1
        # early stopping on val in-scope macro-F1 (tie-break: higher in-scope log-likelihood)
        val = T.predict_logits(model, tokenizer, val_texts, max_length=config.max_length,
                               device=device)["intent"]
        m = y_val >= 0
        pred = val.argmax(axis=1)
        f1 = _macro_f1(pred[m], y_val[m], len(classes))
        z = val[m] - val[m].max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        score = (f1, float(logp[np.arange(int(m.sum())), y_val[m]].mean()))
        if score > best:
            best, best_epoch, bad_epochs = score, epoch + 1, 0
            best_state = {k: v.detach().to("cpu", copy=True)
                          for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tokenizer, {"best_epoch": float(best_epoch), "epochs_run": float(epochs_run),
                              "val_macro_f1_at_best_epoch": float(best[0])}


def _train_setfit(splits: dict[str, Split], classes: list[str], config: IntentHeadConfig,
                  device: str):
    import torch
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    from caliban_ml.router import _torch as T

    texts, labels = splits["train"]
    keep = [i for i, y in enumerate(labels) if y != OOS_LABEL]  # OOS is a gate, not a class
    y = encode_labels([labels[i] for i in keep], classes)
    missing = sorted(set(range(len(classes))) - set(y.tolist()))
    if missing:
        raise ValueError(f"intents without training exemplars: {[classes[i] for i in missing]}")
    import dataclasses

    model = SetFitModel.from_pretrained(str(config.base_model_path), local_files_only=True)
    model.to(device)
    model.model_body.max_seq_length = config.max_length
    wanted = {"batch_size": config.batch_size, "num_epochs": config.epochs,
              "num_iterations": config.setfit_iterations,
              "body_learning_rate": config.learning_rate, "seed": config.seed,
              "max_length": config.max_length, "report_to": "none",
              "use_amp": device == "cuda", "show_progress_bar": False}
    known = {f.name for f in dataclasses.fields(TrainingArguments)}
    for required in ("batch_size", "num_epochs", "num_iterations", "body_learning_rate"):
        if required not in known:
            raise RuntimeError(f"unsupported setfit version: TrainingArguments has no "
                               f"{required!r}")
    args = TrainingArguments(**{k: v for k, v in wanted.items() if k in known})
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_dict(
        {"text": [texts[i] for i in keep], "label": y.tolist()}))
    trainer.train()

    pooling, normalize, encoder, tokenizer = _setfit_body_parts(model)
    head = model.model_head
    coef = getattr(head, "coef_", None)
    if coef is None:
        raise ValueError(f"unsupported SetFit head {type(head).__name__}: expected sklearn "
                         "LogisticRegression")
    if [int(c) for c in head.classes_] != list(range(len(classes))):
        raise ValueError("SetFit head classes do not cover the label space in order")
    w = np.asarray(coef, dtype=np.float32)
    b = np.asarray(head.intercept_, dtype=np.float32)
    if w.shape[0] == 1:  # binary LR: one logit z for class 1 -> logits [0, z], same softmax
        w = np.concatenate([np.zeros_like(w), w])
        b = np.concatenate([np.zeros_like(b), b])
    wrapped = T.PooledHeads(encoder, {"intent": len(classes)}, pooling=pooling,
                            normalize=normalize)
    with torch.no_grad():
        wrapped.heads["intent"].weight.copy_(torch.from_numpy(w))
        wrapped.heads["intent"].bias.copy_(torch.from_numpy(b))
    return wrapped.to(device), tokenizer, {"setfit_train_pairs": float(
        config.setfit_iterations * 2 * len(keep) * config.epochs)}


def _setfit_body_parts(model):
    """Reduce a SetFit body (SentenceTransformer) to (pooling, normalize, encoder, tokenizer).

    Only ``Transformer -> Pooling(mean|cls) [-> Normalize]`` bodies are supported: anything
    else (Dense layers, weighted pooling) would not match the exported graph. Modules are
    matched by class name so this survives sentence-transformers reorganising its packages.
    """
    mods = list(model.model_body)
    first = mods[0] if mods else None
    encoder = getattr(first, "auto_model", None) or getattr(first, "model", None)
    tokenizer = getattr(first, "tokenizer", None)
    if encoder is None or tokenizer is None:
        raise ValueError("SetFit body must start with a sentence-transformers Transformer")
    pooling, normalize = None, bool(getattr(model, "normalize_embeddings", False))
    for m in mods[1:]:
        kind = type(m).__name__
        if kind == "Pooling":
            mode = (m.get_pooling_mode_str() if hasattr(m, "get_pooling_mode_str")
                    else getattr(m, "pooling_mode", None))
            if mode not in ("mean", "cls"):
                raise ValueError(f"unsupported SetFit pooling {mode!r} (mean or cls only)")
            pooling = mode
        elif kind == "Normalize":
            normalize = True
        else:
            raise ValueError(f"unsupported SetFit body module {kind}")
    if pooling is None:
        raise ValueError("SetFit body has no Pooling module")
    return pooling, normalize, encoder, tokenizer
