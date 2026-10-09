"""Pure-numpy kNN intent baseline (Stage 1 "semantic router") with calibration.

This mirrors what the Rust data plane does at Stage 1, so thresholds picked here
transfer 1:1:

1. Embed the query with the shipped embedder (L2-normalised), retrieve the top-``k``
   exemplars by cosine similarity (HNSW in Rust, brute force here).
2. Class probabilities are a temperature softmax over the *neighbours*, summed per class,
   with a small smoothing mass so unseen classes are not exactly zero::

       w_j = exp(s_j / T)            for the k neighbours j
       p_c = (1 - eps) * sum_{j: y_j = c} w_j / sum_j w_j + eps / C

   ``T`` is the calibrated ``Calibration.temperature``.
3. **OOS gate** (explicit, never argmax): if the top-1 similarity is below
   ``Calibration.oos_threshold`` the query is out-of-scope (``oos_score = top1_similarity``).
4. **Exit gate**: Stage 1 accepts its prediction iff ``p_max >= thresholds[label]``
   (else ``default_threshold``). Otherwise the request falls through to the Stage-2
   classifier. A threshold of ``1.0`` means "never exit at Stage 1 for this label".

``k``, ``eps`` and the aggregation rule are written to the artifact's ``config.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from caliban_ml.artifacts import (
    ArtifactKind,
    ArtifactManifest,
    ArtifactRef,
    Calibration,
    build_manifest,
    write_manifest,
)
from caliban_ml.router.dataset import OOS_LABEL

FloatArray = np.ndarray


def l2_normalize(x: FloatArray, eps: float = 1e-12) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


@dataclass
class Neighbours:
    sims: FloatArray  # (n, k) cosine similarities, descending
    labels: FloatArray  # (n, k) int class indices


class KnnIntentClassifier:
    def __init__(self, k: int = 5, temperature: float = 0.05, smoothing: float = 1e-3):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.temperature = temperature
        self.smoothing = smoothing
        self.classes_: list[str] = []
        self._emb: FloatArray | None = None
        self._y: FloatArray | None = None

    def fit(self, embeddings: FloatArray, labels: list[str]) -> KnnIntentClassifier:
        if OOS_LABEL in labels:
            raise ValueError("OOS examples must not be used as kNN exemplars")
        emb = np.asarray(embeddings, dtype=np.float64)
        if emb.ndim != 2 or emb.shape[0] != len(labels):
            raise ValueError("embeddings must be (n, d) and match labels")
        self.classes_ = sorted(set(labels))
        index = {c: i for i, c in enumerate(self.classes_)}
        self._emb = l2_normalize(emb)
        self._y = np.array([index[c] for c in labels], dtype=np.int64)
        return self

    @property
    def n_classes(self) -> int:
        return len(self.classes_)

    def class_index(self, labels: list[str]) -> FloatArray:
        index = {c: i for i, c in enumerate(self.classes_)}
        unknown = sorted({lab for lab in labels if lab not in index and lab != OOS_LABEL})
        if unknown:
            raise ValueError(f"labels not seen in training: {unknown}")
        return np.array([index.get(lab, -1) for lab in labels], dtype=np.int64)

    def neighbours(self, queries: FloatArray) -> Neighbours:
        if self._emb is None or self._y is None:
            raise RuntimeError("classifier is not fitted")
        q = l2_normalize(np.atleast_2d(queries))
        sims = q @ self._emb.T
        k = min(self.k, sims.shape[1])
        top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        top_sims = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_sims, axis=1, kind="stable")
        top = np.take_along_axis(top, order, axis=1)
        return Neighbours(sims=np.take_along_axis(top_sims, order, axis=1), labels=self._y[top])

    def proba_from_neighbours(self, nb: Neighbours, temperature: float | None = None
                              ) -> FloatArray:
        t = self.temperature if temperature is None else temperature
        logits = nb.sims / t
        w = np.exp(logits - logits.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        p = np.zeros((nb.sims.shape[0], self.n_classes))
        np.add.at(p, (np.arange(p.shape[0])[:, None], nb.labels), w)
        return (1.0 - self.smoothing) * p + self.smoothing / self.n_classes

    def predict_proba(self, queries: FloatArray, temperature: float | None = None
                      ) -> tuple[FloatArray, FloatArray]:
        """Return (probabilities (n, C), top-1 similarity (n,))."""
        nb = self.neighbours(queries)
        return self.proba_from_neighbours(nb, temperature), nb.sims[:, 0]

    def predict(self, queries: FloatArray) -> list[str]:
        p, _ = self.predict_proba(queries)
        return [self.classes_[i] for i in p.argmax(axis=1)]


# ---------------------------------------------------------------------- calibration


def nll(proba: FloatArray, y: FloatArray) -> float:
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0))))


def expected_calibration_error(proba: FloatArray, y: FloatArray, n_bins: int = 10) -> float:
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(np.float64)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


DEFAULT_T_GRID = np.geomspace(0.002, 2.0, 80)


def fit_temperature(clf: KnnIntentClassifier, nb: Neighbours, y: FloatArray,
                    grid: FloatArray = DEFAULT_T_GRID) -> float:
    """Pick T minimising NLL on held-out in-scope examples (deterministic grid search)."""
    if len(y) == 0:
        raise ValueError("need at least one in-scope validation example")
    losses = [nll(clf.proba_from_neighbours(nb, t), y) for t in grid]
    return float(grid[int(np.argmin(losses))])


@dataclass
class OosThreshold:
    threshold: float
    strategy: str
    in_scope_acceptance: float  # fraction of in-scope queries that pass the gate
    oos_rejection: float | None  # fraction of OOS queries rejected (None if no OOS data)


def select_oos_threshold(in_scope_scores: FloatArray, oos_scores: FloatArray | None = None,
                         *, strategy: Literal["recall", "youden"] = "recall",
                         target_in_scope_recall: float = 0.95) -> OosThreshold:
    """Choose the OOS gate on a similarity-like score (higher = more in-scope).

    * ``recall``: the largest threshold that still accepts >= ``target_in_scope_recall``
      of in-scope queries. Needs no OOS data; OOS rejection is reported when available.
    * ``youden``: maximise ``in_scope_acceptance + oos_rejection - 1`` over all observed
      scores (ties broken towards the lower threshold, i.e. towards recall).
    """
    s_in = np.sort(np.asarray(in_scope_scores, dtype=np.float64))
    s_oos = None if oos_scores is None else np.asarray(oos_scores, dtype=np.float64)
    if s_in.size == 0:
        raise ValueError("need in-scope scores")
    if not 0.0 < target_in_scope_recall <= 1.0:
        raise ValueError("target_in_scope_recall must be in (0, 1]")

    if strategy == "recall":
        n_reject = int(np.floor((1.0 - target_in_scope_recall) * s_in.size + 1e-9))
        thr = float(s_in[n_reject])
    elif strategy == "youden":
        if s_oos is None or s_oos.size == 0:
            raise ValueError("youden strategy needs OOS scores")
        cands = np.unique(np.concatenate([s_in, s_oos]))
        acc = (s_in[None, :] >= cands[:, None]).mean(axis=1)
        rej = (s_oos[None, :] < cands[:, None]).mean(axis=1)
        thr = float(cands[int(np.argmax(acc + rej))])
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    acceptance = float((s_in >= thr).mean())
    rejection = None if s_oos is None or s_oos.size == 0 else float((s_oos < thr).mean())
    return OosThreshold(thr, strategy, acceptance, rejection)


def select_label_thresholds(proba: FloatArray, y: FloatArray, classes: list[str], *,
                            target_precision: float = 0.95, min_support: int = 3
                            ) -> tuple[dict[str, float], float]:
    """Per-label Stage-1 exit thresholds on calibrated max-probability.

    ``y`` uses -1 for OOS examples that slipped past the OOS gate: they count as wrong.
    For each predicted label, the threshold is the lowest confidence ``t`` such that the
    precision of predictions with ``conf >= t`` is >= ``target_precision``; ``1.0`` if no
    such ``t`` exists (never exit at Stage 1). Labels with fewer than ``min_support``
    predictions get no entry and use the returned global default threshold.
    """
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    correct = pred == y

    def lowest_ok(c: FloatArray, ok: FloatArray) -> float:
        order = np.argsort(-c, kind="stable")
        c, ok = c[order], ok[order]
        prec = np.cumsum(ok) / np.arange(1, len(ok) + 1)
        best = 1.0
        for i in range(len(c)):
            # only cut between distinct confidence values
            if (i + 1 == len(c) or c[i + 1] < c[i]) and prec[i] >= target_precision:
                best = float(c[i])
        return best

    thresholds: dict[str, float] = {}
    for ci, name in enumerate(classes):
        m = pred == ci
        if m.sum() >= min_support:
            thresholds[name] = lowest_ok(conf[m], correct[m])
    default = lowest_ok(conf, correct) if len(conf) else 1.0
    return thresholds, default


# ---------------------------------------------------------------------- end-to-end


@dataclass
class KnnCalibrationResult:
    classifier: KnnIntentClassifier
    calibration: Calibration
    oos: OosThreshold
    metrics: dict[str, float] = field(default_factory=dict)


def _macro_f1(pred: FloatArray, y: FloatArray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate(clf: KnnIntentClassifier, calibration: Calibration, emb: FloatArray,
             labels: list[str], prefix: str = "") -> dict[str, float]:
    y = clf.class_index(labels)
    nb = clf.neighbours(emb)
    p_cal = clf.proba_from_neighbours(nb, calibration.temperature)
    top1 = nb.sims[:, 0]
    in_scope = y >= 0
    gate = top1 >= (calibration.oos_threshold if calibration.oos_threshold is not None
                    else -np.inf)
    pred = p_cal.argmax(axis=1)
    conf = p_cal.max(axis=1)
    thr = np.array([
        calibration.thresholds.get(clf.classes_[c], calibration.default_threshold or 1.0)
        for c in pred
    ])
    exits = gate & (conf >= thr)
    # end-to-end decision with OOS as its own class (-1)
    final = np.where(gate, pred, -1)

    m: dict[str, float] = {}
    if in_scope.any():
        yi = y[in_scope]
        m["in_scope_accuracy"] = float((pred[in_scope] == yi).mean())
        m["in_scope_macro_f1"] = _macro_f1(pred[in_scope], yi, clf.n_classes)
        m["in_scope_acceptance"] = float(gate[in_scope].mean())
        m["nll"] = nll(p_cal[in_scope], yi)
        m["ece"] = expected_calibration_error(p_cal[in_scope], yi)
    if (~in_scope).any():
        m["oos_rejection"] = float((~gate[~in_scope]).mean())
    m["end_to_end_accuracy"] = float((final == y).mean())
    m["stage1_exit_rate"] = float(exits.mean())
    m["stage1_exit_precision"] = float((pred[exits] == y[exits]).mean()) if exits.any() else 0.0
    return {f"{prefix}{k}": v for k, v in m.items()}


def fit_and_calibrate(train_emb: FloatArray, train_labels: list[str], val_emb: FloatArray,
                      val_labels: list[str], *, k: int = 5,
                      oos_strategy: Literal["recall", "youden"] = "recall",
                      target_in_scope_recall: float = 0.95,
                      target_precision: float = 0.95) -> KnnCalibrationResult:
    """Fit the kNN head, calibrate T, pick the OOS gate and per-label exit thresholds."""
    clf = KnnIntentClassifier(k=k).fit(train_emb, train_labels)
    y = clf.class_index(val_labels)
    nb = clf.neighbours(val_emb)
    in_scope = y >= 0

    t_uncal = clf.temperature
    p_uncal = clf.proba_from_neighbours(nb, t_uncal)
    nb_in = Neighbours(nb.sims[in_scope], nb.labels[in_scope])
    t = fit_temperature(clf, nb_in, y[in_scope])
    clf.temperature = t
    p = clf.proba_from_neighbours(nb, t)

    top1 = nb.sims[:, 0]
    oos = select_oos_threshold(top1[in_scope], top1[~in_scope] if (~in_scope).any() else None,
                               strategy=oos_strategy,
                               target_in_scope_recall=target_in_scope_recall)
    passed = top1 >= oos.threshold
    thresholds, default = select_label_thresholds(
        p[passed], y[passed], clf.classes_, target_precision=target_precision
    )
    calibration = Calibration(
        method="temperature",
        temperature=t,
        thresholds=thresholds,
        default_threshold=default,
        oos_threshold=oos.threshold,
        oos_score="top1_similarity",
    )
    metrics = evaluate(clf, calibration, val_emb, val_labels, prefix="val_")
    metrics["val_nll_uncalibrated"] = nll(p_uncal[in_scope], y[in_scope])
    metrics["val_ece_uncalibrated"] = expected_calibration_error(p_uncal[in_scope], y[in_scope])
    return KnnCalibrationResult(clf, calibration, oos, metrics)


def write_knn_head_artifact(result: KnnCalibrationResult, out_dir: Path, *, name: str,
                            version: str, embedder: ArtifactRef, data_card: dict,
                            description: str = "") -> ArtifactManifest:
    """Write a kNN ``intent_head`` artifact: labels.json + config.json + manifest.json.

    The exemplars themselves are tenant data and live in the per-tenant HNSW index
    (embedded at runtime by ``embedder``); they are NOT part of the shipped artifact.
    """
    if embedder.kind != ArtifactKind.EMBEDDER:
        raise ValueError("embedder ref must have kind 'embedder'")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clf = result.classifier
    (out_dir / "labels.json").write_text(json.dumps(clf.classes_, indent=2) + "\n")
    (out_dir / "config.json").write_text(json.dumps({
        "type": "knn",
        "k": clf.k,
        "smoothing": clf.smoothing,
        "aggregation": "softmax_over_neighbours_sum_by_class",
        "similarity": "cosine",
    }, indent=2) + "\n")
    manifest = build_manifest(
        out_dir,
        kind=ArtifactKind.INTENT_HEAD,
        name=name,
        version=version,
        description=description or "kNN intent head (Stage 1) calibration",
        labels=clf.classes_,
        calibration=result.calibration.model_dump(),
        metrics=result.metrics,
        data_card=data_card,
        requires=[embedder.model_dump()],
    )
    write_manifest(manifest, out_dir)
    return manifest
