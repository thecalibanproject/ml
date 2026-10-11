"""Calibration of logit heads (Stage 2), pure numpy.

The Stage-2 classifier emits raw logits; the Rust side applies the manifest's
:class:`~caliban_ml.artifacts.Calibration` exactly like it does for Stage 1::

    p          = softmax(intent_logits / temperature)
    OOS        if p_max < oos_threshold                       (oos_score = max_probability)
    exit       if p_max >= thresholds[label] (else default_threshold)

Threshold selection reuses the kNN helpers (:func:`select_oos_threshold`,
:func:`select_label_thresholds`), so both stages share one definition of "exit" and "OOS".

An optional sigmoid ``oos`` head (multi-head backbone) is calibrated separately: its
:class:`HeadSpec` calibration carries a temperature and ``thresholds["oos"]``; the head
*exits* (declares the query out-of-scope) when ``sigmoid(oos_logit / T) >= thresholds["oos"]``,
the same "exit when p[label] >= threshold" rule as every other head.

``y`` arrays use class indices into ``classes`` and ``-1`` for out-of-scope examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from caliban_ml.artifacts import Calibration
from caliban_ml.router.knn import (
    OosThreshold,
    _macro_f1,
    expected_calibration_error,
    nll,
    select_label_thresholds,
    select_oos_threshold,
)

DEFAULT_T_GRID = np.geomspace(0.05, 20.0, 120)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / temperature
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable


def fit_softmax_temperature(logits: np.ndarray, y: np.ndarray,
                            grid: np.ndarray = DEFAULT_T_GRID) -> float:
    """Temperature minimising NLL of in-scope examples (deterministic grid search)."""
    y = np.asarray(y)
    if y.size == 0:
        raise ValueError("need at least one in-scope example to fit a temperature")
    losses = [nll(softmax(logits, t), y) for t in grid]
    return float(grid[int(np.argmin(losses))])


def _bce(p: np.ndarray, t: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def fit_sigmoid_temperature(logits: np.ndarray, targets: np.ndarray,
                            grid: np.ndarray = DEFAULT_T_GRID) -> float:
    targets = np.asarray(targets, dtype=np.float64)
    if targets.size == 0:
        raise ValueError("need examples to fit a temperature")
    losses = [_bce(sigmoid(logits, t), targets) for t in grid]
    return float(grid[int(np.argmin(losses))])


@dataclass
class HeadCalibrationResult:
    calibration: Calibration
    oos: OosThreshold
    metrics: dict[str, float] = field(default_factory=dict)


def evaluate_logit_head(logits: np.ndarray, y: np.ndarray, classes: list[str],
                        calibration: Calibration, prefix: str = "") -> dict[str, float]:
    """Metrics for the primary softmax head under ``calibration`` (OOS via max_probability)."""
    y = np.asarray(y, dtype=np.int64)
    p = softmax(logits, calibration.temperature if calibration.method == "temperature" else 1.0)
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    in_scope = y >= 0
    gate = conf >= (calibration.oos_threshold if calibration.oos_threshold is not None
                    else -np.inf)
    thr = np.array([calibration.thresholds.get(classes[c], calibration.default_threshold or 1.0)
                    for c in pred])
    exits = gate & (conf >= thr)
    final = np.where(gate, pred, -1)
    m: dict[str, float] = {}
    if in_scope.any():
        yi = y[in_scope]
        m["in_scope_accuracy"] = float((pred[in_scope] == yi).mean())
        m["in_scope_macro_f1"] = _macro_f1(pred[in_scope], yi, len(classes))
        m["in_scope_acceptance"] = float(gate[in_scope].mean())
        m["nll"] = nll(p[in_scope], yi)
        m["ece"] = expected_calibration_error(p[in_scope], yi)
    if (~in_scope).any():
        m["oos_rejection"] = float((~gate[~in_scope]).mean())
    m["end_to_end_accuracy"] = float((final == y).mean())
    m["stage2_exit_rate"] = float(exits.mean())
    m["stage2_exit_precision"] = float((pred[exits] == y[exits]).mean()) if exits.any() else 0.0
    m["n"] = float(len(y))
    return {f"{prefix}{k}": v for k, v in m.items()}


def calibrate_logit_head(logits: np.ndarray, y: np.ndarray, classes: list[str], *,
                         oos_strategy: Literal["recall", "youden"] = "recall",
                         target_in_scope_recall: float = 0.95,
                         target_precision: float = 0.95) -> HeadCalibrationResult:
    """Temperature, OOS gate (on calibrated max-probability) and per-label exit thresholds,
    fitted on held-out validation logits."""
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if logits.ndim != 2 or logits.shape != (len(y), len(classes)):
        raise ValueError(f"logits must be (n, {len(classes)}) and match y")
    in_scope = y >= 0
    t = fit_softmax_temperature(logits[in_scope], y[in_scope])
    p = softmax(logits, t)
    conf = p.max(axis=1)
    if oos_strategy == "youden" and not (~in_scope).any():
        oos_strategy = "recall"  # no OOS examples in val: fall back to the recall target
    oos = select_oos_threshold(conf[in_scope], conf[~in_scope] if (~in_scope).any() else None,
                               strategy=oos_strategy,
                               target_in_scope_recall=target_in_scope_recall)
    passed = conf >= oos.threshold
    thresholds, default = select_label_thresholds(p[passed], y[passed], classes,
                                                  target_precision=target_precision)
    calibration = Calibration(method="temperature", temperature=t, thresholds=thresholds,
                              default_threshold=default, oos_threshold=oos.threshold,
                              oos_score="max_probability")
    metrics = evaluate_logit_head(logits, y, classes, calibration, prefix="val_")
    metrics["val_nll_uncalibrated"] = nll(softmax(logits[in_scope]), y[in_scope])
    return HeadCalibrationResult(calibration, oos, metrics)


def calibrate_oos_head(logits: np.ndarray, is_oos: np.ndarray, *,
                       target_in_scope_recall: float = 0.95) -> tuple[Calibration,
                                                                      dict[str, float]]:
    """Calibrate a sigmoid OOS head: temperature by BCE, then the largest P(oos) threshold
    that still keeps ``target_in_scope_recall`` of in-scope validation queries."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    is_oos = np.asarray(is_oos, dtype=bool).reshape(-1)
    t = fit_sigmoid_temperature(logits, is_oos.astype(np.float64))
    p_oos = sigmoid(logits, t)
    # select_oos_threshold works on "higher = more in-scope" scores.
    sel = select_oos_threshold(1.0 - p_oos[~is_oos],
                               (1.0 - p_oos[is_oos]) if is_oos.any() else None,
                               strategy="recall", target_in_scope_recall=target_in_scope_recall)
    # in-scope accepted iff 1 - p >= s  <=>  p <= 1 - s; declare OOS when p > 1 - s.
    # Nudge up so the boundary example stays in scope under ">= threshold => OOS".
    thr = float(min(1.0, np.nextafter(1.0 - sel.threshold, 2.0)))
    cal = Calibration(method="temperature", temperature=t, thresholds={"oos": thr},
                      default_threshold=thr)
    m = {"val_oos_head_in_scope_acceptance": float((p_oos[~is_oos] < thr).mean())
         if (~is_oos).any() else 0.0}
    if is_oos.any():
        m["val_oos_head_oos_rejection"] = float((p_oos[is_oos] >= thr).mean())
    return cal, m
