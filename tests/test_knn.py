from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from caliban_ml.artifacts import ArtifactRef, verify_artifact_dir
from caliban_ml.router import (
    OOS_LABEL,
    IntentDataset,
    KnnIntentClassifier,
    fit_and_calibrate,
    select_label_thresholds,
    select_oos_threshold,
    write_knn_head_artifact,
)
from caliban_ml.router.knn import expected_calibration_error, nll
from conftest import DATA

INTENTS = ["code", "extraction", "sql_analytics", "summarization"]


def synthetic(seed: int = 0, dim: int = 32, per_class: int = 40, n_oos: int = 40,
              noise: float = 0.35):
    """Gaussian blobs around random unit centroids; OOS points come from other directions."""
    rng = np.random.default_rng(seed)
    centroids = rng.normal(size=(len(INTENTS), dim))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    x, y = [], []
    for c, name in zip(centroids, INTENTS, strict=True):
        x.append(c + noise * rng.normal(size=(per_class, dim)) / np.sqrt(dim) * 3)
        y += [name] * per_class
    oos_dirs = rng.normal(size=(n_oos, dim))
    x.append(oos_dirs)
    y += [OOS_LABEL] * n_oos
    return np.vstack(x), y


def split(x, y, seed=1, val_frac=0.5):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_val = int(len(y) * val_frac)
    va, tr = idx[:n_val], idx[n_val:]
    tr = [i for i in tr if y[i] != OOS_LABEL]  # OOS never used as exemplars
    return x[tr], [y[i] for i in tr], x[va], [y[i] for i in va]


def test_knn_predicts_blobs():
    x, y = synthetic()
    trx, try_, vax, vay = split(x, y)
    clf = KnnIntentClassifier(k=5).fit(trx, try_)
    assert clf.classes_ == INTENTS
    mask = [lab != OOS_LABEL for lab in vay]
    preds = clf.predict(vax[mask])
    acc = np.mean([p == t for p, t in zip(preds, np.array(vay)[mask], strict=True)])
    assert acc > 0.95


def test_proba_rows_sum_to_one_and_smoothing():
    x, y = synthetic()
    trx, try_, vax, _ = split(x, y)
    clf = KnnIntentClassifier(k=3, smoothing=1e-3).fit(trx, try_)
    p, top1 = clf.predict_proba(vax)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert (p > 0).all()  # smoothing: unseen classes are never exactly 0
    assert np.all(top1 <= 1.0 + 1e-9)


def test_oos_examples_cannot_be_exemplars():
    with pytest.raises(ValueError, match="OOS"):
        KnnIntentClassifier().fit(np.eye(2), ["a", OOS_LABEL])


def test_temperature_calibration_does_not_increase_nll():
    x, y = synthetic(noise=0.6)
    trx, try_, vax, vay = split(x, y)
    res = fit_and_calibrate(trx, try_, vax, vay, k=7)
    assert res.calibration.temperature > 0
    assert res.metrics["val_nll"] <= res.metrics["val_nll_uncalibrated"] + 1e-12
    # Hand check against the module functions
    clf = res.classifier
    yv = clf.class_index(vay)
    m = yv >= 0
    p_cal, _ = clf.predict_proba(vax[m])
    assert nll(p_cal, yv[m]) == pytest.approx(res.metrics["val_nll"])
    assert 0.0 <= expected_calibration_error(p_cal, yv[m]) <= 1.0


def test_oos_gate_meets_recall_target_and_rejects_oos():
    x, y = synthetic()
    trx, try_, vax, vay = split(x, y)
    res = fit_and_calibrate(trx, try_, vax, vay, k=5, target_in_scope_recall=0.95)
    assert res.calibration.oos_score == "top1_similarity"
    assert res.oos.in_scope_acceptance >= 0.95
    assert res.metrics["val_in_scope_acceptance"] >= 0.95
    assert res.metrics["val_oos_rejection"] > 0.9  # random directions are far from blobs


def test_select_oos_threshold_recall_exact():
    in_scope = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95])
    oos = np.array([0.20, 0.55, 0.65])
    t = select_oos_threshold(in_scope, oos, strategy="recall", target_in_scope_recall=0.9)
    # 10% of 10 = reject exactly 1 in-scope point -> threshold is the 2nd lowest score
    assert t.threshold == pytest.approx(0.60)
    assert t.in_scope_acceptance == pytest.approx(0.9)
    assert t.oos_rejection == pytest.approx(2 / 3)
    t100 = select_oos_threshold(in_scope, None, target_in_scope_recall=1.0)
    assert t100.threshold == pytest.approx(0.50) and t100.oos_rejection is None


def test_select_oos_threshold_youden():
    in_scope = np.array([0.7, 0.8, 0.9])
    oos = np.array([0.1, 0.2, 0.75])
    t = select_oos_threshold(in_scope, oos, strategy="youden")
    # threshold 0.7 -> acc 1.0, rej 2/3 ; 0.8 -> acc 2/3, rej 1.0 ; tie -> lower threshold
    assert t.threshold == pytest.approx(0.7)
    with pytest.raises(ValueError):
        select_oos_threshold(in_scope, None, strategy="youden")


def test_label_thresholds_hit_precision_target():
    # class 0: confident & right at 0.9+, wrong at 0.6 ; class 1: always right
    proba = np.array([
        [0.95, 0.05], [0.92, 0.08], [0.90, 0.10], [0.60, 0.40], [0.58, 0.42],
        [0.10, 0.90], [0.20, 0.80], [0.30, 0.70],
    ])
    y = np.array([0, 0, 0, 1, -1, 1, 1, 1])  # -1 = OOS that slipped through: always wrong
    thr, default = select_label_thresholds(proba, y, ["a", "b"], target_precision=0.95,
                                           min_support=3)
    assert thr["a"] == pytest.approx(0.90)
    assert thr["b"] == pytest.approx(0.70)
    assert 0.0 < default <= 1.0


def test_label_threshold_unreachable_is_one():
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    y = np.array([1, 1, 1])  # always wrong
    thr, _ = select_label_thresholds(proba, y, ["a", "b"], min_support=3)
    assert thr["a"] == 1.0


def test_intent_dataset_loads_and_splits():
    ds = IntentDataset.from_yaml(DATA / "intents" / "sample.yaml")
    assert ds.labels == INTENTS
    (tr_t, tr_y), (va_t, va_y) = ds.split(val_fraction=0.25, seed=0)
    assert set(tr_y) == set(INTENTS)  # every intent keeps train exemplars
    assert OOS_LABEL not in tr_y and va_y.count(OOS_LABEL) == 3
    assert not set(tr_t) & set(va_t)


def test_intent_dataset_rejects_conflicts():
    with pytest.raises(ValueError, match="appears under both"):
        IntentDataset.model_validate({"intents": {"a": {"utterances": ["Hi there"]},
                                                  "b": {"utterances": ["hi  THERE"]}}})
    with pytest.raises(ValueError, match="invalid intent id"):
        IntentDataset.model_validate({"intents": {"Bad-Id": {"utterances": ["x"]}}})


def test_knn_head_artifact_roundtrip(tmp_path: Path):
    x, y = synthetic()
    trx, try_, vax, vay = split(x, y)
    res = fit_and_calibrate(trx, try_, vax, vay)
    out = tmp_path / "knn-head"
    m = write_knn_head_artifact(
        res, out, name="global-intents-knn", version="0.1.0",
        embedder=ArtifactRef(kind="embedder", name="bge-small-en", version="1.5.0"),
        data_card={"intended_use": "Stage-1 exit thresholds",
                   "training_datasets": [{"name": "synthetic", "source": "tests",
                                          "licence": "caliban-internal", "synthetic": True}]},
    )
    assert m.kind.value == "intent_head" and m.labels == INTENTS
    assert verify_artifact_dir(out, strict=True).ok
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["k"] == 5 and cfg["type"] == "knn"
