"""Stage-2 intent head: calibration, splits, checkpoint and manifest contract, export helpers.

Everything here runs on the default install (no torch, no onnxruntime). The torch/ONNX
paths are covered by ``test_stage2_integration.py`` when the extras are installed.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from caliban_ml._extras import MissingExtraError
from caliban_ml.artifacts import Calibration, load_manifest, verify_artifact_dir
from caliban_ml.cli import app
from caliban_ml.router import OOS_LABEL, IntentDataset
from caliban_ml.router.calibrate import (
    calibrate_logit_head,
    calibrate_oos_head,
    evaluate_logit_head,
    fit_softmax_temperature,
    sigmoid,
    softmax,
)
from caliban_ml.router.checkpoint import (
    CheckpointHead,
    CheckpointMeta,
    read_splits,
    write_splits,
)
from caliban_ml.router.embed import embed_onnx_artifact, pool_hidden
from caliban_ml.router.export import (
    ONNX_CLASSIFIER_TYPE,
    ParityError,
    build_intent_head_manifest,
    check_no_remote_code,
    clean_tokenizer_json,
    onnx_spec,
    pad_batch,
    parity_metrics,
    parity_texts,
)
from caliban_ml.router.train import (
    IntentHeadConfig,
    check_heads,
    dataset_from_examples,
    encode_labels,
    hold_out_intents,
    knn_baseline_metrics,
    scored_eval_sets,
    three_way_split,
    train_intent_head,
)
from conftest import DATA, FAKE_REV, REPO

CLASSES = ["code", "extraction", "sql_analytics", "summarization"]
HAS_TORCH = importlib.util.find_spec("torch") is not None


def sample() -> IntentDataset:
    return IntentDataset.from_yaml(DATA / "intents" / "sample.yaml")


def synthetic_logits(n_per: int = 40, c: int = 4, sharp: float = 6.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(c), n_per)
    logits = rng.normal(0, 1, (len(y), c))
    logits[np.arange(len(y)), y] += sharp * rng.uniform(0.2, 1.0, len(y))
    return logits, y


# ---------------------------------------------------------------------- calibration


def test_softmax_and_sigmoid_are_stable():
    p = softmax(np.array([[1000.0, 0.0], [0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1) and p[0, 0] == pytest.approx(1.0)
    assert sigmoid(np.array([-1000.0, 0.0, 1000.0])) == pytest.approx([0.0, 0.5, 1.0])


def test_temperature_softens_overconfident_logits():
    logits, y = synthetic_logits()
    noisy = y.copy()
    noisy[::5] = (noisy[::5] + 1) % 4  # 20% label noise: sharp logits are overconfident
    assert fit_softmax_temperature(logits * 10, noisy) > 1.0


def test_calibrate_logit_head_uses_stage1_semantics():
    logits, y = synthetic_logits()
    oos_logits = np.random.default_rng(1).normal(0, 0.3, (20, 4))  # flat: low max-prob
    lg = np.concatenate([logits, oos_logits])
    yy = np.concatenate([y, -np.ones(20, dtype=np.int64)])
    res = calibrate_logit_head(lg, yy, CLASSES, oos_strategy="youden",
                               target_precision=0.9)
    cal = res.calibration
    assert cal.oos_score == "max_probability" and cal.method == "temperature"
    assert set(cal.thresholds) <= set(CLASSES)
    m = res.metrics
    assert m["val_in_scope_accuracy"] > 0.9
    assert m["val_oos_rejection"] > 0.8
    assert 0 <= m["val_stage2_exit_rate"] <= 1
    # evaluating with the same calibration reproduces the val numbers
    again = evaluate_logit_head(lg, yy, CLASSES, cal, prefix="val_")
    assert again["val_end_to_end_accuracy"] == pytest.approx(m["val_end_to_end_accuracy"])


def test_calibrate_logit_head_without_oos_falls_back_to_recall():
    logits, y = synthetic_logits()
    res = calibrate_logit_head(logits, y, CLASSES, oos_strategy="youden",
                               target_in_scope_recall=0.9)
    assert res.oos.strategy == "recall"
    assert res.metrics["val_in_scope_acceptance"] >= 0.9
    with pytest.raises(ValueError, match="logits must be"):
        calibrate_logit_head(logits[:, :3], y, CLASSES)


def test_calibrate_oos_head_threshold_direction():
    rng = np.random.default_rng(0)
    lg = np.concatenate([rng.normal(-4, 1, 50), rng.normal(4, 1, 10)])
    is_oos = np.array([False] * 50 + [True] * 10)
    cal, m = calibrate_oos_head(lg, is_oos, target_in_scope_recall=0.95)
    assert isinstance(cal, Calibration) and set(cal.thresholds) == {"oos"}
    p = sigmoid(lg, cal.temperature)
    thr = cal.thresholds["oos"]
    # "exit" (declare OOS) when p >= threshold, like every other head
    assert (p[~is_oos] < thr).mean() >= 0.95
    assert m["val_oos_head_oos_rejection"] == pytest.approx((p[is_oos] >= thr).mean())
    assert m["val_oos_head_oos_rejection"] > 0.9


# ---------------------------------------------------------------------- splits / eval sets


def test_three_way_split_is_disjoint_stratified_and_deterministic():
    ds = sample()
    s = three_way_split(ds, val_fraction=0.25, test_fraction=0.25, seed=3)
    texts = [t for part in s.values() for t in part[0]]
    assert len(texts) == len(set(texts)) == len(ds.examples()[0])
    assert set(s["train"][1]) == set(CLASSES)  # every intent keeps a training exemplar
    assert OOS_LABEL not in s["train"][1]  # 3 OOS examples: all held out
    assert s == three_way_split(ds, val_fraction=0.25, test_fraction=0.25, seed=3)
    s0 = three_way_split(ds, val_fraction=0.25, test_fraction=0.0, seed=3)
    assert s0["test"] == ([], [])


def test_dataset_from_examples_roundtrip():
    ds = sample()
    again = dataset_from_examples(*ds.examples(), like=ds)
    assert again.labels == ds.labels and again.oos == ds.oos
    assert again.intents["code"].description == ds.intents["code"].description


def test_hold_out_intents_and_eval_sets():
    ds = sample()
    kept, (texts, labels) = hold_out_intents(ds, ("code",))
    assert kept.labels == ["extraction", "sql_analytics", "summarization"]
    assert texts == ds.intents["code"].utterances and set(labels) == {OOS_LABEL}
    with pytest.raises(ValueError, match="not in the dataset"):
        hold_out_intents(ds, ("nope",))
    with pytest.raises(ValueError, match="at least two"):
        hold_out_intents(ds, ("code", "extraction", "sql_analytics"))
    splits = three_way_split(kept, val_fraction=0.25, test_fraction=0.25, seed=0)
    sets = scored_eval_sets(splits, (texts, labels), {"fr": (["bonjour"], [OOS_LABEL])},
                            kept.labels)
    assert list(sets) == ["test", "heldout_intents", "fr"]
    with pytest.raises(ValueError, match="not in the label space"):
        scored_eval_sets(splits, ([], []), {"x": (["a"], ["code"])}, kept.labels)
    with pytest.raises(ValueError, match="invalid extra eval set name"):
        scored_eval_sets(splits, ([], []), {"test": (["a"], [OOS_LABEL])}, kept.labels)


def test_encode_labels():
    assert encode_labels(["code", OOS_LABEL, "summarization"], CLASSES).tolist() == [0, -1, 3]
    with pytest.raises(ValueError):
        encode_labels(["chat"], CLASSES)


def test_knn_baseline_metrics_on_separable_embeddings():
    rng = np.random.default_rng(0)
    centers = np.eye(8)[:4] * 5

    def make(n, labels):
        idx = [CLASSES.index(lab) if lab != OOS_LABEL else None for lab in labels]
        return np.stack([centers[i] + rng.normal(0, 0.3, 8) if i is not None
                         else rng.normal(0, 1, 8) * np.r_[0, 0, 0, 0, 1, 1, 1, 1] * 5
                         for i in idx])

    lab_tr = [c for c in CLASSES for _ in range(10)] + [OOS_LABEL] * 3
    lab_va = [c for c in CLASSES for _ in range(5)] + [OOS_LABEL] * 5
    lab_te = [c for c in CLASSES for _ in range(5)]
    splits = {"train": ([""] * len(lab_tr), lab_tr), "val": ([""] * len(lab_va), lab_va),
              "test": ([""] * len(lab_te), lab_te)}
    emb = {k: make(len(v[1]), v[1]) for k, v in splits.items()}
    cfg = IntentHeadConfig(Path("."), "x", FAKE_REV, "apache-2.0", oos_strategy="youden")
    m = knn_baseline_metrics(emb, splits, {"test": splits["test"]}, k=3, prefix="knn_x_",
                             config=cfg)
    assert m["knn_x_test_in_scope_accuracy"] == 1.0
    assert m["knn_x_val_oos_rejection"] == 1.0


# ---------------------------------------------------------------------- checkpoint + manifest


def meta(**over) -> CheckpointMeta:
    cal = Calibration(temperature=1.4, thresholds={"code": 0.8}, default_threshold=0.7,
                      oos_threshold=0.5, oos_score="max_probability")
    fields = {
        "backbone": "modernbert_multihead",
        "base_model": {"id": "answerdotai/ModernBERT-base", "revision": FAKE_REV,
                       "licence": "apache-2.0"},
        "pooling": "mean", "max_length": 128, "hidden_size": 768,
        "heads": [
            CheckpointHead(name="intent", activation="softmax", labels=CLASSES,
                           calibration=cal),
            CheckpointHead(name="oos", activation="sigmoid", labels=["oos"],
                           calibration=Calibration(temperature=1.0, thresholds={"oos": 0.6},
                                                   default_threshold=0.6)),
        ],
        "metrics": {"test_in_scope_accuracy": 0.9},
        "data_card": {"training_datasets": [{"name": "core-default-exemplars",
                                             "source": "core", "licence": "caliban-internal",
                                             "synthetic": True}],
                      "intended_use": "Stage-2 intent classification."},
    }
    fields.update(over)
    return CheckpointMeta.model_validate(fields)


def test_checkpoint_meta_roundtrip(tmp_path: Path):
    m = meta()
    m.write(tmp_path)
    again = CheckpointMeta.read(tmp_path)
    assert again == m
    assert json.loads((tmp_path / "calibration.json").read_text())["oos"]["thresholds"] == {
        "oos": 0.6}
    assert m.heads[1].output == "oos_logits" and m.heads[1].width == 1
    write_splits(tmp_path, {"val": (["a"], ["code"])})
    assert read_splits(tmp_path) == {"val": (["a"], ["code"])}
    with pytest.raises(ValueError, match="first head"):
        meta(heads=list(reversed(m.heads)))


def _fake_artifact(d: Path) -> None:
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"\x08\x07fake" * 10)
    (d / "tokenizer.json").write_text('{"version": "1.0", "model": {"type": "BPE"}}')
    (d / "labels.json").write_text(json.dumps(CLASSES))
    (d / "config.json").write_text(json.dumps({"type": ONNX_CLASSIFIER_TYPE,
                                               "primary_head": "intent"}))


def test_intent_head_manifest_contract(tmp_path: Path):
    d = tmp_path / "art"
    _fake_artifact(d)
    m = build_intent_head_manifest(d, meta(), name="tenant-intents", version="1.0.0",
                                   opset=17, quantization="int8_dynamic",
                                   metrics={"export_argmax_agree": 1.0})
    rep = verify_artifact_dir(d, strict=True)
    assert rep.ok, rep.errors
    loaded = load_manifest(d)
    assert loaded.kind.value == "intent_head" and loaded.labels == CLASSES
    assert loaded.calibration.oos_score == "max_probability"
    assert [t.name for t in loaded.onnx.inputs] == ["input_ids", "attention_mask"]
    assert {t.dtype for t in loaded.onnx.inputs} == {"int64"}
    assert [t.name for t in loaded.onnx.outputs] == ["intent_logits", "oos_logits"]
    assert loaded.onnx.outputs[0].shape == ["batch", 4]
    assert [h.activation for h in loaded.onnx.heads] == ["softmax", "sigmoid"]
    assert loaded.requires == []  # self-contained: no embedder dependency
    assert m.metrics["export_argmax_agree"] == 1.0 and "test_in_scope_accuracy" in m.metrics
    roles = {f.path: f.role.value for f in loaded.files}
    assert roles == {"model.onnx": "model", "tokenizer.json": "tokenizer",
                     "labels.json": "labels", "config.json": "config"}


def test_intent_head_manifest_refuses_noncommercial_base(tmp_path: Path):
    d = tmp_path / "art"
    _fake_artifact(d)
    bad = meta(base_model={"id": "x/nc", "revision": FAKE_REV, "licence": "cc-by-nc-4.0"})
    with pytest.raises(ValueError, match="licence policy"):
        build_intent_head_manifest(d, bad, name="x", version="1.0.0", opset=17,
                                   quantization="none")


def test_onnx_spec_matches_heads():
    spec = onnx_spec(meta(), opset=18, quantization="none")
    assert spec["opset"] == 18
    assert [h["output"] for h in spec["heads"]] == [o["name"] for o in spec["outputs"]]


# ---------------------------------------------------------------------- export helpers


def test_parity_metrics():
    a = np.array([[1.0, 2.0], [3.0, 1.0]])
    assert parity_metrics(a, a + 1e-4) == (pytest.approx(1e-4), 1.0)
    assert parity_metrics(a, a[:, ::-1])[1] == 0.0
    assert parity_metrics(np.array([[0.5], [-1.0]]), np.array([[0.4], [1.0]]))[1] == 0.5
    with pytest.raises(ParityError):
        parity_metrics(a, a[:1])


def test_pad_batch():
    ids, mask = pad_batch([[5, 6, 7], [8]], pad_id=0)
    assert ids.tolist() == [[5, 6, 7], [8, 0, 0]] and mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert ids.dtype == np.int64


def test_clean_tokenizer_json_and_remote_code(tmp_path: Path):
    src = tmp_path / "tokenizer.json"
    src.write_text(json.dumps({"version": "1.0", "model": {}, "truncation": {"max_length": 8},
                               "padding": {"pad_id": 0}}))
    out = clean_tokenizer_json(src, tmp_path / "out.json")
    assert out["truncation"] is None and out["padding"] is None
    with pytest.raises(ValueError, match="not an HF tokenizers"):
        (tmp_path / "bad.json").write_text("{}")
        clean_tokenizer_json(tmp_path / "bad.json", tmp_path / "x.json")
    check_no_remote_code(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"auto_map": {}}))
    with pytest.raises(ValueError, match="trust_remote_code"):
        check_no_remote_code(tmp_path)


def test_parity_texts_reads_splits(tmp_path: Path):
    write_splits(tmp_path, {"train": (["t1"], ["code"]), "val": (["v1", "v1"], ["code"] * 2),
                            "test": (["x1"], ["code"])})
    texts = parity_texts(tmp_path)
    assert texts[:3] == ["v1", "x1", "t1"] and len(texts[-1]) > len("v1 x1 t1")
    with pytest.raises(ValueError, match="no parity texts"):
        parity_texts(tmp_path / "missing")


# ---------------------------------------------------------------------- embed (ONNX backend)


def test_pool_hidden_modes():
    h = np.arange(2 * 3 * 2, dtype=np.float64).reshape(2, 3, 2)
    mask = np.array([[1, 1, 0], [1, 0, 0]])
    assert pool_hidden(h, mask, "cls").tolist() == [[0, 1], [6, 7]]
    assert pool_hidden(h, mask, "mean").tolist() == [[1, 2], [6, 7]]
    assert pool_hidden(h, mask, "last").tolist() == [[2, 3], [6, 7]]
    assert pool_hidden(h[:, 0], mask, "none").tolist() == [[0, 1], [6, 7]]
    with pytest.raises(ValueError):
        pool_hidden(h, mask, "none")


def test_embed_onnx_artifact_verifies_before_loading(tmp_path: Path, artifact_dir: Path):
    from caliban_ml.artifacts import build_manifest, write_manifest
    from conftest import pii_ner_fields

    write_manifest(build_manifest(artifact_dir, **pii_ner_fields()), artifact_dir)
    with pytest.raises(ValueError, match="not an embedder"):
        embed_onnx_artifact(artifact_dir, ["hello"])
    (artifact_dir / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match="failed verification"):
        embed_onnx_artifact(artifact_dir, ["hello"])


# ---------------------------------------------------------------------- training entry point


def _cfg(**over) -> IntentHeadConfig:
    fields = {"base_model_path": Path("/nonexistent"), "base_model_id": "x/base",
              "base_model_revision": FAKE_REV, "base_model_licence": "apache-2.0"}
    fields.update(over)
    return IntentHeadConfig(**fields)


def test_train_refuses_bad_licences_and_heads_before_importing_torch():
    with pytest.raises(ValueError, match="denied"):
        train_intent_head(sample(), _cfg(base_model_licence="cc-by-nc-4.0"))
    with pytest.raises(ValueError, match="review_required"):
        train_intent_head(sample(), _cfg(dataset_licence="unknown"))
    with pytest.raises(ValueError, match="need labels"):
        check_heads(_cfg(heads=("intent", "difficulty")))
    with pytest.raises(ValueError, match="start with 'intent'"):
        check_heads(_cfg(heads=("oos",)))


@pytest.mark.skipif(HAS_TORCH, reason="torch is installed")
def test_train_names_the_missing_extra():
    with pytest.raises(MissingExtraError, match=r"caliban-ml\[train\]"):
        train_intent_head(sample(), _cfg())


def test_light_package_never_imports_heavy_deps():
    code = ("import sys; import caliban_ml.cli, caliban_ml.router.train, "
            "caliban_ml.router.export, caliban_ml.router.embed, caliban_ml.router.calibrate, "
            "caliban_ml.router.checkpoint; "
            "bad = [m for m in ('torch', 'onnxruntime', 'onnx', 'transformers', 'setfit') "
            "if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_cli_router_train_and_export_help():
    runner = CliRunner()
    for cmd in ("train", "export"):
        r = runner.invoke(app, ["router", cmd, "--help"])
        assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["router", "embed", "--dataset", str(DATA / "intents" /
                                                                 "sample.yaml"),
                            "--out-prefix", "x"])
    assert r.exit_code != 0 and "exactly one" in r.output


def test_multilingual_eval_set_parses():
    ds = IntentDataset.from_yaml(DATA / "intents" / "multilingual_eval.yaml")
    assert ds.labels == ["analytics", "chat", "code", "extraction", "reasoning", "summarize",
                         "translate"]
    texts, _ = ds.examples()
    assert any("؀" <= ch <= "ۿ" for t in texts for ch in t)  # Arabic present
