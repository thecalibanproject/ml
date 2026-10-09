from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from caliban_ml.artifacts import load_manifest, verify_artifact_dir
from caliban_ml.router import ProfileCluster, RouterProfile, compute_profile, kmeans
from caliban_ml.router.profile import write_profile_artifact

RESULTS = [
    # model A: sql 1,1,0 ; code 0.5
    {"model": "A", "probe_id": "s1", "cluster": "sql", "score": 1.0},
    {"model": "A", "probe_id": "s2", "cluster": "sql", "score": 1.0},
    {"model": "A", "probe_id": "s3", "cluster": "sql", "score": 0.0},
    {"model": "A", "probe_id": "c1", "cluster": "code", "score": 0.5},
    # model B: sql 0 ; code 1,1 ; one failed call
    {"model": "B", "probe_id": "s1", "cluster": "sql", "score": 0.0},
    {"model": "B", "probe_id": "c1", "cluster": "code", "score": 1.0},
    {"model": "B", "probe_id": "c2", "cluster": "code", "score": 1.0},
    {"model": "B", "probe_id": "s2", "cluster": "sql", "error": "HTTP 503"},
]


def row(p: RouterProfile, model: str):
    return next(m for m in p.models if m.model == model)


def test_raw_means_and_counts_without_shrinkage():
    p = compute_profile(RESULTS, name="t", probe_set="x", prior_weight=0.0)
    assert [c.id for c in p.clusters] == ["code", "sql"]
    a, b = row(p, "A"), row(p, "B")
    assert a.counts == [1, 3] and a.raw_mean == pytest.approx([0.5, 2 / 3])
    assert a.quality == pytest.approx([0.5, 2 / 3])
    assert a.prior == pytest.approx(2.5 / 4)
    assert b.counts == [2, 1] and b.quality == pytest.approx([1.0, 0.0])
    assert b.n_errors == 1  # excluded, not scored as 0
    assert [c.n_probes for c in p.clusters] == [2, 3]


def test_shrinkage_toward_model_prior():
    w = 2.0
    p = compute_profile(RESULTS, name="t", probe_set="x", prior_weight=w)
    a = row(p, "A")
    prior = 2.5 / 4
    assert a.quality[0] == pytest.approx((0.5 + w * prior) / (1 + w))
    assert a.quality[1] == pytest.approx((2.0 + w * prior) / (3 + w))
    # shrunk values sit between the raw mean and the prior
    for q, r in zip(a.quality, a.raw_mean, strict=True):
        assert min(r, prior) - 1e-12 <= q <= max(r, prior) + 1e-12


def test_unseen_cluster_falls_back_to_prior():
    clusters = [ProfileCluster(id="code", n_probes=0), ProfileCluster(id="sql", n_probes=0),
                ProfileCluster(id="math", n_probes=0)]
    p = compute_profile(RESULTS, name="t", probe_set="x", prior_weight=0.0, clusters=clusters)
    a = row(p, "A")
    assert a.counts[2] == 0 and a.raw_mean[2] is None
    assert a.quality[2] == pytest.approx(a.prior)


def test_errors_as_zero():
    p = compute_profile(RESULTS, name="t", probe_set="x", prior_weight=0.0, errors_as_zero=True)
    b = row(p, "B")
    assert b.counts == [2, 2] and b.quality == pytest.approx([1.0, 0.0])


def test_unknown_cluster_rejected():
    with pytest.raises(ValueError, match="unknown clusters"):
        compute_profile(RESULTS, name="t", probe_set="x",
                        clusters=[ProfileCluster(id="sql", n_probes=0)])


def test_profile_validation_alignment():
    p = compute_profile(RESULTS, name="t", probe_set="x")
    data = p.model_dump()
    data["models"][0]["quality"] = [0.5]
    with pytest.raises(ValidationError, match="entries"):
        RouterProfile.model_validate(data)
    data = p.model_dump()
    data["clusters"][0]["centroid"] = [0.1, 0.2]
    with pytest.raises(ValidationError):
        RouterProfile.model_validate(data)  # partial centroids + no embedder


def test_kmeans_recovers_separated_clusters():
    rng = np.random.default_rng(0)
    centers = np.eye(8)[:3] * 10
    x = np.vstack([c + rng.normal(scale=0.3, size=(30, 8)) for c in centers])
    truth = np.repeat(np.arange(3), 30)
    cent, assign = kmeans(x, 3, seed=1)
    assert cent.shape == (3, 8)
    np.testing.assert_allclose(np.linalg.norm(cent, axis=1), 1.0)
    # same partition up to relabelling
    for k in range(3):
        assert len(set(assign[truth == k])) == 1
    assert len(set(assign)) == 3
    _, again = kmeans(x, 3, seed=1)
    np.testing.assert_array_equal(assign, again)  # deterministic


def test_profile_artifact(tmp_path: Path):
    p = compute_profile(RESULTS, name="sample-profile", probe_set="caliban-sample-v1")
    out = tmp_path / "prof"
    m = write_profile_artifact(p, out, version="1.0.0", data_card={
        "intended_use": "test", "eval_datasets": [
            {"name": "caliban-sample-v1", "source": "data/probes/sample.yaml",
             "licence": "caliban-internal"}]})
    assert m.kind.value == "router_profile"
    assert verify_artifact_dir(out, strict=True).ok
    assert load_manifest(out).files[0].role.value == "profile"
    loaded = RouterProfile.model_validate(json.loads((out / "profile.json").read_text()))
    np.testing.assert_allclose(loaded.quality_matrix(), p.quality_matrix())
