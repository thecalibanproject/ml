"""UniRoute-style router profiles: per-model, per-cluster quality vectors.

Inputs are probe results (``model x prompt -> score in [0, 1]``) where every prompt has a
cluster id (from the probe set's labels, or from :func:`kmeans` over prompt embeddings).
Output is a JSON *profile table* the Rust Stage-4 scorer consumes::

    quality[model][cluster] = (sum of scores + w * prior[model]) / (count + w)

``prior[model]`` is the model's mean score over all its probes and ``w`` is
``prior_weight`` (Bayesian shrinkage, so a cluster seen 2 times cannot swing to 0 or 1).
With ``w = 0`` this is the plain per-cluster mean; clusters with no probes then fall back
to the prior. ``counts`` are always emitted so Rust can apply its own confidence rules
(e.g. exploration bonus for low counts).

A new BYO model becomes routable as soon as its profile row exists: no router retraining
(UniRoute, arXiv:2502.08773). When ``centroids`` are present, Rust assigns an incoming
prompt embedding to the nearest centroid (cosine) of the *same embedder* named in
``embedder``; otherwise clusters are only addressable by id (e.g. via the intent).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from caliban_ml.artifacts import (
    ArtifactKind,
    ArtifactManifest,
    ArtifactRef,
    FileRole,
    build_manifest,
    write_manifest,
)

PROFILE_VERSION = 1
PROFILE_FILENAME = "profile.json"


class ProbeScore(BaseModel):
    """One judged probe outcome. ``caliban_ml.probes`` writes these as JSONL."""

    model_config = ConfigDict(extra="ignore")
    model: str
    probe_id: str
    cluster: str
    score: float | None = Field(default=None, ge=0, le=1)
    error: str | None = None


class ProfileCluster(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    description: str = ""
    n_probes: int = Field(ge=0, description="Distinct probe prompts in this cluster.")
    centroid: list[float] | None = None


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(description="Model id as registered in the tenant model registry.")
    prior: float = Field(ge=0, le=1, description="Mean score over all of this model's probes.")
    quality: list[float] = Field(description="Shrunk quality per cluster, aligned to clusters.")
    raw_mean: list[float | None] = Field(description="Unshrunk mean, null where count == 0.")
    counts: list[int]
    n_errors: int = Field(default=0, ge=0, description="Probe calls that failed (excluded).")

    @field_validator("quality")
    @classmethod
    def _range(cls, v: list[float]) -> list[float]:
        if any(not (0.0 <= x <= 1.0) or math.isnan(x) for x in v):
            raise ValueError("quality values must be in [0, 1]")
        return v


class RouterProfile(BaseModel):
    """Profile table consumed by the Rust Stage-4 scorer.

    JSON Schema: ``schemas/router-profile.schema.json``.
    """

    model_config = ConfigDict(extra="forbid")
    profile_version: int = PROFILE_VERSION
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    probe_set: str = Field(description="Name/version of the probe set the scores came from.")
    prior_weight: float = Field(ge=0)
    embedder: ArtifactRef | None = Field(
        default=None, description="Embedder whose space the centroids live in."
    )
    clusters: list[ProfileCluster] = Field(min_length=1)
    models: list[ModelProfile] = Field(min_length=1)

    @model_validator(mode="after")
    def _aligned(self) -> RouterProfile:
        n = len(self.clusters)
        ids = [c.id for c in self.clusters]
        if len(set(ids)) != n:
            raise ValueError("duplicate cluster ids")
        names = [m.model for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError("duplicate model ids")
        for m in self.models:
            if not (len(m.quality) == len(m.raw_mean) == len(m.counts) == n):
                raise ValueError(f"model {m.model!r}: vectors must have {n} entries")
        dims = {len(c.centroid) for c in self.clusters if c.centroid is not None}
        if len(dims) > 1:
            raise ValueError("centroids must all have the same dimension")
        if dims and any(c.centroid is None for c in self.clusters):
            raise ValueError("either all clusters have centroids or none do")
        if dims and self.embedder is None:
            raise ValueError("centroids require an embedder reference")
        return self

    def quality_matrix(self) -> np.ndarray:
        return np.array([m.quality for m in self.models], dtype=np.float64)


def compute_profile(results: Iterable[ProbeScore | dict], *, name: str, probe_set: str,
                    prior_weight: float = 2.0, clusters: list[ProfileCluster] | None = None,
                    embedder: ArtifactRef | None = None,
                    errors_as_zero: bool = False) -> RouterProfile:
    """Aggregate probe results into a :class:`RouterProfile`.

    Records with ``error`` set (timeouts, 5xx, rate limits) are excluded by default:
    infrastructure failures are not a quality signal. ``errors_as_zero`` counts them as 0.
    If a model scored the same probe more than once (repeated sampling), all samples
    count; ``n_probes`` counts distinct prompts.
    """
    if prior_weight < 0:
        raise ValueError("prior_weight must be >= 0")
    recs = [r if isinstance(r, ProbeScore) else ProbeScore.model_validate(r) for r in results]
    if not recs:
        raise ValueError("no probe results")

    sums: dict[tuple[str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    errors: dict[str, int] = defaultdict(int)
    probes_per_cluster: dict[str, set[str]] = defaultdict(set)
    models: list[str] = []
    for r in recs:
        if r.model not in models:
            models.append(r.model)
        probes_per_cluster[r.cluster].add(r.probe_id)
        if r.error is not None or r.score is None:
            errors[r.model] += 1
            if not errors_as_zero:
                continue
            score = 0.0
        else:
            score = r.score
        sums[(r.model, r.cluster)] += score
        counts[(r.model, r.cluster)] += 1

    if clusters is None:
        clusters = [ProfileCluster(id=c, n_probes=len(probes_per_cluster[c]))
                    for c in sorted(probes_per_cluster)]
    else:
        known = {c.id for c in clusters}
        unknown = sorted(set(probes_per_cluster) - known)
        if unknown:
            raise ValueError(f"results reference unknown clusters: {unknown}")
        clusters = [c.model_copy(update={"n_probes": len(probes_per_cluster.get(c.id, ()))})
                    for c in clusters]

    rows: list[ModelProfile] = []
    for model in sorted(models):
        total_n = sum(counts[(model, c.id)] for c in clusters)
        total_s = sum(sums[(model, c.id)] for c in clusters)
        prior = total_s / total_n if total_n else 0.0
        quality, raw, cnts = [], [], []
        for c in clusters:
            n, s = counts[(model, c.id)], sums[(model, c.id)]
            cnts.append(n)
            raw.append(s / n if n else None)
            denom = n + prior_weight
            quality.append((s + prior_weight * prior) / denom if denom > 0 else prior)
        rows.append(ModelProfile(model=model, prior=prior, quality=quality, raw_mean=raw,
                                 counts=cnts, n_errors=errors[model]))
    return RouterProfile(name=name, probe_set=probe_set, prior_weight=prior_weight,
                         embedder=embedder, clusters=clusters, models=rows)


def load_probe_scores(path: Path) -> list[ProbeScore]:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(ProbeScore.model_validate_json(line))
    return out


def write_profile_artifact(profile: RouterProfile, out_dir: Path, *, version: str,
                           data_card: dict, description: str = "") -> ArtifactManifest:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / PROFILE_FILENAME).write_text(profile.model_dump_json(indent=2) + "\n")
    requires = [profile.embedder.model_dump()] if profile.embedder else []
    metrics = {f"prior.{m.model}": m.prior for m in profile.models}
    manifest = build_manifest(
        out_dir,
        roles={PROFILE_FILENAME: FileRole.PROFILE},
        kind=ArtifactKind.ROUTER_PROFILE,
        name=profile.name,
        version=version,
        description=description or f"Per-cluster quality profile from {profile.probe_set}",
        metrics=metrics,
        data_card=data_card,
        requires=requires,
    )
    write_manifest(manifest, out_dir)
    return manifest


# ---------------------------------------------------------------------- clustering


def kmeans(x: np.ndarray, k: int, *, seed: int = 0, n_iter: int = 100, cosine: bool = True
           ) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic k-means++ (spherical when ``cosine``). Returns (centroids, assignments).

    UniRoute clusters a representative prompt set in embedding space and profiles every
    model per cluster; use this to derive cluster ids when the probe set has none.
    """
    x = np.asarray(x, dtype=np.float64)
    if cosine:
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n = x.shape[0]
    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= n_points")
    rng = np.random.default_rng(seed)
    centroids = [x[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((x[:, None, :] - np.array(centroids)[None]) ** 2).sum(-1), axis=1)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        centroids.append(x[rng.choice(n, p=probs)])
    c = np.array(centroids)
    assign = np.full(n, -1)
    for _ in range(n_iter):
        d2 = ((x[:, None, :] - c[None]) ** 2).sum(-1)
        new = d2.argmin(axis=1)
        if np.array_equal(new, assign):
            break
        assign = new
        for j in range(k):
            members = x[assign == j]
            if len(members):
                c[j] = members.mean(axis=0)
        if cosine:
            c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    return c, assign

