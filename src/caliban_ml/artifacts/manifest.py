"""The artifact manifest: the contract between ``caliban-ml`` and the Rust data plane.

An *artifact* is a directory::

    <name>-<version>/
        manifest.json              # this model, serialized
        manifest.json.minisig      # optional detached signature over manifest.json
        model.onnx                 # ONNX graph (embedder / intent_head / pii_ner)
        tokenizer.json             # HF `tokenizers` JSON, loaded by the Rust `tokenizers` crate
        labels.json, profile.json, ...

The manifest lists every file with its SHA-256, so signing ``manifest.json`` alone covers
the whole directory. The Rust side must:

1. verify the signature over ``manifest.json`` (when the deployment requires it),
2. verify every listed file's size and SHA-256 before loading,
3. refuse an artifact whose ``manifest_version`` it does not understand,
4. bind ONNX inputs/outputs by the tensor names declared here (never by position).

The JSON Schema for this model is exported to ``schemas/artifact-manifest.schema.json``
(``caliban-ml manifest schema``) and is what ``core`` validates against.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from caliban_ml import __version__
from caliban_ml.artifacts.licenses import LicenceVerdict, classify_licence

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "manifest.json"

NAME_RE = r"^[a-z0-9][a-z0-9._-]{0,63}$"
SEMVER_RE = (
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
SHA256_RE = r"^[0-9a-f]{64}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False, populate_by_name=True)


class ArtifactKind(StrEnum):
    EMBEDDER = "embedder"
    INTENT_HEAD = "intent_head"
    PII_NER = "pii_ner"
    ROUTER_PROFILE = "router_profile"


class FileRole(StrEnum):
    MODEL = "model"
    MODEL_DATA = "model_data"  # ONNX external data (``*.onnx.data``) for >2GB graphs
    TOKENIZER = "tokenizer"
    CONFIG = "config"
    LABELS = "labels"
    PROFILE = "profile"
    EXEMPLARS = "exemplars"
    OTHER = "other"


TensorDType = Literal["float32", "float16", "bfloat16", "int64", "int32", "int8", "uint8", "bool"]
# A dimension is either a fixed positive size or a symbolic name ("batch", "seq").
Dim = Annotated[int, Field(ge=1)] | Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]


def _check_rel_path(p: str) -> str:
    if not p or p.startswith(("/", "\\")) or "\\" in p or re.match(r"^[A-Za-z]:", p):
        raise ValueError(f"file path must be a relative POSIX path: {p!r}")
    parts = p.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"file path must not contain empty, '.' or '..' segments: {p!r}")
    return p


class FileEntry(_Strict):
    path: str = Field(description="Relative POSIX path inside the artifact directory.")
    sha256: str = Field(pattern=SHA256_RE, description="Lower-case hex SHA-256 of the file.")
    size_bytes: int = Field(ge=0)
    role: FileRole = FileRole.OTHER

    @field_validator("path")
    @classmethod
    def _path(cls, v: str) -> str:
        _check_rel_path(v)
        if v == MANIFEST_FILENAME or v.startswith(MANIFEST_FILENAME + "."):
            raise ValueError("the manifest and its signatures are not listed in files")
        return v


class TensorSpec(_Strict):
    name: str = Field(min_length=1, description="ONNX graph input/output name.")
    dtype: TensorDType
    shape: list[Dim] = Field(description="Ints are fixed sizes, strings are symbolic dims.")


class Calibration(_Strict):
    """Post-hoc calibration the Rust side applies to raw head outputs.

    ``p = softmax(logits / temperature)``. The head *exits* (accepts its own prediction)
    when ``p[label] >= thresholds.get(label, default_threshold)`` and, when set, the
    margin between the top-2 probabilities is ``>= margin_threshold``. If ``oos_threshold``
    is set, a query whose ``oos_score`` is below it is out-of-scope (an explicit decision,
    never an argmax over known intents).
    """

    method: Literal["none", "temperature"] = "temperature"
    temperature: float = Field(default=1.0, gt=0)
    thresholds: dict[str, Annotated[float, Field(ge=0, le=1)]] = Field(default_factory=dict)
    default_threshold: float | None = Field(default=None, ge=0, le=1)
    margin_threshold: float | None = Field(default=None, ge=0, le=1)
    oos_threshold: float | None = None
    oos_score: Literal["max_probability", "top1_similarity", "oos_head_probability"] | None = None

    @model_validator(mode="after")
    def _oos(self) -> Calibration:
        if (self.oos_threshold is None) != (self.oos_score is None):
            raise ValueError("oos_threshold and oos_score must be set together")
        if self.method == "none" and self.temperature != 1.0:
            raise ValueError("temperature must be 1.0 when method is 'none'")
        return self


class HeadSpec(_Strict):
    """Maps one ONNX output tensor to a label space (multi-head classifiers)."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="e.g. intent, difficulty, oos")
    output: str = Field(description="Name of the ONNX output tensor holding this head's logits.")
    activation: Literal["softmax", "sigmoid", "none"] = "softmax"
    labels: list[str] = Field(min_length=1)
    calibration: Calibration | None = None


class OnnxSpec(_Strict):
    file: str = Field(description="Path of the .onnx file; must appear in `files`.")
    opset: int = Field(ge=7)
    inputs: list[TensorSpec] = Field(min_length=1)
    outputs: list[TensorSpec] = Field(min_length=1)
    quantization: Literal["none", "int8_dynamic", "int8_static", "fp16"] = "none"
    heads: list[HeadSpec] = Field(
        default_factory=list,
        description="Optional extra heads (multi-head ModernBERT). The primary head is "
        "described by the top-level `labels` + `calibration`.",
    )

    @model_validator(mode="after")
    def _names(self) -> OnnxSpec:
        for group, specs in (("inputs", self.inputs), ("outputs", self.outputs)):
            names = [t.name for t in specs]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate tensor names in onnx.{group}: {names}")
        out_names = {t.name for t in self.outputs}
        for h in self.heads:
            if h.output not in out_names:
                raise ValueError(f"head {h.name!r} references unknown output {h.output!r}")
        return self


class TokenizerSpec(_Strict):
    file: str = Field(description="HF `tokenizers` JSON (tokenizer.json); must appear in `files`.")
    format: Literal["hf_tokenizers_json"] = "hf_tokenizers_json"
    max_length: int = Field(ge=1, le=131072)
    truncation_side: Literal["right", "left"] = "right"


class EmbeddingSpec(_Strict):
    dim: int = Field(ge=1)
    output: str = Field(description="ONNX output tensor holding token or pooled embeddings.")
    pooling: Literal["cls", "mean", "last", "none"] = Field(
        description="'none' means the graph already outputs a pooled vector."
    )
    normalize: bool = True
    query_prefix: str = Field(default="", description="e.g. 'query: ' for E5-family models")
    passage_prefix: str = ""


class DatasetRef(_Strict):
    name: str = Field(min_length=1)
    source: str = Field(description="URL, HF dataset id, or internal path. Pin a revision.")
    revision: str | None = Field(default=None, description="Commit / version / sha256.")
    licence: str = Field(description="SPDX id where possible, e.g. 'apache-2.0', 'cc-by-4.0'.")
    n_examples: int | None = Field(default=None, ge=0)
    synthetic: bool = False


class DataCard(_Strict):
    training_datasets: list[DatasetRef] = Field(default_factory=list)
    eval_datasets: list[DatasetRef] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list, description="BCP-47 codes.")
    intended_use: str = Field(min_length=1)
    limitations: str | None = None
    contains_personal_data: bool = Field(
        default=False,
        description="True if training data included real personal data. Shipping such an "
        "artifact needs a DPIA reference in `notes`.",
    )
    notes: str | None = None


class BaseModelRef(_Strict):
    id: str = Field(min_length=1, description="e.g. 'answerdotai/ModernBERT-base'")
    revision: str = Field(min_length=1, description="Pinned commit hash of the weights used.")
    licence: str = Field(min_length=1, description="SPDX id from the model card.")
    url: str | None = None


class LicenceOverride(_Strict):
    """A recorded human decision to ship something whose licence is not on the allow-list."""

    subject: str = Field(description="Base model id or dataset name this override applies to.")
    licence: str
    approved_by: str = Field(min_length=1)
    approved_on: date
    reason: str = Field(min_length=10)


class ArtifactRef(_Strict):
    kind: ArtifactKind
    name: str = Field(pattern=NAME_RE)
    version: str = Field(pattern=SEMVER_RE)


class Producer(_Strict):
    tool: Literal["caliban-ml"] = "caliban-ml"
    version: str = __version__
    git_commit: str | None = None


class ArtifactManifest(_Strict):
    manifest_version: Literal[1] = MANIFEST_VERSION
    kind: ArtifactKind
    name: str = Field(pattern=NAME_RE)
    version: str = Field(pattern=SEMVER_RE)
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    producer: Producer = Field(default_factory=Producer)

    files: list[FileEntry] = Field(min_length=1)
    onnx: OnnxSpec | None = None
    tokenizer: TokenizerSpec | None = None
    embedding: EmbeddingSpec | None = None
    labels: list[str] = Field(default_factory=list)
    calibration: Calibration | None = None
    metrics: dict[str, float] = Field(default_factory=dict)

    data_card: DataCard
    base_model: BaseModelRef | None = None
    licence_overrides: list[LicenceOverride] = Field(default_factory=list)
    requires: list[ArtifactRef] = Field(
        default_factory=list,
        description="Artifacts this one depends on, e.g. an intent head or router profile "
        "built in a specific embedder's vector space.",
    )

    # ------------------------------------------------------------------ validators

    @field_validator("labels")
    @classmethod
    def _unique_labels(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("labels must be unique")
        if any(not s for s in v):
            raise ValueError("labels must be non-empty strings")
        return v

    @field_validator("metrics")
    @classmethod
    def _finite_metrics(cls, v: dict[str, float]) -> dict[str, float]:
        bad = [k for k, x in v.items() if not math.isfinite(x)]
        if bad:
            raise ValueError(f"metrics must be finite numbers: {bad}")
        return v

    @model_validator(mode="after")
    def _structure(self) -> ArtifactManifest:
        paths = [f.path for f in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate paths in files")
        listed = set(paths)

        def must_list(path: str, what: str) -> None:
            if path not in listed:
                raise ValueError(f"{what} file {path!r} is not listed in files")

        if self.onnx is not None:
            must_list(self.onnx.file, "onnx")
        if self.tokenizer is not None:
            must_list(self.tokenizer.file, "tokenizer")

        k = self.kind
        if k in (ArtifactKind.EMBEDDER, ArtifactKind.PII_NER):
            if self.onnx is None or self.tokenizer is None:
                raise ValueError(f"{k.value} artifacts require onnx and tokenizer")
            if self.base_model is None:
                raise ValueError(f"{k.value} artifacts require base_model")
        if k == ArtifactKind.EMBEDDER:
            if self.embedding is None:
                raise ValueError("embedder artifacts require embedding")
            out_names = {t.name for t in self.onnx.outputs}  # type: ignore[union-attr]
            if self.embedding.output not in out_names:
                raise ValueError(f"embedding.output {self.embedding.output!r} is not an output")
        elif self.embedding is not None:
            raise ValueError("embedding is only valid for embedder artifacts")
        if k in (ArtifactKind.INTENT_HEAD, ArtifactKind.PII_NER) and not self.labels:
            raise ValueError(f"{k.value} artifacts require a non-empty labels list")
        if k == ArtifactKind.INTENT_HEAD:
            has_embedder = any(r.kind == ArtifactKind.EMBEDDER for r in self.requires)
            if self.onnx is None and not has_embedder:
                raise ValueError(
                    "intent_head needs either an onnx graph or a `requires` embedder "
                    "(kNN heads score in the embedder's vector space)"
                )
            if self.onnx is not None and (self.tokenizer is None or self.base_model is None):
                raise ValueError("onnx intent heads require tokenizer and base_model")
            if self.calibration is None:
                raise ValueError("intent_head artifacts require calibration")
        if k == ArtifactKind.ROUTER_PROFILE:
            if self.onnx is not None or self.tokenizer is not None:
                raise ValueError("router_profile artifacts carry no onnx/tokenizer")
            if not any(f.role == FileRole.PROFILE for f in self.files):
                raise ValueError("router_profile artifacts require a file with role 'profile'")

        if self.calibration is not None:
            unknown = set(self.calibration.thresholds) - set(self.labels)
            if unknown:
                raise ValueError(f"calibration thresholds for unknown labels: {sorted(unknown)}")
        return self

    @model_validator(mode="after")
    def _licences(self, info: ValidationInfo) -> ArtifactManifest:
        ctx: dict[str, Any] = info.context or {}
        allow_nc = bool(ctx.get("allow_noncommercial", False))
        overrides = {(o.subject, o.licence.strip().lower()) for o in self.licence_overrides}

        subjects: list[tuple[str, str]] = []
        if self.base_model is not None:
            subjects.append((self.base_model.id, self.base_model.licence))
        for ds in [*self.data_card.training_datasets, *self.data_card.eval_datasets]:
            subjects.append((ds.name, ds.licence))

        problems: list[str] = []
        for subject, licence in subjects:
            d = classify_licence(licence)
            if d.verdict is LicenceVerdict.ALLOWED:
                continue
            overridden = (subject, licence.strip().lower()) in overrides
            if d.verdict is LicenceVerdict.REVIEW_REQUIRED and overridden:
                continue
            if d.verdict is LicenceVerdict.DENIED and overridden and allow_nc:
                continue
            hint = (
                "add a licence_overrides entry"
                if d.verdict is LicenceVerdict.REVIEW_REQUIRED
                else "non-commercial: cannot ship in the default bundle"
            )
            problems.append(f"{subject}: licence {licence!r} is {d.verdict.value} ({hint})")
        if problems:
            raise ValueError("licence policy violation: " + "; ".join(problems))
        return self

    # ------------------------------------------------------------------ helpers

    def ref(self) -> ArtifactRef:
        return ArtifactRef(kind=self.kind, name=self.name, version=self.version)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2, exclude_none=True) + "\n"
