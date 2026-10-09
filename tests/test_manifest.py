from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from caliban_ml.artifacts import (
    ArtifactManifest,
    LicenceVerdict,
    build_manifest,
    classify_licence,
    load_manifest,
    sha256_file,
    verify_artifact_dir,
    write_manifest,
)
from caliban_ml.schemas import SCHEMAS, render
from conftest import REPO, pii_ner_fields


def _write(artifact_dir: Path, **over) -> ArtifactManifest:
    m = build_manifest(artifact_dir, **pii_ner_fields(**over))
    write_manifest(m, artifact_dir)
    return m


def test_sha256_matches_hashlib(tmp_path: Path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"caliban" * 1_000_000)  # > 1 chunk
    assert sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_build_and_verify_roundtrip(artifact_dir: Path):
    m = _write(artifact_dir)
    assert {f.path for f in m.files} == {"model.onnx", "tokenizer.json"}
    roles = {f.path: f.role.value for f in m.files}
    assert roles == {"model.onnx": "model", "tokenizer.json": "tokenizer"}
    rep = verify_artifact_dir(artifact_dir)
    assert rep.ok, rep.errors
    assert load_manifest(artifact_dir) == m


def test_verify_detects_tampering(artifact_dir: Path):
    _write(artifact_dir)
    data = bytearray((artifact_dir / "model.onnx").read_bytes())
    data[10] ^= 0xFF  # same size, different content
    (artifact_dir / "model.onnx").write_bytes(bytes(data))
    rep = verify_artifact_dir(artifact_dir)
    assert not rep.ok
    assert any("sha256 mismatch" in e for e in rep.errors)


def test_verify_detects_missing_and_resized(artifact_dir: Path):
    _write(artifact_dir)
    (artifact_dir / "tokenizer.json").unlink()
    (artifact_dir / "model.onnx").write_bytes(b"short")
    errs = verify_artifact_dir(artifact_dir).errors
    assert any("tokenizer.json: listed in manifest but missing" in e for e in errs)
    assert any("model.onnx: size" in e for e in errs)


def test_unlisted_files_warn_or_fail_in_strict(artifact_dir: Path):
    _write(artifact_dir)
    (artifact_dir / "extra.bin").write_bytes(b"sneaky")
    assert verify_artifact_dir(artifact_dir).ok
    assert verify_artifact_dir(artifact_dir).warnings
    assert not verify_artifact_dir(artifact_dir, strict=True).ok


def test_manifest_and_signature_files_are_not_payload(artifact_dir: Path):
    _write(artifact_dir)
    (artifact_dir / "manifest.json.minisig").write_text("untrusted comment: sig\n")
    rep = verify_artifact_dir(artifact_dir, strict=True)
    assert rep.ok, rep.errors


def test_invalid_manifest_json_reported(artifact_dir: Path):
    (artifact_dir / "manifest.json").write_text("{not json")
    rep = verify_artifact_dir(artifact_dir)
    assert not rep.ok and "not valid JSON" in rep.errors[0]


@pytest.mark.parametrize("bad", ["../model.onnx", "/etc/passwd", "a//b", "a\\b", "./x"])
def test_path_traversal_rejected(bad: str):
    fields = pii_ner_fields()
    fields["files"] = [{"path": bad, "sha256": "0" * 64, "size_bytes": 1}]
    with pytest.raises(ValidationError):
        ArtifactManifest.model_validate(fields)


def test_symlink_escape_rejected(artifact_dir: Path, tmp_path: Path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret")
    _write(artifact_dir)
    (artifact_dir / "model.onnx").unlink()
    (artifact_dir / "model.onnx").symlink_to(outside)
    rep = verify_artifact_dir(artifact_dir)
    assert any("outside the artifact directory" in e for e in rep.errors)


# ------------------------------------------------------------------ structure rules


def test_onnx_file_must_be_listed(artifact_dir: Path):
    fields = pii_ner_fields()
    fields["onnx"]["file"] = "missing.onnx"
    with pytest.raises(ValidationError, match="not listed in files"):
        build_manifest(artifact_dir, **fields)


def test_pii_ner_requires_labels(artifact_dir: Path):
    with pytest.raises(ValidationError, match="labels"):
        build_manifest(artifact_dir, **pii_ner_fields(labels=[]))


def test_threshold_for_unknown_label_rejected(artifact_dir: Path):
    cal = {"method": "temperature", "temperature": 1.0, "thresholds": {"IBAN": 0.5}}
    with pytest.raises(ValidationError, match="unknown labels"):
        build_manifest(artifact_dir, **pii_ner_fields(calibration=cal))


def test_duplicate_tensor_names_rejected(artifact_dir: Path):
    fields = pii_ner_fields()
    fields["onnx"]["inputs"].append(dict(fields["onnx"]["inputs"][0]))
    with pytest.raises(ValidationError, match="duplicate tensor names"):
        build_manifest(artifact_dir, **fields)


def test_oos_threshold_requires_score(artifact_dir: Path):
    cal = {"method": "temperature", "temperature": 1.0, "oos_threshold": 0.3}
    with pytest.raises(ValidationError, match="oos_threshold and oos_score"):
        build_manifest(artifact_dir, **pii_ner_fields(calibration=cal))


def test_knn_intent_head_needs_embedder_reference(tmp_path: Path):
    (tmp_path / "labels.json").write_text('["a", "b"]')
    base = {
        "kind": "intent_head", "name": "knn-head", "version": "1.0.0",
        "labels": ["a", "b"], "calibration": {"temperature": 0.05},
        "data_card": {"intended_use": "test"},
    }
    with pytest.raises(ValidationError, match="requires"):
        build_manifest(tmp_path, **base)
    m = build_manifest(tmp_path, **base, requires=[
        {"kind": "embedder", "name": "bge-small-en", "version": "1.5.0"}])
    assert m.requires[0].name == "bge-small-en"


# ------------------------------------------------------------------ licence policy


@pytest.mark.parametrize("licence,verdict", [
    ("apache-2.0", LicenceVerdict.ALLOWED),
    ("MIT", LicenceVerdict.ALLOWED),
    ("cc-by-4.0", LicenceVerdict.ALLOWED),
    ("cc-by-nc-4.0", LicenceVerdict.DENIED),
    ("CC-BY-NC-SA-4.0", LicenceVerdict.DENIED),
    ("research-only", LicenceVerdict.DENIED),
    ("llama3.1", LicenceVerdict.REVIEW_REQUIRED),
    ("nvidia-open-model-license", LicenceVerdict.REVIEW_REQUIRED),
    ("some-custom-licence", LicenceVerdict.REVIEW_REQUIRED),
])
def test_classify_licence(licence: str, verdict: LicenceVerdict):
    assert classify_licence(licence).verdict is verdict


def test_noncommercial_base_model_rejected(artifact_dir: Path):
    fields = pii_ner_fields()
    fields["base_model"]["licence"] = "cc-by-nc-4.0"
    with pytest.raises(ValidationError, match="non-commercial"):
        build_manifest(artifact_dir, **fields)


def test_noncommercial_dataset_rejected(artifact_dir: Path):
    fields = pii_ner_fields()
    fields["data_card"]["training_datasets"][0]["licence"] = "CC-BY-NC-4.0"
    with pytest.raises(ValidationError, match="licence policy violation"):
        build_manifest(artifact_dir, **fields)


def test_review_licence_needs_override(artifact_dir: Path):
    fields = pii_ner_fields()
    fields["base_model"]["licence"] = "nvidia-open-model-license"
    with pytest.raises(ValidationError, match="licence_overrides"):
        build_manifest(artifact_dir, **fields)
    fields["licence_overrides"] = [{
        "subject": "example/gliner2-pii", "licence": "nvidia-open-model-license",
        "approved_by": "legal@caliban", "approved_on": "2026-10-01",
        "reason": "Reviewed NVIDIA Open Model License: commercial use and redistribution OK.",
    }]
    assert build_manifest(artifact_dir, **fields).base_model is not None


def test_noncommercial_override_only_with_explicit_flag(artifact_dir: Path):
    fields = pii_ner_fields()
    fields["base_model"]["licence"] = "cc-by-nc-4.0"
    fields["licence_overrides"] = [{
        "subject": "example/gliner2-pii", "licence": "cc-by-nc-4.0",
        "approved_by": "research@caliban", "approved_on": "2026-10-01",
        "reason": "Internal research comparison only, never bundled.",
    }]
    with pytest.raises(ValidationError):
        build_manifest(artifact_dir, **fields)
    m = build_manifest(artifact_dir, allow_noncommercial=True, **fields)
    write_manifest(m, artifact_dir)
    assert not verify_artifact_dir(artifact_dir).ok  # default verification still refuses
    assert verify_artifact_dir(artifact_dir, allow_noncommercial=True).ok


# ------------------------------------------------------------------ schema


@pytest.mark.parametrize("filename", sorted(SCHEMAS))
def test_committed_schema_in_sync(filename: str):
    committed = (REPO / "schemas" / filename).read_text(encoding="utf-8")
    assert committed == render(filename), "run: caliban-ml manifest schema"


def test_schema_describes_contract_fields():
    schema = json.loads(render("artifact-manifest.schema.json"))
    assert schema["$id"].endswith("artifact-manifest.schema.json")
    for key in ("kind", "files", "onnx", "tokenizer", "labels", "calibration", "metrics",
                "data_card", "base_model"):
        assert key in schema["properties"]
    assert set(schema["$defs"]["ArtifactKind"]["enum"]) == {
        "embedder", "intent_head", "pii_ner", "router_profile"}
