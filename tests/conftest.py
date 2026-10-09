from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

FAKE_REV = "0123456789abcdef0123456789abcdef01234567"


def pii_ner_fields(**over):
    """Valid ArtifactManifest fields for a pii_ner artifact (files are hashed separately)."""
    fields = {
        "kind": "pii_ner",
        "name": "gliner2-pii-int8",
        "version": "0.1.0",
        "description": "test artifact",
        "onnx": {
            "file": "model.onnx",
            "opset": 17,
            "quantization": "int8_dynamic",
            "inputs": [
                {"name": "input_ids", "dtype": "int64", "shape": ["batch", "seq"]},
                {"name": "attention_mask", "dtype": "int64", "shape": ["batch", "seq"]},
            ],
            "outputs": [{"name": "logits", "dtype": "float32",
                         "shape": ["batch", "seq", "seq", 12]}],
        },
        "tokenizer": {"file": "tokenizer.json", "max_length": 512},
        "labels": ["PERSON", "EMAIL", "PHONE"],
        "calibration": {"method": "temperature", "temperature": 1.3,
                        "thresholds": {"PERSON": 0.4, "EMAIL": 0.5}},
        "metrics": {"span_f1_overlap": 0.81},
        "data_card": {
            "training_datasets": [{"name": "synthetic-pii-v1", "source": "internal://pii/v1",
                                   "licence": "caliban-internal", "n_examples": 5000,
                                   "synthetic": True}],
            "languages": ["en"],
            "intended_use": "L1 PII span detection in the gateway request path.",
        },
        "base_model": {"id": "example/gliner2-pii", "revision": FAKE_REV,
                       "licence": "apache-2.0"},
    }
    fields.update(over)
    return fields


@pytest.fixture
def artifact_dir(tmp_path: Path) -> Path:
    d = tmp_path / "gliner2-pii-int8-0.1.0"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"\x08\x07fake-onnx-graph" * 100)
    (d / "tokenizer.json").write_text('{"version": "1.0", "model": {"type": "WordPiece"}}')
    return d
