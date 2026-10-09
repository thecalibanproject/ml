"""JSON Schema export for the contracts consumed by caliban core (Rust).

``schemas/*.schema.json`` are generated from the pydantic models; a test fails if the
committed files drift from the models. Regenerate with ``caliban-ml manifest schema``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from caliban_ml.artifacts.manifest import ArtifactManifest
from caliban_ml.router.profile import RouterProfile

SCHEMA_BASE = "https://caliban.local/schemas/"

SCHEMAS: dict[str, tuple[type, str]] = {
    "artifact-manifest.schema.json": (
        ArtifactManifest,
        "Caliban model artifact manifest (v1). Produced by caliban-ml, verified by core "
        "before loading ONNX/JSON artifacts. Licence policy and kind-specific rules are "
        "enforced by caliban-ml; core re-checks hashes and tensor names.",
    ),
    "router-profile.schema.json": (
        RouterProfile,
        "UniRoute-style per-model, per-cluster quality table (profile.json inside a "
        "router_profile artifact). Consumed by the Stage-4 model scorer.",
    ),
}


def build_schema(model: type, filename: str, description: str) -> dict[str, Any]:
    schema = model.model_json_schema(mode="validation")  # type: ignore[attr-defined]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_BASE + filename,
        **{k: v for k, v in schema.items() if k != "description"},
        "description": description,
    }


def render(filename: str) -> str:
    model, description = SCHEMAS[filename]
    return json.dumps(build_schema(model, filename, description), indent=2) + "\n"


def write_schemas(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for filename in SCHEMAS:
        p = out_dir / filename
        p.write_text(render(filename), encoding="utf-8")
        written.append(p)
    return written
