"""Hashing, writing, loading and verifying artifact directories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from caliban_ml.artifacts.manifest import (
    MANIFEST_FILENAME,
    ArtifactManifest,
    FileEntry,
    FileRole,
)

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _is_manifest_or_signature(rel: str) -> bool:
    return rel == MANIFEST_FILENAME or rel.startswith(MANIFEST_FILENAME + ".")


def _iter_payload_files(root: Path) -> list[str]:
    out: list[str] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if not _is_manifest_or_signature(rel) and not rel.split("/")[-1].startswith("."):
                out.append(rel)
    return out


def collect_files(root: Path, roles: dict[str, FileRole | str] | None = None) -> list[FileEntry]:
    """Hash every payload file under ``root`` (manifest, signatures and dotfiles excluded)."""
    roles = roles or {}
    entries = []
    for rel in _iter_payload_files(root):
        p = root / rel
        entries.append(
            FileEntry(
                path=rel,
                sha256=sha256_file(p),
                size_bytes=p.stat().st_size,
                role=FileRole(roles.get(rel, _guess_role(rel))),
            )
        )
    return entries


def _guess_role(rel: str) -> FileRole:
    name = rel.split("/")[-1]
    if name.endswith(".onnx"):
        return FileRole.MODEL
    if name.endswith(".onnx.data"):
        return FileRole.MODEL_DATA
    if name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        return FileRole.TOKENIZER
    if name == "labels.json":
        return FileRole.LABELS
    if name == "profile.json":
        return FileRole.PROFILE
    if name == "config.json":
        return FileRole.CONFIG
    return FileRole.OTHER


def build_manifest(root: Path, roles: dict[str, FileRole | str] | None = None,
                   *, allow_noncommercial: bool = False, **fields: Any) -> ArtifactManifest:
    """Hash the payload files in ``root`` and validate a manifest from ``fields``."""
    data = {**fields, "files": [e.model_dump() for e in collect_files(root, roles)]}
    return ArtifactManifest.model_validate(
        data, context={"allow_noncommercial": allow_noncommercial}
    )


def write_manifest(manifest: ArtifactManifest, root: Path) -> Path:
    path = root / MANIFEST_FILENAME
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def load_manifest(root: Path, *, allow_noncommercial: bool = False) -> ArtifactManifest:
    raw = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    ctx = {"allow_noncommercial": allow_noncommercial}
    return ArtifactManifest.model_validate(raw, context=ctx)


@dataclass
class VerifyReport:
    root: Path
    manifest: ArtifactManifest | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def verify_artifact_dir(root: Path, *, strict: bool = False,
                        allow_noncommercial: bool = False) -> VerifyReport:
    """Validate ``manifest.json`` and check every listed file's size and SHA-256.

    ``strict`` turns unlisted payload files into errors (the default bundle build uses it:
    anything shipped must be covered by the signed manifest).
    """
    root = Path(root)
    rep = VerifyReport(root=root)
    mpath = root / MANIFEST_FILENAME
    if not mpath.is_file():
        rep.errors.append(f"missing {MANIFEST_FILENAME} in {root}")
        return rep
    try:
        rep.manifest = load_manifest(root, allow_noncommercial=allow_noncommercial)
    except json.JSONDecodeError as exc:
        rep.errors.append(f"{MANIFEST_FILENAME} is not valid JSON: {exc}")
        return rep
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"]) or "<root>"
            rep.errors.append(f"manifest invalid at {loc}: {err['msg']}")
        return rep

    real_root = root.resolve()
    for entry in rep.manifest.files:
        p = root / entry.path
        try:
            resolved = p.resolve()
        except OSError as exc:  # pragma: no cover
            rep.errors.append(f"{entry.path}: cannot resolve ({exc})")
            continue
        if not resolved.is_relative_to(real_root):
            rep.errors.append(f"{entry.path}: resolves outside the artifact directory")
            continue
        if not p.is_file():
            rep.errors.append(f"{entry.path}: listed in manifest but missing")
            continue
        size = p.stat().st_size
        if size != entry.size_bytes:
            rep.errors.append(f"{entry.path}: size {size} != manifest {entry.size_bytes}")
            continue
        digest = sha256_file(p)
        if digest != entry.sha256:
            rep.errors.append(f"{entry.path}: sha256 mismatch ({digest} != {entry.sha256})")

    listed = {e.path for e in rep.manifest.files}
    for rel in _iter_payload_files(root):
        if rel not in listed:
            msg = f"{rel}: present but not listed in manifest"
            (rep.errors if strict else rep.warnings).append(msg)
    return rep
