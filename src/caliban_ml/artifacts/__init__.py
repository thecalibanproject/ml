"""Artifact manifest contract (ONNX + JSON) between caliban-ml and caliban core."""

from caliban_ml.artifacts.io import (
    VerifyReport,
    build_manifest,
    collect_files,
    load_manifest,
    sha256_file,
    verify_artifact_dir,
    write_manifest,
)
from caliban_ml.artifacts.licenses import LicenceVerdict, classify_licence
from caliban_ml.artifacts.manifest import (
    MANIFEST_FILENAME,
    ArtifactKind,
    ArtifactManifest,
    ArtifactRef,
    BaseModelRef,
    Calibration,
    DataCard,
    DatasetRef,
    FileEntry,
    FileRole,
    LicenceOverride,
    OnnxSpec,
    TensorSpec,
    TokenizerSpec,
)

__all__ = [
    "MANIFEST_FILENAME",
    "ArtifactKind",
    "ArtifactManifest",
    "ArtifactRef",
    "BaseModelRef",
    "Calibration",
    "DataCard",
    "DatasetRef",
    "FileEntry",
    "FileRole",
    "LicenceOverride",
    "LicenceVerdict",
    "OnnxSpec",
    "TensorSpec",
    "TokenizerSpec",
    "VerifyReport",
    "build_manifest",
    "classify_licence",
    "collect_files",
    "load_manifest",
    "sha256_file",
    "verify_artifact_dir",
    "write_manifest",
]
