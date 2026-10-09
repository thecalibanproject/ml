"""caliban-ml: offline training, evaluation and export for Caliban's in-process models.

The only contract with ``caliban core`` (Rust) is a directory of ONNX files plus JSON
(an :class:`~caliban_ml.artifacts.ArtifactManifest` and, for routers, a profile table).
Nothing in this package is imported by the data plane.
"""

__version__ = "0.1.0"
