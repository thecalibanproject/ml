"""ONNX export of trained heads/encoders (behind the ``[export]`` extra).

Target runtime is ONNX Runtime via the Rust ``ort`` crate (CPU INT8 by default) or
``candle`` for the pure-Rust build. The export must produce:

* ``model.onnx``: opset >= 17, dynamic ``batch`` and ``seq`` axes, inputs named
  ``input_ids`` / ``attention_mask`` (int64), one output per head (``<head>_logits``).
* ``tokenizer.json``: the HF fast-tokenizer file, loadable by the Rust ``tokenizers``
  crate (no Python-only tokenizer code paths, no ``trust_remote_code``).
* ``manifest.json``: built with :func:`build_export_manifest` so tensor names, shapes,
  dtypes, labels, calibration and licence are machine-checked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from caliban_ml._extras import require
from caliban_ml.artifacts import ArtifactManifest, build_manifest, write_manifest

Quantization = Literal["none", "int8_dynamic", "fp16"]


def export_onnx(checkpoint_dir: Path, out_dir: Path, *, opset: int = 17,
                quantization: Quantization = "int8_dynamic",
                parity_atol: float = 1e-3) -> Path:
    """Export a trained checkpoint to ``out_dir/model.onnx`` (+ ``tokenizer.json``)."""
    require("onnx", "export")
    require("onnxruntime", "export")
    require("optimum", "export")
    # TODO(export):
    #   1. optimum.exporters.onnx.main_export(checkpoint_dir, out_dir, opset=opset,
    #      task="feature-extraction" | "text-classification", no_post_process=False)
    #      For multi-head models, export a torch.nn.Module wrapper with torch.onnx.export
    #      and dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, ...}.
    #   2. quantization == "int8_dynamic": onnxruntime.quantization.quantize_dynamic(
    #      model.onnx, model.int8.onnx, weight_type=QInt8); keep only the quantized file.
    #   3. Parity check: run N validation texts through torch and ORT; assert
    #      max|logits diff| <= parity_atol (fp32) and argmax agreement >= 99.5% (int8).
    #      Record both numbers in the manifest metrics (export_max_abs_diff, export_argmax_agree).
    #   4. Copy tokenizer.json; refuse tokenizers that need trust_remote_code.
    #   5. Call build_export_manifest(...) and write_manifest(...).
    raise NotImplementedError("ONNX export: see TODO(export)")


def build_export_manifest(out_dir: Path, **fields: Any) -> ArtifactManifest:
    """Hash ``out_dir`` and write ``manifest.json``. ``fields`` are ArtifactManifest fields
    (kind, name, version, onnx, tokenizer, labels, calibration, metrics, data_card,
    base_model, ...). Pure Python: usable without the ``[export]`` extra."""
    manifest = build_manifest(Path(out_dir), **fields)
    write_manifest(manifest, Path(out_dir))
    return manifest
