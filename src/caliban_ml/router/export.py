"""ONNX export of trained Stage-2 intent heads (``[train]`` + ``[export]`` extras).

Target runtime is ONNX Runtime via the Rust ``ort`` crate (CPU INT8 by default) or
``candle`` for the pure-Rust build. :func:`export_onnx` turns a checkpoint written by
:func:`caliban_ml.router.train.train_intent_head` into an ``intent_head`` artifact::

    <out_dir>/
        manifest.json     kind intent_head; onnx (tensor names, shapes, heads), tokenizer,
                          labels + calibration of the primary head, metrics, data card,
                          base model + licence
        model.onnx        opset >= 17, inputs ``input_ids`` / ``attention_mask`` (int64,
                          [batch, seq]), one float32 output per head: ``intent_logits``
                          [batch, n_labels] (raw logits, softmax applied by the runtime) and,
                          for the multi-head backbone, ``oos_logits`` [batch, 1] (sigmoid)
        tokenizer.json    HF fast-tokenizer file for the Rust ``tokenizers`` crate, with
                          truncation and padding stripped (the runtime truncates to
                          ``tokenizer.max_length`` and pads with ``config.json`` ``pad_id``)
        labels.json       primary head labels, in logit order
        config.json       ``type = "onnx_classifier"``, primary head, pooling, pad id

Parity is checked before anything is written: torch fp32 logits vs ONNX Runtime fp32
(``max |diff| <= parity_atol``) and, when quantised, the INT8 graph's argmax agreement
(``>= min_argmax_agree``). The ONNX side is fed by the *shipped* ``tokenizer.json``
(through the ``tokenizers`` library, as Rust does), and its token ids must equal the
training tokenizer's. All numbers land in the manifest metrics (``export_*``).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np

from caliban_ml._extras import require
from caliban_ml.artifacts import (
    ArtifactKind,
    ArtifactManifest,
    FileRole,
    build_manifest,
    verify_artifact_dir,
    write_manifest,
)
from caliban_ml.router.checkpoint import ENCODER_DIR, CheckpointMeta, read_splits

Quantization = Literal["none", "int8_dynamic", "fp16"]
INPUT_NAMES = ("input_ids", "attention_mask")
ONNX_CLASSIFIER_TYPE = "onnx_classifier"
MAX_PARITY_TEXTS = 256


class ParityError(RuntimeError):
    """The exported graph does not reproduce the torch model closely enough."""


# ---------------------------------------------------------------------- pure helpers


def parity_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """``(max |reference - candidate|, argmax agreement)`` over the last axis."""
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(candidate, dtype=np.float64)
    if ref.shape != got.shape:
        raise ParityError(f"output shape {got.shape} != reference {ref.shape}")
    diff = float(np.max(np.abs(ref - got))) if ref.size else 0.0
    if ref.shape[-1] == 1:  # sigmoid head: agreement on the sign of the logit
        agree = float(np.mean((ref > 0) == (got > 0))) if ref.size else 1.0
    else:
        agree = float(np.mean(ref.argmax(-1) == got.argmax(-1))) if ref.size else 1.0
    return diff, agree


def clean_tokenizer_json(src: Path, dst: Path) -> dict:
    """Copy an HF ``tokenizer.json`` with truncation and padding removed (the runtime owns
    both). Returns the parsed JSON."""
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    if "model" not in data or "version" not in data:
        raise ValueError(f"{src} is not an HF tokenizers JSON file")
    data["truncation"] = None
    data["padding"] = None
    Path(dst).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def check_no_remote_code(encoder_dir: Path) -> None:
    """Refuse tokenizers/models that would need ``trust_remote_code``."""
    for name in ("tokenizer_config.json", "config.json"):
        p = Path(encoder_dir) / name
        if p.is_file() and "auto_map" in json.loads(p.read_text(encoding="utf-8")):
            raise ValueError(f"{p} declares auto_map: needs trust_remote_code, refused")
    if not (Path(encoder_dir) / "tokenizer.json").is_file():
        raise ValueError(f"{encoder_dir} has no tokenizer.json (a fast tokenizer is required)")


def onnx_spec(meta: CheckpointMeta, *, opset: int, quantization: str) -> dict[str, Any]:
    """The manifest ``onnx`` block for a checkpoint's graph."""
    return {
        "file": "model.onnx",
        "opset": opset,
        "quantization": quantization,
        "inputs": [{"name": n, "dtype": "int64", "shape": ["batch", "seq"]}
                   for n in INPUT_NAMES],
        "outputs": [{"name": h.output, "dtype": "float32", "shape": ["batch", h.width]}
                    for h in meta.heads],
        "heads": [{"name": h.name, "output": h.output, "activation": h.activation,
                   "labels": h.labels, "calibration": h.calibration.model_dump()}
                  for h in meta.heads],
    }


def build_intent_head_manifest(out_dir: Path, meta: CheckpointMeta, *, name: str,
                               version: str, opset: int, quantization: str,
                               metrics: dict[str, float] | None = None,
                               description: str = "") -> ArtifactManifest:
    """Hash ``out_dir`` (model.onnx, tokenizer.json, labels.json, config.json) and write the
    Stage-2 ``intent_head`` manifest. Pure Python: no torch or onnx needed."""
    primary = meta.primary
    manifest = build_manifest(
        Path(out_dir),
        roles={"model.onnx": FileRole.MODEL, "tokenizer.json": FileRole.TOKENIZER,
               "labels.json": FileRole.LABELS, "config.json": FileRole.CONFIG},
        kind=ArtifactKind.INTENT_HEAD,
        name=name,
        version=version,
        description=description or f"Stage-2 intent classifier ({meta.backbone}, "
                                   f"{meta.base_model.id})",
        onnx=onnx_spec(meta, opset=opset, quantization=quantization),
        tokenizer={"file": "tokenizer.json", "max_length": meta.max_length,
                   "truncation_side": "right"},
        labels=primary.labels,
        calibration=primary.calibration.model_dump(),
        metrics={**meta.metrics, **(metrics or {})},
        data_card=meta.data_card.model_dump(),
        base_model=meta.base_model.model_dump(),
    )
    write_manifest(manifest, Path(out_dir))
    return manifest


def build_export_manifest(out_dir: Path, **fields: Any) -> ArtifactManifest:
    """Hash ``out_dir`` and write ``manifest.json``. ``fields`` are ArtifactManifest fields
    (kind, name, version, onnx, tokenizer, labels, calibration, metrics, data_card,
    base_model, ...). Pure Python: usable without the ``[export]`` extra."""
    manifest = build_manifest(Path(out_dir), **fields)
    write_manifest(manifest, Path(out_dir))
    return manifest


def pad_batch(ids: list[list[int]], pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad token id lists into ``(input_ids, attention_mask)`` int64 arrays."""
    seq = max((len(x) for x in ids), default=0)
    input_ids = np.full((len(ids), seq), pad_id, dtype=np.int64)
    mask = np.zeros((len(ids), seq), dtype=np.int64)
    for i, x in enumerate(ids):
        input_ids[i, :len(x)] = x
        mask[i, :len(x)] = 1
    return input_ids, mask


def parity_texts(ckpt_dir: Path, texts: list[str] | None = None,
                 limit: int = MAX_PARITY_TEXTS) -> list[str]:
    """Texts for the parity check: the caller's, else the checkpoint's val + test (+ train)
    splits, plus one over-long text so truncation at ``max_length`` is exercised."""
    if texts is None:
        splits = read_splits(ckpt_dir)
        texts = [t for k in ("val", "test", "train") for t in splits.get(k, ([], []))[0]]
    texts = list(dict.fromkeys(texts))[:limit]
    if not texts:
        raise ValueError("no parity texts: pass texts= or keep splits.json in the checkpoint")
    return [*texts, " ".join(texts[:50]) * 8]


# ---------------------------------------------------------------------- export


def export_onnx(checkpoint_dir: Path, out_dir: Path, *, name: str, version: str,
                opset: int = 17, quantization: Quantization = "int8_dynamic",
                parity_atol: float = 1e-3, min_argmax_agree: float = 0.995,
                texts: list[str] | None = None, description: str = "",
                overwrite: bool = False) -> Path:
    """Export a trained checkpoint to an ``intent_head`` artifact in ``out_dir``.

    Raises :class:`ParityError` (and writes nothing) when the ONNX graph does not match.
    """
    torch = require("torch", "train")
    require("transformers", "train")
    onnx = require("onnx", "export")
    ort = require("onnxruntime", "export")
    tokenizers = require("tokenizers", "export")
    from transformers import AutoTokenizer

    from caliban_ml.router import _torch as T

    if opset < 17:
        raise ValueError("opset must be >= 17")
    if quantization == "fp16":
        raise NotImplementedError("fp16 export is not implemented; use none or int8_dynamic")
    checkpoint_dir, out_dir = Path(checkpoint_dir), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_dir} is not empty (pass overwrite=True)")
        shutil.rmtree(out_dir)
    encoder_dir = checkpoint_dir / ENCODER_DIR
    check_no_remote_code(encoder_dir)
    meta, model = T.load_checkpoint(checkpoint_dir, attn_implementation="eager")
    hf_tok = AutoTokenizer.from_pretrained(str(encoder_dir), local_files_only=True)
    pad_id = hf_tok.pad_token_id if hf_tok.pad_token_id is not None else 0
    outputs = [h.output for h in meta.heads]
    texts = parity_texts(checkpoint_dir, texts)

    with tempfile.TemporaryDirectory(prefix="caliban-export-") as tmp_s:
        tmp = Path(tmp_s)
        tok_json = tmp / "tokenizer.json"
        clean_tokenizer_json(encoder_dir / "tokenizer.json", tok_json)

        # Tokenize like Rust: the shipped tokenizer.json, truncation at max_length.
        rt_tok = tokenizers.Tokenizer.from_file(str(tok_json))
        rt_tok.enable_truncation(meta.max_length)
        rt_ids = [e.ids for e in rt_tok.encode_batch(texts)]
        hf_ids = hf_tok(texts, truncation=True, max_length=meta.max_length)["input_ids"]
        mismatches = sum(a != list(b) for a, b in zip(rt_ids, hf_ids, strict=True))
        if mismatches:
            raise ParityError(f"tokenizer.json disagrees with the training tokenizer on "
                              f"{mismatches}/{len(texts)} texts")

        # Reference logits from torch (fp32, CPU), on the same padded batches.
        batches = [pad_batch(rt_ids[i:i + 32], pad_id) for i in range(0, len(rt_ids), 32)]
        ref: dict[str, list[np.ndarray]] = {o: [] for o in outputs}
        with torch.no_grad():
            for ids, mask in batches:
                outs = model(torch.from_numpy(ids), torch.from_numpy(mask))
                for o, t in zip(outputs, outs, strict=True):
                    ref[o].append(t.float().numpy())

        fp32_path = tmp / "model.fp32.onnx"
        _torch_export(torch, model, batches[0], fp32_path, outputs, opset)
        onnx.checker.check_model(str(fp32_path))
        proto = onnx.load(str(fp32_path), load_external_data=False)
        _check_graph_io(proto, outputs)
        opset = _graph_opset(proto)
        del proto

        metrics: dict[str, float] = {"export_parity_texts": float(len(texts)),
                                     "export_tokenizer_mismatches": float(mismatches)}
        diff, agree = _parity(ort, fp32_path, batches, ref, outputs)
        metrics["export_fp32_max_abs_diff"], metrics["export_fp32_argmax_agree"] = diff, agree
        if diff > parity_atol or agree < 1.0:
            raise ParityError(f"fp32 ONNX parity failed: max|diff|={diff:.3g} "
                              f"(atol {parity_atol}), argmax agreement {agree:.4f}")
        final = fp32_path
        if quantization == "int8_dynamic":
            from onnxruntime.quantization import QuantType, quantize_dynamic

            final = tmp / "model.int8.onnx"
            quantize_dynamic(str(fp32_path), str(final), weight_type=QuantType.QInt8)
            diff, agree = _parity(ort, final, batches, ref, outputs)
            metrics["export_int8_max_abs_diff"] = diff
            metrics["export_int8_argmax_agree"] = agree
            if agree < min_argmax_agree:
                raise ParityError(f"int8 argmax agreement {agree:.4f} < {min_argmax_agree}")
        metrics["export_max_abs_diff"] = diff
        metrics["export_argmax_agree"] = agree

        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(final, out_dir / "model.onnx")
        shutil.copyfile(tok_json, out_dir / "tokenizer.json")
    (out_dir / "labels.json").write_text(json.dumps(meta.primary.labels, indent=2) + "\n")
    (out_dir / "config.json").write_text(json.dumps({
        "type": ONNX_CLASSIFIER_TYPE,
        "primary_head": meta.primary.name,
        "backbone": meta.backbone,
        "pooling": meta.pooling,
        "normalize": meta.normalize,
        "pad_id": int(pad_id),
        "max_length": meta.max_length,
    }, indent=2) + "\n")
    metrics["model_onnx_mb"] = (out_dir / "model.onnx").stat().st_size / 2**20
    build_intent_head_manifest(out_dir, meta, name=name, version=version, opset=opset,
                               quantization=quantization, metrics=metrics,
                               description=description)
    rep = verify_artifact_dir(out_dir, strict=True)
    if not rep.ok:  # pragma: no cover - build_manifest just hashed these files
        raise RuntimeError(f"exported artifact failed verification: {rep.errors}")
    return out_dir


def _torch_export(torch, model, batch: tuple[np.ndarray, np.ndarray], path: Path,
                  outputs: list[str], opset: int) -> None:
    """Export with dynamic batch/seq axes as a single self-contained file.

    Uses the TorchScript exporter (``dynamo=False``) where torch still ships it and falls
    back to the dynamo exporter (opset >= 18) otherwise. The opset actually written is read
    back from the graph for the manifest."""
    ids, mask = (torch.from_numpy(x) for x in batch)
    dynamic_axes = {n: {0: "batch", 1: "seq"} for n in INPUT_NAMES}
    dynamic_axes.update({o: {0: "batch"} for o in outputs})
    common = {"input_names": list(INPUT_NAMES), "output_names": outputs,
              "dynamic_axes": dynamic_axes}
    with torch.no_grad():
        try:
            torch.onnx.export(model, (ids, mask), str(path), opset_version=opset,
                              do_constant_folding=True, dynamo=False, **common)
        except Exception as legacy_error:  # removed or failing TorchScript exporter
            for p in path.parent.glob(path.name + "*"):
                p.unlink()
            try:
                torch.onnx.export(model, (ids, mask), str(path), opset_version=max(opset, 18),
                                  dynamo=True, external_data=False, **common)
            except Exception as exc:
                raise RuntimeError(f"ONNX export failed (TorchScript: {legacy_error!r}; "
                                   f"dynamo: {exc!r})") from exc
    import onnx

    if any(p.suffix == ".data" for p in path.parent.iterdir()):  # inline external data
        model_proto = onnx.load(str(path), load_external_data=True)
        onnx.save_model(model_proto, str(path), save_as_external_data=False)


def _graph_opset(model_proto) -> int:
    return max(o.version for o in model_proto.opset_import if o.domain in ("", "ai.onnx"))


def _check_graph_io(model_proto, outputs: list[str]) -> None:
    ins = [i.name for i in model_proto.graph.input]
    outs = [o.name for o in model_proto.graph.output]
    if ins != list(INPUT_NAMES) or outs != outputs:
        raise ParityError(f"graph io {ins} -> {outs} != expected {list(INPUT_NAMES)} -> "
                          f"{outputs}")


def _parity(ort, path: Path, batches, ref: dict[str, list[np.ndarray]],
            outputs: list[str]) -> tuple[float, float]:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    got: dict[str, list[np.ndarray]] = {o: [] for o in outputs}
    for ids, mask in batches:
        res = sess.run(outputs, {"input_ids": ids, "attention_mask": mask})
        for o, r in zip(outputs, res, strict=True):
            got[o].append(r)
    diffs, agrees = [], []
    for o in outputs:
        d, a = parity_metrics(np.concatenate(ref[o]), np.concatenate(got[o]))
        diffs.append(d)
        agrees.append(a)
    return max(diffs), min(agrees)
