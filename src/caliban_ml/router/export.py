"""ONNX export of trained Stage-2 checkpoints (``[train]`` + ``[export]`` extras).

Target runtime is ONNX Runtime via the Rust ``ort`` crate or ``candle`` for the pure-Rust
build. :func:`export_onnx` turns a checkpoint written by
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

:func:`export_embedder` exports only the fine-tuned encoder (pooled and L2-normalised in
the graph, output ``embedding``) as an ``embedder`` artifact for the Stage-1 kNN.

Quantization is ``none`` (fp32) by default. ``int8_dynamic`` quantizes weight MatMuls only
(:data:`INT8_DYNAMIC_OPTIONS`); ORT's default dynamic INT8 broke the real ModernBERT-base
head (30% argmax agreement), see the comment on :data:`INT8_DYNAMIC_OPTIONS`.

Parity is checked before anything is written: torch fp32 vs ONNX Runtime fp32
(``max |diff| <= parity_atol``) and, when quantised, the INT8 graph's argmax agreement
(``>= min_argmax_agree``; minimum cosine similarity for embedders). The ONNX side is fed by
the *shipped* ``tokenizer.json`` (through the ``tokenizers`` library, as Rust does), and its
token ids must equal the training tokenizer's. All numbers land in the manifest metrics
(``export_*``).
"""

from __future__ import annotations

import json
import re
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

Quantization = Literal["none", "int8_dynamic"]
EMBEDDING_OUTPUT = "embedding"
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


# ---------------------------------------------------------------------- INT8 config

# Dynamic INT8 settings. On the AWS run, ORT's default dynamic INT8 gave 30% argmax
# agreement with torch on the fine-tuned ModernBERT-base head (fp32 ONNX matched torch to
# 3e-5, so the export itself was fine). ORT's defaults quantize the embedding Gather and
# every weight MatMul with one per-tensor scale, and quantize each MatMul's input activation
# with one dynamic per-tensor uint8 scale. ModernBERT-family encoders are 22 layers deep,
# have large-magnitude outlier dimensions in the residual stream and large GeGLU outputs
# feeding the MLP down-projection (``mlp/Wo``); with per-tensor scales the outliers take up
# almost all of the 256 levels and the error compounds layer by layer.
#
# The conservative settings below quantize weight MatMuls only (constant B), per output
# channel, and keep in fp32 the embeddings, LayerNorms, attention score/value products, the
# MLP down-projections (whose inputs carry the GeGLU outliers) and the classifier heads. On a
# tiny ModernBERT with injected residual outliers they keep the embedding output at a
# minimum cosine of 0.9999 to fp32 where ORT's defaults drop to about 0.95
# (tests/test_stage2_integration.py). Activation quantization is still per-tensor, so this is
# not proven to fix the real 22-layer models: export is gated on parity, the default
# quantization is "none", and scripts/router_stage2_eval.py probes INT8 (both these settings
# and ORT's defaults) on every GPU run so the next run decides.
INT8_DYNAMIC_OPTIONS: dict[str, Any] = {
    "per_channel": True,
    "op_types_to_quantize": ["MatMul"],
    "extra_options": {"MatMulConstBOnly": True},
}
# Node-name patterns (TorchScript export names) kept in fp32 under int8_dynamic.
INT8_EXCLUDE_PATTERNS = (r"/mlp/Wo/", r"/heads/")


def int8_nodes_to_exclude(model_path: Path) -> list[str]:
    import onnx

    pats = [re.compile(p) for p in INT8_EXCLUDE_PATTERNS]
    graph = onnx.load(str(model_path), load_external_data=False).graph
    return [n.name for n in graph.node if any(p.search(n.name) for p in pats)]


def quantize_int8_dynamic(src: Path, dst: Path, *, legacy_defaults: bool = False) -> None:
    """Dynamic INT8 (weights QInt8) with :data:`INT8_DYNAMIC_OPTIONS` and
    :data:`INT8_EXCLUDE_PATTERNS`; ``legacy_defaults`` uses ORT's defaults instead (kept to
    measure and reproduce the failure)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    opts: dict[str, Any] = {}
    if not legacy_defaults:
        opts = {**INT8_DYNAMIC_OPTIONS, "nodes_to_exclude": int8_nodes_to_exclude(src)}
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, **opts)


def cosine_parity(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """``(max |diff|, min row-wise cosine similarity)`` for embedding outputs."""
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(candidate, dtype=np.float64)
    if ref.shape != got.shape:
        raise ParityError(f"output shape {got.shape} != reference {ref.shape}")
    if not ref.size:
        return 0.0, 1.0
    num = (ref * got).sum(-1)
    den = np.maximum(np.linalg.norm(ref, axis=-1) * np.linalg.norm(got, axis=-1), 1e-12)
    return float(np.max(np.abs(ref - got))), float(np.min(num / den))


# ---------------------------------------------------------------------- export


def export_onnx(checkpoint_dir: Path, out_dir: Path, *, name: str, version: str,
                opset: int = 17, quantization: Quantization = "none",
                parity_atol: float = 1e-3, min_argmax_agree: float = 0.995,
                texts: list[str] | None = None, description: str = "",
                overwrite: bool = False) -> Path:
    """Export a trained checkpoint to an ``intent_head`` artifact in ``out_dir``.

    Raises :class:`ParityError` (and writes nothing) when the ONNX graph does not match.
    """
    ctx = _ExportContext.open(checkpoint_dir, out_dir, opset=opset, quantization=quantization,
                              overwrite=overwrite, texts=texts)
    meta = ctx.meta
    outputs = [h.output for h in meta.heads]
    metrics = ctx.export(ctx.model, outputs, parity_atol=parity_atol,
                         int8_gate=("argmax", min_argmax_agree))
    out_dir = ctx.out_dir
    (out_dir / "labels.json").write_text(json.dumps(meta.primary.labels, indent=2) + "\n")
    (out_dir / "config.json").write_text(json.dumps({
        "type": ONNX_CLASSIFIER_TYPE,
        "primary_head": meta.primary.name,
        "backbone": meta.backbone,
        "pooling": meta.pooling,
        "normalize": meta.normalize,
        "pad_id": int(ctx.pad_id),
        "max_length": meta.max_length,
    }, indent=2) + "\n")
    metrics["model_onnx_mb"] = (out_dir / "model.onnx").stat().st_size / 2**20
    build_intent_head_manifest(out_dir, meta, name=name, version=version, opset=ctx.opset,
                               quantization=quantization, metrics=metrics,
                               description=description)
    return _verified(out_dir)


def export_embedder(checkpoint_dir: Path, out_dir: Path, *, name: str, version: str,
                    opset: int = 17, quantization: Quantization = "none",
                    parity_atol: float = 1e-3, min_cosine: float = 0.99,
                    texts: list[str] | None = None, description: str = "",
                    overwrite: bool = False) -> Path:
    """Export only the checkpoint's fine-tuned encoder (pool + L2 normalise in the graph) as
    an ``embedder`` artifact, for the Stage-1 kNN.

    The graph outputs ``embedding`` [batch, hidden], already pooled and normalised, so the
    manifest declares ``pooling = "none"``. A kNN calibration fitted in this space
    (``router embed --embedder-artifact`` then ``router knn-eval``) ``requires`` this
    artifact. INT8 is gated on the minimum torch-vs-ORT cosine similarity.
    """
    ctx = _ExportContext.open(checkpoint_dir, out_dir, opset=opset, quantization=quantization,
                              overwrite=overwrite, texts=texts)
    from caliban_ml.router import _torch as T

    metrics = ctx.export(T.EmbeddingOnly(ctx.model), [EMBEDDING_OUTPUT],
                         parity_atol=parity_atol, int8_gate=("cosine", min_cosine))
    (ctx.out_dir / "config.json").write_text(json.dumps({
        "type": "onnx_embedder", "pad_id": int(ctx.pad_id), "max_length": ctx.meta.max_length,
        "pooling_in_graph": ctx.meta.pooling, "normalized_in_graph": True,
    }, indent=2) + "\n")
    metrics["model_onnx_mb"] = (ctx.out_dir / "model.onnx").stat().st_size / 2**20
    build_embedder_manifest(ctx.out_dir, ctx.meta, name=name, version=version, opset=ctx.opset,
                            quantization=quantization, metrics=metrics,
                            description=description)
    return _verified(ctx.out_dir)


def build_embedder_manifest(out_dir: Path, meta: CheckpointMeta, *, name: str, version: str,
                            opset: int, quantization: str,
                            metrics: dict[str, float] | None = None,
                            description: str = "") -> ArtifactManifest:
    """Hash ``out_dir`` and write an ``embedder`` manifest for a fine-tuned encoder.
    Pure Python: no torch or onnx needed."""
    manifest = build_manifest(
        Path(out_dir),
        roles={"model.onnx": FileRole.MODEL, "tokenizer.json": FileRole.TOKENIZER,
               "config.json": FileRole.CONFIG},
        kind=ArtifactKind.EMBEDDER,
        name=name,
        version=version,
        description=description or f"Fine-tuned {meta.base_model.id} encoder ({meta.backbone} "
                                   f"checkpoint) as a Stage-1 kNN embedder",
        onnx={"file": "model.onnx", "opset": opset, "quantization": quantization,
              "inputs": [{"name": n, "dtype": "int64", "shape": ["batch", "seq"]}
                         for n in INPUT_NAMES],
              "outputs": [{"name": EMBEDDING_OUTPUT, "dtype": "float32",
                           "shape": ["batch", meta.hidden_size]}]},
        tokenizer={"file": "tokenizer.json", "max_length": meta.max_length,
                   "truncation_side": "right"},
        embedding={"dim": meta.hidden_size, "output": EMBEDDING_OUTPUT, "pooling": "none",
                   "normalize": True},
        metrics={**meta.metrics, **(metrics or {})},
        data_card=meta.data_card.model_dump(),
        base_model=meta.base_model.model_dump(),
    )
    write_manifest(manifest, Path(out_dir))
    return manifest


def _verified(out_dir: Path) -> Path:
    rep = verify_artifact_dir(out_dir, strict=True)
    if not rep.ok:  # pragma: no cover - build_manifest just hashed these files
        raise RuntimeError(f"exported artifact failed verification: {rep.errors}")
    return out_dir


class _ExportContext:
    """Shared export steps: load the checkpoint, tokenize the parity texts the way Rust does,
    export, check parity, optionally quantize, and copy model.onnx + tokenizer.json."""

    @classmethod
    def open(cls, checkpoint_dir: Path, out_dir: Path, *, opset: int, quantization: str,
             overwrite: bool, texts: list[str] | None) -> _ExportContext:
        if opset < 17:
            raise ValueError("opset must be >= 17")
        if quantization not in ("none", "int8_dynamic"):
            raise ValueError(f"unsupported quantization {quantization!r}: use none or "
                             "int8_dynamic")
        self = cls()
        self.torch = require("torch", "train")
        require("transformers", "train")
        self.onnx = require("onnx", "export")
        self.ort = require("onnxruntime", "export")
        tokenizers = require("tokenizers", "export")
        from transformers import AutoTokenizer

        from caliban_ml.router import _torch as T

        self.quantization, self.opset = quantization, opset
        checkpoint_dir, self.out_dir = Path(checkpoint_dir), Path(out_dir)
        if self.out_dir.exists() and any(self.out_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{self.out_dir} is not empty (pass overwrite=True)")
            shutil.rmtree(self.out_dir)
        self.encoder_dir = checkpoint_dir / ENCODER_DIR
        check_no_remote_code(self.encoder_dir)
        self.meta, self.model = T.load_checkpoint(checkpoint_dir, attn_implementation="eager")
        hf_tok = AutoTokenizer.from_pretrained(str(self.encoder_dir), local_files_only=True)
        self.pad_id = hf_tok.pad_token_id if hf_tok.pad_token_id is not None else 0
        self.texts = parity_texts(checkpoint_dir, texts)
        self.tmp = Path(tempfile.mkdtemp(prefix="caliban-export-"))
        self.tok_json = self.tmp / "tokenizer.json"
        clean_tokenizer_json(self.encoder_dir / "tokenizer.json", self.tok_json)
        # Tokenize like Rust: the shipped tokenizer.json, truncation at max_length.
        rt_tok = tokenizers.Tokenizer.from_file(str(self.tok_json))
        rt_tok.enable_truncation(self.meta.max_length)
        rt_ids = [e.ids for e in rt_tok.encode_batch(self.texts)]
        hf_ids = hf_tok(self.texts, truncation=True,
                        max_length=self.meta.max_length)["input_ids"]
        self.mismatches = sum(a != list(b) for a, b in zip(rt_ids, hf_ids, strict=True))
        if self.mismatches:
            shutil.rmtree(self.tmp, ignore_errors=True)
            raise ParityError(f"tokenizer.json disagrees with the training tokenizer on "
                              f"{self.mismatches}/{len(self.texts)} texts")
        # Padded batches of mixed lengths, plus unpadded single texts: the TorchScript trace
        # of the attention mask code must hold on both paths (batch-1 is core's hot path).
        self.batches = [pad_batch(rt_ids[i:i + 32], self.pad_id)
                        for i in range(0, len(rt_ids), 32)]
        self.batches += [pad_batch([ids], self.pad_id) for ids in rt_ids[:8]]
        return self

    def export(self, module, outputs: list[str], *, parity_atol: float,
               int8_gate: tuple[str, float]) -> dict[str, float]:
        torch, tmp = self.torch, self.tmp
        try:
            ref: dict[str, list[np.ndarray]] = {o: [] for o in outputs}
            with torch.no_grad():
                for ids, mask in self.batches:
                    outs = module(torch.from_numpy(ids), torch.from_numpy(mask))
                    outs = outs if isinstance(outs, tuple) else (outs,)
                    for o, t in zip(outputs, outs, strict=True):
                        ref[o].append(t.float().numpy())
            fp32_path = tmp / "model.fp32.onnx"
            _torch_export(torch, module, self.batches[0], fp32_path, outputs, self.opset)
            self.onnx.checker.check_model(str(fp32_path))
            proto = self.onnx.load(str(fp32_path), load_external_data=False)
            _check_graph_io(proto, outputs)
            self.opset = _graph_opset(proto)
            del proto

            gate, threshold = int8_gate
            metrics: dict[str, float] = {"export_parity_texts": float(len(self.texts)),
                                         "export_tokenizer_mismatches": float(self.mismatches)}
            diff, agree = _parity(self.ort, fp32_path, self.batches, ref, outputs, gate)
            metrics["export_fp32_max_abs_diff"] = diff
            metrics[f"export_fp32_{gate}_agree"] = agree
            if diff > parity_atol or (gate == "argmax" and agree < 1.0):
                raise ParityError(f"fp32 ONNX parity failed: max|diff|={diff:.3g} "
                                  f"(atol {parity_atol}), {gate} agreement {agree:.4f}")
            final = fp32_path
            if self.quantization == "int8_dynamic":
                final = tmp / "model.int8.onnx"
                quantize_int8_dynamic(fp32_path, final)
                diff, agree = _parity(self.ort, final, self.batches, ref, outputs, gate)
                metrics["export_int8_max_abs_diff"] = diff
                metrics[f"export_int8_{gate}_agree"] = agree
                if agree < threshold:
                    raise ParityError(f"int8 {gate} agreement {agree:.4f} < {threshold}")
            metrics["export_max_abs_diff"] = diff
            metrics[f"export_{gate}_agree"] = agree
            self.out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(final, self.out_dir / "model.onnx")
            shutil.copyfile(self.tok_json, self.out_dir / "tokenizer.json")
            return metrics
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


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
            outputs: list[str], gate: str = "argmax") -> tuple[float, float]:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    got: dict[str, list[np.ndarray]] = {o: [] for o in outputs}
    for ids, mask in batches:
        res = sess.run(outputs, {"input_ids": ids, "attention_mask": mask})
        for o, r in zip(outputs, res, strict=True):
            got[o].append(r)
    measure = parity_metrics if gate == "argmax" else cosine_parity
    diffs, agrees = [], []
    for o in outputs:
        d, a = measure(np.concatenate(ref[o]), np.concatenate(got[o]))
        diffs.append(d)
        agrees.append(a)
    return max(diffs), min(agrees)
