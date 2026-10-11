"""Embed intent exemplars into an ``.npz`` for ``caliban-ml router knn-eval``.

Two backends, both offline:

* ``onnx`` (``[export]`` extra): runs the *shipped* embedder artifact (manifest +
  model.onnx + tokenizer.json) with onnxruntime. This is the preferred path: thresholds
  are then picked in exactly the vector space the Rust data plane uses.
* ``sentence-transformers`` (``[train]`` extra): a local model directory. Handy for
  experiments; re-calibrate with the ONNX backend before shipping.

The ``.npz`` layout is ``embeddings: float32 (n, d)``, ``labels: str (n,)``,
``texts: str (n,)``; OOS rows carry the label ``__oos__``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from caliban_ml._extras import require


def save_npz(path: Path, embeddings: np.ndarray, labels: list[str], texts: list[str]) -> None:
    np.savez_compressed(path, embeddings=np.asarray(embeddings, dtype=np.float32),
                        labels=np.array(labels, dtype=str), texts=np.array(texts, dtype=str))


def load_npz(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as z:
        return z["embeddings"].astype(np.float64), [str(x) for x in z["labels"]]


def embed_sentence_transformers(model_dir: Path, texts: list[str], prefix: str = ""
                                ) -> np.ndarray:
    st = require("sentence_transformers", "train")
    model = st.SentenceTransformer(str(model_dir), local_files_only=True)
    return model.encode([prefix + t for t in texts], normalize_embeddings=True,
                        convert_to_numpy=True)


def pool_hidden(hidden: np.ndarray, attention_mask: np.ndarray, pooling: str) -> np.ndarray:
    """Pool ``(n, seq, d)`` token states into ``(n, d)`` per ``EmbeddingSpec.pooling``.

    ``none`` expects an already pooled ``(n, d)`` output; ``last`` takes the last non-pad
    token of each row (right padding)."""
    h = np.asarray(hidden, dtype=np.float64)
    if pooling == "none":
        if h.ndim != 2:
            raise ValueError(f"pooling 'none' expects a (n, d) output, got shape {h.shape}")
        return h
    if h.ndim != 3:
        raise ValueError(f"pooling {pooling!r} expects a (n, seq, d) output, got {h.shape}")
    m = np.asarray(attention_mask, dtype=np.float64)
    if pooling == "cls":
        return h[:, 0]
    if pooling == "mean":
        return (h * m[..., None]).sum(axis=1) / np.maximum(m.sum(axis=1, keepdims=True), 1e-9)
    if pooling == "last":
        last = np.maximum(m.sum(axis=1).astype(np.int64) - 1, 0)
        return h[np.arange(h.shape[0]), last]
    raise ValueError(f"unknown pooling {pooling!r}")


def embed_onnx_artifact(artifact_dir: Path, texts: list[str], *, batch_size: int = 32,
                        passage: bool = False) -> np.ndarray:
    """Embed ``texts`` with a shipped ``embedder`` artifact, the way the Rust side does.

    The artifact is verified first (manifest schema, licence policy, every file's size and
    SHA-256); the tokenizer and graph are then loaded from those verified files. Inputs are
    bound by the names the manifest declares (``input_ids``, ``attention_mask`` and, when
    declared, ``token_type_ids`` as zeros), never by position.
    """
    from caliban_ml.artifacts import ArtifactKind, verify_artifact_dir

    artifact_dir = Path(artifact_dir)
    rep = verify_artifact_dir(artifact_dir)
    if not rep.ok or rep.manifest is None:
        raise ValueError(f"embedder artifact {artifact_dir} failed verification: {rep.errors}")
    m = rep.manifest
    if m.kind != ArtifactKind.EMBEDDER or m.onnx is None or m.tokenizer is None \
            or m.embedding is None:
        raise ValueError(f"{artifact_dir} is a {m.kind.value} artifact, not an embedder")
    ort = require("onnxruntime", "export")
    tokenizers = require("tokenizers", "export")

    names = {t.name: t for t in m.onnx.inputs}
    for required in ("input_ids", "attention_mask"):
        if required not in names:
            raise ValueError(f"embedder declares no {required!r} input")
    extra = sorted(set(names) - {"input_ids", "attention_mask", "token_type_ids"})
    if extra:
        raise ValueError(f"embedder declares unsupported inputs {extra}")

    tok = tokenizers.Tokenizer.from_file(str(artifact_dir / m.tokenizer.file))
    # Pad id: the tokenizer's own padding config, else a conventional pad token, else 0.
    # Padded positions are masked out, so the id only has to be a valid vocabulary entry.
    pad_id = (tok.padding or {}).get("pad_id")
    if pad_id is None:
        pad_id = next((tid for pad in ("[PAD]", "<pad>", "<|pad|>")
                       if (tid := tok.token_to_id(pad)) is not None), 0)
    tok.no_padding()
    tok.enable_truncation(m.tokenizer.max_length, direction=m.tokenizer.truncation_side)

    sess = ort.InferenceSession(str(artifact_dir / m.onnx.file),
                                providers=["CPUExecutionProvider"])
    spec = m.embedding
    prefix = spec.passage_prefix if passage else spec.query_prefix
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        enc = tok.encode_batch([prefix + t for t in texts[i:i + batch_size]])
        seq = max(len(e.ids) for e in enc)
        ids = np.full((len(enc), seq), pad_id, dtype=np.int64)
        mask = np.zeros((len(enc), seq), dtype=np.int64)
        for j, e in enumerate(enc):
            ids[j, :len(e.ids)] = e.ids
            mask[j, :len(e.ids)] = 1
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(ids)
        (hidden,) = sess.run([spec.output], feed)
        out.append(pool_hidden(hidden, mask, spec.pooling))
    if not out:
        return np.zeros((0, spec.dim), dtype=np.float32)
    emb = np.concatenate(out, axis=0)
    if emb.shape[1] != spec.dim:
        raise ValueError(f"embedder produced dim {emb.shape[1]}, manifest says {spec.dim}")
    if spec.normalize:
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return emb.astype(np.float32)
