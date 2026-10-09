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


def embed_onnx_artifact(artifact_dir: Path, texts: list[str]) -> np.ndarray:
    require("onnxruntime", "export")
    # TODO(embed/onnx): load manifest (verify hashes first), tokenizers.Tokenizer.from_file(
    # tokenizer.json) with truncation to manifest.tokenizer.max_length, run the session with
    # the declared input names, pool per manifest.embedding.pooling, L2-normalise when
    # manifest.embedding.normalize, prepend manifest.embedding.query_prefix.
    raise NotImplementedError("ONNX embedder backend: see TODO(embed/onnx)")
