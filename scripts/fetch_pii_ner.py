#!/usr/bin/env python3
"""Fetch a pinned PII NER (token-classification) ONNX model and package it as a Caliban artifact.

Downloads ``model.onnx`` + ``tokenizer.json`` + ``config.json`` from the Hugging Face Hub at a
**pinned revision**, checks every file against hashes pinned in this script (LFS sha256, or the
git blob id for small non-LFS files), and writes ``manifest.json`` (kind ``pii_ner``) next to them::

    ml/artifacts/pii_ner/<name>/<version>/
        manifest.json
        model.onnx
        tokenizer.json
        config.json

The manifest follows ``schemas/artifact-manifest.schema.json``: every file with sha256 + size,
the ONNX inputs/outputs/opset read from the graph itself, the BIO label list from
``config.json`` ``id2label`` (index = class id), and the licence of the weights and of every
training dataset. Finally it runs ``caliban-ml manifest verify --strict`` when available.

Stdlib only (no ``huggingface_hub`` / ``onnx`` needed): plain HTTPS downloads from
``https://huggingface.co/<repo>/resolve/<revision>/<file>`` and a tiny protobuf reader for the
ONNX header. ``HF_TOKEN`` is sent if set. Nothing here runs at Caliban runtime.

Licence policy: if a preset lists something that needs review (e.g. CC-BY-SA source text used
for fine-tuning), the manifest only validates once a human records a sign-off with
``--approved-by`` / ``--approval-reason``. Without it the files are still downloaded and hashed
(the Rust loader can use them in development), but the script exits with status 3 and the
artifact must not be bundled.

Usage::

    python3 scripts/fetch_pii_ner.py                       # default preset
    python3 scripts/fetch_pii_ner.py --preset distilbert-ner-en
    python3 scripts/fetch_pii_ner.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ML_ROOT / "artifacts" / "pii_ner"
HF = "https://huggingface.co"


@dataclass(frozen=True)
class RemoteFile:
    src: str  # path inside the HF repo
    dst: str  # path inside the artifact directory
    role: str
    size: int
    sha256: str | None = None  # LFS files: pinned sha256
    git_blob: str | None = None  # non-LFS files: pinned git blob sha1


@dataclass(frozen=True)
class Dataset:
    name: str
    source: str
    licence: str
    revision: str | None = None
    synthetic: bool = False
    n_examples: int | None = None


@dataclass(frozen=True)
class Preset:
    name: str
    version: str
    repo: str
    revision: str
    licence: str  # SPDX id of the weights, from cardData.license at `revision`
    description: str
    files: tuple[RemoteFile, ...]
    quantization: str
    max_length: int
    languages: tuple[str, ...]
    intended_use: str
    limitations: str
    notes: str
    datasets: tuple[Dataset, ...] = field(default_factory=tuple)


PRESETS: dict[str, Preset] = {
    # Default L1 model. MIT weights (card: license: mit), MIT base model (jhu-clsp/mmBERT-small),
    # MIT synthetic training data. ~23 languages, 40 PII types in IOB2 (81 labels).
    "nym-pii-multilingual-small-int8": Preset(
        name="nym-pii-multilingual-small-int8",
        version="3.0.0",
        repo="Wismut/nym-pii-multilingual-small",
        revision="4348999cd3c2e20c49615e9af7c6bbb45b64cd85",
        licence="mit",
        description=(
            "Multilingual PII token classifier (mmBERT-small, 16 layers, IOB2, 40 entity types), "
            "int8 embeddings + fp16 body weights, fp32 compute. L1 detector for caliban-pii."
        ),
        files=(
            RemoteFile("int8/model_int8.onnx", "model.onnx", "model", 138730982,
                       sha256="139006aea2cbd8e709d322f056232570de54661f624143be4893aaa387190286"),
            RemoteFile("int8/tokenizer.json", "tokenizer.json", "tokenizer", 12389891,
                       sha256="c299144e68dfec1dc536204a7ae3712710c5c6cade9269a83f5042250d47d8de"),
            RemoteFile("int8/config.json", "config.json", "config", 5688,
                       git_blob="8a0d83d46217097196bedc48fbae080bff294b21"),
        ),
        quantization="int8_dynamic",
        max_length=512,
        languages=("en", "de", "fr", "es", "it", "pt", "nl", "pl", "sv", "cs", "ro", "tr", "fi",
                   "da", "el", "ru", "uk", "ja", "zh", "ko", "ar", "hi"),
        intended_use=(
            "In-process L1 PII span detection in the Caliban gateway request path (union with "
            "L0 regex/dictionaries), before reversible pseudonymization."
        ),
        limitations=(
            "Recall-leaning; over-flags ambiguous terms. Reference-style IDs are the weakest "
            "class. Card benchmarks: real-text F1 76.4, ai4privacy OOD 67.2 (int8). No "
            "organisation-in-prose training signal beyond COMPANY_NAME."
        ),
        notes=(
            "Upstream: Wismut/nym-pii-multilingual-small (v3), a distilled student of "
            "Wismut/nym-pii-multilingual; base jhu-clsp/mmBERT-small (MIT). Fine-tuning also "
            "used ~77.5k Wikipedia passages auto-labelled by google/gemma-4-26B-A4B "
            "(Apache-2.0); that corpus is not redistributed."
        ),
        datasets=(
            Dataset("nym-pii-multilingual-data", "hf:datasets/Wismut/nym-pii-multilingual-data",
                    "mit", revision="abe23bf08c305f5824b1643dc4f079f06ccb2ad1", synthetic=True,
                    n_examples=724500),
            Dataset("wikipedia-llm-labelled", "https://www.wikipedia.org (passages labelled by "
                    "google/gemma-4-26B-A4B; not redistributed)", "cc-by-sa-4.0",
                    n_examples=77500),
        ),
    ),
    # English CoNLL-2003 fallback: PER/ORG/LOC/MISC. Apache-2.0 weights.
    "distilbert-ner-en": Preset(
        name="distilbert-ner-en",
        version="1.0.0",
        repo="dslim/distilbert-NER",
        revision="dfa2838a127384aabb82ed7719e16dab84c42a2a",
        licence="apache-2.0",
        description="English CoNLL-2003 NER (DistilBERT-cased, PER/ORG/LOC/MISC, IOB2), fp32.",
        files=(
            RemoteFile("onnx/model.onnx", "model.onnx", "model", 260926482,
                       sha256="4440f9fc64cd28ac75d83a38d89716f25947799640cd0e5f1f9f6e57b9c14160"),
            RemoteFile("tokenizer.json", "tokenizer.json", "tokenizer", 669021,
                       git_blob="de7ac9b6bda3ed337df237120512a771c1c3f519"),
            RemoteFile("config.json", "config.json", "config", 926,
                       git_blob="8f34a873ec20f0e84bbd0eaeffdc02dc93c71e0a"),
        ),
        quantization="none",
        max_length=512,
        languages=("en",),
        intended_use="English-only L1 PER/ORG/LOC detection in the Caliban request path.",
        limitations="English news domain (Reuters 1996); weak on lowercase/chat text and "
                    "non-English.",
        notes="Upstream card reports CoNLL-2003 test F1 ~0.92.",
        datasets=(
            # The Reuters RCV1 text behind CoNLL-2003 is under a research-use agreement.
            Dataset("conll2003", "hf:datasets/eriktks/conll2003", "other"),
        ),
    ),
}


_NYM = PRESETS["nym-pii-multilingual-small-int8"]
_NYM_CONFIG = RemoteFile("config.json", "config.json", "config", 5688,
                         git_blob="8a0d83d46217097196bedc48fbae080bff294b21")
_NYM_TOKENIZER = RemoteFile("tokenizer.json", "tokenizer.json", "tokenizer", 12389891,
                            sha256="c299144e68dfec1dc536204a7ae3712710c5c6cade9269a83f5042250d47d8de")
# Same model, full dynamic int8 (smallest/fastest; card: ~-4 points ai4privacy OOD).
PRESETS["nym-pii-multilingual-small-edge-int8"] = replace(
    _NYM,
    name="nym-pii-multilingual-small-edge-int8",
    description="Multilingual PII token classifier (mmBERT-small, 16 layers, IOB2, 40 entity "
                "types), full dynamic int8. Fastest L1 variant; card: ~-4 pts OOD vs int8/.",
    files=(
        RemoteFile("edge-int8/model_int8.onnx", "model.onnx", "model", 108233527,
                   sha256="e9a3a8c8cd55b3bcf329de5a9307cfae5053ce93c00330c33facd022e60daa17"),
        _NYM_TOKENIZER, _NYM_CONFIG,
    ),
    quantization="int8_dynamic",
)
# Same model, fp32 reference (accuracy-first; 429 MB).
PRESETS["nym-pii-multilingual-small-fp32"] = replace(
    _NYM,
    name="nym-pii-multilingual-small-fp32",
    description="Multilingual PII token classifier (mmBERT-small, 16 layers, IOB2, 40 entity "
                "types), fp32 reference export.",
    files=(
        RemoteFile("model.onnx", "model.onnx", "model", 429218752,
                   sha256="60c2ae1a5992e43d4022a29dcc81cc45250bdb08fd2ee472cfb4d8800bbe87aa"),
        _NYM_TOKENIZER, _NYM_CONFIG,
    ),
    quantization="none",
)


# --------------------------------------------------------------------------- download


def _request(url: str) -> urllib.request.Request:
    headers = {"User-Agent": "caliban-ml/fetch_pii_ner"}
    if tok := os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {tok}"
    return urllib.request.Request(url, headers=headers)


def download(repo: str, revision: str, f: RemoteFile, dest: Path) -> None:
    """Stream to a temp file, verify the pinned hash, then atomically move into place."""
    if dest.is_file() and dest.stat().st_size == f.size and _matches(dest, f):
        print(f"  = {f.dst} (already present, hash ok)")
        return
    url = f"{HF}/{repo}/resolve/{revision}/{f.src}"
    print(f"  ↓ {f.src} → {f.dst} ({f.size / 1e6:.1f} MB)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(_request(url), timeout=60) as r:
            shutil.copyfileobj(r, out, length=1 << 20)
        tmp_path = Path(tmp)
        if tmp_path.stat().st_size != f.size:
            raise SystemExit(f"{f.src}: size {tmp_path.stat().st_size} != pinned {f.size}")
        if not _matches(tmp_path, f):
            raise SystemExit(f"{f.src}: hash does not match the pinned value; refusing")
        tmp_path.replace(dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _matches(path: Path, f: RemoteFile) -> bool:
    if f.sha256:
        return sha256_file(path) == f.sha256
    if f.git_blob:
        h = hashlib.sha1(usedforsecurity=False)
        h.update(f"blob {path.stat().st_size}\0".encode())
        h.update(path.read_bytes())
        return h.hexdigest() == f.git_blob
    raise SystemExit(f"{f.src}: no pinned hash in the preset")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- ONNX header

_ONNX_DTYPES = {1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64", 9: "bool",
                10: "float16", 16: "bfloat16"}


def _varint(buf: memoryview, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf: memoryview):
    """Yield (field_number, wire_type, value) for one protobuf message."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
            yield fno, wt, v
        elif wt == 1:
            yield fno, wt, buf[i:i + 8]
            i += 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            yield fno, wt, buf[i:i + ln]
            i += ln
        elif wt == 5:
            yield fno, wt, buf[i:i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")


def _value_info(buf: memoryview, pos: int) -> dict:
    name, dtype, shape = "", None, []
    for fno, _, v in _fields(buf):
        if fno == 1:
            name = bytes(v).decode()
        elif fno == 2:  # TypeProto
            for tf, _, tv in _fields(v):
                if tf != 1:  # tensor_type
                    continue
                for ef, _, ev in _fields(tv):
                    if ef == 1:
                        dtype = _ONNX_DTYPES.get(ev)
                    elif ef == 2:  # TensorShapeProto
                        for df, _, dv in _fields(ev):
                            if df != 1:
                                continue
                            dim: int | str | None = None
                            for xf, _, xv in _fields(dv):
                                if xf == 1:
                                    dim = int(xv)
                                elif xf == 2:
                                    dim = bytes(xv).decode()
                            if isinstance(dim, int) and dim >= 1:
                                shape.append(dim)
                            else:
                                dim = dim if isinstance(dim, str) else None
                                s = re.sub(r"[^A-Za-z0-9_]", "_", dim or "")
                                if not s or not re.match(r"[A-Za-z_]", s):
                                    s = f"d{pos}_{len(shape)}"
                                shape.append(s)
    if dtype is None:
        raise ValueError(f"tensor {name!r}: unsupported or missing dtype")
    return {"name": name, "dtype": dtype, "shape": shape}


def onnx_header(path: Path) -> tuple[int, list[dict], list[dict]]:
    """(opset of the default domain, graph inputs minus initializers, graph outputs)."""
    buf = memoryview(path.read_bytes())
    opset, graph = None, None
    for fno, _, v in _fields(buf):
        if fno == 8:  # opset_import
            domain, version = "", None
            for of, _, ov in _fields(v):
                if of == 1:
                    domain = bytes(ov).decode()
                elif of == 2:
                    version = ov
            if domain in ("", "ai.onnx") and version is not None:
                opset = version
        elif fno == 7:
            graph = v
    if opset is None or graph is None:
        raise SystemExit(f"{path}: not an ONNX model (no graph/opset)")
    inits: set[str] = set()
    raw_in, raw_out = [], []
    for fno, _, v in _fields(graph):
        if fno == 5:  # initializer (TensorProto.name = 8)
            for tf, _, tv in _fields(v):
                if tf == 8:
                    inits.add(bytes(tv).decode())
        elif fno == 11:
            raw_in.append(v)
        elif fno == 12:
            raw_out.append(v)
    inputs = [_value_info(v, i) for i, v in enumerate(raw_in)]
    inputs = [t for t in inputs if t["name"] not in inits]
    outputs = [_value_info(v, 100 + i) for i, v in enumerate(raw_out)]
    return int(opset), inputs, outputs


# --------------------------------------------------------------------------- manifest


def labels_from_config(cfg_path: Path) -> list[str]:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    if sorted(id2label) != list(range(len(id2label))):
        raise SystemExit("config.json id2label ids are not contiguous from 0")
    return [id2label[i] for i in range(len(id2label))]


def producer_version() -> str:
    init = ML_ROOT / "src" / "caliban_ml" / "__init__.py"
    m = re.search(r'__version__\s*=\s*"([^"]+)"', init.read_text(encoding="utf-8"))
    return m.group(1) if m else "0.0.0"


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ML_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(p: Preset, root: Path, approved_by: str | None, reason: str | None) -> dict:
    model = root / "model.onnx"
    opset, inputs, outputs = onnx_header(model)
    labels = labels_from_config(root / "config.json")
    logits = next((t for t in outputs if t["name"] == "logits"), outputs[0])
    last = logits["shape"][-1] if logits["shape"] else None
    if isinstance(last, int) and last != len(labels):
        raise SystemExit(f"logits last dim {last} != {len(labels)} labels")

    files = []
    for f in sorted(p.files, key=lambda f: f.dst):
        path = root / f.dst
        files.append({"path": f.dst, "sha256": sha256_file(path),
                      "size_bytes": path.stat().st_size, "role": f.role})

    producer = {"tool": "caliban-ml", "version": producer_version()}
    if commit := git_commit():
        producer["git_commit"] = commit

    m: dict = {
        "manifest_version": 1,
        "kind": "pii_ner",
        "name": p.name,
        "version": p.version,
        "description": p.description,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "producer": producer,
        "files": files,
        "onnx": {"file": "model.onnx", "opset": opset, "inputs": inputs, "outputs": outputs,
                 "quantization": p.quantization},
        "tokenizer": {"file": "tokenizer.json", "format": "hf_tokenizers_json",
                      "max_length": p.max_length, "truncation_side": "right"},
        "labels": labels,
        "data_card": {
            "training_datasets": [
                {k: v for k, v in {
                    "name": d.name, "source": d.source, "revision": d.revision,
                    "licence": d.licence, "n_examples": d.n_examples, "synthetic": d.synthetic,
                }.items() if v is not None}
                for d in p.datasets
            ],
            "languages": list(p.languages),
            "intended_use": p.intended_use,
            "limitations": p.limitations,
            "contains_personal_data": False,
            "notes": p.notes,
        },
        "base_model": {"id": p.repo, "revision": p.revision, "licence": p.licence,
                       "url": f"{HF}/{p.repo}/tree/{p.revision}"},
    }
    if approved_by:
        review = [d for d in p.datasets if d.licence not in ALLOWED_HINT]
        m["licence_overrides"] = [
            {"subject": d.name, "licence": d.licence, "approved_by": approved_by,
             "approved_on": date.today().isoformat(),
             "reason": reason or "Fine-tuning corpus is not redistributed; weights are licensed "
                                 "under the upstream card licence."}
            for d in review
        ]
    return m


# Mirrors caliban_ml.artifacts.licenses.ALLOWED_LICENCES (only used to decide which datasets
# need an override entry; the authoritative check is `caliban-ml manifest verify`).
ALLOWED_HINT = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0",
                "unlicense", "cc-by-4.0", "cc-by-3.0", "zlib", "bsl-1.0", "psf-2.0"}


def verify(root: Path) -> int | None:
    exe = ML_ROOT / ".venv" / "bin" / "caliban-ml"
    found = str(exe) if exe.exists() else shutil.which("caliban-ml")
    if not found:
        print("caliban-ml not found; skipping `manifest verify`")
        return None
    r = subprocess.run([found, "manifest", "verify", str(root), "--strict"])
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preset", default="nym-pii-multilingual-small-int8", choices=sorted(PRESETS))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="artifacts/pii_ner root")
    ap.add_argument("--list", action="store_true", help="list presets and exit")
    ap.add_argument("--approved-by", help="record a human licence sign-off for review items")
    ap.add_argument("--approval-reason", help="why the review items are acceptable")
    args = ap.parse_args()

    if args.list:
        for name, p in sorted(PRESETS.items()):
            rev = [f"{d.name}={d.licence}" for d in p.datasets if d.licence not in ALLOWED_HINT]
            print(f"{name:34} {p.repo}@{p.revision[:12]}  weights={p.licence}  "
                  f"review={rev or 'none'}")
        return 0

    p = PRESETS[args.preset]
    root = args.out / p.name / p.version
    root.mkdir(parents=True, exist_ok=True)
    print(f"{p.repo}@{p.revision} → {root}")
    for f in p.files:
        download(p.repo, p.revision, f, root / f.dst)

    manifest = build_manifest(p, root, args.approved_by, args.approval_reason)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {root / 'manifest.json'} ({len(manifest['labels'])} labels, "
          f"opset {manifest['onnx']['opset']}, inputs "
          f"{[t['name'] for t in manifest['onnx']['inputs']]})")

    pending = [d for d in p.datasets if d.licence not in ALLOWED_HINT]
    rc = verify(root)
    if pending and not args.approved_by:
        print("\nLICENCE REVIEW PENDING (artifact usable for development, NOT for bundling):")
        for d in pending:
            print(f"  - {d.name}: {d.licence} ({d.source})")
        print("Re-run with --approved-by 'Name <email>' --approval-reason '...' after review.")
        return 3
    return 0 if rc in (None, 0) else rc


if __name__ == "__main__":
    sys.exit(main())
