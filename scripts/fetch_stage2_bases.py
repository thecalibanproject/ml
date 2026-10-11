#!/usr/bin/env python3
"""Fetch the Stage-2 base encoders at pinned revisions (stdlib only).

Downloads each base model's config, tokenizer and weights from
``https://huggingface.co/<repo>/resolve/<revision>/<file>`` into
``<out>/<repo with / -> -->/<revision>/`` and checks the weight files' SHA-256 against the
values pinned here (the LFS oids at that revision). Sends ``HF_TOKEN`` when set. Prints one
``NAME=PATH:REPO:REVISION:LICENCE`` line per model, the ``--base`` format of
``scripts/router_stage2_eval.py``.

Licences are the model cards' at the pinned revision; both pass the default licence policy.
Neither card covers the licences of the pre-training data: see the README licence table.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BASES = {
    "modernbert": {
        "repo": "answerdotai/ModernBERT-base",
        "revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
        "licence": "apache-2.0",
        "files": {
            "config.json": None,
            "tokenizer.json": None,
            "tokenizer_config.json": None,
            "special_tokens_map.json": None,
            "model.safetensors":
                "340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9",
        },
    },
    # Multilingual ModernBERT-architecture encoder (arXiv 2509.06888). Ships only a pickle
    # checkpoint; transformers loads it with torch.load(weights_only=True).
    "mmbert": {
        "repo": "jhu-clsp/mmBERT-base",
        "revision": "c5955035435e2bf121cde7f3c8863ef52ff35d82",
        "licence": "mit",
        "files": {
            "config.json": None,
            "tokenizer.json":
                "197d4cc5406ee12cc50c8b5511f2393cc32d9db321545979ce041c1199178356",
            "tokenizer_config.json": None,
            "special_tokens_map.json": None,
            "pytorch_model.bin":
                "8ea64ec1ea4eb8fca0fc14b69a2ae571de6bfbc25fd214bb932dd4aba6a3a04e",
        },
    },
}


def fetch(url: str, dest: Path, sha256: str | None) -> None:
    if dest.is_file() and sha256 and _sha(dest) == sha256:
        return
    req = urllib.request.Request(url)
    if token := os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {token}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
    if sha256 and h.hexdigest() != sha256:
        tmp.unlink()
        raise SystemExit(f"{url}: sha256 {h.hexdigest()} != pinned {sha256}")
    tmp.replace(dest)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help=f"Subset of {sorted(BASES)} (default: all).")
    ap.add_argument("--out", type=Path, default=REPO / ".hf_cache")
    args = ap.parse_args()
    unknown = sorted(set(args.names) - set(BASES))
    if unknown:
        ap.error(f"unknown base models {unknown}; choose from {sorted(BASES)}")
    for name in args.names or list(BASES):
        b = BASES[name]
        d = args.out / b["repo"].replace("/", "--") / b["revision"]
        d.mkdir(parents=True, exist_ok=True)
        for fname, sha in b["files"].items():
            url = f"https://huggingface.co/{b['repo']}/resolve/{b['revision']}/{fname}"
            print(f"[{name}] {fname}", file=sys.stderr, flush=True)
            fetch(url, d / fname, sha)
        print(f"{name}={d}:{b['repo']}:{b['revision']}:{b['licence']}")


if __name__ == "__main__":
    main()
