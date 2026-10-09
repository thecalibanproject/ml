"""Signing hook for artifact manifests (STUB: thin wrappers over external CLIs).

What is signed: only ``manifest.json``. It lists every payload file with its SHA-256,
so a valid signature over the manifest plus a passing ``caliban-ml manifest verify``
(or the equivalent check in Rust) covers the whole artifact.

Two schemes are anticipated; both must work fully offline / air-gapped:

* **minisign** (default for on-prem bundles; tiny, no transparency log)::

      minisign -S -s caliban-artifacts.key -m manifest.json -x manifest.json.minisig
      minisign -V -p caliban-artifacts.pub -m manifest.json -x manifest.json.minisig

* **cosign** with a key pair and the transparency log disabled (air-gap)::

      cosign sign-blob --yes --key cosign.key --tlog-upload=false \
          --bundle manifest.json.sigstore.json manifest.json
      cosign verify-blob --key cosign.pub --insecure-ignore-tlog=true \
          --bundle manifest.json.sigstore.json manifest.json

Keys live in the release pipeline's secret store, never in this repo (``*.key`` is
git-ignored). The public key ships in the on-prem bundle and is pinned in
``caliban.toml``; the Rust loader verifies before ``ArcSwap``-ing a new model in.

TODO(signing): the command lines above are not exercised in CI yet. Add an integration
test on the release runner that signs a fixture artifact and verifies it with both the
CLI and the Rust loader (minisign verification in Rust: the ``minisign-verify`` crate).
"""

from __future__ import annotations

import shutil
import subprocess
from enum import StrEnum
from pathlib import Path

from caliban_ml.artifacts.manifest import MANIFEST_FILENAME


class SignatureScheme(StrEnum):
    MINISIGN = "minisign"
    COSIGN = "cosign"


SIGNATURE_FILES: dict[SignatureScheme, str] = {
    SignatureScheme.MINISIGN: MANIFEST_FILENAME + ".minisig",
    SignatureScheme.COSIGN: MANIFEST_FILENAME + ".sigstore.json",
}


class SigningUnavailableError(RuntimeError):
    pass


def _sign_cmd(scheme: SignatureScheme, root: Path, key: Path) -> list[str]:
    m, sig = root / MANIFEST_FILENAME, root / SIGNATURE_FILES[scheme]
    if scheme is SignatureScheme.MINISIGN:
        return ["minisign", "-S", "-s", str(key), "-m", str(m), "-x", str(sig)]
    return ["cosign", "sign-blob", "--yes", "--key", str(key), "--tlog-upload=false",
            "--bundle", str(sig), str(m)]


def _verify_cmd(scheme: SignatureScheme, root: Path, pubkey: Path) -> list[str]:
    m, sig = root / MANIFEST_FILENAME, root / SIGNATURE_FILES[scheme]
    if scheme is SignatureScheme.MINISIGN:
        return ["minisign", "-V", "-p", str(pubkey), "-m", str(m), "-x", str(sig)]
    return ["cosign", "verify-blob", "--key", str(pubkey), "--insecure-ignore-tlog=true",
            "--bundle", str(sig), str(m)]


def _run(cmd: list[str]) -> None:
    if shutil.which(cmd[0]) is None:
        raise SigningUnavailableError(f"'{cmd[0]}' not found on PATH")
    subprocess.run(cmd, check=True)


def sign_manifest(root: Path, scheme: SignatureScheme, key: Path) -> Path:
    """Write a detached signature next to ``manifest.json``. Returns its path."""
    _run(_sign_cmd(scheme, Path(root), Path(key)))
    return Path(root) / SIGNATURE_FILES[scheme]


def verify_signature(root: Path, scheme: SignatureScheme, pubkey: Path) -> None:
    """Raise if the detached signature over ``manifest.json`` does not verify."""
    root = Path(root)
    if not (root / SIGNATURE_FILES[scheme]).is_file():
        raise FileNotFoundError(f"no {SIGNATURE_FILES[scheme]} in {root}")
    _run(_verify_cmd(scheme, root, Path(pubkey)))
