"""GLiNER-family detector (``[train]`` extra) - STUB, not exercised in CI.

Default L1 candidate is GLiNER2-PII (see docs/research/05). Load weights from a local,
licence-checked directory only (``local_files_only=True``): the eval harness must not
download anything. Check each checkpoint's licence: early ``urchade/gliner_*`` v1
checkpoints are CC-BY-NC-4.0 and are NOT shippable; ``nvidia/gliner-pii`` is under the
NVIDIA Open Model License (review required).

TODO(pii/gliner2): GLiNER2 ships its own package/API (schema-based extraction). Add a
``Gliner2Detector`` once the pinned release is chosen; this class covers the
``gliner`` package API (``GLiNER.predict_entities``).
TODO(pii/gliner): sliding 512-token windows with overlap, mirroring the Rust L1 tier.
"""

from __future__ import annotations

from pathlib import Path

from caliban_ml._extras import require
from caliban_ml.pii.dataset import Span

# Natural-language label prompts -> canonical labels.
DEFAULT_LABELS: dict[str, str] = {
    "person": "PERSON",
    "location": "LOCATION",
    "organization": "ORGANIZATION",
    "email address": "EMAIL",
    "phone number": "PHONE",
    "credit card number": "CREDIT_CARD",
    "iban": "IBAN",
    "social security number": "US_SSN",
    "ip address": "IP_ADDRESS",
    "url": "URL",
    "api key": "SECRET",
    "date of birth": "DATE_OF_BIRTH",
}


class GlinerDetector:
    name = "gliner"

    def __init__(self, model_dir: Path | str, labels: dict[str, str] | None = None,
                 threshold: float = 0.4):
        gliner = require("gliner", "train")
        self.model = gliner.GLiNER.from_pretrained(str(model_dir), local_files_only=True)
        self.labels = labels or DEFAULT_LABELS
        self.threshold = threshold

    def detect(self, text: str) -> list[Span]:
        ents = self.model.predict_entities(text, list(self.labels), threshold=self.threshold)
        return [Span(start=e["start"], end=e["end"], label=self.labels.get(e["label"], e["label"]),
                     score=e.get("score")) for e in ents]
