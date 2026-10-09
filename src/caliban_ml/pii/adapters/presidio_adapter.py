"""Microsoft Presidio detector (``[presidio]`` extra) - STUB, not exercised in CI.

Presidio (MIT) is a useful comparison point for L0+NER, but its default NLP engine needs a
spaCy model package; install that wheel from the offline mirror, never via
``spacy download`` at runtime.

TODO(pii/presidio): configure the NLP engine explicitly (model name + languages) instead of
relying on AnalyzerEngine defaults, and add per-language runs.
"""

from __future__ import annotations

from caliban_ml._extras import require
from caliban_ml.pii.dataset import Span

PRESIDIO_TO_CANONICAL: dict[str, str] = {
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "NRP": "ORGANIZATION",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CREDIT_CARD",
    "IBAN_CODE": "IBAN",
    "US_SSN": "US_SSN",
    "IP_ADDRESS": "IP_ADDRESS",
    "URL": "URL",
}


class PresidioDetector:
    name = "presidio"

    def __init__(self, language: str = "en", score_threshold: float = 0.35):
        analyzer = require("presidio_analyzer", "presidio")
        self.engine = analyzer.AnalyzerEngine()
        self.language = language
        self.score_threshold = score_threshold

    def detect(self, text: str) -> list[Span]:
        results = self.engine.analyze(text=text, language=self.language,
                                      score_threshold=self.score_threshold)
        return [Span(start=r.start, end=r.end,
                     label=PRESIDIO_TO_CANONICAL.get(r.entity_type, r.entity_type),
                     score=r.score) for r in results]
