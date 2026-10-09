"""PII detector evaluation harness (span-level P/R/F1 per entity type)."""

from __future__ import annotations

from collections.abc import Sequence

from caliban_ml.pii.adapters import PiiDetector, RegexBaselineDetector, get_detector
from caliban_ml.pii.dataset import CANONICAL_LABELS, PiiExample, Span, load_pii_dataset
from caliban_ml.pii.metrics import LabelStats, MatchMode, PiiReport, evaluate_spans, match_spans


def evaluate_detector(detector: PiiDetector, examples: Sequence[PiiExample],
                      mode: MatchMode = "exact") -> PiiReport:
    preds = [detector.detect(ex.text) for ex in examples]
    return evaluate_spans([ex.spans for ex in examples], preds, mode)


__all__ = [
    "CANONICAL_LABELS",
    "LabelStats",
    "MatchMode",
    "PiiDetector",
    "PiiExample",
    "PiiReport",
    "RegexBaselineDetector",
    "Span",
    "evaluate_detector",
    "evaluate_spans",
    "get_detector",
    "load_pii_dataset",
    "match_spans",
]
