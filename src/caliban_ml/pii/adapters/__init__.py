"""PII detector adapters. Each exposes ``name`` and ``detect(text) -> list[Span]``."""

from __future__ import annotations

from typing import Any, Protocol

from caliban_ml.pii.adapters.regex_baseline import RegexBaselineDetector
from caliban_ml.pii.dataset import Span


class PiiDetector(Protocol):
    name: str

    def detect(self, text: str) -> list[Span]: ...


def get_detector(name: str, **kwargs: Any) -> PiiDetector:
    if name == "regex":
        return RegexBaselineDetector(**kwargs)
    if name == "gliner":
        from caliban_ml.pii.adapters.gliner_adapter import GlinerDetector

        return GlinerDetector(**kwargs)
    if name == "presidio":
        from caliban_ml.pii.adapters.presidio_adapter import PresidioDetector

        return PresidioDetector(**kwargs)
    raise ValueError(f"unknown detector {name!r} (expected regex | gliner | presidio)")


__all__ = ["PiiDetector", "RegexBaselineDetector", "get_detector"]
