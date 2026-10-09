"""PII span dataset format: JSONL, one document per line::

    {"id": "d1", "text": "Mail jane@example.com",
     "spans": [{"start": 5, "end": 21, "label": "EMAIL"}]}

(shown wrapped; in the file each document is a single line).

Offsets are Python ``str`` indices (Unicode code points), end-exclusive. Note that the
Rust side works in UTF-8 byte offsets: convert at the boundary, never in the data.

Canonical labels (detectors map their native types onto these): PERSON, LOCATION,
ORGANIZATION, EMAIL, PHONE, CREDIT_CARD, IBAN, US_SSN, IP_ADDRESS, URL, SECRET,
DATE_OF_BIRTH.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

CANONICAL_LABELS = (
    "PERSON", "LOCATION", "ORGANIZATION", "EMAIL", "PHONE", "CREDIT_CARD", "IBAN",
    "US_SSN", "IP_ADDRESS", "URL", "SECRET", "DATE_OF_BIRTH",
)


class Span(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    start: int = Field(ge=0)
    end: int
    label: str = Field(min_length=1)
    score: float | None = None

    @model_validator(mode="after")
    def _order(self) -> Span:
        if self.end <= self.start:
            raise ValueError(f"span end ({self.end}) must be > start ({self.start})")
        return self


class PiiExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | None = None
    text: str
    spans: list[Span] = Field(default_factory=list)
    lang: str | None = None

    @model_validator(mode="after")
    def _bounds(self) -> PiiExample:
        for s in self.spans:
            if s.end > len(self.text):
                raise ValueError(f"span {s.start}:{s.end} is beyond text length {len(self.text)}")
        return self


def load_pii_dataset(path: Path | str) -> list[PiiExample]:
    out: list[PiiExample] = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                out.append(PiiExample.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{n}: {exc}") from exc
    return out
