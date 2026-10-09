"""L0 deterministic detector: regex + validators (the Python twin of the Rust L0 tier).

Covers structured identifiers only (email, phone, card+Luhn, IBAN+mod-97, US SSN rules,
IPv4, URL, well-known secret formats). It cannot find names, places or organisations;
that gap is what the L1 NER model (GLiNER2-PII) closes, and this baseline makes the gap
measurable.

Overlap resolution follows the gateway merge rule: prefer the longest span, then the
higher-risk type.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from caliban_ml.pii.dataset import Span

# Higher number = higher risk; used to break ties between equally long overlapping spans.
RISK: dict[str, int] = {
    "SECRET": 9, "CREDIT_CARD": 8, "IBAN": 7, "US_SSN": 7, "EMAIL": 5, "PHONE": 4,
    "IP_ADDRESS": 3, "URL": 2,
}


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_ok(raw: str) -> bool:
    s = raw.replace(" ", "").upper()
    if not 15 <= len(s) <= 34:
        return False
    rearranged = s[4:] + s[:4]
    num = "".join(str(int(c, 36)) for c in rearranged)
    return int(num) % 97 == 1


def ssn_ok(area: str, group: str, serial: str) -> bool:
    return area not in ("000", "666") and not area.startswith("9") \
        and group != "00" and serial != "0000"


def ipv4_ok(s: str) -> bool:
    parts = s.split(".")
    return len(parts) == 4 and all(p.isdigit() and int(p) <= 255 and (p == "0" or
                                   not p.startswith("0")) for p in parts)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def phone_ok(s: str) -> bool:
    """7-15 digits (E.164 bound); short local forms need a '+' or '(...)' marker, which
    keeps invoice numbers and ISO dates out."""
    n = len(_digits(s))
    if not 7 <= n <= 15 or _ISO_DATE.match(s):
        return False
    return n >= 10 or s.startswith(("+", "("))


Validator = Callable[[re.Match[str]], bool]

PATTERNS: list[tuple[str, re.Pattern[str], Validator | None]] = [
    ("SECRET", re.compile(
        r"(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36}"
        r"|xox[abprs]-[A-Za-z0-9-]{10,})(?![A-Za-z0-9_-])"), None),
    ("EMAIL", re.compile(
        r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?!\w)"),
     None),
    ("URL", re.compile(r"\bhttps?://[^\s<>\"']+[^\s<>\"'.,;:!?)\]]"), None),
    ("CREDIT_CARD", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
     lambda m: 13 <= len(_digits(m.group())) <= 19 and luhn_ok(_digits(m.group()))),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b"),
     lambda m: iban_ok(m.group())),
    ("US_SSN", re.compile(r"(?<!\d)(\d{3})-(\d{2})-(\d{4})(?!\d)"),
     lambda m: ssn_ok(m.group(1), m.group(2), m.group(3))),
    ("IP_ADDRESS", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
     lambda m: ipv4_ok(m.group())),
    ("PHONE", re.compile(
        r"(?<![\w+])(?:\+\d{1,3}[ .-]?)?(?:\(\d{1,4}\)[ .-]?)?\d{2,4}(?:[ .-]\d{2,4}){1,3}(?!\w)"),
     lambda m: phone_ok(m.group())),
]


class RegexBaselineDetector:
    name = "regex"

    def __init__(self, labels: set[str] | None = None):
        self.labels = labels

    def detect(self, text: str) -> list[Span]:
        cands: list[Span] = []
        for label, pat, validate in PATTERNS:
            if self.labels is not None and label not in self.labels:
                continue
            for m in pat.finditer(text):
                if validate is None or validate(m):
                    cands.append(Span(start=m.start(), end=m.end(), label=label))
        return resolve_overlaps(cands)


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Greedy: longest first, then higher risk, then leftmost."""
    order = sorted(spans, key=lambda s: (-(s.end - s.start), -RISK.get(s.label, 0), s.start))
    kept: list[Span] = []
    for s in order:
        if all(s.end <= k.start or s.start >= k.end for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda s: (s.start, s.end))
