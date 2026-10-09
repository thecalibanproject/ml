"""Licence policy for base models and training data.

Caliban ships on-prem, commercially, inside customer networks. Every base model and
every training dataset baked into a shipped artifact must allow commercial use and
redistribution. The policy here is deliberately conservative:

* ``allowed``         - permissive / commercial-OK licences (Apache-2.0, MIT, BSD, ...).
* ``review_required`` - commercial use is possible but terms need a human read
                        (copyleft, RAIL use restrictions, vendor model licences,
                        Llama-style MAU caps) or the licence is unknown.
* ``denied``          - non-commercial / research-only (CC-BY-NC-*, "research only", ...).

Anything that is not ``allowed`` is rejected by default. A manifest can carry an explicit
:class:`~caliban_ml.artifacts.manifest.LicenceOverride` (who approved it and why), which
turns ``review_required`` into an accepted decision. ``denied`` licences additionally
require ``allow_noncommercial=True`` at validation time, which the default bundle
build never sets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class LicenceVerdict(StrEnum):
    ALLOWED = "allowed"
    REVIEW_REQUIRED = "review_required"
    DENIED = "denied"


# Lower-case SPDX ids (or well-known names) that are fine for commercial on-prem shipping.
ALLOWED_LICENCES: frozenset[str] = frozenset(
    {
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
        "isc",
        "cc0-1.0",
        "unlicense",
        "cc-by-4.0",
        "cc-by-3.0",
        "zlib",
        "bsl-1.0",  # Boost Software License (NOT the Business Source License)
        "psf-2.0",
        # First-party data written and owned by Caliban (e.g. the global probe set).
        "caliban-internal",
    }
)

# Commercial use possible, but obligations or use restrictions need a human decision.
REVIEW_LICENCES: frozenset[str] = frozenset(
    {
        "mpl-2.0",
        "lgpl-2.1",
        "lgpl-3.0",
        "gpl-2.0",
        "gpl-3.0",
        "agpl-3.0",
        "cc-by-sa-3.0",
        "cc-by-sa-4.0",
        "openrail",
        "openrail-m",
        "bigscience-openrail-m",
        "creativeml-openrail-m",
        "llama2",
        "llama3",
        "llama3.1",
        "llama3.2",
        "llama3.3",
        "gemma",
        "nvidia-open-model-license",
        "other",
        "unknown",
    }
)

# Substrings that mark a licence as non-commercial / research-only.
_DENY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|[-_ ])nc([-_ ]|$)"),  # cc-by-nc-4.0, cc-by-nc-sa-4.0, cc-by-nc-nd-4.0
    re.compile(r"non[-_ ]?commercial"),
    re.compile(r"research[-_ ]?only"),
    re.compile(r"academic[-_ ]?only"),
    re.compile(r"evaluation[-_ ]?only"),
)


@dataclass(frozen=True)
class LicenceDecision:
    licence: str
    verdict: LicenceVerdict
    reason: str

    @property
    def ok(self) -> bool:
        return self.verdict is LicenceVerdict.ALLOWED


def normalize_licence(licence: str) -> str:
    return licence.strip().lower().replace(" ", "-")


def classify_licence(licence: str) -> LicenceDecision:
    """Classify a licence identifier (SPDX id preferred, as on the HF model card)."""
    norm = normalize_licence(licence)
    if not norm:
        return LicenceDecision(licence, LicenceVerdict.REVIEW_REQUIRED, "empty licence")
    for pat in _DENY_PATTERNS:
        if pat.search(norm):
            return LicenceDecision(
                licence,
                LicenceVerdict.DENIED,
                "non-commercial / research-only licences cannot ship in the on-prem bundle",
            )
    if norm in ALLOWED_LICENCES:
        return LicenceDecision(licence, LicenceVerdict.ALLOWED, "permissive licence")
    if norm in REVIEW_LICENCES:
        return LicenceDecision(
            licence,
            LicenceVerdict.REVIEW_REQUIRED,
            "commercial use possible but terms need review (copyleft, use restrictions, "
            "vendor licence or unknown)",
        )
    return LicenceDecision(
        licence, LicenceVerdict.REVIEW_REQUIRED, "unrecognised licence; review required"
    )
