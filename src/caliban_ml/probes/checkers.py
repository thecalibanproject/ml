"""Deterministic probe checkers. All return a score in [0, 1] and a pass flag."""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from typing import Any

from caliban_ml.probes.probeset import (
    ContainsCheck,
    ExactCheck,
    JsonCheck,
    RegexCheck,
)


@dataclass
class CheckResult:
    score: float
    passed: bool
    detail: str = ""


_PUNCT = str.maketrans("", "", string.punctuation)
_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```\s*$", re.DOTALL)


def _norm(s: str, *, case_sensitive: bool, strip_punctuation: bool = False) -> str:
    s = " ".join(s.split())
    if strip_punctuation:
        s = s.translate(_PUNCT).strip()
    return s if case_sensitive else s.lower()


def check_exact(c: ExactCheck, output: str) -> CheckResult:
    ok = _norm(output, case_sensitive=c.case_sensitive, strip_punctuation=c.strip_punctuation) \
        == _norm(c.value, case_sensitive=c.case_sensitive, strip_punctuation=c.strip_punctuation)
    return CheckResult(float(ok), ok)


def check_contains(c: ContainsCheck, output: str) -> CheckResult:
    hay = _norm(output, case_sensitive=c.case_sensitive)
    found = [v for v in c.values if _norm(v, case_sensitive=c.case_sensitive) in hay]
    if c.mode == "any":
        ok = bool(found)
        return CheckResult(float(ok), ok, f"found={found}")
    score = len(found) / len(c.values)
    return CheckResult(score, score == 1.0, f"found {len(found)}/{len(c.values)}")


def check_regex(c: RegexCheck, output: str) -> CheckResult:
    pat = re.compile(c.pattern)
    ok = bool(pat.fullmatch(output.strip()) if c.fullmatch else pat.search(output))
    return CheckResult(float(ok), ok)


def extract_json(output: str) -> Any:
    """Parse JSON from a model reply, tolerating ```json fences and surrounding prose."""
    text = output.strip()
    m = _FENCE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back to the first {...} or [...] block
    for open_, close in (("{", "}"), ("[", "]")):
        i, j = text.find(open_), text.rfind(close)
        if 0 <= i < j:
            try:
                return json.loads(text[i : j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON value found in output")


def check_json(c: JsonCheck, output: str) -> CheckResult:
    try:
        obj = extract_json(output)
    except ValueError as exc:
        return CheckResult(0.0, False, str(exc))
    if not c.required_keys and not c.equals:
        return CheckResult(1.0, True, "valid JSON")
    if not isinstance(obj, dict):
        return CheckResult(0.0, False, "expected a JSON object")
    total = len(c.required_keys) + len(c.equals)
    hits = sum(1 for k in c.required_keys if k in obj)
    hits += sum(1 for k, v in c.equals.items() if k in obj and obj[k] == v)
    missing = [k for k in c.required_keys if k not in obj]
    wrong = [k for k, v in c.equals.items() if obj.get(k, object()) != v]
    return CheckResult(hits / total, hits == total, f"missing={missing} wrong={wrong}")


def run_deterministic_check(check: Any, output: str) -> CheckResult:
    if isinstance(check, ExactCheck):
        return check_exact(check, output)
    if isinstance(check, ContainsCheck):
        return check_contains(check, output)
    if isinstance(check, RegexCheck):
        return check_regex(check, output)
    if isinstance(check, JsonCheck):
        return check_json(check, output)
    raise TypeError(f"not a deterministic check: {type(check).__name__}")
