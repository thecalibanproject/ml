"""Span-level precision / recall / F1 per entity type.

Match modes (all one-to-one: a predicted span can satisfy at most one gold span):

* ``exact``     - same ``start``, ``end`` and ``label``.
* ``overlap``   - same ``label`` and the character ranges intersect (greedy by overlap size).
* ``redaction`` - ranges intersect, label ignored. This is the privacy-relevant number:
                  a phone number tagged as ``US_SSN`` is still redacted. TP and FN are
                  counted under the gold label, FP under the predicted label.

Recall matters more than precision for Caliban (a missed entity leaks; an extra one only
costs some utility), so reports lead with recall.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from caliban_ml.pii.dataset import Span

MatchMode = Literal["exact", "overlap", "redaction"]


@dataclass
class LabelStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        d = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / d if d else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    def as_dict(self) -> dict[str, float]:
        return {"precision": self.precision, "recall": self.recall, "f1": self.f1,
                "tp": self.tp, "fp": self.fp, "fn": self.fn, "support": self.support}


@dataclass
class PiiReport:
    mode: MatchMode
    n_docs: int
    per_label: dict[str, LabelStats] = field(default_factory=dict)

    @property
    def micro(self) -> LabelStats:
        tot = LabelStats()
        for s in self.per_label.values():
            tot.tp += s.tp
            tot.fp += s.fp
            tot.fn += s.fn
        return tot

    @property
    def macro_f1(self) -> float:
        vals = [s.f1 for s in self.per_label.values() if s.tp + s.fp + s.fn]
        return sum(vals) / len(vals) if vals else 0.0

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "n_docs": self.n_docs,
            "micro": self.micro.as_dict(),
            "macro_f1": self.macro_f1,
            "per_label": {k: v.as_dict() for k, v in sorted(self.per_label.items())},
        }


def _overlap(a: Span, b: Span) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def match_spans(gold: Sequence[Span], pred: Sequence[Span], mode: MatchMode
                ) -> list[tuple[int, int]]:
    """Return one-to-one (gold_idx, pred_idx) matches."""
    cands: list[tuple[int, int, int]] = []  # (-overlap, gi, pi) for deterministic greedy
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if mode == "exact":
                if (g.start, g.end, g.label) == (p.start, p.end, p.label):
                    cands.append((-(g.end - g.start), gi, pi))
            else:
                ov = _overlap(g, p)
                if ov > 0 and (mode == "redaction" or g.label == p.label):
                    cands.append((-ov, gi, pi))
    cands.sort()
    used_g: set[int] = set()
    used_p: set[int] = set()
    out = []
    for _, gi, pi in cands:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            out.append((gi, pi))
    return out


def evaluate_spans(gold_docs: Sequence[Sequence[Span]], pred_docs: Sequence[Sequence[Span]],
                   mode: MatchMode = "exact") -> PiiReport:
    if len(gold_docs) != len(pred_docs):
        raise ValueError("gold and prediction document counts differ")
    rep = PiiReport(mode=mode, n_docs=len(gold_docs))

    def stats(label: str) -> LabelStats:
        return rep.per_label.setdefault(label, LabelStats())

    for gold, pred in zip(gold_docs, pred_docs, strict=True):
        matches = match_spans(gold, pred, mode)
        mg = {gi for gi, _ in matches}
        mp = {pi for _, pi in matches}
        for gi, _ in matches:
            stats(gold[gi].label).tp += 1
        for gi, g in enumerate(gold):
            if gi not in mg:
                stats(g.label).fn += 1
        for pi, p in enumerate(pred):
            if pi not in mp:
                stats(p.label).fp += 1
    return rep
