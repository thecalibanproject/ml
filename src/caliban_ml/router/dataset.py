"""Intent exemplar dataset format (YAML).

::

    version: 1
    intents:
      sql_analytics:
        description: Questions answered by querying a tenant datasource.
        utterances:
          - how many invoices were overdue last month
          - top 10 customers by revenue in Q3
      summarization:
        utterances: [...]
    oos:                      # out-of-scope examples: used ONLY to pick the OOS threshold
      - what's the weather like on mars
      - write me a sonnet about my cat

Intent ids follow the route-registry ``route_id`` convention (lower snake case, optionally
dotted for the domain -> action hierarchy, e.g. ``finance.invoice_triage``). OOS is an
explicit label (:data:`OOS_LABEL`) and never one of the intents.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

OOS_LABEL = "__oos__"
INTENT_ID_RE = r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$"


class IntentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = ""
    utterances: list[str] = Field(min_length=1)


class IntentDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    intents: dict[str, IntentSpec] = Field(min_length=1)
    oos: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> IntentDataset:
        import re

        seen: dict[str, str] = {}
        for intent, spec in self.intents.items():
            if not re.match(INTENT_ID_RE, intent):
                raise ValueError(f"invalid intent id {intent!r} (expected {INTENT_ID_RE})")
            for u in spec.utterances:
                key = _norm(u)
                if not key:
                    raise ValueError(f"empty utterance in intent {intent!r}")
                if key in seen and seen[key] != intent:
                    raise ValueError(
                        f"utterance {u!r} appears under both {seen[key]!r} and {intent!r}"
                    )
                seen[key] = intent
        for u in self.oos:
            if _norm(u) in seen:
                raise ValueError(f"OOS example {u!r} is also an in-scope utterance")
        return self

    @classmethod
    def from_yaml(cls, path: Path | str) -> IntentDataset:
        with Path(path).open(encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    @property
    def labels(self) -> list[str]:
        return sorted(self.intents)

    def examples(self, include_oos: bool = True) -> tuple[list[str], list[str]]:
        texts: list[str] = []
        labels: list[str] = []
        for intent in self.labels:
            for u in self.intents[intent].utterances:
                texts.append(u)
                labels.append(intent)
        if include_oos:
            texts.extend(self.oos)
            labels.extend([OOS_LABEL] * len(self.oos))
        return texts, labels

    def split(self, val_fraction: float = 0.3, seed: int = 0
              ) -> tuple[tuple[list[str], list[str]], tuple[list[str], list[str]]]:
        """Stratified split. Every intent keeps >=1 train exemplar; OOS goes to val only
        unless there are >= 4 OOS examples, in which case it is split too (OOS exemplars
        are never used as kNN neighbours, but a held-out OOS set keeps evaluation honest)."""
        rng = np.random.default_rng(seed)
        texts, labels = self.examples(include_oos=True)
        by_label: dict[str, list[int]] = {}
        for i, lab in enumerate(labels):
            by_label.setdefault(lab, []).append(i)
        train_idx: list[int] = []
        val_idx: list[int] = []
        for lab, idx in sorted(by_label.items()):
            idx = list(rng.permutation(idx))
            if lab == OOS_LABEL and len(idx) < 4:
                val_idx.extend(idx)
                continue
            n_val = round(len(idx) * val_fraction)
            n_val = min(n_val, len(idx) - 1) if lab != OOS_LABEL else n_val
            val_idx.extend(idx[:n_val])
            train_idx.extend(idx[n_val:])
        tr = ([texts[i] for i in sorted(train_idx)], [labels[i] for i in sorted(train_idx)])
        va = ([texts[i] for i in sorted(val_idx)], [labels[i] for i in sorted(val_idx)])
        return tr, va


def _norm(s: str) -> str:
    return " ".join(s.lower().split())
