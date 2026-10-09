"""Probe set format (YAML, or JSONL with one probe per line).

::

    version: 1
    name: caliban-sample
    clusters:
      - id: sql
        description: Text-to-SQL and analytics questions
    probes:
      - id: sql-001
        cluster: sql
        prompt: "Write a SQL query that ..."          # or `messages: [{role, content}]`
        check: {type: regex, pattern: "(?i)select\\s+count"}
        max_tokens: 256

Check types: ``exact``, ``contains``, ``regex``, ``json`` (deterministic, in
:mod:`caliban_ml.probes.checkers`) and ``judge`` (an LLM judge hook, see
:mod:`caliban_ml.probes.runner`). Every probe scores in ``[0, 1]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExactCheck(_M):
    type: Literal["exact"] = "exact"
    value: str
    case_sensitive: bool = False
    strip_punctuation: bool = True


class ContainsCheck(_M):
    type: Literal["contains"] = "contains"
    values: list[str] = Field(min_length=1)
    mode: Literal["any", "all"] = "all"
    case_sensitive: bool = False


class RegexCheck(_M):
    type: Literal["regex"] = "regex"
    pattern: str
    fullmatch: bool = False


class JsonCheck(_M):
    type: Literal["json"] = "json"
    required_keys: list[str] = Field(default_factory=list)
    equals: dict[str, Any] = Field(default_factory=dict, description="Subset match on values.")


class JudgeCheck(_M):
    type: Literal["judge"] = "judge"
    rubric: str = Field(min_length=1)
    reference: str | None = None
    pass_threshold: float = Field(default=0.5, ge=0, le=1)


Check = Annotated[
    ExactCheck | ContainsCheck | RegexCheck | JsonCheck | JudgeCheck,
    Field(discriminator="type"),
]


class Message(_M):
    role: Literal["system", "user", "assistant"]
    content: str


class Probe(_M):
    id: str = Field(min_length=1)
    cluster: str = Field(min_length=1)
    prompt: str | None = None
    messages: list[Message] | None = None
    check: Check
    max_tokens: int = Field(default=512, ge=1)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_input(self) -> Probe:
        if (self.prompt is None) == (self.messages is None):
            raise ValueError(f"probe {self.id!r}: set exactly one of prompt / messages")
        return self

    def chat_messages(self) -> list[dict[str, str]]:
        if self.messages is not None:
            return [m.model_dump() for m in self.messages]
        return [{"role": "user", "content": self.prompt or ""}]


class ProbeCluster(_M):
    id: str = Field(min_length=1)
    description: str = ""


class ProbeSet(_M):
    version: int = 1
    name: str = Field(min_length=1)
    clusters: list[ProbeCluster] = Field(default_factory=list)
    probes: list[Probe] = Field(min_length=1)

    @model_validator(mode="after")
    def _refs(self) -> ProbeSet:
        ids = [p.id for p in self.probes]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate probe ids: {dupes}")
        if self.clusters:
            known = {c.id for c in self.clusters}
            unknown = sorted({p.cluster for p in self.probes} - known)
            if unknown:
                raise ValueError(f"probes reference undeclared clusters: {unknown}")
        else:
            self.clusters = [ProbeCluster(id=c) for c in sorted({p.cluster for p in self.probes})]
        return self


def load_probe_set(path: Path | str) -> ProbeSet:
    path = Path(path)
    if path.suffix == ".jsonl":
        probes = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        return ProbeSet.model_validate({"name": path.stem, "probes": probes})
    with path.open(encoding="utf-8") as f:
        return ProbeSet.model_validate(yaml.safe_load(f))
