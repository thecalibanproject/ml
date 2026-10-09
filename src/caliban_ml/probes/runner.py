"""BYO-model probe runner against any OpenAI-compatible ``/chat/completions`` endpoint.

Point it at the Caliban data plane (``http://caliban:8080/v1``, with a tenant key and an
explicit model id so the router is bypassed) or straight at a local vLLM / llama.cpp
server. The runner never talks to anything except the configured ``base_url`` (and the
judge endpoint, if one is configured), so it works on an air-gapped network.

Results are JSONL, one :class:`ProbeResult` per (model, probe); they feed
``caliban-ml router profile``. Model outputs are stored by default for debugging; pass
``store_output=False`` when probing with sensitive tenant probe sets.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import BaseModel

from caliban_ml.probes.checkers import CheckResult, extract_json, run_deterministic_check
from caliban_ml.probes.probeset import JudgeCheck, Probe, ProbeSet

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ProbeResult(BaseModel):
    probe_set: str
    model: str
    probe_id: str
    cluster: str
    check_type: str
    score: float | None = None
    passed: bool | None = None
    detail: str = ""
    error: str | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    output: str | None = None


@dataclass
class EndpointConfig:
    base_url: str  # e.g. http://localhost:8080/v1 (Caliban) or http://vllm:8000/v1
    model: str
    api_key: str | None = None
    timeout_s: float = 120.0
    max_retries: int = 2
    temperature: float = 0.0
    extra_body: dict[str, Any] = field(default_factory=dict)


class ChatError(RuntimeError):
    pass


class ChatClient:
    """Minimal OpenAI-compatible chat client (sync, httpx)."""

    def __init__(self, endpoint: EndpointConfig, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep, backoff_s: float = 0.5):
        self.endpoint = endpoint
        self._client = client or httpx.Client(timeout=endpoint.timeout_s)
        self._sleep = sleep
        self._backoff = backoff_s

    def chat(self, messages: list[dict[str, str]], max_tokens: int,
             temperature: float | None = None) -> tuple[str, dict[str, Any], float]:
        ep = self.endpoint
        body: dict[str, Any] = {
            "model": ep.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": ep.temperature if temperature is None else temperature,
            "stream": False,
            **ep.extra_body,
        }
        headers = {"Content-Type": "application/json"}
        if ep.api_key:
            headers["Authorization"] = f"Bearer {ep.api_key}"
        url = ep.base_url.rstrip("/") + "/chat/completions"
        last: str = ""
        for attempt in range(ep.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client.post(url, json=body, headers=headers,
                                         timeout=ep.timeout_s)
            except httpx.TransportError as exc:
                last = f"transport error: {exc!r}"
            else:
                latency = (time.perf_counter() - t0) * 1000.0
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        content = data["choices"][0]["message"].get("content") or ""
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        raise ChatError(f"malformed response: {exc!r}") from exc
                    return content, data.get("usage") or {}, latency
                last = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in RETRY_STATUS:
                    raise ChatError(last)
            if attempt < ep.max_retries:
                self._sleep(self._backoff * (2**attempt))
        raise ChatError(f"gave up after {ep.max_retries + 1} attempts: {last}")


class Judge(Protocol):
    def __call__(self, probe: Probe, check: JudgeCheck, output: str) -> CheckResult: ...


JUDGE_SYSTEM = (
    "You are a strict grader. Grade the RESPONSE to the PROMPT against the RUBRIC"
    " (and the REFERENCE, if given). Reply with JSON only: "
    '{"score": <integer 0-10>, "reason": "<one sentence>"}'
)


class LLMJudge:
    """Judge hook backed by an OpenAI-compatible endpoint (a *local* judge model on-prem).

    TODO(judge): per-cluster judge calibration against a small human-labelled set, and
    position/verbosity-bias controls, before judge scores are mixed with checker scores.
    """

    def __init__(self, client: ChatClient):
        self.client = client

    def __call__(self, probe: Probe, check: JudgeCheck, output: str) -> CheckResult:
        prompt = "\n".join(m["content"] for m in probe.chat_messages() if m["role"] == "user")
        user = f"PROMPT:\n{prompt}\n\nRUBRIC:\n{check.rubric}\n\n"
        if check.reference:
            user += f"REFERENCE:\n{check.reference}\n\n"
        user += f"RESPONSE:\n{output}"
        text, _, _ = self.client.chat(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
            max_tokens=128, temperature=0.0,
        )
        try:
            verdict = extract_json(text)
            score = max(0.0, min(10.0, float(verdict["score"]))) / 10.0
        except (ValueError, KeyError, TypeError) as exc:
            raise ChatError(f"unparseable judge verdict: {text[:200]!r}") from exc
        return CheckResult(score, score >= check.pass_threshold, str(verdict.get("reason", "")))


class ProbeRunner:
    def __init__(self, client: ChatClient, *, judge: Judge | None = None,
                 store_output: bool = True):
        self.client = client
        self.judge = judge
        self.store_output = store_output

    def run_probe(self, probe: Probe, probe_set: str) -> ProbeResult:
        res = ProbeResult(probe_set=probe_set, model=self.client.endpoint.model,
                          probe_id=probe.id, cluster=probe.cluster, check_type=probe.check.type)
        try:
            output, usage, latency = self.client.chat(probe.chat_messages(), probe.max_tokens)
        except ChatError as exc:
            res.error = str(exc)
            return res
        res.latency_ms = round(latency, 2)
        res.prompt_tokens = usage.get("prompt_tokens")
        res.completion_tokens = usage.get("completion_tokens")
        if self.store_output:
            res.output = output
        try:
            if isinstance(probe.check, JudgeCheck):
                if self.judge is None:
                    res.error = "judge check but no judge configured"
                    return res
                cr = self.judge(probe, probe.check, output)
            else:
                cr = run_deterministic_check(probe.check, output)
        except ChatError as exc:
            res.error = f"judge: {exc}"
            return res
        res.score, res.passed, res.detail = cr.score, cr.passed, cr.detail
        return res

    def run(self, probe_set: ProbeSet, concurrency: int = 1) -> list[ProbeResult]:
        if concurrency <= 1:
            return [self.run_probe(p, probe_set.name) for p in probe_set.probes]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(lambda p: self.run_probe(p, probe_set.name), probe_set.probes))


def write_results(results: list[ProbeResult], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.model_dump(exclude_none=True), ensure_ascii=False) + "\n")


def summarize(results: list[ProbeResult]) -> dict[str, dict[str, float]]:
    """Per-cluster mean score / pass rate / error count (for CLI output)."""
    out: dict[str, dict[str, float]] = {}
    for r in results:
        s = out.setdefault(r.cluster, {"n": 0, "errors": 0, "score_sum": 0.0, "passed": 0})
        s["n"] += 1
        if r.error is not None:
            s["errors"] += 1
            continue
        s["score_sum"] += r.score or 0.0
        s["passed"] += int(bool(r.passed))
    for s in out.values():
        ok = s["n"] - s["errors"]
        s["mean_score"] = s.pop("score_sum") / ok if ok else 0.0
        s["pass_rate"] = s.pop("passed") / ok if ok else 0.0
    return out
