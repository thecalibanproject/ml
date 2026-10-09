from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from caliban_ml.probes import (
    ChatClient,
    EndpointConfig,
    LLMJudge,
    ProbeRunner,
    ProbeSet,
    load_probe_set,
    run_deterministic_check,
    write_results,
)
from caliban_ml.probes.probeset import ContainsCheck, ExactCheck, JsonCheck, RegexCheck
from caliban_ml.router import compute_profile, load_probe_scores
from conftest import DATA

# Correct answers for the sample probe set, keyed by a substring of the prompt.
GOOD_ANSWERS = {
    "counts invoices": "SELECT COUNT(*) FROM invoices WHERE status = 'overdue';",
    "3 customers": "SELECT customer, SUM(total) AS t FROM orders GROUP BY customer "
                   "ORDER BY t DESC LIMIT 3;",
    "Acme Widgets": '```json\n{"vendor": "Acme Widgets", "amount": 1250.5, "currency": "EUR"}\n```',
    "support ticket": '{"priority": "high"}',
    "board meeting": "The Q3 board meeting moved to October 11; agenda unchanged.",
    "Churn rose": "Churn rose from 2.1% to 3.4%, driven by SMB after the price change.",
    "17 * 23": "391",
    "40 USD": "408",
    "is_palindrome": "def is_palindrome(s: str) -> bool:\n    ...",
    "clamp01": "fn clamp01(x: f64) -> f64 { x.clamp(0.0, 1.0) }",
}


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    })


def make_client(handler, model="local/qwen", **kw) -> ChatClient:
    transport = httpx.MockTransport(handler)
    ep = EndpointConfig(base_url="http://caliban.test:8080/v1", model=model, api_key="cal_test",
                        **kw)
    return ChatClient(ep, client=httpx.Client(transport=transport), sleep=lambda s: None)


# ------------------------------------------------------------------ checkers


@pytest.mark.parametrize("check,output,score,passed", [
    (ExactCheck(value="391"), " 391. ", 1.0, True),
    (ExactCheck(value="391"), "The answer is 391", 0.0, False),
    (ExactCheck(value="Paris", case_sensitive=True), "paris", 0.0, False),
    (ContainsCheck(values=["GROUP BY", "LIMIT 3"]), "select ... group by x limit 3", 1.0, True),
    (ContainsCheck(values=["a1", "b2", "c3", "d4"]), "a1 and b2", 0.5, False),
    (ContainsCheck(values=["x", "y"], mode="any"), "only y", 1.0, True),
    (RegexCheck(pattern=r"(?i)select\s+count"), "SELECT  COUNT(*)", 1.0, True),
    (RegexCheck(pattern=r"\d+", fullmatch=True), " 42 ", 1.0, True),
    (RegexCheck(pattern=r"\d+", fullmatch=True), "42 apples", 0.0, False),
    (JsonCheck(required_keys=["a"], equals={"b": 2}), 'Sure! {"a": 1, "b": 2} done', 1.0, True),
    (JsonCheck(required_keys=["a", "c"], equals={"b": 3}), '{"a": 1, "b": 2}', 1 / 3, False),
    (JsonCheck(), "not json at all", 0.0, False),
    (JsonCheck(required_keys=["a"]), "[1, 2]", 0.0, False),
])
def test_checkers(check, output, score, passed):
    r = run_deterministic_check(check, output)
    assert r.score == pytest.approx(score) and r.passed is passed


# ------------------------------------------------------------------ probe set


def test_sample_probe_set_shape():
    ps = load_probe_set(DATA / "probes" / "sample.yaml")
    assert len(ps.probes) == 10
    assert len(ps.clusters) == 5
    per_cluster = {c.id: sum(p.cluster == c.id for p in ps.probes) for c in ps.clusters}
    assert set(per_cluster.values()) == {2}
    assert {p.check.type for p in ps.probes} == {"exact", "contains", "regex", "json", "judge"}


def test_probe_set_validation():
    base = {"name": "x", "probes": [{"id": "p", "cluster": "c", "prompt": "hi",
                                     "check": {"type": "exact", "value": "x"}}]}
    assert ProbeSet.model_validate(base).clusters[0].id == "c"  # inferred
    with pytest.raises(ValueError, match="undeclared"):
        ProbeSet.model_validate({**base, "clusters": [{"id": "other"}]})
    bad = json.loads(json.dumps(base))
    bad["probes"][0]["messages"] = [{"role": "user", "content": "hi"}]
    with pytest.raises(ValueError, match="exactly one"):
        ProbeSet.model_validate(bad)


# ------------------------------------------------------------------ runner


def answer_for(body: dict) -> str:
    text = " ".join(m["content"] for m in body["messages"])
    return next(a for k, a in GOOD_ANSWERS.items() if k in text)


class FakeJudge:
    def __init__(self):
        self.calls = 0

    def __call__(self, probe, check, output):
        from caliban_ml.probes import CheckResult

        self.calls += 1
        return CheckResult(0.8, True, "fake")


def test_runner_scores_sample_set_with_mock_transport(tmp_path: Path):
    ps = load_probe_set(DATA / "probes" / "sample.yaml")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer cal_test"
        body = json.loads(request.content)
        assert body["model"] == "local/qwen" and body["temperature"] == 0.0
        assert body["stream"] is False and body["seed"] == 7
        return chat_response(answer_for(body))

    judge = FakeJudge()
    runner = ProbeRunner(make_client(handler, extra_body={"seed": 7}), judge=judge)
    results = runner.run(ps)
    assert len(results) == 10 and len(seen) == 10
    assert all(r.error is None for r in results), [r.error for r in results]
    by_id = {r.probe_id: r for r in results}
    assert by_id["summarization-002"].score == pytest.approx(0.8) and judge.calls == 1
    assert all(r.score == 1.0 for r in results if r.check_type != "judge")
    assert by_id["math-001"].prompt_tokens == 11

    out = tmp_path / "results.jsonl"
    write_results(results, out)
    prof = compute_profile(load_probe_scores(out), name="p", probe_set=ps.name, prior_weight=0)
    assert [c.id for c in prof.clusters] == sorted(c.id for c in ps.clusters)
    assert prof.models[0].quality == pytest.approx([1.0, 1.0, 1.0, 1.0, 0.9])


def test_runner_concurrency_preserves_order():
    ps = load_probe_set(DATA / "probes" / "sample.yaml")
    runner = ProbeRunner(make_client(lambda r: chat_response(answer_for(json.loads(r.content)))),
                         judge=FakeJudge())
    res = runner.run(ps, concurrency=4)
    assert [r.probe_id for r in res] == [p.id for p in ps.probes]


def test_retry_on_5xx_then_success():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, text="busy") if calls["n"] < 3 else chat_response("391")

    client = make_client(handler, max_retries=2)
    text, _usage, _ = client.chat([{"role": "user", "content": "17 * 23"}], 10)
    assert text == "391" and calls["n"] == 3


def test_non_retryable_error_recorded_not_scored():
    ps = load_probe_set(DATA / "probes" / "sample.yaml")
    runner = ProbeRunner(make_client(lambda r: httpx.Response(401, text="bad key")))
    r = runner.run_probe(ps.probes[0], ps.name)
    assert r.error and "401" in r.error and r.score is None


def test_judge_probe_without_judge_is_error():
    ps = load_probe_set(DATA / "probes" / "sample.yaml")
    probe = next(p for p in ps.probes if p.check.type == "judge")
    runner = ProbeRunner(make_client(lambda r: chat_response("summary")))
    r = runner.run_probe(probe, ps.name)
    assert r.error == "judge check but no judge configured"


def test_llm_judge_hook_parses_verdict():
    ps = load_probe_set(DATA / "probes" / "sample.yaml")
    probe = next(p for p in ps.probes if p.check.type == "judge")

    def judge_handler(request):
        body = json.loads(request.content)
        assert body["model"] == "local/judge"
        assert "RUBRIC" in body["messages"][1]["content"]
        return chat_response('{"score": 7, "reason": "misses the driver"}')

    judge = LLMJudge(make_client(judge_handler, model="local/judge"))
    runner = ProbeRunner(make_client(lambda r: chat_response("Churn went up.")), judge=judge,
                         store_output=False)
    r = runner.run_probe(probe, ps.name)
    assert r.score == pytest.approx(0.7) and r.passed is True and r.output is None


def test_store_output_toggle():
    ps = load_probe_set(DATA / "probes" / "sample.yaml")
    runner = ProbeRunner(make_client(lambda r: chat_response("391")), store_output=True)
    probe = next(p for p in ps.probes if p.id == "math-001")
    assert runner.run_probe(probe, ps.name).output == "391"
