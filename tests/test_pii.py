from __future__ import annotations

import pytest
from pydantic import ValidationError

from caliban_ml.pii import (
    CANONICAL_LABELS,
    PiiExample,
    RegexBaselineDetector,
    Span,
    evaluate_detector,
    evaluate_spans,
    load_pii_dataset,
    match_spans,
)
from caliban_ml.pii.adapters.regex_baseline import iban_ok, luhn_ok, ssn_ok
from conftest import DATA


def S(start, end, label):
    return Span(start=start, end=end, label=label)


def test_exact_vs_overlap_vs_redaction():
    gold = [[S(0, 10, "PERSON"), S(20, 30, "EMAIL")]]
    pred = [[S(0, 8, "PERSON"),     # partial boundary
             S(20, 30, "PHONE")]]   # right span, wrong type
    ex = evaluate_spans(gold, pred, "exact")
    assert (ex.micro.tp, ex.micro.fp, ex.micro.fn) == (0, 2, 2)
    ov = evaluate_spans(gold, pred, "overlap")
    assert ov.per_label["PERSON"].tp == 1
    assert ov.per_label["EMAIL"].fn == 1 and ov.per_label["PHONE"].fp == 1
    red = evaluate_spans(gold, pred, "redaction")
    assert (red.micro.tp, red.micro.fp, red.micro.fn) == (2, 0, 0)
    assert red.per_label["EMAIL"].tp == 1  # counted under the gold label


def test_one_to_one_matching():
    gold = [S(0, 10, "PERSON")]
    pred = [S(0, 5, "PERSON"), S(4, 10, "PERSON")]
    m = match_spans(gold, pred, "overlap")
    assert m == [(0, 1)]  # larger overlap wins (6 > 5); the other pred is a FP
    rep = evaluate_spans([gold], [pred], "overlap")
    assert (rep.micro.tp, rep.micro.fp, rep.micro.fn) == (1, 1, 0)


def test_duplicate_predictions_are_false_positives_in_exact():
    rep = evaluate_spans([[S(0, 4, "EMAIL")]], [[S(0, 4, "EMAIL"), S(0, 4, "EMAIL")]], "exact")
    assert (rep.micro.tp, rep.micro.fp) == (1, 1)


def test_prf_math_and_macro():
    gold = [[S(0, 1, "A"), S(2, 3, "A"), S(4, 5, "B")]]
    pred = [[S(0, 1, "A"), S(6, 7, "A"), S(4, 5, "B")]]
    rep = evaluate_spans(gold, pred, "exact")
    a = rep.per_label["A"]
    assert (a.tp, a.fp, a.fn) == (1, 1, 1)
    assert a.precision == pytest.approx(0.5) and a.recall == pytest.approx(0.5)
    assert a.f1 == pytest.approx(0.5)
    assert rep.per_label["B"].f1 == pytest.approx(1.0)
    assert rep.macro_f1 == pytest.approx(0.75)
    assert rep.micro.f1 == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))


def test_empty_docs():
    rep = evaluate_spans([[]], [[]], "overlap")
    assert rep.micro.f1 == 0.0 and rep.per_label == {}
    with pytest.raises(ValueError):
        evaluate_spans([[]], [], "exact")


def test_dataset_validation():
    with pytest.raises(ValidationError):
        PiiExample.model_validate({"text": "abc", "spans": [{"start": 1, "end": 9, "label": "X"}]})
    with pytest.raises(ValidationError):
        Span(start=3, end=3, label="X")


def test_sample_dataset_is_consistent():
    examples = load_pii_dataset(DATA / "pii" / "sample.jsonl")
    assert 18 <= len(examples) <= 25
    labels = {s.label for ex in examples for s in ex.spans}
    assert labels <= set(CANONICAL_LABELS)
    for ex in examples:
        for s in ex.spans:
            chunk = ex.text[s.start:s.end]
            assert chunk == chunk.strip() and chunk, (ex.id, s)
        # obviously-fake data only
        assert "@" not in ex.text or "example." in ex.text


def test_validators():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    assert iban_ok("GB82 WEST 1234 5698 7654 32") and not iban_ok("GB82 WEST 1234 5698 7654 33")
    assert ssn_ok("123", "45", "6789")
    assert not ssn_ok("000", "45", "6789") and not ssn_ok("912", "45", "6789")


def test_regex_baseline_on_sample():
    examples = load_pii_dataset(DATA / "pii" / "sample.jsonl")
    rep = evaluate_detector(RegexBaselineDetector(), examples, "exact")
    structured = ["EMAIL", "PHONE", "CREDIT_CARD", "IBAN", "US_SSN", "IP_ADDRESS", "SECRET", "URL"]
    for label in structured:
        assert rep.per_label[label].recall == 1.0, label
        assert rep.per_label[label].precision == 1.0, label
    # Names/places are out of reach for L0 regex: that is the L1 NER's job.
    assert rep.per_label["PERSON"].recall == 0.0


def test_regex_overlap_resolution_prefers_longest_then_risk():
    d = RegexBaselineDetector()
    spans = d.detect("card 4111-1111-1111-1111 now")
    assert [(s.label, s.start, s.end) for s in spans] == [("CREDIT_CARD", 5, 24)]
    assert d.detect("call 555-0100 or 2026-01-15") == []  # too short / ISO date
