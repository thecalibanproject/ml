from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from caliban_ml.artifacts import build_manifest, write_manifest
from caliban_ml.cli import app
from caliban_ml.router.embed import save_npz
from conftest import DATA, REPO, pii_ner_fields
from test_knn import split, synthetic

runner = CliRunner()


def test_manifest_verify_ok_and_tampered(artifact_dir: Path):
    write_manifest(build_manifest(artifact_dir, **pii_ner_fields()), artifact_dir)
    r = runner.invoke(app, ["manifest", "verify", str(artifact_dir)])
    assert r.exit_code == 0, r.output
    assert "OK" in r.output
    (artifact_dir / "tokenizer.json").write_text("{}")
    r = runner.invoke(app, ["manifest", "verify", str(artifact_dir), "--json"])
    assert r.exit_code == 1
    assert json.loads(r.stdout)["ok"] is False


def test_manifest_schema_check():
    r = runner.invoke(app, ["manifest", "schema", "--check", "--out", str(REPO / "schemas")])
    assert r.exit_code == 0, r.output


def test_pii_eval(tmp_path: Path):
    out = tmp_path / "report.json"
    r = runner.invoke(app, ["pii", "eval", str(DATA / "pii" / "sample.jsonl"),
                            "--mode", "overlap", "--out", str(out)])
    assert r.exit_code == 0, r.output
    rep = json.loads(out.read_text())
    assert rep["reports"]["overlap"]["per_label"]["EMAIL"]["recall"] == 1.0


def test_router_knn_eval(tmp_path: Path):
    x, y = synthetic()
    trx, try_, vax, vay = split(x, y)
    save_npz(tmp_path / "train.npz", trx, try_, ["t"] * len(try_))
    save_npz(tmp_path / "val.npz", vax, vay, ["t"] * len(vay))
    out = tmp_path / "calib.json"
    r = runner.invoke(app, ["router", "knn-eval", "--train", str(tmp_path / "train.npz"),
                            "--val", str(tmp_path / "val.npz"), "--test",
                            str(tmp_path / "val.npz"), "--out", str(out)])
    assert r.exit_code == 0, r.output
    payload = json.loads(out.read_text())
    assert payload["calibration"]["oos_score"] == "top1_similarity"
    assert "test_end_to_end_accuracy" in payload["metrics"]


def test_router_profile(tmp_path: Path):
    res = tmp_path / "results.jsonl"
    rows = [
        {"probe_set": "s", "model": "m1", "probe_id": "a", "cluster": "sql", "score": 1.0},
        {"probe_set": "s", "model": "m1", "probe_id": "b", "cluster": "code", "score": 0.0},
        {"probe_set": "s", "model": "m2", "probe_id": "a", "cluster": "sql", "score": 0.5},
    ]
    res.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "profile"
    r = runner.invoke(app, ["router", "profile", str(res), "--name", "test-profile",
                            "--version", "0.1.0", "--out", str(out)])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["manifest", "verify", str(out), "--strict"])
    assert r.exit_code == 0, r.output
