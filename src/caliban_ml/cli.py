"""``caliban-ml`` command line."""

from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path

import typer

from caliban_ml import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Caliban offline ML tooling: artifacts, router, probes, PII eval.")
manifest_app = typer.Typer(no_args_is_help=True, help="Artifact manifests (ONNX + JSON contract).")
router_app = typer.Typer(no_args_is_help=True, help="Intent heads and router profiles.")
probes_app = typer.Typer(no_args_is_help=True, help="BYO-model probe runs.")
pii_app = typer.Typer(no_args_is_help=True, help="PII detector evaluation.")
app.add_typer(manifest_app, name="manifest")
app.add_typer(router_app, name="router")
app.add_typer(probes_app, name="probes")
app.add_typer(pii_app, name="pii")


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(version: bool = typer.Option(False, "--version", callback=_version, is_eager=True,
                                      help="Print version and exit.")) -> None:
    pass


# ---------------------------------------------------------------------------- manifest


class Scheme(StrEnum):
    minisign = "minisign"
    cosign = "cosign"


@manifest_app.command("verify")
def manifest_verify(
    artifact_dir: Path = typer.Argument(..., exists=True, file_okay=False,
                                        help="Artifact directory containing manifest.json."),
    strict: bool = typer.Option(False, help="Fail on payload files not listed in the manifest."),
    allow_noncommercial: bool = typer.Option(
        False, help="Accept NC licences that carry an override (never for the default bundle)."),
    signature: Scheme | None = typer.Option(None, help="Also verify a detached signature."),
    pubkey: Path | None = typer.Option(None, help="Public key for --signature."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Validate manifest.json and check every file's size + SHA-256."""
    from caliban_ml.artifacts import verify_artifact_dir

    rep = verify_artifact_dir(artifact_dir, strict=strict, allow_noncommercial=allow_noncommercial)
    if rep.ok and signature is not None:
        from caliban_ml.artifacts.signing import (
            SignatureScheme,
            SigningUnavailableError,
            verify_signature,
        )

        if pubkey is None:
            rep.errors.append("--signature requires --pubkey")
        else:
            try:
                verify_signature(artifact_dir, SignatureScheme(signature.value), pubkey)
            except (SigningUnavailableError, FileNotFoundError, OSError) as exc:
                rep.errors.append(f"signature: {exc}")
            except Exception as exc:  # subprocess.CalledProcessError and friends
                rep.errors.append(f"signature verification failed: {exc}")
    if as_json:
        ref = None if rep.manifest is None else rep.manifest.ref().model_dump(mode="json")
        typer.echo(json.dumps({"ok": rep.ok, "errors": rep.errors, "warnings": rep.warnings,
                               "artifact": ref}, indent=2))
    else:
        if rep.manifest is not None:
            m = rep.manifest
            typer.echo(f"{m.kind.value} {m.name}@{m.version}: {len(m.files)} files")
        for w in rep.warnings:
            typer.echo(f"WARN  {w}")
        for e in rep.errors:
            typer.echo(f"ERROR {e}", err=True)
        typer.echo("OK" if rep.ok else "FAILED")
    raise typer.Exit(0 if rep.ok else 1)


@manifest_app.command("schema")
def manifest_schema(
    out: Path = typer.Option(Path("schemas"), help="Output directory."),
    check: bool = typer.Option(False, help="Exit 1 if committed schemas are out of date."),
) -> None:
    """Export JSON Schemas (artifact manifest, router profile) for caliban core."""
    from caliban_ml.schemas import SCHEMAS, render, write_schemas

    if check:
        stale = [f for f in SCHEMAS if not (out / f).is_file()
                 or (out / f).read_text(encoding="utf-8") != render(f)]
        for f in stale:
            typer.echo(f"stale: {out / f}", err=True)
        raise typer.Exit(1 if stale else 0)
    for p in write_schemas(out):
        typer.echo(f"wrote {p}")


# ---------------------------------------------------------------------------- router


class OosStrategy(StrEnum):
    recall = "recall"
    youden = "youden"


@router_app.command("knn-eval")
def router_knn_eval(
    train: Path = typer.Option(..., exists=True, help="Train .npz (embeddings, labels)."),
    val: Path = typer.Option(..., exists=True, help="Validation .npz (may include __oos__)."),
    test: Path | None = typer.Option(None, exists=True, help="Optional held-out test .npz."),
    k: int = typer.Option(5, min=1),
    oos_strategy: OosStrategy = typer.Option(OosStrategy.recall),
    target_recall: float = typer.Option(0.95, help="In-scope acceptance target for the OOS gate."),
    target_precision: float = typer.Option(0.95, help="Stage-1 exit precision target."),
    out: Path | None = typer.Option(None, help="Write calibration + metrics JSON here."),
) -> None:
    """Fit the numpy kNN baseline, calibrate T, pick OOS + exit thresholds, report metrics."""
    from caliban_ml.router.embed import load_npz
    from caliban_ml.router.knn import evaluate, fit_and_calibrate

    tr_x, tr_y = load_npz(train)
    va_x, va_y = load_npz(val)
    res = fit_and_calibrate(tr_x, tr_y, va_x, va_y, k=k, oos_strategy=oos_strategy.value,
                            target_in_scope_recall=target_recall,
                            target_precision=target_precision)
    metrics = dict(res.metrics)
    if test is not None:
        te_x, te_y = load_npz(test)
        metrics.update(evaluate(res.classifier, res.calibration, te_x, te_y, prefix="test_"))
    payload = {"labels": res.classifier.classes_, "k": k,
               "calibration": res.calibration.model_dump(exclude_none=True), "metrics": metrics}
    if out:
        out.write_text(json.dumps(payload, indent=2) + "\n")
    typer.echo(json.dumps(payload, indent=2))


@router_app.command("embed")
def router_embed(
    dataset: Path = typer.Option(..., exists=True, help="Intent exemplar YAML."),
    model_dir: Path = typer.Option(..., exists=True, file_okay=False,
                                   help="Local sentence-transformers model dir ([train] extra)."),
    out_prefix: Path = typer.Option(..., help="Writes <prefix>.train.npz and <prefix>.val.npz."),
    val_fraction: float = typer.Option(0.3),
    seed: int = typer.Option(0),
    prefix: str = typer.Option("", help="Query prefix, e.g. 'query: ' for E5 models."),
) -> None:
    """Split + embed an intent dataset into .npz files for knn-eval."""
    from caliban_ml.router.dataset import IntentDataset
    from caliban_ml.router.embed import embed_sentence_transformers, save_npz

    ds = IntentDataset.from_yaml(dataset)
    for split, (texts, labels) in zip(("train", "val"), ds.split(val_fraction, seed),
                                      strict=True):
        emb = embed_sentence_transformers(model_dir, texts, prefix=prefix)
        path = Path(f"{out_prefix}.{split}.npz")
        save_npz(path, emb, labels, texts)
        typer.echo(f"wrote {path} ({len(texts)} rows)")


@router_app.command("profile")
def router_profile(
    results: list[Path] = typer.Argument(..., exists=True, help="Probe result JSONL file(s)."),
    name: str = typer.Option(..., help="Artifact name, e.g. tenant-acme-profile."),
    version: str = typer.Option(..., help="Semver, e.g. 1.0.0."),
    out: Path = typer.Option(..., help="Artifact output directory."),
    probe_set: str | None = typer.Option(None, help="Defaults to the probe_set in the results."),
    prior_weight: float = typer.Option(2.0, min=0.0, help="Shrinkage strength (pseudo-counts)."),
    errors_as_zero: bool = typer.Option(False, help="Count failed calls as score 0."),
    data_licence: str = typer.Option("caliban-internal", help="Licence of the probe set."),
) -> None:
    """Aggregate probe results into a router_profile artifact (profile.json + manifest)."""
    from caliban_ml.router.profile import compute_profile, load_probe_scores, write_profile_artifact

    rows = []
    probe_sets = set()
    for p in results:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                probe_sets.add(json.loads(line).get("probe_set", p.stem))
        rows.extend(load_probe_scores(p))
    ps = probe_set or ",".join(sorted(probe_sets))
    prof = compute_profile(rows, name=name, probe_set=ps, prior_weight=prior_weight,
                           errors_as_zero=errors_as_zero)
    data_card = {
        "eval_datasets": [{"name": ps, "source": ",".join(str(p) for p in results),
                           "licence": data_licence, "n_examples": len(rows)}],
        "intended_use": "Stage-4 per-cluster model quality prior for caliban/auto routing.",
    }
    m = write_profile_artifact(prof, out, version=version, data_card=data_card)
    typer.echo(f"wrote {out} ({m.kind.value} {m.name}@{m.version})")
    header = "model".ljust(28) + "".join(c.id[:12].rjust(13) for c in prof.clusters)
    typer.echo(header)
    for mp in prof.models:
        typer.echo(mp.model[:27].ljust(28) + "".join(f"{q:13.3f}" for q in mp.quality))


# ---------------------------------------------------------------------------- probes


@probes_app.command("run")
def probes_run(
    probe_set: Path = typer.Argument(..., exists=True, help="Probe set YAML/JSONL."),
    base_url: str = typer.Option(..., help="OpenAI-compatible base URL, e.g. http://localhost:8080/v1"),
    model: list[str] = typer.Option(..., help="Model id(s) to probe (repeatable)."),
    out: Path = typer.Option(..., help="Results JSONL."),
    api_key_env: str = typer.Option("CALIBAN_PROBE_API_KEY",
                                    help="Env var holding the endpoint API key."),
    judge_base_url: str | None = typer.Option(None, help="Judge endpoint (local model)."),
    judge_model: str | None = typer.Option(None),
    judge_api_key_env: str = typer.Option("CALIBAN_JUDGE_API_KEY"),
    concurrency: int = typer.Option(1, min=1),
    timeout: float = typer.Option(120.0),
    store_output: bool = typer.Option(True, help="Store model outputs in results."),
    extra_body: str | None = typer.Option(
        None, help='Extra JSON merged into each request, e.g. \'{"seed": 7}\''),
) -> None:
    """Run a probe set against one or more models and write scored results."""
    from caliban_ml.probes import (
        ChatClient,
        EndpointConfig,
        LLMJudge,
        ProbeRunner,
        load_probe_set,
        summarize,
        write_results,
    )

    ps = load_probe_set(probe_set)
    extra = json.loads(extra_body) if extra_body else {}
    judge = None
    if judge_base_url and judge_model:
        judge = LLMJudge(ChatClient(EndpointConfig(
            base_url=judge_base_url, model=judge_model,
            api_key=os.environ.get(judge_api_key_env), timeout_s=timeout)))
    all_results = []
    for mid in model:
        client = ChatClient(EndpointConfig(base_url=base_url, model=mid,
                                           api_key=os.environ.get(api_key_env),
                                           timeout_s=timeout, extra_body=extra))
        res = ProbeRunner(client, judge=judge, store_output=store_output).run(ps, concurrency)
        all_results.extend(res)
        typer.echo(f"== {mid}")
        for cluster, s in sorted(summarize(res).items()):
            typer.echo(f"  {cluster:<20} n={int(s['n']):<4} errors={int(s['errors']):<3} "
                       f"mean={s['mean_score']:.3f} pass={s['pass_rate']:.3f}")
    write_results(all_results, out)
    typer.echo(f"wrote {out} ({len(all_results)} results)")


# ---------------------------------------------------------------------------- pii


class Detector(StrEnum):
    regex = "regex"
    gliner = "gliner"
    presidio = "presidio"


class Mode(StrEnum):
    exact = "exact"
    overlap = "overlap"
    redaction = "redaction"


@pii_app.command("eval")
def pii_eval(
    dataset: Path = typer.Argument(..., exists=True, help="JSONL of {text, spans}."),
    detector: Detector = typer.Option(Detector.regex),
    mode: list[Mode] = typer.Option([Mode.exact, Mode.overlap, Mode.redaction]),
    model_dir: Path | None = typer.Option(None, help="Local model dir (gliner)."),
    out: Path | None = typer.Option(None, help="Write the JSON report here."),
) -> None:
    """Span-level P/R/F1 per entity type for a PII detector."""
    from caliban_ml.pii import evaluate_spans, get_detector, load_pii_dataset

    kwargs = {"model_dir": model_dir} if detector is Detector.gliner else {}
    if detector is Detector.gliner and model_dir is None:
        raise typer.BadParameter("--model-dir is required for gliner")
    det = get_detector(detector.value, **kwargs)
    examples = load_pii_dataset(dataset)
    preds = [det.detect(ex.text) for ex in examples]
    reports = {}
    for md in mode:
        rep = evaluate_spans([ex.spans for ex in examples], preds, md.value)
        reports[md.value] = rep.as_dict()
        mi = rep.micro
        typer.echo(f"== {det.name} / {md.value}: P={mi.precision:.3f} R={mi.recall:.3f} "
                   f"F1={mi.f1:.3f} macroF1={rep.macro_f1:.3f} (docs={rep.n_docs})")
        typer.echo(f"  {'label':<16}{'P':>7}{'R':>7}{'F1':>7}{'support':>9}")
        for label, s in sorted(rep.per_label.items()):
            typer.echo(f"  {label:<16}{s.precision:7.3f}{s.recall:7.3f}{s.f1:7.3f}"
                       f"{s.support:9d}")
    if out:
        out.write_text(json.dumps({"detector": det.name, "dataset": str(dataset),
                                   "reports": reports}, indent=2) + "\n")


if __name__ == "__main__":  # pragma: no cover
    app()
