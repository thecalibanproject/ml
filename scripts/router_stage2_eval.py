#!/usr/bin/env python3
"""Train and evaluate Stage-2 intent heads against the Stage-1 kNN baseline.

Needs the ``[train]`` and ``[export]`` extras and local, licence-checked base models. Writes
everything under ``--out`` (default ``runs/stage2-eval``, git-ignored) and prints a summary.

For every ``--base`` and ``--backbone``:

1. a main run on the exemplar set (three-way split), scored on the held-out test split and
   on extra eval sets: independently written English paraphrases (``data/intents/sample.yaml``
   mapped onto core's labels) and French / Arabic prompts (``data/intents/multilingual_eval.yaml``);
   then ONNX export (INT8) with the torch-vs-ONNX parity check and a CPU latency probe;
2. leave-one-intent-out folds: each intent is removed from training and its utterances
   are scored as out-of-scope (held-out categories, per research note 02).

Every number is reported next to the Stage-1 kNN on the same split, over the base encoder's
and the fine-tuned encoder's embeddings.

Example::

    python scripts/router_stage2_eval.py \\
        --exemplars ../core/crates/caliban-route/data/exemplars.default.json \\
        --base modernbert=<local dir>:answerdotai/ModernBERT-base:<revision>:apache-2.0 \\
        --base mmbert=<local dir>:jhu-clsp/mmBERT-base:<revision>:mit
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMPLE_TO_CORE = {"sql_analytics": "analytics", "summarization": "summarize",
                  "extraction": "extraction", "code": "code"}


def wait_for_gpu(min_free_mb: int, poll_s: int = 30, max_wait_s: int = 4 * 3600) -> int:
    """Poll nvidia-smi until ``min_free_mb`` is free (another process may hold the GPU)."""
    waited = 0
    while True:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True)
        free = int(out.stdout.split()[0])
        if free >= min_free_mb:
            return free
        if waited >= max_wait_s:
            raise SystemExit(f"GPU never had {min_free_mb} MiB free (last: {free})")
        print(f"[gpu] {free} MiB free < {min_free_mb}; waiting {poll_s}s", flush=True)
        time.sleep(poll_s)
        waited += poll_s


def is_arabic(text: str) -> bool:
    return any("؀" <= ch <= "ۿ" for ch in text)


def extra_eval_sets(classes: set[str]) -> dict[str, tuple[list[str], list[str]]]:
    from caliban_ml.router import IntentDataset

    sample = IntentDataset.from_yaml(REPO / "data" / "intents" / "sample.yaml")
    # In-scope only: the sample's OOS prompts (a joke, the weather) are arguably "chat" in
    # core's taxonomy, so they would test label ambiguity, not OOS detection.
    texts, labels = sample.examples(include_oos=False)
    labels = [SAMPLE_TO_CORE[y] for y in labels]
    sets = {"para_en": (texts, labels)}
    ml = IntentDataset.from_yaml(REPO / "data" / "intents" / "multilingual_eval.yaml")
    mt, my = ml.examples(include_oos=True)
    sets["fr"] = ([t for t in mt if not is_arabic(t)], [y for t, y in zip(mt, my, strict=True)
                                                        if not is_arabic(t)])
    sets["ar"] = ([t for t in mt if is_arabic(t)], [y for t, y in zip(mt, my, strict=True)
                                                    if is_arabic(t)])
    for name, (_, ys) in sets.items():
        unknown = {y for y in ys if y != "__oos__"} - classes
        if unknown:
            raise SystemExit(f"eval set {name} has labels outside the exemplar set: {unknown}")
    return sets


def ort_latency_ms(art: Path, texts: list[str], model: Path | None = None) -> float:
    """Median single-query ONNX Runtime CPU latency (4 threads) of ``model`` (default: the
    artifact's model.onnx)."""
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    from caliban_ml.artifacts import load_manifest

    m = load_manifest(art)
    tok = Tokenizer.from_file(str(art / "tokenizer.json"))
    tok.enable_truncation(m.tokenizer.max_length)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(model or art / "model.onnx"), so,
                                providers=["CPUExecutionProvider"])
    outs = [o.name for o in m.onnx.outputs]
    times = []
    for i, t in enumerate(texts * 2):
        ids = np.array([tok.encode(t).ids], dtype=np.int64)
        t0 = time.perf_counter()
        sess.run(outs, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
        if i >= len(texts) // 2:  # skip warm-up
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def _logits(sess, art: Path, texts: list[str]):
    import numpy as np
    from tokenizers import Tokenizer

    from caliban_ml.artifacts import load_manifest

    m = load_manifest(art)
    cfg = json.loads((art / "config.json").read_text())
    tok = Tokenizer.from_file(str(art / "tokenizer.json"))
    tok.enable_truncation(m.tokenizer.max_length)
    out = []
    for i in range(0, len(texts), 32):
        enc = tok.encode_batch(texts[i:i + 32])
        ids = np.full((len(enc), max(len(e.ids) for e in enc)), cfg["pad_id"], dtype=np.int64)
        mask = np.zeros_like(ids)
        for j, e in enumerate(enc):
            ids[j, :len(e.ids)] = e.ids
            mask[j, :len(e.ids)] = 1
        out.append(sess.run([m.onnx.outputs[0].name],
                            {"input_ids": ids, "attention_mask": mask})[0])
    return np.concatenate(out)


def int8_probe(art: Path, texts: list[str]) -> dict[str, float]:
    """Quantize a fp32 intent_head artifact two ways (the conservative settings in
    caliban_ml.router.export and ORT's defaults) and measure argmax agreement with fp32 on
    ``texts``. Measurement only: nothing is shipped."""
    import tempfile

    import numpy as np
    import onnxruntime as ort

    from caliban_ml.router.export import quantize_int8_dynamic

    def sess(p: Path):
        return ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])

    ref = _logits(sess(art / "model.onnx"), art, texts)
    out: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for label, legacy in (("conservative", False), ("ort_defaults", True)):
            q = Path(tmp) / f"{label}.onnx"
            quantize_int8_dynamic(art / "model.onnx", q, legacy_defaults=legacy)
            got = _logits(sess(q), art, texts)
            out[f"int8_{label}_argmax_agree"] = float((ref.argmax(1) == got.argmax(1)).mean())
            out[f"int8_{label}_max_abs_diff"] = float(np.abs(ref - got).max())
            out[f"int8_{label}_mb"] = q.stat().st_size / 2**20
            if not legacy:
                out["int8_conservative_latency_ms_p50"] = ort_latency_ms(art, texts[:50], q)
    return out


SCORERS = (("trained head", ""), ("kNN, base encoder", "knn_base_"),
           ("kNN, fine-tuned encoder", "knn_tuned_"))
COLUMNS = (("test acc", "test_in_scope_accuracy"),
           ("test macro-F1", "test_in_scope_macro_f1"),
           ("test e2e acc", "test_end_to_end_accuracy"),
           ("para_en acc", "para_en_in_scope_accuracy"),
           ("fr acc", "fr_in_scope_accuracy"),
           ("ar acc", "ar_in_scope_accuracy"),
           ("fr+ar in-scope accept", "accept"),
           ("fr+ar OOS rej", None))


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def render_report(summary: dict) -> str:
    """Markdown tables: every scorer on the same splits, plus held-out-intent folds."""
    vers = ", ".join(f"{k} {v}" for k, v in summary.get("versions", {}).items())
    lines = [f"GPU: {summary['gpu']}, torch {summary['torch']}"
             + (f" (CUDA {summary['cuda']})" if summary.get("cuda") else "")
             + (f", {vers}" if vers else "") + f". Exemplars: {summary['n_exemplars']} "
             f"utterances over {len(summary['intents'])} intents.", ""]
    for key, run in summary["runs"].items():
        m, exp, folds = run["main"], run["export"], run["folds"]
        nan = float("nan")
        latency = exp.get("ort_cpu_latency_ms_p50", exp.get("ort_cpu_int8_latency_ms_p50", nan))
        lines += [f"### {key}", "",
                  f"Train {m.get('train_seconds', 0):.1f} s, peak VRAM "
                  f"{m.get('train_peak_vram_allocated_mb', 0):.0f} MiB allocated / "
                  f"{m.get('train_peak_vram_reserved_mb', 0):.0f} MiB reserved. Export "
                  f"({exp.get('quantization', 'n/a')}): fp32 max|diff| "
                  f"{exp.get('export_fp32_max_abs_diff', nan):.2e}, model.onnx "
                  f"{exp.get('model_onnx_mb', 0):.0f} MiB, ORT CPU p50 "
                  f"{latency:.1f} ms per query.", ""]
        probe = run.get("int8_probe") or {}
        if probe:
            lines += [f"INT8 probe (argmax agreement with fp32 / size / latency): conservative "
                      f"{probe.get('int8_conservative_argmax_agree', nan):.3f} / "
                      f"{probe.get('int8_conservative_mb', nan):.0f} MiB / "
                      f"{probe.get('int8_conservative_latency_ms_p50', nan):.1f} ms, ORT "
                      f"defaults {probe.get('int8_ort_defaults_argmax_agree', nan):.3f}.", ""]
        emb = run.get("embedder_export") or {}
        if emb:
            lines += [f"Embedder export: fp32 min cosine "
                      f"{emb.get('export_fp32_cosine_agree', nan):.6f}, model.onnx "
                      f"{emb.get('model_onnx_mb', 0):.0f} MiB.", ""]
        lines += [
                  "| scorer | " + " | ".join(c for c, _ in COLUMNS)
                  + " | held-out intent OOS rej (mean of folds) | fold test acc (mean) |",
                  "|---" * (len(COLUMNS) + 3) + "|"]
        for label, prefix in SCORERS:
            cells = []
            for _, col in COLUMNS:
                if col is None:
                    cells.append(_fmt(_mean([m.get(f"{prefix}fr_oos_rejection"),
                                             m.get(f"{prefix}ar_oos_rejection")])))
                elif col == "accept":
                    cells.append(_fmt(_mean([m.get(f"{prefix}fr_in_scope_acceptance"),
                                             m.get(f"{prefix}ar_in_scope_acceptance")])))
                else:
                    cells.append(_fmt(m.get(prefix + col)))
            rej = _mean([f.get(f"{prefix}heldout_intents_oos_rejection") for f in folds.values()])
            acc = _mean([f.get(f"{prefix}test_in_scope_accuracy") for f in folds.values()])
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {_fmt(rej)} | {_fmt(acc)} |")
        if any("heldout_intents_oos_head_oos_rejection" in f for f in folds.values()):
            rej = _mean([f.get("heldout_intents_oos_head_oos_rejection") for f in folds.values()])
            lines.append("| trained head, oos head only |" + " |" * len(COLUMNS)
                         + f" {_fmt(rej)} | |")
        if folds:
            lines += ["", "Held-out intent OOS rejection per fold (head / kNN base / kNN tuned): "
                      + ", ".join(f"{k} {_fmt(f.get('heldout_intents_oos_rejection'))}/"
                                  f"{_fmt(f.get('knn_base_heldout_intents_oos_rejection'))}/"
                                  f"{_fmt(f.get('knn_tuned_heldout_intents_oos_rejection'))}"
                                  for k, f in folds.items())]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exemplars", type=Path, required=True,
                    help="Intent exemplar file (core's exemplars.default.json).")
    ap.add_argument("--exemplars-name", default="core-exemplars-default")
    ap.add_argument("--base", action="append", required=True,
                    help="NAME=PATH:HF_ID:REVISION:LICENCE (repeatable).")
    ap.add_argument("--backbone", action="append",
                    choices=["modernbert_multihead", "setfit"])
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "stage2-eval")
    ap.add_argument("--no-folds", action="store_true", help="Skip leave-one-intent-out folds.")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--min-free-mb", type=int, default=6000)
    ap.add_argument("--quantization", choices=["none", "int8_dynamic"], default="none",
                    help="Quantization of the exported intent_head artifacts (gated on parity).")
    ap.add_argument("--no-int8-probe", action="store_true",
                    help="Skip measuring INT8 agreement on fp32 exports.")
    ap.add_argument("--no-embedder-export", action="store_true",
                    help="Skip exporting the fine-tuned encoder as an embedder artifact.")
    args = ap.parse_args()
    backbones = args.backbone or ["modernbert_multihead", "setfit"]

    import torch

    from caliban_ml.router import IntentDataset
    from caliban_ml.router.export import export_embedder, export_onnx
    from caliban_ml.router.train import IntentHeadConfig, train_intent_head

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    ds = IntentDataset.from_yaml(args.exemplars)
    extra = extra_eval_sets(set(ds.labels))
    args.out.mkdir(parents=True, exist_ok=True)
    import importlib.metadata as md
    import platform

    versions = {"python": platform.python_version()}
    for dist in ("transformers", "sentence-transformers", "setfit", "onnx", "onnxruntime",
                 "tokenizers", "numpy"):
        try:
            versions[dist] = md.version(dist)
        except md.PackageNotFoundError:
            versions[dist] = "missing"
    summary: dict = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "versions": versions,
                     "exemplars": str(args.exemplars), "n_exemplars": len(ds.examples()[0]),
                     "intents": ds.labels, "runs": {}}
    for spec in args.base:
        name, _, rest = spec.partition("=")
        path, hf_id, rev, licence = rest.split(":")
        for backbone in backbones:
            key = f"{name}/{backbone}"
            run_dir = args.out / name / backbone
            base_cfg = IntentHeadConfig(
                base_model_path=Path(path), base_model_id=hf_id, base_model_revision=rev,
                base_model_licence=licence, backbone=backbone, max_length=128,
                epochs=15 if backbone != "setfit" else 1,
                learning_rate=5e-5 if backbone != "setfit" else 2e-5, patience=4,
                setfit_iterations=20, batch_size=16, seed=args.seed, device="cuda",
                dataset_name=args.exemplars_name, dataset_source=str(args.exemplars),
                dataset_licence="caliban-internal", dataset_synthetic=True,
                languages=("en",), output_dir=run_dir / "main" / "checkpoint")
            print(f"== {key}: main run", flush=True)
            wait_for_gpu(args.min_free_mb)
            ckpt = train_intent_head(ds, base_cfg, extra_eval=extra)
            metrics = json.loads((ckpt / "metrics.json").read_text())
            art = run_dir / "main" / "artifact"
            slug = f"stage2-{name}-{backbone.replace('_', '-')}"
            export_onnx(ckpt, art, name=slug, version="0.1.0", quantization=args.quantization,
                        overwrite=True)
            from caliban_ml.artifacts import load_manifest

            def export_metrics(d: Path) -> dict:
                return {k: v for k, v in load_manifest(d).metrics.items()
                        if k.startswith(("export_", "model_"))}

            exp = {**export_metrics(art), "quantization": args.quantization}
            exp["ort_cpu_latency_ms_p50"] = ort_latency_ms(art, extra["para_en"][0])
            run = {"main": metrics, "export": exp, "folds": {}}
            if args.quantization == "none" and not args.no_int8_probe:
                probe_texts = [t for s in ("test", "para_en", "fr", "ar")
                               for t in (json.loads((ckpt / "splits.json").read_text())
                                         ["test"]["texts"] if s == "test" else extra[s][0])]
                run["int8_probe"] = int8_probe(art, probe_texts)
            if not args.no_embedder_export:
                emb_dir = run_dir / "main" / "embedder"
                export_embedder(ckpt, emb_dir, name=f"{slug}-embedder", version="0.1.0",
                                quantization=args.quantization, overwrite=True)
                run["embedder_export"] = export_metrics(emb_dir)
            if not args.no_folds:
                for intent in ds.labels:
                    print(f"== {key}: hold out {intent}", flush=True)
                    wait_for_gpu(args.min_free_mb)
                    fold_dir = run_dir / "folds" / intent
                    cfg = replace(base_cfg, held_out_intents=(intent,), output_dir=fold_dir)
                    fold_ckpt = train_intent_head(ds, cfg)
                    run["folds"][intent] = json.loads((fold_ckpt / "metrics.json").read_text())
                    shutil.rmtree(fold_ckpt / "encoder")  # weights: keep only metrics
            summary["runs"][key] = run
            (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            (args.out / "report.md").write_text(render_report(summary) + "\n")
    print(render_report(summary))
    print(f"wrote {args.out / 'summary.json'} and {args.out / 'report.md'}")


if __name__ == "__main__":
    sys.exit(main())
