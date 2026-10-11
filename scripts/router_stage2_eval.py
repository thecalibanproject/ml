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


def ort_latency_ms(art: Path, texts: list[str]) -> float:
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    from caliban_ml.artifacts import load_manifest

    m = load_manifest(art)
    tok = Tokenizer.from_file(str(art / "tokenizer.json"))
    tok.enable_truncation(m.tokenizer.max_length)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(art / "model.onnx"), so, providers=["CPUExecutionProvider"])
    outs = [o.name for o in m.onnx.outputs]
    times = []
    for i, t in enumerate(texts * 2):
        ids = np.array([tok.encode(t).ids], dtype=np.int64)
        t0 = time.perf_counter()
        sess.run(outs, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
        if i >= len(texts) // 2:  # skip warm-up
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


SCORERS = (("trained head", ""), ("kNN, base encoder", "knn_base_"),
           ("kNN, fine-tuned encoder", "knn_tuned_"))
COLUMNS = (("test acc", "test_in_scope_accuracy"),
           ("test macro-F1", "test_in_scope_macro_f1"),
           ("test e2e acc", "test_end_to_end_accuracy"),
           ("para_en acc", "para_en_in_scope_accuracy"),
           ("fr acc", "fr_in_scope_accuracy"),
           ("ar acc", "ar_in_scope_accuracy"),
           ("fr+ar OOS rej", None))


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def render_report(summary: dict) -> str:
    """Markdown tables: every scorer on the same splits, plus held-out-intent folds."""
    lines = [f"GPU: {summary['gpu']}, torch {summary['torch']}, exemplars: "
             f"{summary['n_exemplars']} utterances over {len(summary['intents'])} intents.", ""]
    for key, run in summary["runs"].items():
        m, exp, folds = run["main"], run["export"], run["folds"]
        lines += [f"### {key}", "",
                  f"Train {m.get('train_seconds', 0):.1f} s, peak VRAM "
                  f"{m.get('train_peak_vram_allocated_mb', 0):.0f} MiB allocated / "
                  f"{m.get('train_peak_vram_reserved_mb', 0):.0f} MiB reserved. Export: fp32 "
                  f"max|diff| {exp.get('export_fp32_max_abs_diff', float('nan')):.2e}, INT8 "
                  f"argmax agreement {exp.get('export_int8_argmax_agree', float('nan')):.4f}, "
                  f"model.onnx {exp.get('model_onnx_mb', 0):.0f} MiB, ORT CPU p50 "
                  f"{exp.get('ort_cpu_int8_latency_ms_p50', 0):.1f} ms per query.", "",
                  "| scorer | " + " | ".join(c for c, _ in COLUMNS)
                  + " | held-out intent OOS rej (mean of folds) | fold test acc (mean) |",
                  "|---" * (len(COLUMNS) + 3) + "|"]
        for label, prefix in SCORERS:
            cells = []
            for _, col in COLUMNS:
                if col is None:
                    cells.append(_fmt(_mean([m.get(f"{prefix}fr_oos_rejection"),
                                             m.get(f"{prefix}ar_oos_rejection")])))
                else:
                    cells.append(_fmt(m.get(prefix + col)))
            rej = _mean([f.get(f"{prefix}heldout_intents_oos_rejection") for f in folds.values()])
            acc = _mean([f.get(f"{prefix}test_in_scope_accuracy") for f in folds.values()])
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {_fmt(rej)} | {_fmt(acc)} |")
        if any("heldout_intents_oos_head_oos_rejection" in f for f in folds.values()):
            rej = _mean([f.get("heldout_intents_oos_head_oos_rejection") for f in folds.values()])
            lines.append(f"| trained head, oos head only | | | | | | | | {_fmt(rej)} | |")
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
    args = ap.parse_args()
    backbones = args.backbone or ["modernbert_multihead", "setfit"]

    import torch

    from caliban_ml.router import IntentDataset
    from caliban_ml.router.export import export_onnx
    from caliban_ml.router.train import IntentHeadConfig, train_intent_head

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    ds = IntentDataset.from_yaml(args.exemplars)
    extra = extra_eval_sets(set(ds.labels))
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
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
            export_onnx(ckpt, art, name=f"stage2-{name}-{backbone.replace('_', '-')}",
                        version="0.1.0", quantization="int8_dynamic", overwrite=True)
            from caliban_ml.artifacts import load_manifest

            exp = {k: v for k, v in load_manifest(art).metrics.items()
                   if k.startswith(("export_", "model_"))}
            exp["ort_cpu_int8_latency_ms_p50"] = ort_latency_ms(art, extra["para_en"][0])
            run = {"main": metrics, "export": exp, "folds": {}}
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
