# Stage-2 intent heads: design and results

Status: **metrics pending the AWS run.** The training, calibration, export and evaluation code is in place and unit tested; the GPU run on ModernBERT-base and mmBERT-base has not been done yet. It was stopped on the original training box because the network there (about 1 MB/s, shared) made the roughly 4 GB of CUDA wheels and base weights impractical to fetch. The tables below are the exact shape `scripts/router_stage2_eval.py` writes to `runs/stage2-eval/report.md`; paste that file in here once the run is done.

## How to reproduce

On a fresh CUDA Linux box with Python 3.11+ and a core checkout next to this repo (or a copy of core's `crates/caliban-route/data/exemplars.default.json`):

```bash
scripts/stage2_gpu_run.sh                       # or: scripts/stage2_gpu_run.sh <exemplars.json> <out dir>
```

The script installs `[train,export,dev]` into `.venv-train`, runs the CPU smoke test (a tiny random ModernBERT is trained, exported and parity-checked, so library API drift shows up in minutes, before any GPU time), fetches the pinned base models with `scripts/fetch_stage2_bases.py` (SHA-256 checked), runs the GPU smoke test, then the full evaluation, then `caliban-ml manifest verify --strict` on every exported artifact. It waits for `MIN_FREE_MB` (default 6000) of free GPU memory before each training run instead of competing with other processes. A 24 GB card (A10G, L4) is plenty: base-size encoders, sequence length 128, batch 16, bf16 autocast.

## What is compared, and why

Research note 02 ([docs/research/02](https://github.com/thecalibanproject/docs/blob/main/research/02-intent-classification-and-routing.md)) places the encoder at Stage 2, behind rules and the Stage-1 embedding kNN that core already runs, and warns that simple routers are hard to beat: a tuned kNN over prompt embeddings matches or beats learned routers ([arXiv 2505.12601](https://arxiv.org/abs/2505.12601), [LLMRouterBench, arXiv 2601.07206](https://arxiv.org/abs/2601.07206)), and gains measured in-distribution vanish on held-out categories ([False Floors, arXiv 2610.01535](https://arxiv.org/abs/2610.01535)). So every trained head is scored next to the Stage-1 kNN (the same numpy code core mirrors, calibrated the same way) on the same splits:

| Scorer | What it is |
|---|---|
| trained head | the Stage-2 head, calibrated on the validation split (temperature, OOS gate on max-probability, per-label exit thresholds) |
| kNN, base encoder | Stage-1 kNN over the *untuned* base encoder's mean-pooled embeddings |
| kNN, fine-tuned encoder | Stage-1 kNN over the fine-tuned encoder's pooled embeddings |

Runs: base model in {ModernBERT-base ([arXiv 2412.13663](https://arxiv.org/abs/2412.13663)), mmBERT-base ([arXiv 2509.06888](https://arxiv.org/abs/2509.06888), multilingual, the backbone vLLM Semantic Router v0.4 moved to)} x backbone in {`modernbert_multihead` (intent softmax + explicit OOS sigmoid head), `setfit` ([arXiv 2209.11055](https://arxiv.org/abs/2209.11055))}. Base size only: scaling the encoder barely helps routing ([RegexRoute, arXiv 2609.34326](https://arxiv.org/abs/2609.34326)).

Data: core's built-in exemplar set (7 intents x 30 synthetic utterances plus 10 OOS, written for Caliban, `caliban-internal`). Eval sets, none used for training or calibration:

- `test`: a stratified 20% split of the exemplars.
- held-out intents: seven leave-one-intent-out folds; the removed intent's 30 utterances must be rejected as out-of-scope.
- `para_en`: this repo's `data/intents/sample.yaml`, written independently, mapped onto core's labels (in-scope only).
- `fr`, `ar`: `data/intents/multilingual_eval.yaml`, French and Arabic prompts for the same intents plus OOS, written for Caliban. The exemplars are English only, so this measures cross-lingual transfer.

Caveats to keep in mind when the numbers arrive: the test split has about 42 in-scope and 2 OOS examples, so differences of a few points are within noise (note 02 cites 5% label flips between runs on real routing data); one seed; the multilingual sets are small (32 in-scope + 4 OOS per language).

## Results (pending AWS run)

Per base model and backbone the report has one block like this:

| scorer | test acc | test macro-F1 | test e2e acc | para_en acc | fr acc | ar acc | fr+ar OOS rej | held-out intent OOS rej (mean of folds) | fold test acc (mean) |
|---|---|---|---|---|---|---|---|---|---|
| trained head | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| kNN, base encoder | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| kNN, fine-tuned encoder | pending | pending | pending | pending | pending | pending | pending | pending | pending |

plus train time, peak VRAM (allocated and reserved), export parity (fp32 max |diff|, INT8 argmax agreement), `model.onnx` size and ONNX Runtime CPU latency per query. If the trained head does not beat the kNN baselines, that is the result and it will be stated as such here.

## Artifact and how core should load it

`caliban-ml router export` writes an `intent_head` artifact: `manifest.json` (schema `schemas/artifact-manifest.schema.json`), `model.onnx` (opset 17, INT8 dynamic by default), `tokenizer.json` (truncation and padding stripped), `labels.json`, `config.json` (`type = "onnx_classifier"`, `primary_head`, `pooling`, `pad_id`, `max_length`). Inputs `input_ids` and `attention_mask` (int64, `[batch, seq]`), outputs `intent_logits` (`[batch, n_labels]`) and, for the multi-head backbone, `oos_logits` (`[batch, 1]`), all described in `manifest.onnx.outputs` and `manifest.onnx.heads`. Calibration semantics are the Stage-1 ones (`oos_score = max_probability`), see the README section "Stage-2 intent heads". The export refuses to write anything unless the shipped `tokenizer.json` reproduces the training tokenizer's ids and ONNX Runtime matches torch (fp32 max |diff| <= 1e-3, INT8 argmax agreement >= 99.5%).

Core today: `caliban-route` documents Stage 2 as a TODO and its `load_knn_calibration` already refuses an `intent_head` that carries `onnx`, so these artifacts cannot be mistaken for a Stage-1 calibration. To use them core needs a Stage-2 loader, which can reuse the PII NER loader's pattern (`crates/caliban-pii/src/ner/artifact.rs`, `ort` + `tokenizers` behind a cargo feature):

1. verify the manifest like the other loaders (kind `intent_head`, `onnx` present, `config.json` `type = "onnx_classifier"`);
2. bind `input_ids` / `attention_mask` and every `onnx.heads[].output` by name; enable truncation to `tokenizer.max_length` (right) and pad with `config.json` `pad_id`;
3. apply `softmax(intent_logits / calibration.temperature)`, the OOS gate (`p_max < oos_threshold`), the per-label exit thresholds, and optionally the `oos` head (`sigmoid(logit / T) >= thresholds["oos"]` means out-of-scope);
4. run it only when Stage 1 abstains, within the routing budget, and fall back to the keyword rules on timeout or error, like Stage 1;
5. check that the labels match the tenant's route ids (the artifact is trained on a fixed label space).
