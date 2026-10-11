# Stage-2 intent heads: design and results

Status: **measured** on an AWS L4 on 2026-10-11 (one seed, small data). Conclusion first: on this data the trained Stage-2 head is **not worth shipping**. It never beats the Stage-1 kNN run over the same fine-tuned encoder. The promising path is a **fine-tuned mmBERT-base encoder used as the kNN embedder**. The data set is too small to decide anything finer than that, and **out-of-scope rejection is the main gap** for every method.

## What was compared, and why

Research note 02 ([docs/research/02](https://github.com/thecalibanproject/docs/blob/main/research/02-intent-classification-and-routing.md)) places the encoder at Stage 2, behind rules and the Stage-1 embedding kNN that core already runs. It also warns that simple routers are hard to beat: a tuned kNN over prompt embeddings matches or beats learned routers ([arXiv 2505.12601](https://arxiv.org/abs/2505.12601), [LLMRouterBench, arXiv 2601.07206](https://arxiv.org/abs/2601.07206)), and gains measured in-distribution vanish on held-out categories ([False Floors, arXiv 2610.01535](https://arxiv.org/abs/2610.01535)). So every trained head is scored on the same splits next to the Stage-1 kNN. The kNN is the same numpy code core mirrors, calibrated the same way.

| Scorer | What it is |
|---|---|
| trained head | the Stage-2 head, calibrated on the validation split (temperature, OOS gate on max-probability, per-label exit thresholds) |
| kNN, base encoder | Stage-1 kNN over the *untuned* base encoder's mean-pooled embeddings |
| kNN, fine-tuned encoder | Stage-1 kNN over the same run's fine-tuned encoder (pooled embeddings) |

Runs crossed two base models with two backbones:

- **Base models:** ModernBERT-base ([arXiv 2412.13663](https://arxiv.org/abs/2412.13663), English) and mmBERT-base ([arXiv 2509.06888](https://arxiv.org/abs/2509.06888), multilingual, the backbone vLLM Semantic Router v0.4 moved to).
- **Backbones:** `modernbert_multihead` (an intent softmax head plus an explicit OOS sigmoid head) and `setfit` ([arXiv 2209.11055](https://arxiv.org/abs/2209.11055)).

Both base models are base size only: scaling the encoder barely helps routing ([RegexRoute, arXiv 2609.34326](https://arxiv.org/abs/2609.34326)).

The training data is core's built-in exemplar set: 7 intents x 30 synthetic utterances plus 10 OOS, written for Caliban (`caliban-internal`). None of the eval sets below is used for training or calibration:

- `test`: a stratified split of the exemplars: 44 examples (42 in-scope, 2 OOS).
- **Held-out intents:** seven leave-one-intent-out folds. The removed intent's 30 utterances must be rejected as out-of-scope.
- `para_en`: this repo's `data/intents/sample.yaml`, written independently and mapped onto core's labels (16 in-scope).
- `fr`, `ar`: `data/intents/multilingual_eval.yaml`, French and Arabic prompts for the same intents (28 in-scope + 4 OOS per language). The exemplars are English only, so these sets measure cross-lingual transfer.

## Environment and commands

- **Hardware:** AWS instance with an NVIDIA L4 (23034 MiB), driver 595.91.07.
- **Software:** Python 3.12, torch 2.14.1+cu130 (from the PyTorch CUDA 13.0 index). The other libraries came from PyPI at install time; their versions were not captured, and `router_stage2_eval.py` now records them in `summary.json`.
- **Code:** branch `router-training` at `eccad1a`.

First, the end-to-end script:

```bash
scripts/stage2_gpu_run.sh exemplars.default.json      # core's crates/caliban-route/data/exemplars.default.json
```

It passed the CPU smoke test and the base-model fetch, then stopped at the GPU smoke test: the INT8 export of the real ModernBERT-base head had 30% argmax agreement with torch (gate: 99.5%), while fp32 ONNX matched torch to 3e-5. See INT8 below.

The full run was then done with the export patched to `quantization="none"` on the host. That is now the default and the `--quantization` flag:

```bash
.venv-train/bin/python scripts/router_stage2_eval.py --exemplars exemplars.default.json --out out2 \
  --base modernbert=.hf_cache/answerdotai--ModernBERT-base/8949b909ec900327062f0ebf497f51aef5e6f0c8:answerdotai/ModernBERT-base:8949b909ec900327062f0ebf497f51aef5e6f0c8:apache-2.0 \
  --base mmbert=.hf_cache/jhu-clsp--mmBERT-base/c5955035435e2bf121cde7f3c8863ef52ff35d82:jhu-clsp/mmBERT-base:c5955035435e2bf121cde7f3c8863ef52ff35d82:mit
```

The `--base` strings are what `scripts/fetch_stage2_bases.py` prints. Hyperparameters:

| | multi-head | SetFit |
|---|---|---|
| max length | 128 | 128 |
| batch size | 16 | 16 |
| seed | 13 | 13 |
| epochs | up to 15, early stopping with patience 4 | 1 |
| learning rate | encoder 5e-5, heads 1e-3 | 2e-5 |
| other | | 20 pair iterations |

On the multi-head runs, ModernBERT stopped at epoch 10 (best epoch 6). mmBERT used all 15 epochs (best epoch 15), so it may be under-trained.

To reproduce on a fresh CUDA box: `scripts/stage2_gpu_run.sh` (defaults now: fp32 export, an INT8 probe, and an embedder export).

## Results

Accuracy columns are argmax accuracy on in-scope examples. "e2e acc" counts an OOS-gated in-scope query as an error and a correctly rejected OOS query as correct. Always read "fr+ar OOS rej" next to "fr+ar in-scope accept": the base-encoder kNN "rejects" all French and Arabic OOS prompts only because it rejects almost every French and Arabic prompt (acceptance 0.00 to 0.09).

#### modernbert/modernbert_multihead

Train 11.6 s, peak VRAM 2952 MiB allocated / 3202 MiB reserved. Export (fp32, quantization none): fp32 max|diff| 2.83e-05, model.onnx 569 MiB, ORT CPU p50 16.4 ms per query.

| scorer | test acc | test macro-F1 | test e2e acc | para_en acc | fr acc | ar acc | fr+ar in-scope accept | fr+ar OOS rej | held-out intent OOS rej (mean of folds) | fold test acc (mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| trained head | 0.905 | 0.903 | 0.841 | 1.000 | 0.786 | 0.143 | 0.929 | 0.000 | 0.414 | 0.897 |
| kNN, base encoder | 0.810 | 0.808 | 0.750 | 1.000 | 0.429 | 0.179 | 0.000 | 1.000 | 0.281 | 0.841 |
| kNN, fine-tuned encoder | 0.905 | 0.903 | 0.886 | 1.000 | 0.821 | 0.143 | 0.393 | 0.750 | 0.476 | 0.913 |
| trained head, oos head only | | | | | | | | | 0.148 | |

Held-out intent OOS rejection per fold (head / kNN base / kNN tuned): analytics 0.500/0.167/0.533, chat 0.433/0.600/0.933, code 0.400/0.400/0.400, extraction 0.500/0.133/0.233, reasoning 0.533/0.333/0.700, summarize 0.333/0.133/0.200, translate 0.200/0.200/0.333

#### modernbert/setfit

Train 50.6 s, peak VRAM 2876 MiB allocated / 3492 MiB reserved. Export (fp32, quantization none): fp32 max|diff| 6.68e-05, model.onnx 569 MiB, ORT CPU p50 15.9 ms per query.

| scorer | test acc | test macro-F1 | test e2e acc | para_en acc | fr acc | ar acc | fr+ar in-scope accept | fr+ar OOS rej | held-out intent OOS rej (mean of folds) | fold test acc (mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| trained head | 0.881 | 0.862 | 0.841 | 1.000 | 0.750 | 0.071 | 0.839 | 0.250 | 0.243 | 0.913 |
| kNN, base encoder | 0.810 | 0.808 | 0.750 | 1.000 | 0.429 | 0.179 | 0.000 | 1.000 | 0.281 | 0.841 |
| kNN, fine-tuned encoder | 0.905 | 0.886 | 0.841 | 1.000 | 0.750 | 0.071 | 0.625 | 0.375 | 0.252 | 0.913 |

Held-out intent OOS rejection per fold (head / kNN base / kNN tuned): analytics 0.033/0.167/0.067, chat 0.300/0.600/0.333, code 0.467/0.400/0.500, extraction 0.067/0.133/0.067, reasoning 0.167/0.333/0.133, summarize 0.400/0.133/0.433, translate 0.267/0.200/0.233

#### mmbert/modernbert_multihead

Train 32.2 s, peak VRAM 7731 MiB allocated / 7974 MiB reserved. Export (fp32, quantization none): fp32 max|diff| 8.85e-06, model.onnx 1171 MiB, ORT CPU p50 16.3 ms per query.

| scorer | test acc | test macro-F1 | test e2e acc | para_en acc | fr acc | ar acc | fr+ar in-scope accept | fr+ar OOS rej | held-out intent OOS rej (mean of folds) | fold test acc (mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| trained head | 0.905 | 0.903 | 0.795 | 1.000 | 1.000 | 0.857 | 0.929 | 0.250 | 0.352 | 0.952 |
| kNN, base encoder | 0.881 | 0.884 | 0.864 | 1.000 | 1.000 | 0.786 | 0.089 | 1.000 | 0.290 | 0.925 |
| kNN, fine-tuned encoder | 0.905 | 0.903 | 0.909 | 1.000 | 1.000 | 0.857 | 0.857 | 1.000 | 0.576 | 0.952 |
| trained head, oos head only | | | | | | | | | 0.224 | |

Held-out intent OOS rejection per fold (head / kNN base / kNN tuned): analytics 0.067/0.333/0.200, chat 0.333/0.267/0.933, code 0.533/0.400/0.500, extraction 0.467/0.100/0.467, reasoning 0.500/0.333/0.667, summarize 0.467/0.133/0.500, translate 0.100/0.467/0.767

#### mmbert/setfit

Train 57.5 s, peak VRAM 5532 MiB allocated / 6722 MiB reserved. Export (fp32, quantization none): fp32 max|diff| 1.91e-05, model.onnx 1171 MiB, ORT CPU p50 16.2 ms per query.

| scorer | test acc | test macro-F1 | test e2e acc | para_en acc | fr acc | ar acc | fr+ar in-scope accept | fr+ar OOS rej | held-out intent OOS rej (mean of folds) | fold test acc (mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| trained head | 0.881 | 0.875 | 0.864 | 1.000 | 0.929 | 0.643 | 0.875 | 0.125 | 0.357 | 0.940 |
| kNN, base encoder | 0.881 | 0.884 | 0.864 | 1.000 | 1.000 | 0.786 | 0.089 | 1.000 | 0.290 | 0.925 |
| kNN, fine-tuned encoder | 0.881 | 0.875 | 0.841 | 1.000 | 0.964 | 0.643 | 0.821 | 0.250 | 0.390 | 0.940 |

Held-out intent OOS rejection per fold (head / kNN base / kNN tuned): analytics 0.367/0.333/0.433, chat 0.467/0.267/0.567, code 0.500/0.400/0.367, extraction 0.233/0.100/0.333, reasoning 0.300/0.333/0.400, summarize 0.467/0.133/0.467, translate 0.167/0.467/0.167

### Summary

| | test acc | fr / ar acc | fr+ar accept | held-out intent OOS rej | ORT CPU p50 | fp32 model.onnx | train | peak VRAM (alloc) |
|---|---|---|---|---|---|---|---|---|
| ModernBERT multi-head: head | 0.905 | 0.79 / 0.14 | 0.93 | 0.41 | 16.4 ms | 569 MiB | 11.6 s | 2.9 GiB |
| ModernBERT multi-head: kNN on tuned encoder | 0.905 | 0.82 / 0.14 | 0.39 | 0.48 | | | | |
| mmBERT multi-head: head | 0.905 | 1.00 / 0.86 | 0.93 | 0.35 | 16.3 ms | 1171 MiB | 32.2 s | 7.6 GiB |
| **mmBERT multi-head: kNN on tuned encoder** | **0.905** | **1.00 / 0.86** | 0.86 | **0.58** | | | | |
| mmBERT base encoder, kNN (no training) | 0.881 | 1.00 / 0.79 | 0.09 | 0.29 | | | | |

## Conclusions

1. **The trained head is not worth shipping on this data.**
   - No trained head clearly beats the kNN over the same fine-tuned encoder. In all four runs the head's test accuracy is equal or lower, and so are its held-out-intent rejection and its French and Arabic OOS rejection.
   - Where a head is ahead, it is either within one example (mmBERT SetFit end-to-end accuracy, 0.864 against 0.841) or it comes from accepting nearly everything. The ModernBERT multi-head accepts 93% of French and Arabic in-scope prompts, but rejects none of their OOS prompts.
   - This is note 02's "simple routers are hard to beat" result, reproduced on Caliban's own taxonomy.
   - A Stage-2 head would add a second ONNX model and a second calibration to core for no measured gain.
2. **The promising path is a fine-tuned mmBERT-base encoder used as the Stage-1 kNN embedder.**
   - mmBERT plus kNN on the fine-tuned encoder is the best configuration overall:
     - test 0.905;
     - French 1.00, Arabic 0.86;
     - French and Arabic in-scope acceptance 0.86 with all 8 OOS prompts rejected;
     - held-out-intent rejection 0.58, the best of all runs.
   - Fine-tuning is what makes the multilingual space usable: the untuned mmBERT kNN classifies French well (1.00) but its calibrated gate accepts only 9% of French and Arabic prompts.
   - ModernBERT-base is English-centric: Arabic accuracy is 0.14 for every scorer.
   - `caliban-ml router export --embedder` now exports the fine-tuned encoder alone as an `embedder` artifact for this use; see below for what core needs.
3. **The data set is too small to decide anything finer.**
   - 42 in-scope test examples means one example is 2.4 points, and there are only 2 OOS test examples. The fold means average 7 folds of 30 held-out utterances each.
   - Every run used one seed, and note 02 cites about 5% label flips between runs on real routing data.
   - Most differences between the trained configurations are within one or two examples.
   - The next step is more exemplars per intent (real tenant data, paraphrases, more languages) and several seeds before any further model choice.
4. **Out-of-scope rejection is the main gap.**
   - The best held-out-intent rejection is 0.58: a whole unseen intent still passes the gate 42% of the time, at about 95% in-scope acceptance.
   - The explicit OOS head is worse than the max-probability gate (0.15 to 0.22 held-out rejection): with 6 OOS training examples it has nothing to learn from.
   - Per note 02 ([arXiv 2410.01627](https://arxiv.org/abs/2410.01627), [arXiv 2507.01541](https://arxiv.org/abs/2507.01541)), uncertain and novel traffic should go to the Stage-3 LLM fallback with top-k candidate routes, and OOS needs its own data (many more and more varied OOS examples, ideally from real traffic).

## INT8

**The failure.** The GPU smoke test exported the fine-tuned ModernBERT-base multi-head with ORT's default dynamic INT8 (`quantize_dynamic(weight_type=QInt8)`). Agreement with torch was 30% (gate 99.5%), while the fp32 graph matched torch to 3e-5. So the export was correct and the quantization broke the model.

**The diagnosis**, from the graph and from ModernBERT's structure; it could not be re-measured on the real model here:

- ORT's defaults quantize the embedding Gather and every weight MatMul with one per-tensor scale.
- They also quantize each MatMul's *input activation* with one dynamic per-tensor uint8 scale.
- ModernBERT-family encoders are 22 layers deep and carry large-magnitude outlier dimensions in the residual stream and in the GeGLU outputs that feed the MLP down-projection (`mlp/Wo`).
- With per-tensor scales the outliers take up most of the 256 levels and flatten everything else, and the error compounds layer by layer.
- The attention score and value products are not quantized by ORT's dynamic defaults in this graph: we checked the op counts. So attention itself is not the cause.

**The change.** `int8_dynamic` now quantizes weight MatMuls only (`MatMulConstBOnly`), per output channel. It keeps the following in fp32: embeddings, LayerNorms, attention products, the MLP down-projections and the classifier heads (`INT8_DYNAMIC_OPTIONS`, `INT8_EXCLUDE_PATTERNS` in `router/export.py`).

On a tiny ModernBERT with injected 100x residual outliers (`tests/test_stage2_integration.py::test_int8_settings_survive_residual_outliers`):

| | min cosine to fp32 |
|---|---|
| new settings | 0.9999 or better |
| ORT's defaults | about 0.95 |

**Not proven on the real model.** Activations are still quantized per tensor, which is the classic outlier failure (LLM.int8). So:

- the default export quantization is now **`none`** (fp32);
- `int8_dynamic` stays opt-in and is gated on parity;
- every fp32 run of `router_stage2_eval.py` measures argmax agreement for both the new settings and ORT's defaults on the real models (the "INT8 probe" line in the report).

If the probe still fails, the next step is static QDQ quantization with a calibration set, or keeping fp32. On CPU, fp32 already meets note 02's Stage-2 budget: about 16 ms per query with 4 threads, against a 5 to 20 ms target.

**Size.** INT8 matters most for size. About 750 MiB of mmBERT's 1171 MiB fp32 graph is the 256k-token embedding table (256000 x 768 x 4 bytes), which the new settings leave in fp32. Shrinking it would need weight-only quantization of that Gather or vocabulary pruning.

**fp16 export was removed**, not implemented. Core runs ONNX Runtime on CPU, where fp16 kernels are sparse and slower than fp32, so fp16 only makes sense for a GPU execution provider. The manifest schema still accepts `quantization: "fp16"` for other producers.

## Exporting the fine-tuned encoder as an embedder

`caliban-ml router export --embedder` (`router.export.export_embedder`) writes an `embedder` artifact:

- `model.onnx` with output `embedding` [batch, hidden], already mean-pooled and L2-normalised in the graph;
- `tokenizer.json`;
- `config.json` (`type = "onnx_embedder"`, `pad_id`);
- a manifest with `embedding.pooling = "none"` and `normalize = true`.

This fits the existing manifest kinds, so no schema change is needed. Parity is checked as minimum torch-vs-ONNX cosine similarity. `router embed --embedder-artifact` and `router knn-eval` then fit a Stage-1 calibration in exactly that space, and the calibration `requires` the embedder.

**What core needs to use it:**

- Core's Stage-1 embedder today is provider-backed: the gateway adapts its HTTP `Embedder` to `caliban_route::PromptEmbedder`.
- Core needs an in-process `PromptEmbedder` backed by `ort` and `tokenizers` that loads an `embedder` artifact. It should:
  - verify it like the PII NER loader;
  - bind `input_ids` / `attention_mask` and `embedding.output` by name;
  - truncate to `tokenizer.max_length` and pad with `pad_id`;
  - apply `embedding.pooling` and `normalize` (both no-ops for this artifact);
  - use `name@version` as `space_id`.
- `routing.embedder_artifact` already names the embedder a calibration must match (`load_knn_calibration` enforces it), so the config side exists.
- The exemplar index is rebuilt per vector space already.
- Cost: about 16 ms per prompt on CPU (fp32), and 1.2 GB of RAM for mmBERT.

## Artifact and how core should load a Stage-2 head

Kept for completeness. Given the results above, this is not the recommended next step for core.

`caliban-ml router export` writes an `intent_head` artifact:

- `manifest.json`;
- `model.onnx` (opset 17, fp32 by default);
- `tokenizer.json` (truncation and padding stripped);
- `labels.json`;
- `config.json` (`type = "onnx_classifier"`, `primary_head`, `pooling`, `pad_id`, `max_length`).

Inputs are `input_ids` and `attention_mask` (int64, `[batch, seq]`). Outputs are `intent_logits` (`[batch, n_labels]`) and, for the multi-head backbone, `oos_logits` (`[batch, 1]`). Calibration semantics are the Stage-1 ones (`oos_score = max_probability`).

Core's `load_knn_calibration` already refuses an `intent_head` that carries `onnx`. A Stage-2 loader would:

- follow the PII NER loader pattern;
- apply `softmax(intent_logits / T)`, the OOS gate and the per-label thresholds;
- run only when Stage 1 abstains, within the routing budget;
- fall back to the keyword rules on timeout or error.

## Notes for running on AWS

- **HOME must be set.** pip, torch and the Hugging Face libraries cache under `$HOME`. AWS SSM Run Command starts shells without `HOME`. `scripts/stage2_gpu_run.sh` now derives it from the passwd entry, falling back to `.home/` in the repo, which is git-ignored. Set it yourself when calling the Python scripts directly.
- **Trace warnings.** The TorchScript export warns that transformers' mask code converts a tensor to a Python boolean during tracing. Parity now also runs unpadded single-text batches (core's hot path) besides padded mixed-length batches, so a trace that baked in one mask path would fail the export.
