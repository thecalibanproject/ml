# caliban-ml

Offline training, evaluation and export for the small models that Caliban's Rust data plane
runs in-process: the embedder, the intent/router heads, the PII NER model, and the router
quality profiles.

**Contract with `core`: this repo only ever hands over ONNX files plus JSON** (an artifact
manifest, and profile tables for routers). The Rust data plane loads them with `ort`
(ONNX Runtime) or `candle`. Python never runs in the request path, and nothing here is
imported by `core`.

Everything runs offline. Training may download a base model **once**, onto the training box,
from a pinned revision with a checked licence. After that, artifacts are built, verified,
signed and shipped. Nothing downloads anything at runtime.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'        # light deps only: pydantic, pyyaml, numpy, httpx, typer
.venv/bin/ruff check .
.venv/bin/pytest

# heavy extras (training box only, never in CI or the data plane)
.venv/bin/pip install -e '.[train]'      # torch, transformers, sentence-transformers, setfit, datasets, gliner
.venv/bin/pip install -e '.[export]'     # onnx, onnxruntime, optimum
.venv/bin/pip install -e '.[presidio]'   # presidio-analyzer (needs a locally installed spaCy model)
```

CLI (`caliban-ml --help`):

| Command | What it does |
|---|---|
| `caliban-ml manifest verify <dir> [--strict] [--signature minisign --pubkey k.pub]` | Validates `manifest.json` (schema, kind rules, licence policy) and checks every file's size and SHA-256. `--strict` also fails on files the manifest doesn't list. |
| `caliban-ml manifest schema [--check]` | Regenerates (or checks) `schemas/*.schema.json` from the pydantic models. |
| `caliban-ml router embed --dataset data/intents/sample.yaml --model-dir <local ST model> --out-prefix runs/intents` | Splits the intent exemplars and embeds them into `.npz` files (`[train]`). |
| `caliban-ml router knn-eval --train a.npz --val b.npz [--test c.npz] --out calib.json` | Fits the pure-numpy kNN baseline, calibrates the temperature, and picks the OOS gate and per-label exit thresholds. Prints metrics. |
| `caliban-ml router profile results.jsonl --name p --version 1.0.0 --out artifacts/router_profile/p/1.0.0` | Turns probe results into a UniRoute-style `router_profile` artifact. |
| `caliban-ml probes run data/probes/sample.yaml --base-url http://localhost:8080/v1 --model <id> --out results/x.jsonl` | Runs a probe set against any OpenAI-compatible endpoint and scores the replies. |
| `caliban-ml pii eval data/pii/sample.jsonl --detector regex` | Reports span-level P/R/F1 per entity type, in `exact`, `overlap` and `redaction` modes. |

## Layout

```
schemas/                       JSON Schemas consumed by core (generated, drift-tested)
  artifact-manifest.schema.json
  router-profile.schema.json
data/
  intents/sample.yaml          intent exemplars + OOS examples (synthetic)
  probes/sample.yaml           10 probes x 5 clusters (synthetic)
  pii/sample.jsonl             20 docs with gold spans (obviously fake data)
src/caliban_ml/
  artifacts/                   manifest model, licence policy, hashing/verify, signing hook
  router/                      intent dataset, numpy kNN + calibration, UniRoute profiles,
                               train/export stubs ([train]/[export])
  probes/                      probe-set format, checkers, OpenAI-compatible runner, judge hook
  pii/                         span metrics, dataset format, detector adapters
  cli.py                       typer CLI
artifacts/                     local build output (binaries are git-ignored)
```

## The artifact contract

An artifact is a directory, published at `artifacts/<kind>/<name>/<version>/`:

```
manifest.json            the contract (schemas/artifact-manifest.schema.json)
manifest.json.minisig    detached signature over manifest.json (release builds)
model.onnx               embedder / intent_head / pii_ner
tokenizer.json           HF `tokenizers` JSON (Rust `tokenizers` crate); no trust_remote_code
labels.json, config.json, profile.json, ...
```

| Field | Meaning |
|---|---|
| `manifest_version` | Currently `1`. Core refuses any version it doesn't know. |
| `kind` | `embedder`, `intent_head`, `pii_ner` or `router_profile`. Each kind has its own required fields; see below. |
| `name`, `version` | A slug and a semver. Bump the **major** version whenever tensor names, shapes, labels or semantics change. |
| `files[]` | `path` (relative POSIX; `..` and absolute paths are rejected), `sha256`, `size_bytes`, `role`. The manifest and its signatures are not listed. |
| `onnx` | `file`, `opset`, `inputs[]` and `outputs[]` as `{name, dtype, shape}` (a string dim is symbolic, e.g. `batch`), `quantization`, and optional `heads[]` that map each output tensor to its labels for multi-head models. |
| `tokenizer` | `file`, `max_length`, `truncation_side`. |
| `embedding` | Embedder only: `dim`, `output`, `pooling`, `normalize`, and query/passage prefixes. |
| `labels` | Label space of the primary head (intents, or PII entity types). |
| `calibration` | `temperature`, per-label exit `thresholds` (with a `default_threshold`), an optional `margin_threshold`, and `oos_threshold` + `oos_score` (`top1_similarity`, `max_probability` or `oos_head_probability`). |
| `metrics` | Eval numbers (finite floats), e.g. `val_end_to_end_accuracy` or `span_f1_overlap`. |
| `data_card` | Training/eval datasets (source, revision, **licence**, size, synthetic flag), languages, intended use, limitations, and a `contains_personal_data` flag. |
| `base_model` | `id`, the pinned `revision` and the **licence**. Required for every ONNX-bearing kind. |
| `licence_overrides[]` | A recorded human sign-off (who, when, why) for a licence that needs review. |
| `requires[]` | Other artifacts this one depends on, e.g. a kNN intent head or a profile with centroids requires the exact embedder whose vector space it uses. |

Kind rules (enforced in `ArtifactManifest`):
- `embedder`: `onnx`, `tokenizer`, `embedding` and `base_model` are required.
- `pii_ner`: `onnx`, `tokenizer`, `labels` and `base_model` are required.
- `intent_head`: `labels` and `calibration` are required, plus either an ONNX graph (Stage-2 classifier) or a `requires` embedder (Stage-1 kNN calibration, whose exemplars are tenant data in the HNSW index and are not shipped).
- `router_profile`: needs a `profile.json` with role `profile`, and carries no ONNX.

What the Rust loader must do: (1) verify the signature over `manifest.json` when the
deployment requires it, (2) check the size and SHA-256 of every listed file before loading,
(3) reject an unknown `manifest_version`, (4) bind ONNX tensors **by name** as declared,
never by position, (5) check that the `requires` artifacts are loaded at the exact versions
listed, and (6) hot-swap via `ArcSwap` only after all of the above pass.

### Stage-1 kNN semantics (shared with Rust)

```
neighbours  = top-k exemplars by cosine similarity (k in config.json)
w_j         = exp(s_j / T)                                    T = calibration.temperature
p_c         = (1 - eps) * sum_{j: y_j = c} w_j / sum_j w_j + eps / C
OOS         if s_top1 < calibration.oos_threshold             (explicit decision, never argmax)
exit        if p_max >= thresholds[label] (else default_threshold); otherwise go to Stage 2
```

`knn-eval` picks `T` by minimising NLL on held-out in-scope data. It sets the OOS gate either
to the largest threshold that keeps the target in-scope acceptance (`recall`), or by Youden's
J (`youden`). It sets each label's exit threshold to the lowest confidence that still reaches
the target precision; OOS queries that slip past the gate count as errors, and a threshold of
`1.0` means "never exit at Stage 1".

### Router profiles (UniRoute-style)

`profile.json` (`schemas/router-profile.schema.json`) holds a list of clusters, plus one row
per model with `quality`, `raw_mean`, `counts`, `prior` and `n_errors`, aligned to the
clusters:

```
quality[m][c] = (sum of scores[m, c] + w * prior[m]) / (count[m, c] + w)      w = prior_weight
```

`prior[m]` is the model's mean over all of its probes. Failed calls (timeouts, 5xx, rate
limits) are excluded by default because they are not a quality signal. If clusters carry
`centroid`s, the profile must name the `embedder` they came from (`router.profile.kmeans`
builds them). A new BYO model becomes routable as soon as its row exists, with no router
retraining.

### Signing

Only `manifest.json` is signed. Because it pins every file's SHA-256, the signature covers
the whole artifact. Both supported schemes work air-gapped: minisign, the default for
bundles, and cosign with a key pair and `--tlog-upload=false`. The exact commands are in
`src/caliban_ml/artifacts/signing.py`. Private keys live in the release pipeline's secret
store, and `*.key` is git-ignored. The public key ships in the bundle and is pinned in the
Caliban config.

## Training flow

```
data (intents YAML / probe results / PII JSONL; licence-checked, revision-pinned)
  -> train            [train]   SetFit or ModernBERT multi-head (intent, difficulty, oos, ...)
  -> calibrate                  temperature + OOS gate + per-label exit thresholds (same code as knn)
  -> export_onnx      [export]  opset >= 17, dynamic batch/seq, INT8 dynamic quantization
  -> parity check               torch vs ORT logits (max abs diff, argmax agreement) -> metrics
  -> build manifest             hashes, tensor names, labels, calibration, data card, licence
  -> manifest verify --strict
  -> sign (release pipeline)
  -> publish artifacts/<kind>/<name>/<version>/ to the internal artifact store
```

The training and export entry points (`router.train.train_intent_head` and
`router.export.export_onnx`) are skeletons with `TODO(...)` markers. The pure-Python parts are
implemented and tested: manifest building, the licence gate, calibration and threshold
selection, profiles, the probe runner and PII metrics.

## How artifacts get into the on-prem bundle (`deploy/`)

This is the proposed flow; the bundle scripts live in the `deploy` repo:

1. The release pipeline pins a set of artifact versions (e.g. `models.lock`: kind, name,
   version and manifest SHA-256).
2. The air-gap bundle script fetches each artifact directory from the internal artifact
   store, runs `caliban-ml manifest verify --strict --signature minisign --pubkey ...`, and
   copies it into the bundle under `models/<kind>/<name>/<version>/`.
3. The bundle also carries the signing public key and the `schemas/` it was validated
   against. `caliban standalone` and the data plane load models only from that directory and
   re-verify the hashes and signature at startup and on hot-swap.
4. Customer-specific artifacts (tenant intent heads, BYO-model profiles) are produced on site
   by the same CLI. They go through the same verify step and are signed with the customer's
   own key if they require it.

No artifact is ever fetched from a model hub at runtime.

## Licensing rule for base models and data

Caliban ships on-prem, commercially. The default bundle must contain only weights and
training data that allow commercial use and redistribution. `artifacts/licenses.py` enforces
this on every manifest:

- **allowed**: Apache-2.0, MIT, BSD-2/3, ISC, CC0, Unlicense, CC-BY-3.0/4.0, Zlib, BSL-1.0
  (Boost), PSF-2.0, and `caliban-internal` (first-party data).
- **review required**, which needs a `licence_overrides` entry with approver, date and
  reason: copyleft (MPL, LGPL, GPL, AGPL, CC-BY-SA), OpenRAIL variants, Llama and Gemma
  licences, the NVIDIA Open Model License, `other`, `unknown`, and anything unrecognised.
- **denied**: any `*-nc-*` licence (CC-BY-NC, CC-BY-NC-SA and so on), "non-commercial",
  "research only", "academic only" or "evaluation only". An override is accepted only when
  validation also runs with `--allow-noncommercial`, which the default bundle build never
  uses.

The same rule applies to **training and eval datasets** in the data card, not just to
weights. Check each model at the exact revision you pin. Model cards change, and a model's
own licence says nothing about the licence of its training data.

| Candidate | Use | Licence (per upstream card; re-verify at the pinned revision) | Status |
|---|---|---|---|
| `answerdotai/ModernBERT-base` / `-large` | Stage-2 multi-head backbone | Apache-2.0 | OK |
| GLiNER2-PII (default L1 PII model) | PII NER | Check the card and the fine-tuning data. The GLiNER2 family is published under Apache-2.0. | OK once verified |
| `nvidia/gliner-pii` | PII NER alternative | NVIDIA Open Model License (commercial use allowed per the research notes) | Review + override |
| `urchade/gliner_base`, `urchade/gliner_multi` (v1) | PII NER | CC-BY-NC-4.0 | **Denied** |
| `urchade/gliner_*-v2.1` | PII NER | Apache-2.0 on the card | OK once verified |
| OpenAI Privacy Filter | Secrets/PII, English | Apache-2.0 (ONNX export not yet confirmed) | OK once verified |
| `BAAI/bge-small-en-v1.5`, `intfloat/multilingual-e5-small` | Embedder | MIT | OK, but check training-data terms |
| `katanemo/Arch-Router-1.5B` | Stage-3 LLM router | Custom terms (the paper says CC-BY-4.0) | Review + override |
| Llama / Gemma family judges | Probe judge | Vendor licences with use restrictions | Review + override |

Judge models used only on the training box (to score probes) are not redistributed. They
still need a licence that permits the use, and on-prem probe runs must use a judge the
customer is allowed to run.

## Probes (BYO-model onboarding)

```bash
export CALIBAN_PROBE_API_KEY=cal_xxx
caliban-ml probes run data/probes/sample.yaml \
  --base-url http://localhost:8080/v1 --model acme-local-llama --model gpt-4.1-mini \
  --judge-base-url http://vllm-judge:8000/v1 --judge-model local-judge \
  --out results/sample.jsonl
caliban-ml router profile results/sample.jsonl --name acme-profile --version 1.0.0 \
  --out artifacts/router_profile/acme-profile/1.0.0
```

Probe the Caliban data plane with an explicit model ID, not `caliban/auto`, so the router is
bypassed. Alternatively, probe a local vLLM or llama.cpp server directly. The runner only
calls the configured endpoints. Use `--no-store-output` for sensitive tenant probe sets.
Check types are `exact`, `contains`, `regex`, `json` and `judge`.

## PII evaluation

The dataset is JSONL of `{"id", "text", "spans": [{"start", "end", "label"}]}`, where offsets
are Python string (code-point) indices and the end is exclusive. There are three match
modes: `exact`, `overlap` (same label, intersecting ranges, one-to-one), and `redaction`
(intersecting ranges with the label ignored). `redaction` is the number that matters for
leakage. Here is the regex L0 baseline on `data/pii/sample.jsonl`:

```
regex / overlap: P=1.000 R=0.633 F1=0.776   (structured types 1.0; PERSON/LOCATION 0.0: the L1 NER's job)
```

## TODOs

- `TODO(train/setfit)`, `TODO(train/modernbert_multihead)`: real training loops in `router/train.py`.
- `TODO(export)`: optimum/torch.onnx export, INT8 quantization and the parity check in `router/export.py`.
- `TODO(embed/onnx)`: ONNX embedder backend in `router/embed.py`, so calibration runs in the exact shipped vector space.
- `TODO(pii/gliner2)`, `TODO(pii/gliner)`, `TODO(pii/presidio)`: a GLiNER2 adapter, sliding windows, and an explicit Presidio NLP engine config.
- `TODO(judge)`: judge calibration against human labels, and bias controls.
- `TODO(signing)`: a CI integration test for minisign/cosign, plus `minisign-verify` in the Rust loader.
- Shared fixtures: publish example manifests and profiles for `core`'s loader tests.
