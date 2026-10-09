<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/thecalibanproject/website/main/public/brand/logo-white.svg">
  <img src="https://raw.githubusercontent.com/thecalibanproject/website/main/public/brand/logo.svg" alt="Caliban" width="200">
</picture>

# Caliban ML

Offline tooling that packages, evaluates and verifies the small models Caliban runs in-process: the PII NER model, intent/router heads, embedders and router quality profiles.

[core](https://github.com/thecalibanproject/core) · [deploy](https://github.com/thecalibanproject/deploy) · [docs](https://github.com/thecalibanproject/docs) · [sdk-python](https://github.com/thecalibanproject/sdk-python) · [website](https://github.com/thecalibanproject/website)

Caliban is a sovereign AI gateway: one OpenAI- and Anthropic-compatible endpoint for a company, with intent routing, PII pseudonymisation, an ontology layer, caching, agents ("nodes") and on-prem open-weight models. Its core is a single Rust binary. This repo is the offline side of the models that binary loads.

> **Status:** in active development with design partners. The PII NER packaging path, the artifact contract, the licence gate, kNN calibration, router profiles, the probe runner and the PII metrics are implemented and tested. Training and ONNX export for Caliban's own classifiers are still skeletons (see [Current state](#current-state)).

## Contents

- [What this repo does](#what-this-repo-does)
- [Model weights are not in this repo](#model-weights-are-not-in-this-repo)
- [Quickstart](#quickstart)
- [PII NER: fetch, verify, use from core](#pii-ner-fetch-verify-use-from-core)
- [Router and intent classifiers](#router-and-intent-classifiers)
- [The artifact contract (manifests and model cards)](#the-artifact-contract-manifests-and-model-cards)
- [Licence policy for weights and data](#licence-policy-for-weights-and-data)
- [Probes (BYO-model onboarding)](#probes-byo-model-onboarding)
- [PII evaluation](#pii-evaluation)
- [Getting artifacts into an on-prem bundle](#getting-artifacts-into-an-on-prem-bundle)
- [Repository layout](#repository-layout)
- [Current state](#current-state)
- [Licence](#licence)

## What this repo does

**The contract with [core](https://github.com/thecalibanproject/core) is ONNX files plus JSON:** an artifact manifest for every model, and profile tables for routers. The Rust data plane loads them with ONNX Runtime (`ort`). Python never runs in the request path, and nothing here is imported by core.

Everything runs offline. A base model may be downloaded **once**, onto a build or training machine, from a pinned revision with a checked licence. After that, artifacts are verified, signed and shipped. Nothing downloads anything at runtime.

## Model weights are not in this repo

`artifacts/` is git-ignored apart from `artifacts/.gitkeep`. No `.onnx`, `.safetensors`, `.bin`, `.npy`/`.npz` file or fetched `artifacts/pii_ner/` directory is committed, so cloning this repo gives you **no model weights and no generated manifests**. You produce them locally with the scripts and CLI below:

| Artifact | How to produce it |
|---|---|
| PII NER (`pii_ner`) | `python3 scripts/fetch_pii_ner.py [--preset ...]` downloads pinned upstream ONNX files from Hugging Face and writes `manifest.json`. |
| Router profile (`router_profile`) | `caliban-ml probes run ...` then `caliban-ml router profile ...` (JSON only, no weights). |
| Stage-1 kNN calibration (`intent_head`) | `caliban-ml router embed ...` then `caliban-ml router knn-eval ...` (needs a local sentence-transformers model). |
| Stage-2 classifier, ONNX embedder | Not yet: the training and export entry points are stubs. |

The upstream models keep their own licences. See [Licence policy](#licence-policy-for-weights-and-data) and the model notes in core's [`crates/caliban-pii/MODELS.md`](https://github.com/thecalibanproject/core/blob/main/crates/caliban-pii/MODELS.md).

## Quickstart

Requires Python 3.11+.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'        # light deps only: pydantic, pyyaml, numpy, httpx, typer
.venv/bin/ruff check .
.venv/bin/pytest

# heavy extras (build/training machine only, never in CI or the data plane)
.venv/bin/pip install -e '.[train]'      # torch, transformers, sentence-transformers, setfit, datasets, gliner
.venv/bin/pip install -e '.[export]'     # onnx, onnxruntime, optimum
.venv/bin/pip install -e '.[presidio]'   # presidio-analyzer (needs a locally installed spaCy model)
```

CLI (`caliban-ml --help`):

| Command | What it does |
|---|---|
| `caliban-ml manifest verify <dir> [--strict] [--signature minisign --pubkey k.pub]` | Validates `manifest.json` (schema, kind rules, licence policy) and checks every file's size and SHA-256. `--strict` also fails on files the manifest doesn't list. `--json` gives machine-readable output. |
| `caliban-ml manifest schema [--check]` | Regenerates (or checks) `schemas/*.schema.json` from the pydantic models. |
| `caliban-ml router embed --dataset data/intents/sample.yaml --model-dir <local ST model> --out-prefix runs/intents` | Splits the intent exemplars and embeds them into `.npz` files (`[train]`). |
| `caliban-ml router knn-eval --train a.npz --val b.npz [--test c.npz] --out calib.json` | Fits the pure-numpy kNN baseline, calibrates the temperature, and picks the OOS gate and per-label exit thresholds. Prints metrics. |
| `caliban-ml router profile results.jsonl --name p --version 1.0.0 --out artifacts/router_profile/p/1.0.0` | Turns probe results into a UniRoute-style `router_profile` artifact. |
| `caliban-ml probes run data/probes/sample.yaml --base-url http://localhost:8080/v1 --model <id> --out results/x.jsonl` | Runs a probe set against any OpenAI-compatible endpoint and scores the replies. |
| `caliban-ml pii eval data/pii/sample.jsonl --detector regex` | Reports span-level P/R/F1 per entity type, in `exact`, `overlap` and `redaction` modes. Detectors: `regex`, `gliner` (with `--model-dir`), `presidio`. |

## PII NER: fetch, verify, use from core

Caliban's PII pipeline is tiered: L0 is patterns and dictionaries in Rust, L1 is a small ONNX token-classification (NER) model run in-process. **This repo does not train the L1 model.** It packages an existing, permissively licensed model that is already exported to ONNX (and already int8-quantised upstream), pins it by revision and hash, and writes the Caliban manifest for it.

### Presets

`scripts/fetch_pii_ner.py` is stdlib-only (no `huggingface_hub` or `onnx` needed). List the presets with `python3 scripts/fetch_pii_ner.py --list`:

| Preset | Upstream (pinned revision) | Quantisation | Weights licence | Notes |
|---|---|---|---|---|
| `nym-pii-multilingual-small-int8` (default) | `Wismut/nym-pii-multilingual-small` @ `4348999c...`, `int8/` | `int8_dynamic` (int8 embeddings, fp16 body weights, fp32 compute) | MIT | mmBERT-small base, about 23 languages, 40 PII entity types in IOB2 (81 labels). |
| `nym-pii-multilingual-small-edge-int8` | same repo and revision, `edge-int8/` | `int8_dynamic` (full dynamic int8) | MIT | Smallest and fastest variant; the upstream card reports lower out-of-domain accuracy. |
| `nym-pii-multilingual-small-fp32` | same repo and revision, root `model.onnx` | none | MIT | fp32 reference export (about 429 MB). |
| `distilbert-ner-en` | `dslim/distilbert-NER` @ `dfa2838a...` | none | Apache-2.0 | English CoNLL-2003 PER/ORG/LOC/MISC fallback, not PII-specific. |

### What the script does

```bash
python3 scripts/fetch_pii_ner.py                                   # default preset
python3 scripts/fetch_pii_ner.py --preset distilbert-ner-en
# -> artifacts/pii_ner/<name>/<version>/{manifest.json, model.onnx, tokenizer.json, config.json}
.venv/bin/caliban-ml manifest verify artifacts/pii_ner/nym-pii-multilingual-small-int8/3.0.0 --strict
```

1. Downloads `model.onnx`, `tokenizer.json` and `config.json` from `https://huggingface.co/<repo>/resolve/<pinned revision>/<file>` (sends `HF_TOKEN` if set).
2. Refuses any file whose size, SHA-256 (LFS files) or git blob id (small files) does not match the value pinned in the script.
3. Reads the ONNX opset, inputs and outputs from the graph itself, takes the BIO labels from `config.json` `id2label`, and checks the logits width against the label count.
4. Writes `manifest.json` (kind `pii_ner`) with file hashes, tensor names, labels, a data card and the base-model licence, then runs `caliban-ml manifest verify --strict` if the CLI is installed.
5. Exits with status **3** while a licence review is pending. The files are still usable for development, but the artifact must not be bundled until a reviewer records a sign-off with `--approved-by 'Name <email>' --approval-reason '...'`.

### Recorded licence approval

The nym models' weights are MIT, but their fine-tuning data includes about 77.5k Wikipedia passages (CC-BY-SA-4.0 text, auto-labelled by an Apache-2.0 model, not redistributed). The licence policy classifies CC-BY-SA as *review required*. For Caliban's own bundle, Elie Sfeir approved this item on 2026-10-03, and the approval is recorded in the `nym-pii-multilingual-small-int8` 3.0.0 manifest as:

```json
"licence_overrides": [{
  "subject": "wikipedia-llm-labelled",
  "licence": "cc-by-sa-4.0",
  "approved_by": "Elie Sfeir <elie@internalizable.dev>",
  "approved_on": "2026-10-03",
  "reason": "Approved 2026-10-03 for commercial on-prem redistribution: weights are MIT; the CC-BY-SA-4.0 Wikipedia text was used only for fine-tuning and is not redistributed."
}]
```

Because generated manifests are git-ignored, this record lives in the built artifact, not in the repo. A manifest you generate yourself has no override until **you** record your own reviewer's sign-off with `--approved-by`; the record above grants no licence to anyone else. The other presets also have pending review items: the same Wikipedia corpus for the `edge-int8` and `fp32` variants, and the CoNLL-2003 data (Reuters text under a research-use agreement, recorded as `other`) for `distilbert-ner-en`.

### How core consumes the artifact

In [core](https://github.com/thecalibanproject/core), the NER detector sits behind the cargo feature `ner`, which is off by default:

```bash
# in a core checkout
cargo build -p caliban --features ner
export CALIBAN_PII_NER_DIR=/path/to/artifacts/pii_ner/nym-pii-multilingual-small-int8/3.0.0
export CALIBAN_PII_NER_SESSIONS=2   # optional: parallel ONNX sessions
```

- `CALIBAN_PII_NER_DIR` points at one artifact directory.
- At startup the loader rejects an unknown `manifest_version` or a kind other than `pii_ner`, checks every listed file's size and SHA-256, and binds ONNX tensors by the names in the manifest. If the variable is set and verification fails, the binary refuses to start. If the variable is set but the binary was built without `ner`, it also refuses to start.
- The loader does not yet check the `manifest.json.minisig` signature; deployments that require signed artifacts must check it before start.

Runtime behaviour (windowing, decoding, label mapping, thresholds) and the full candidate comparison are documented in core's [`crates/caliban-pii/MODELS.md`](https://github.com/thecalibanproject/core/blob/main/crates/caliban-pii/MODELS.md).

## Router and intent classifiers

Caliban's router is staged: rules, then an embedding kNN (Stage 1), then a small encoder classifier (Stage 2), with an LLM only as a fallback, and per-cluster model quality profiles to choose between models. The reasoning is in [docs: research note 02](https://github.com/thecalibanproject/docs/blob/main/research/02-intent-classification-and-routing.md).

No trained router or classifier weights are published. What exists today:

- **Stage-1 kNN calibration** (implemented, pure numpy): `router embed` + `router knn-eval` produce the calibration block for an `intent_head` artifact that `requires` a specific embedder. The exemplars themselves are tenant data that live in core's vector index and are not shipped.
- **Router profiles** (implemented): `router profile` builds a `router_profile` artifact from probe results.
- **Stage-2 classifier training** (stub): `router.train.train_intent_head` has `TODO` markers for SetFit and a ModernBERT multi-head model.
- **ONNX export and INT8 quantisation** (stub): `router.export.export_onnx` is planned to use opset 17 or later, dynamic batch and sequence axes, INT8 dynamic quantisation and a torch-vs-ONNX Runtime parity check.

### Stage-1 kNN semantics (shared with Rust)

```
neighbours  = top-k exemplars by cosine similarity (k in config.json)
w_j         = exp(s_j / T)                                    T = calibration.temperature
p_c         = (1 - eps) * sum_{j: y_j = c} w_j / sum_j w_j + eps / C
OOS         if s_top1 < calibration.oos_threshold             (explicit decision, never argmax)
exit        if p_max >= thresholds[label] (else default_threshold); otherwise go to Stage 2
```

`knn-eval` picks `T` by minimising NLL on held-out in-scope data. It sets the OOS gate either to the largest threshold that keeps the target in-scope acceptance (`recall`), or by Youden's J (`youden`). It sets each label's exit threshold to the lowest confidence that still reaches the target precision; OOS queries that slip past the gate count as errors, and a threshold of `1.0` means "never exit at Stage 1".

### Router profiles (UniRoute-style)

`profile.json` ([`schemas/router-profile.schema.json`](schemas/router-profile.schema.json)) holds a list of clusters, plus one row per model with `quality`, `raw_mean`, `counts`, `prior` and `n_errors`, aligned to the clusters:

```
quality[m][c] = (sum of scores[m, c] + w * prior[m]) / (count[m, c] + w)      w = prior_weight
```

`prior[m]` is the model's mean over all of its probes. Failed calls (timeouts, 5xx, rate limits) are excluded by default because they are not a quality signal. If clusters carry `centroid`s, the profile must name the `embedder` they came from (`router.profile.kmeans` builds them). A new bring-your-own model becomes routable as soon as its row exists, with no router retraining.

### Planned training flow

```
data (intents YAML / probe results; licence-checked, revision-pinned)
  -> train            [train]   SetFit or ModernBERT multi-head (intent, difficulty, oos, ...)   (stub)
  -> calibrate                  temperature + OOS gate + per-label exit thresholds (same code as kNN)
  -> export_onnx      [export]  opset >= 17, dynamic batch/seq, INT8 dynamic quantisation        (stub)
  -> parity check               torch vs ORT logits (max abs diff, argmax agreement) -> metrics  (stub)
  -> build manifest             hashes, tensor names, labels, calibration, data card, licence
  -> manifest verify --strict
  -> sign (release pipeline)
  -> publish artifacts/<kind>/<name>/<version>/
```

## The artifact contract (manifests and model cards)

An artifact is a directory at `artifacts/<kind>/<name>/<version>/`:

```
manifest.json            the contract (schemas/artifact-manifest.schema.json)
manifest.json.minisig    detached signature over manifest.json (release builds)
model.onnx               embedder / intent_head / pii_ner
tokenizer.json           HF `tokenizers` JSON (loaded by the Rust `tokenizers` crate); no trust_remote_code
labels.json, config.json, profile.json, ...
```

The manifest is also the model card: its `data_card` and `base_model` fields record where the weights and data came from, under which licence, and what the model is for. The JSON Schemas in [`schemas/`](schemas/) are generated from the pydantic models and drift-tested (`caliban-ml manifest schema --check`).

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
- `intent_head`: `labels` and `calibration` are required, plus either an ONNX graph (Stage-2 classifier) or a `requires` embedder (Stage-1 kNN calibration).
- `router_profile`: needs a `profile.json` with role `profile`, and carries no ONNX.

What a loader must do: (1) verify the signature over `manifest.json` when the deployment requires it, (2) check the size and SHA-256 of every listed file before loading, (3) reject an unknown `manifest_version`, (4) bind ONNX tensors **by name** as declared, never by position, (5) check that the `requires` artifacts are loaded at the exact versions listed, and (6) hot-swap only after all of the above pass.

### Signing

Only `manifest.json` is signed. Because it pins every file's SHA-256, the signature covers the whole artifact. Both supported schemes work air-gapped: minisign (the default for bundles) and cosign with a key pair and `--tlog-upload=false`. The exact commands are in [`src/caliban_ml/artifacts/signing.py`](src/caliban_ml/artifacts/signing.py). Private keys belong in the release pipeline's secret store, and `*.key` is git-ignored. The public key ships with the bundle.

## Licence policy for weights and data

Caliban ships on-prem, commercially, so a default bundle may only contain weights and training data that allow commercial use and redistribution. [`src/caliban_ml/artifacts/licenses.py`](src/caliban_ml/artifacts/licenses.py) enforces this on every manifest, for the weights **and** for every dataset in the data card:

- **Allowed:** Apache-2.0, MIT, BSD-2/3, ISC, CC0, Unlicense, CC-BY-3.0/4.0, Zlib, BSL-1.0 (Boost), PSF-2.0, and `caliban-internal` (first-party data).
- **Review required** (needs a `licence_overrides` entry with approver, date and reason): copyleft (MPL, LGPL, GPL, AGPL, CC-BY-SA), OpenRAIL variants, Llama and Gemma licences, the NVIDIA Open Model License, `other`, `unknown`, and anything unrecognised.
- **Denied:** any `*-nc-*` licence (CC-BY-NC, CC-BY-NC-SA and so on), "non-commercial", "research only", "academic only" or "evaluation only". An override is accepted only when validation runs with `--allow-noncommercial`, which a default bundle build never uses.

Check each model at the exact revision you pin: model cards change, and a model's own licence says nothing about the licence of its training data. Candidates considered so far (per their upstream cards; re-verify at the pinned revision):

| Candidate | Use | Licence | Status |
|---|---|---|---|
| `Wismut/nym-pii-multilingual-small` | PII NER (default L1) | MIT weights; CC-BY-SA fine-tuning text | Packaged; review item approved for Caliban's bundle (above) |
| `dslim/distilbert-NER` | English NER fallback | Apache-2.0 weights; CoNLL-2003 data | Packaged; data review pending |
| `answerdotai/ModernBERT-base` / `-large` | Stage-2 multi-head backbone | Apache-2.0 | OK |
| `urchade/gliner_*-v2.1` | PII NER alternative | Apache-2.0 on the card | OK once verified |
| `urchade/gliner_base`, `urchade/gliner_multi` (v1) | PII NER | CC-BY-NC-4.0 | **Denied** |
| `nvidia/gliner-pii` | PII NER alternative | NVIDIA Open Model License | Review + override |
| `BAAI/bge-small-en-v1.5`, `intfloat/multilingual-e5-small` | Embedder | MIT | OK, but check training-data terms |
| `katanemo/Arch-Router-1.5B` | Stage-3 LLM router | Custom terms | Review + override |
| Llama / Gemma family judges | Probe judge | Vendor licences with use restrictions | Review + override |

Third-party models and datasets remain under their owners' licences; this repo's licence does not cover them. Judge models used only on a build machine to score probes are not redistributed, but they still need a licence that permits the use.

## Probes (BYO-model onboarding)

```bash
export CALIBAN_PROBE_API_KEY=cal_xxx
caliban-ml probes run data/probes/sample.yaml \
  --base-url http://localhost:8080/v1 --model my-local-model --model gpt-4.1-mini \
  --judge-base-url http://vllm-judge:8000/v1 --judge-model local-judge \
  --out results/sample.jsonl
caliban-ml router profile results/sample.jsonl --name my-profile --version 1.0.0 \
  --out artifacts/router_profile/my-profile/1.0.0
```

Probe the Caliban data plane with an explicit model ID, not `caliban/auto`, so the router is bypassed, or probe a local vLLM or llama.cpp server directly. The runner only calls the configured endpoints. Use `--no-store-output` for sensitive probe sets (`results/` is git-ignored anyway). Check types are `exact`, `contains`, `regex`, `json` and `judge`.

## PII evaluation

The dataset is JSONL of `{"id", "text", "spans": [{"start", "end", "label"}]}`, where offsets are Python string (code-point) indices and the end is exclusive. There are three match modes: `exact`, `overlap` (same label, intersecting ranges, one-to-one), and `redaction` (intersecting ranges, label ignored). `redaction` is the number that matters for leakage. The regex L0 baseline on the synthetic `data/pii/sample.jsonl`:

```
regex / overlap: P=1.000 R=0.633 F1=0.776   (structured types 1.0; PERSON/LOCATION 0.0: the L1 NER's job)
```

This is a 20-document smoke test, not a benchmark.

## Getting artifacts into an on-prem bundle

The bundle tooling lives in [deploy](https://github.com/thecalibanproject/deploy). Today, [`airgap/models.lock.yaml`](https://github.com/thecalibanproject/deploy/blob/main/airgap/models.lock.yaml) pins the default PII NER model (`Wismut/nym-pii-multilingual-small` at the same revision as the preset), and the Compose example sets `CALIBAN_PII_NER_DIR` to the copied artifact. The recommended route is still to fetch with `scripts/fetch_pii_ner.py`, because that writes the hash-verified `manifest.json` core requires, and then copy `artifacts/pii_ner/` to the site's model directory.

The intended release flow for all artifact kinds (not fully automated yet):

1. The release pipeline pins a set of artifact versions (kind, name, version, manifest SHA-256).
2. The air-gap bundle step fetches each artifact directory, runs `caliban-ml manifest verify --strict --signature minisign --pubkey ...`, and copies it into the bundle under `models/<kind>/<name>/<version>/`.
3. The bundle carries the signing public key and the `schemas/` it was validated against. Core loads models only from that directory and re-verifies hashes at startup.
4. Customer-specific artifacts (tenant intent heads, BYO-model profiles) are produced on site with the same CLI, go through the same verify step, and can be signed with the customer's own key.

No artifact is ever fetched from a model hub at runtime.

## Repository layout

```
scripts/
  fetch_pii_ner.py             pinned PII NER fetch + manifest writer (stdlib only)
schemas/                       JSON Schemas consumed by core (generated, drift-tested)
  artifact-manifest.schema.json
  router-profile.schema.json
data/                          synthetic samples only
  intents/sample.yaml          intent exemplars + OOS examples
  probes/sample.yaml           10 probes across 5 clusters
  pii/sample.jsonl             20 docs with gold spans (obviously fake data)
src/caliban_ml/
  artifacts/                   manifest model, licence policy, hashing/verify, signing hook
  router/                      intent dataset, numpy kNN + calibration, UniRoute profiles,
                               train/export stubs ([train]/[export])
  probes/                      probe-set format, checkers, OpenAI-compatible runner, judge hook
  pii/                         span metrics, dataset format, detector adapters (regex, GLiNER, Presidio)
  cli.py                       typer CLI
tests/                         pytest suite
artifacts/                     local build output; git-ignored, empty in the repo
```

## Current state

Open items, marked `TODO(...)` in the code:

- `TODO(train/setfit)`, `TODO(train/modernbert_multihead)`: real training loops in `router/train.py`.
- `TODO(export)`: ONNX export, INT8 quantisation and the parity check in `router/export.py`.
- `TODO(embed/onnx)`: ONNX embedder backend in `router/embed.py`, so calibration runs in the exact shipped vector space.
- `TODO(pii/gliner2)`, `TODO(pii/gliner)`, `TODO(pii/presidio)`: a GLiNER2 adapter, sliding windows, and an explicit Presidio NLP engine config for the eval harness.
- `TODO(judge)`: judge calibration against human labels, and bias controls.
- `TODO(signing)`: a CI integration test for minisign/cosign, plus signature verification in the Rust loader.
- Publish example manifests and profiles as shared fixtures for core's loader tests.

## Licence

Copyright 2026 Elie Sfeir. All rights reserved.

This repository is **proprietary and source-available**, not open source. It is public for reference and evaluation only. No right to use, copy, modify or distribute it is granted without a separate written agreement with the copyright holder. See [LICENSE](LICENSE). For licensing, contact [elie@internalizable.dev](mailto:elie@internalizable.dev).

Third-party models, datasets and dependencies remain under their own licences.
