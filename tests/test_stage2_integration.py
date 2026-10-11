"""End-to-end Stage-2 train -> export -> ONNX parity on a tiny random ModernBERT.

Skipped unless the ``[train]`` and ``[export]`` extras are installed (CI has neither). The
tiny model and its tokenizer are built locally, so nothing is downloaded. The ``gpu`` test
trains a real base model on CUDA and needs ``CALIBAN_ML_GPU_BASE_MODEL`` (a local model dir)
plus ``CALIBAN_ML_GPU_BASE_REVISION`` / ``CALIBAN_ML_GPU_BASE_ID`` for the manifest.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

for _mod in ("torch", "transformers", "safetensors", "onnx", "onnxruntime", "tokenizers"):
    pytest.importorskip(_mod)

from caliban_ml.artifacts import (  # noqa: E402
    ArtifactKind,
    build_manifest,
    load_manifest,
    verify_artifact_dir,
    write_manifest,
)
from caliban_ml.router import IntentDataset  # noqa: E402
from caliban_ml.router.export import export_onnx  # noqa: E402
from caliban_ml.router.train import IntentHeadConfig, train_intent_head  # noqa: E402
from conftest import DATA, FAKE_REV  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tiny_modernbert(tmp_path_factory) -> Path:
    """A 2-layer random ModernBERT with a word-level tokenizer over the sample vocabulary."""
    import torch
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("tiny-modernbert")
    ds = IntentDataset.from_yaml(DATA / "intents" / "sample.yaml")
    words = sorted({w for t in ds.examples()[0] for w in re.findall(r"\w+|[^\w\s]", t.lower())})
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = {w: i for i, w in enumerate(specials + words)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace()])
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    fast.save_pretrained(str(d))
    torch.manual_seed(0)
    cfg = ModernBertConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=2,
                           max_position_embeddings=256, global_attn_every_n_layers=2,
                           local_attention=16, pad_token_id=0, cls_token_id=2, sep_token_id=3,
                           bos_token_id=2, eos_token_id=3)
    ModernBertModel(cfg).save_pretrained(str(d), safe_serialization=True)
    return d


def _config(base: Path, out: Path, **over) -> IntentHeadConfig:
    fields = {"base_model_path": base, "base_model_id": "test/tiny-modernbert",
              "base_model_revision": FAKE_REV, "base_model_licence": "apache-2.0",
              "backbone": "modernbert_multihead", "epochs": 3, "learning_rate": 1e-3,
              "batch_size": 8, "max_length": 32, "val_fraction": 0.25,
              "test_fraction": 0.25, "device": "cpu", "output_dir": out,
              "dataset_name": "sample-intents"}
    fields.update(over)
    return IntentHeadConfig(**fields)


def _ort_logits(art: Path, texts: list[str]) -> dict[str, np.ndarray]:
    """Run an exported intent_head the way core would: tokenizer.json + names from the
    manifest + pad id from config.json."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    m = load_manifest(art)
    cfg = json.loads((art / "config.json").read_text())
    tok = Tokenizer.from_file(str(art / m.tokenizer.file))
    tok.enable_truncation(m.tokenizer.max_length)
    enc = tok.encode_batch(texts)
    seq = max(len(e.ids) for e in enc)
    ids = np.full((len(enc), seq), cfg["pad_id"], dtype=np.int64)
    mask = np.zeros_like(ids)
    for i, e in enumerate(enc):
        ids[i, :len(e.ids)] = e.ids
        mask[i, :len(e.ids)] = 1
    sess = ort.InferenceSession(str(art / m.onnx.file), providers=["CPUExecutionProvider"])
    names = [o.name for o in m.onnx.outputs]
    return dict(zip(names, sess.run(names, {"input_ids": ids, "attention_mask": mask}),
                    strict=True))


def test_multihead_train_export_parity(tiny_modernbert: Path, tmp_path: Path):
    import torch

    from caliban_ml.router import _torch as T
    from caliban_ml.router.checkpoint import ENCODER_DIR

    ds = IntentDataset.from_yaml(DATA / "intents" / "sample.yaml")
    extra = {"paraphrases": (["please summarize this report", "what is the weather"],
                             ["summarization", "__oos__"])}
    ckpt = train_intent_head(ds, _config(tiny_modernbert, tmp_path / "ckpt",
                                         held_out_intents=("code",)), extra_eval=extra)
    meta = json.loads((ckpt / "caliban_checkpoint.json").read_text())
    assert [h["name"] for h in meta["heads"]] == ["intent"]  # no OOS in train: no oos head
    assert "code" not in meta["heads"][0]["labels"]
    metrics = meta["metrics"]
    for key in ("test_in_scope_accuracy", "heldout_intents_oos_rejection",
                "paraphrases_end_to_end_accuracy", "knn_base_test_in_scope_accuracy",
                "knn_tuned_heldout_intents_oos_rejection", "train_seconds"):
        assert key in metrics, key

    art = export_onnx(ckpt, tmp_path / "art", name="sample-intents", version="0.1.0",
                      quantization="none", parity_atol=1e-4)
    rep = verify_artifact_dir(art, strict=True)
    assert rep.ok, rep.errors
    m = rep.manifest
    assert m.kind == ArtifactKind.INTENT_HEAD and m.onnx.quantization == "none"
    assert m.metrics["export_fp32_argmax_agree"] == 1.0
    assert m.metrics["export_tokenizer_mismatches"] == 0.0

    # Unseen texts: ORT through the shipped tokenizer == torch through the HF tokenizer.
    texts = ["write a python function to sort a list", "summarize the meeting",
             "how many customers churned", "pull the dates out of this email"]
    got = _ort_logits(art, texts)["intent_logits"]
    _, model = T.load_checkpoint(ckpt)
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(str(ckpt / ENCODER_DIR))
    enc = hf(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
    with torch.no_grad():
        ref = model(enc["input_ids"], enc["attention_mask"])[0].numpy()
    assert np.abs(ref - got).max() < 1e-4
    assert (ref.argmax(1) == got.argmax(1)).all()


def test_multihead_with_oos_head_and_int8(tiny_modernbert: Path, tmp_path: Path):
    ds = IntentDataset.from_yaml(DATA / "intents" / "sample.yaml")
    ds = ds.model_copy(update={"oos": [*ds.oos, "book me a flight to rome",
                                       "what time is it in tokyo", "play some jazz",
                                       "turn off the lights"]})
    ckpt = train_intent_head(ds, _config(tiny_modernbert, tmp_path / "ckpt", epochs=2,
                                         knn_baseline=False))
    heads = json.loads((ckpt / "caliban_checkpoint.json").read_text())["heads"]
    assert [h["name"] for h in heads] == ["intent", "oos"]
    # A random tiny model has near-tied logits, so int8 agreement is not meaningful here:
    # this exercises the quantisation path and the second output only.
    art = export_onnx(ckpt, tmp_path / "art", name="sample-intents", version="0.1.0",
                      quantization="int8_dynamic", min_argmax_agree=0.0)
    m = load_manifest(art)
    assert [o.name for o in m.onnx.outputs] == ["intent_logits", "oos_logits"]
    assert m.onnx.quantization == "int8_dynamic"
    assert "export_int8_argmax_agree" in m.metrics
    out = _ort_logits(art, ["play some jazz"])
    assert out["oos_logits"].shape == (1, 1)
    with pytest.raises(FileExistsError):
        export_onnx(ckpt, art, name="sample-intents", version="0.1.0")


def test_setfit_train_export_parity(tiny_modernbert: Path, tmp_path: Path):
    pytest.importorskip("setfit")
    ds = IntentDataset.from_yaml(DATA / "intents" / "sample.yaml")
    ckpt = train_intent_head(ds, _config(tiny_modernbert, tmp_path / "ckpt", backbone="setfit",
                                         epochs=1, setfit_iterations=2,
                                         learning_rate=1e-4, knn_baseline=False))
    meta = json.loads((ckpt / "caliban_checkpoint.json").read_text())
    assert meta["backbone"] == "setfit" and meta["pooling"] == "mean"
    art = export_onnx(ckpt, tmp_path / "art", name="sample-setfit", version="0.1.0",
                      quantization="none", parity_atol=1e-4)
    assert verify_artifact_dir(art, strict=True).ok


def test_embed_onnx_artifact_matches_torch(tiny_modernbert: Path, tmp_path: Path):
    import torch

    from caliban_ml.router import _torch as T
    from caliban_ml.router.embed import embed_onnx_artifact
    from caliban_ml.router.export import clean_tokenizer_json

    enc = T.load_encoder(tiny_modernbert, attn_implementation="eager").eval()

    class Hidden(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc

        def forward(self, input_ids, attention_mask):
            return self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    art = tmp_path / "emb"
    art.mkdir()
    ids = torch.tensor([[2, 10, 11, 3], [2, 12, 3, 0]])
    mask = (ids != 0).long()
    torch.onnx.export(Hidden(), (ids, mask), str(art / "model.onnx"),
                      input_names=["input_ids", "attention_mask"], output_names=["hidden"],
                      dynamic_axes={"input_ids": {0: "b", 1: "s"},
                                    "attention_mask": {0: "b", 1: "s"}, "hidden": {0: "b", 1: "s"}},
                      opset_version=17, dynamo=False)
    clean_tokenizer_json(tiny_modernbert / "tokenizer.json", art / "tokenizer.json")
    write_manifest(build_manifest(
        art, kind="embedder", name="tiny-embedder", version="0.1.0",
        onnx={"file": "model.onnx", "opset": 17,
              "inputs": [{"name": "input_ids", "dtype": "int64", "shape": ["b", "s"]},
                         {"name": "attention_mask", "dtype": "int64", "shape": ["b", "s"]}],
              "outputs": [{"name": "hidden", "dtype": "float32", "shape": ["b", "s", 32]}]},
        tokenizer={"file": "tokenizer.json", "max_length": 32},
        embedding={"dim": 32, "output": "hidden", "pooling": "mean", "normalize": True},
        base_model={"id": "test/tiny", "revision": FAKE_REV, "licence": "apache-2.0"},
        data_card={"intended_use": "test"}), art)

    texts = ["summarize this email thread for me", "top 10 customers by revenue",
             "write a python function"]
    got = embed_onnx_artifact(art, texts, batch_size=2)
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(str(tiny_modernbert))
    b = hf(texts, padding=True, return_tensors="pt")
    with torch.no_grad():
        h = enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
    ref = T.pool(h, b["attention_mask"], "mean")
    ref = torch.nn.functional.normalize(ref, dim=-1).numpy()
    assert got.shape == (3, 32)
    assert np.abs(got - ref).max() < 1e-4


@pytest.mark.gpu
def test_gpu_real_base_model(tmp_path: Path):
    import torch

    base = os.environ.get("CALIBAN_ML_GPU_BASE_MODEL")
    if not base or not torch.cuda.is_available():
        pytest.skip("needs CUDA and CALIBAN_ML_GPU_BASE_MODEL")
    ds = IntentDataset.from_yaml(DATA / "intents" / "sample.yaml")
    cfg = _config(Path(base), tmp_path / "ckpt", epochs=2, learning_rate=5e-5, device="cuda",
                  base_model_id=os.environ.get("CALIBAN_ML_GPU_BASE_ID", "local/base"),
                  base_model_revision=os.environ.get("CALIBAN_ML_GPU_BASE_REVISION", FAKE_REV),
                  base_model_licence=os.environ.get("CALIBAN_ML_GPU_BASE_LICENCE", "apache-2.0"))
    ckpt = train_intent_head(ds, cfg)
    metrics = json.loads((ckpt / "metrics.json").read_text())
    assert metrics["train_peak_vram_allocated_mb"] > 0
    art = export_onnx(ckpt, tmp_path / "art", name="gpu-smoke", version="0.1.0")
    assert verify_artifact_dir(art, strict=True).ok
