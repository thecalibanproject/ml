"""Torch pieces shared by training and export. Imports torch at module level, so it must
only ever be imported lazily from code paths that already required the ``[train]`` extra."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from caliban_ml.router.checkpoint import ENCODER_DIR, HEADS_FILENAME, CheckpointMeta


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "cls":
        return hidden[:, 0]
    if mode == "mean":
        m = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
    raise ValueError(f"unsupported pooling {mode!r}")


class PooledHeads(nn.Module):
    """``encoder -> pool -> [L2 normalise] -> dropout -> one Linear per head``.

    ``forward`` returns a tuple of logits, one per head, in ``head_names`` order. This is
    the exact graph that is exported (dropout is a no-op in eval mode).
    """

    def __init__(self, encoder: nn.Module, head_widths: dict[str, int], *, pooling: str,
                 normalize: bool = False, dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.dropout = nn.Dropout(dropout)
        hidden = int(encoder.config.hidden_size)
        self.head_names = list(head_widths)
        self.heads = nn.ModuleDict({n: nn.Linear(hidden, w) for n, w in head_widths.items()})

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = pool(out.last_hidden_state, attention_mask, self.pooling)
        if self.normalize:
            h = nn.functional.normalize(h, p=2.0, dim=-1)
        return h

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor
                ) -> tuple[torch.Tensor, ...]:
        h = self.dropout(self.embed(input_ids, attention_mask))
        return tuple(self.heads[n](h) for n in self.head_names)

    def head_state(self) -> dict[str, torch.Tensor]:
        return {f"{n}.{k}": v.detach().cpu().contiguous()
                for n in self.head_names for k, v in self.heads[n].state_dict().items()}


def load_encoder(path: Path, *, attn_implementation: str | None = None) -> nn.Module:
    from transformers import AutoModel

    kwargs = {"local_files_only": True}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModel.from_pretrained(str(path), **kwargs)
    if hasattr(model.config, "reference_compile"):
        model.config.reference_compile = False  # ModernBERT: never torch.compile in-graph
    return model


def load_checkpoint(ckpt_dir: Path, *, attn_implementation: str | None = "eager"
                    ) -> tuple[CheckpointMeta, PooledHeads]:
    """Rebuild the exact inference graph of a checkpoint (fp32, eval mode, CPU)."""
    from safetensors.torch import load_file

    ckpt_dir = Path(ckpt_dir)
    meta = CheckpointMeta.read(ckpt_dir)
    encoder = load_encoder(ckpt_dir / ENCODER_DIR, attn_implementation=attn_implementation)
    model = PooledHeads(encoder, {h.name: h.width for h in meta.heads}, pooling=meta.pooling,
                        normalize=meta.normalize)
    state = load_file(str(ckpt_dir / HEADS_FILENAME))
    for h in meta.heads:
        model.heads[h.name].load_state_dict({"weight": state[f"{h.name}.weight"],
                                             "bias": state[f"{h.name}.bias"]})
    return meta, model.float().eval()


@torch.no_grad()
def predict_logits(model: PooledHeads, tokenizer, texts: list[str], *, max_length: int,
                   batch_size: int = 64, device: str = "cpu") -> dict[str, np.ndarray]:
    """Run ``model`` over ``texts``; returns ``{head: float32 (n, width)}``."""
    model.eval()
    outs: dict[str, list] = {n: [] for n in model.head_names}
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        for n, t in zip(model.head_names, logits, strict=True):
            outs[n].append(t.float().cpu().numpy())
    return {n: np.concatenate(v, axis=0) for n, v in outs.items()}


class EmbeddingOnly(nn.Module):
    """The encoder of a :class:`PooledHeads` without its heads: pooled, L2-normalised
    sentence embeddings (the graph of an exported ``embedder`` artifact)."""

    def __init__(self, model: PooledHeads):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.model.embed(input_ids, attention_mask)
        return nn.functional.normalize(h, p=2.0, dim=-1)
