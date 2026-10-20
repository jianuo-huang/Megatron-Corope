"""Standalone local LM and strict transfer to the full Domino draft model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

try:
    from ..domino_context import DominoContextTransformer
except ImportError:  # CLI adds specforge/ directly; avoid its eager target imports.
    from domino_context import DominoContextTransformer


@dataclass(frozen=True)
class LocalSpec:
    vocab_size: int
    input_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    bottleneck: int
    block_size: int
    rope_theta: float
    rms_norm_eps: float
    initializer_range: float = 0.02
    verified_layout: str = "prefix"
    verified_prefix_len: int = 0
    share_lm_head: bool = False
    num_key_value_heads: int = 0
    token_rope_start: int | None = None
    moe_config: dict | None = None

    @property
    def window_tokens(self):
        """Tokens per training window: in the token layout the verified prefix
        joins the block as one contiguous text segment at RoPE 8..; the prefix
        layout keeps the original block-only windows."""
        return self.block_size + (
            self.verified_prefix_len if self.verified_layout == "token" else 0
        )

    @classmethod
    def from_draft(cls, config):
        if not isinstance(config, dict):
            config = config.to_dict()
        c = config["dflash_config"]
        if (
            c.get("ensemble_context_model") != "transformer"
            or c.get("ensemble_use_dflash_hidden", False)
            or c.get("shift_label", False)
        ):
            raise ValueError(
                "Pretraining requires the independent Transformer head without shifted labels."
            )
        layout = c.get("ensemble_context_verified_layout", "prefix")
        verified = int(c.get("ensemble_context_verified_prefix_len", 0))
        if layout not in ("prefix", "token") or verified < 0:
            raise ValueError("Invalid verified layout/prefix length.")
        share = bool(c.get("ensemble_share_lm_head", False))
        kv_heads = int(c.get("ensemble_context_num_key_value_heads") or 0)
        return cls(
            config["vocab_size"],
            config["hidden_size"],
            c["ensemble_context_hidden_dim"],
            c["ensemble_context_num_layers"],
            c["ensemble_context_num_attention_heads"],
            c["ensemble_context_intermediate_dim"],
            c["emb_dim"],
            config["block_size"],
            float(
                c.get("ensemble_context_rope_theta")
                or config.get("rope_theta")
                or (config.get("rope_parameters") or {}).get("rope_theta", 10000.0)
            ),
            config["rms_norm_eps"],
            config.get("initializer_range", 0.02),
            layout,
            verified,
            share,
            kv_heads,
            c.get("ensemble_context_token_rope_start"),
            c.get("ensemble_context_moe"),
        )


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for chunk in value.split(4 * 1024 * 1024):
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def tokenizer_identity(root):
    root = Path(root)
    names = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    ]
    files = {name: file_sha(root / name) for name in names if (root / name).is_file()}
    if not files:
        raise ValueError("Target tokenizer files are required.")
    return files


def load_target_embedding(root):
    """Read exactly one tensor; never construct a target model or read its head."""
    root = Path(root)
    key = "model.embed_tokens.weight"
    index = root / "model.safetensors.index.json"
    filename = (
        json.loads(index.read_text())["weight_map"][key] if index.exists() else "model.safetensors"
    )
    with safe_open(root / filename, framework="pt", device="cpu") as stream:
        weight = stream.get_tensor(key)
    identity = {
        "embedding_sha256": tensor_sha(weight),
        "embedding_shape": list(weight.shape),
        "embedding_dtype": str(weight.dtype),
        "tokenizer_files": tokenizer_identity(root),
    }
    return weight, identity


def load_target_head(root):
    """Read only the frozen lm_head weight for shared-head pretraining."""
    root = Path(root)
    key = "lm_head.weight"
    index = root / "model.safetensors.index.json"
    filename = (
        json.loads(index.read_text())["weight_map"][key] if index.exists() else "model.safetensors"
    )
    with safe_open(root / filename, framework="pt", device="cpu") as stream:
        weight = stream.get_tensor(key)
    return weight, {"lm_head_sha256": tensor_sha(weight)}


def transferable(name):
    return name.startswith(
        (
            "context_transformer.input_proj.",
            "context_transformer.layers.",
            "context_transformer.final_norm.",
            "embed_proj.",
        )
    )


class LocalLanguageModel(nn.Module):
    def __init__(self, spec: LocalSpec, embedding: torch.Tensor, *, window_tokens=None):
        super().__init__()
        if list(embedding.shape) != [spec.vocab_size, spec.input_size]:
            raise ValueError("Embedding shape does not match the draft vocabulary/input size.")
        self.spec = spec
        self.window_tokens = spec.window_tokens if window_tokens is None else window_tokens
        if not isinstance(self.window_tokens, int) or self.window_tokens < 2:
            raise ValueError("Pretraining window_tokens must be an integer >= 2.")
        self.token_embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.context_transformer = DominoContextTransformer(
            spec.input_size,
            spec.hidden_size,
            spec.num_layers,
            spec.num_heads,
            spec.intermediate_size,
            spec.rms_norm_eps,
            spec.rope_theta,
            prefix_length=spec.block_size,
            num_key_value_heads=spec.num_key_value_heads or None,
            token_rope_start=spec.token_rope_start,
            moe_config=spec.moe_config,
        )
        if spec.share_lm_head:
            # Trained bridge into the target head's input space; the frozen
            # lm_head weight is injected as a plain attribute, never saved.
            self.embed_proj = nn.Sequential(
                nn.Linear(spec.hidden_size, spec.input_size, bias=False)
            )
            self.lm_head_weight = None
        else:
            self.embed_proj = nn.Sequential(
                nn.Linear(spec.hidden_size, spec.bottleneck, bias=False),
                nn.SiLU(),
                nn.Linear(spec.bottleneck, spec.vocab_size, bias=False),
            )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=spec.initializer_range)
        self.context_transformer.anchor_init_proj.requires_grad_(False)
        self.loss_backend = "chunked"
        self.chunk_size = 128
        self.num_sources = 3
        self._fused_ce = None

    def features(self, ids):
        if ids.ndim != 2 or ids.shape[1] != self.window_tokens - 1:
            raise ValueError("Inputs must be [windows, window_tokens-1].")
        x = self.context_transformer.forward_tokens(self.token_embedding(ids))
        if self.spec.share_lm_head:
            return self.embed_proj(x)
        return self.embed_proj[1](self.embed_proj[0](x))

    def head_weight(self):
        if self.spec.share_lm_head:
            if self.lm_head_weight is None:
                raise RuntimeError(
                    "share_lm_head pretraining requires the frozen target "
                    "lm_head weight (load_target_head)."
                )
            return self.lm_head_weight
        return self.embed_proj[2].weight

    def head_logits(self, features):
        return F.linear(features.to(self.head_weight().dtype), self.head_weight())

    def forward(
        self, ids, labels=None, *, sources=None, return_metrics=False, loss_normalizer=None
    ):
        if self.spec.moe_config:
            for layer in self.context_transformer.layers:
                layer.mlp._balance_mask = labels.ne(-100) if labels is not None else None
        try:
            features = self.features(ids)
        finally:
            if self.spec.moe_config:
                for layer in self.context_transformer.layers:
                    layer.mlp._balance_mask = None
        if labels is None:
            return self.head_logits(features)
        if labels.shape != ids.shape:
            raise ValueError("Labels and input shape differ.")
        if self.loss_backend == "frozen-fused":
            if not self.spec.share_lm_head or loss_normalizer is None:
                raise ValueError(
                    "Frozen fused training requires shared head and an explicit global normalizer"
                )
            from .frozen_head_ce import normalized_frozen_head_ce

            loss = normalized_frozen_head_ce(
                features.flatten(0, 1),
                self.head_weight(),
                labels.flatten(),
                normalizer=loss_normalizer,
                chunk_size=self.chunk_size,
            )
        else:
            if loss_normalizer is not None:
                raise ValueError("Explicit loss_normalizer is only for frozen-fused")
            loss = self.loss_sum(features.flatten(0, 1), labels.flatten())
        if return_metrics:
            if sources is None or sources.shape != (ids.shape[0],):
                raise ValueError("Metrics require one source index per window")
            return loss, self.metric_sums(features.detach(), labels, sources)
        return loss

    @torch.no_grad()
    def metric_sums(self, features, labels, sources):
        depth = features.shape[1]
        flattened = features.flatten(0, 1)
        target = labels.flatten()
        groups = (
            sources.repeat_interleave(depth) * depth
            + torch.arange(target.numel(), device=features.device) % depth
        )
        if sources.min() < 0 or sources.max() >= self.num_sources:
            raise ValueError("Invalid source index")
        sums = torch.zeros(
            (3, self.num_sources, depth), dtype=torch.float64, device=features.device
        )
        for start in range(0, target.numel(), self.chunk_size):
            y = target[start : start + self.chunk_size]
            index = groups[start : start + self.chunk_size]
            valid = y != -100
            logits = self.head_logits(flattened[start : start + self.chunk_size])
            nll = F.cross_entropy(logits.float(), y, reduction="none", ignore_index=-100)
            sums[0].flatten().scatter_add_(0, index, nll.double())
            sums[1].flatten().scatter_add_(0, index, valid.double())
            sums[2].flatten().scatter_add_(0, index, ((logits.argmax(-1) == y) & valid).double())
        return sums

    def loss_sum(self, features, labels):
        if self.loss_backend == "liger":
            if self.spec.share_lm_head:
                raise ValueError("share_lm_head pretraining uses the chunked CE backend.")
            if not features.is_cuda:
                raise ValueError("Liger requires CUDA; use chunked CE for CPU reference.")
            if self._fused_ce is None:
                from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

                self._fused_ce = LigerFusedLinearCrossEntropyLoss(
                    reduction="sum", ignore_index=-100, accum_dtype=torch.float32
                )
            # FP32 model parameters are the master weights; cast only for compute.
            return self._fused_ce(self.embed_proj[2].weight.to(features.dtype), features, labels)
        if self.loss_backend != "chunked":
            raise ValueError(self.loss_backend)

        def ce(x, weight, y):
            return F.cross_entropy(
                F.linear(x.to(weight.dtype), weight).float(), y, reduction="sum", ignore_index=-100
            )

        parts = []
        head = self.head_weight()
        for x, y in zip(features.split(self.chunk_size), labels.split(self.chunk_size)):
            args = (x, head, y)
            parts.append(
                checkpoint(ce, *args, use_reentrant=False) if torch.is_grad_enabled() else ce(*args)
            )
        return torch.stack(parts).sum()

    def export(self, root, identity, extra=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        weights = {
            k: v.detach().cpu().contiguous()
            for k, v in self.state_dict().items()
            if transferable(k)
        }
        save_file(weights, root / "expert.safetensors")
        meta = {
            "format": "domino-local-pretrain-v1",
            "spec": asdict(self.spec),
            "target_identity": identity,
            "weight_sha256": file_sha(root / "expert.safetensors"),
            "keys": sorted(weights),
            "extra": dict(extra or {}, pretrain_window_tokens=self.window_tokens),
        }
        (root / "pretrain.json").write_text(json.dumps(meta, indent=2) + "\n")
        return meta

    def load_expert(self, root, identity):
        weights, _ = read_expert(root, self.spec, identity)
        expected = {k for k in self.state_dict() if transferable(k)}
        if weights.keys() != expected:
            raise ValueError("Pretrained keys differ from local expert.")
        own = self.state_dict()
        for k, value in weights.items():
            if value.shape != own[k].shape:
                raise ValueError(f"Pretrained shape differs: {k}")
        with torch.no_grad():
            for k, value in weights.items():
                own[k].copy_(value)


def read_expert(root, spec, identity):
    root = Path(root)
    meta = json.loads((root / "pretrain.json").read_text())
    # Normalize only declared dataclass defaults; reject unknown/changed fields.
    saved_spec = LocalSpec(**meta["spec"])
    if meta["format"] != "domino-local-pretrain-v1" or saved_spec != spec:
        raise ValueError("Pretraining architecture/config does not match.")
    if meta["target_identity"] != identity:
        raise ValueError("Pretraining tokenizer/target embedding identity does not match.")
    if file_sha(root / "expert.safetensors") != meta["weight_sha256"]:
        raise ValueError("Pretraining weight checksum mismatch.")
    weights = load_file(root / "expert.safetensors")
    if sorted(weights) != meta["keys"] or not all(transferable(k) for k in weights):
        raise ValueError("Unexpected pretrained keys.")
    if not all(torch.isfinite(v).all().item() for v in weights.values()):
        raise ValueError("Non-finite pretrained weight.")
    return weights, meta


def initialize_draft(draft, root, identity):
    """Only copy an exact whitelist, after draft post_init and before FSDP."""
    weights, meta = read_expert(root, LocalSpec.from_draft(draft.config), identity)
    expected = {k for k in draft.state_dict() if transferable(k)}
    if weights.keys() != expected:
        raise ValueError(f"Draft/pretrain key mismatch: {weights.keys() ^ expected}")
    own = draft.state_dict()
    for k, value in weights.items():
        if value.shape != own[k].shape:
            raise ValueError(f"Draft/pretrain shape mismatch: {k}")
    with torch.no_grad():
        for k, value in weights.items():
            own[k].copy_(value)
    return {
        "weight_sha256": meta["weight_sha256"],
        "loaded_keys": sorted(weights),
        "pretrain_extra": meta["extra"],
    }


def initialize_draft_from_target(draft, root, target_path):
    """Derive transfer identity from the actual target, including a shared head."""
    embedding, identity = load_target_embedding(target_path)
    del embedding
    if LocalSpec.from_draft(draft.config).share_lm_head:
        head, head_identity = load_target_head(target_path)
        del head
        identity.update(head_identity)
    return initialize_draft(draft, root, identity)
