"""Shared, target-runtime-free local Transformer for pretraining and Domino."""

import math

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

FULL_ATTENTION = "full_attention"


class DominoContextTransformer(nn.Module):
    """Qwen3 decoder stack with projected condition/token inputs for one block."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
        rope_theta: float,
        prefix_length: int = 1,
        num_key_value_heads: int = None,
        token_rope_start: int = None,
        moe_config: dict = None,
    ) -> None:
        super().__init__()
        if num_key_value_heads is None:
            num_key_value_heads = num_heads
        if num_key_value_heads <= 0 or num_heads % num_key_value_heads:
            raise ValueError(
                "Transformer context KV head count must be positive and divide "
                f"the head count; got {num_key_value_heads} for {num_heads} heads."
            )
        if hidden_size <= 0 or num_layers <= 0 or num_heads <= 0:
            raise ValueError(
                "Transformer context hidden size, layer count, and head count "
                "must all be positive."
            )
        if hidden_size % num_heads:
            raise ValueError(
                "Transformer context hidden size must be divisible by its head count; "
                f"got hidden_size={hidden_size}, num_heads={num_heads}."
            )
        head_dim = hidden_size // num_heads
        if head_dim % 2:
            raise ValueError(
                "Transformer context RoPE head dimension must be even; " f"got head_dim={head_dim}."
            )
        if intermediate_size <= 0:
            raise ValueError(
                "Transformer context intermediate size must be positive; "
                f"got {intermediate_size}."
            )
        if not math.isfinite(float(rope_theta)) or float(rope_theta) <= 0:
            raise ValueError(f"Transformer context rope_theta must be positive; got {rope_theta}.")
        if not isinstance(prefix_length, int) or prefix_length < 1:
            raise ValueError("Transformer context prefix_length must be a positive integer.")
        self.prefix_length = prefix_length
        # RoPE index of the first token-path slot. The default keeps tokens
        # right after the condition slots; 1 co-locates token slot k with
        # condition slot k + 1 (the DFlash hidden of the token slot k predicts).
        if token_rope_start is None:
            token_rope_start = prefix_length
        if not isinstance(token_rope_start, int) or token_rope_start < 0:
            raise ValueError("Transformer context token_rope_start must be a non-negative integer.")
        self.token_rope_start = token_rope_start
        self.hidden_size = int(hidden_size)
        self.config = Qwen3Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act="silu",
            rms_norm_eps=rms_norm_eps,
            rope_theta=float(rope_theta),
            attention_bias=False,
            attention_dropout=0.0,
            use_sliding_window=False,
            layer_types=[FULL_ATTENTION] * num_layers,
            use_cache=False,
        )
        # Match the original context's causal SDPA independently of the draft
        # backbone's attention backend. No token embedding or LM head is created.
        self.config._attn_implementation = "sdpa"
        self.anchor_init_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.input_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(self.config, layer_idx=i) for i in range(num_layers)]
        )
        if moe_config is not None:
            try:
                from .domino_moe import DominoMoE
            except ImportError:
                from domino_moe import DominoMoE
            for layer in self.layers:
                layer.mlp = DominoMoE(hidden_size, moe_config)
        self.moe_config = moe_config
        self.final_norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(self.config)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Regenerate non-persistent RoPE buffers from config: casting rounded
        # BF16 frequencies back to FP32 would not restore their original values.
        self.rotary_emb = Qwen3RotaryEmbedding(self.config, device=self.input_proj.weight.device)
        return self

    def forward(
        self,
        anchor_hidden: torch.Tensor,
        token_embeddings: torch.Tensor,
        routing_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        prefix = anchor_hidden.unsqueeze(1) if anchor_hidden.ndim == 2 else anchor_hidden
        if prefix.ndim != 3 or token_embeddings.ndim != 3:
            raise ValueError("Context prefix and token embeddings must be rank 3.")
        if prefix.shape[0] != token_embeddings.shape[0] or prefix.shape[1] != self.prefix_length:
            raise ValueError("Transformer context prefix batch/length mismatch.")
        condition = self.anchor_init_proj(prefix)
        hidden_states = torch.cat([condition, self.input_proj(token_embeddings)], dim=1)
        if self.moe_config is not None:
            if routing_mask is None:
                routing_mask = torch.ones(
                    token_embeddings.shape[:2], device=hidden_states.device, dtype=torch.bool
                )
            assert routing_mask.shape == token_embeddings.shape[:2]
            mask = torch.cat(
                [
                    torch.zeros(prefix.shape[:2], device=hidden_states.device, dtype=torch.bool),
                    routing_mask.bool(),
                ],
                dim=1,
            )
            for layer in self.layers:
                layer.mlp._balance_mask = mask
        try:
            return self._decode(hidden_states, position_offset=0)[:, self.prefix_length :]
        finally:
            if self.moe_config is not None:
                for layer in self.layers:
                    layer.mlp._balance_mask = None

    def forward_tokens(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Pretrain the token path with no condition slots in attention.

        Offset positions to the same token positions used after prefix injection.
        The condition projection is deliberately absent from this computation.
        Each batch element is an independent causal window.
        """
        if token_embeddings.ndim != 3 or token_embeddings.shape[1] < 1:
            raise ValueError("Token embeddings must be nonempty [batch, length, hidden].")
        return self._decode(self.input_proj(token_embeddings), position_offset=self.prefix_length)

    def rope_positions(self, slots: torch.Tensor) -> torch.Tensor:
        """Map sequence slots to RoPE indices: condition slots keep their index,
        token-path slots start at ``token_rope_start``."""
        shift = self.token_rope_start - self.prefix_length
        if shift == 0:
            return slots
        return torch.where(slots >= self.prefix_length, slots + shift, slots)

    def _decode(self, hidden_states: torch.Tensor, position_offset: int) -> torch.Tensor:
        position_ids = self.rope_positions(
            torch.arange(
                position_offset,
                position_offset + hidden_states.shape[1],
                device=hidden_states.device,
            )
        )[None]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
        return self.final_norm(hidden_states)
