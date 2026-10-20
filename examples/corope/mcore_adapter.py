"""Token-only Context pretraining using genuine Megatron Core / TE layers.

TP=PP=CP=EP=1; data parallelism is handled by the caller. Frozen target tables
are external tensors, excluded from both the optimizer and distributed saves.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from domino_moe import DominoMoE
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig


class MCoreDominoMoE(MegatronModule):
    """Existing Loss-Free MoE semantics, with the MCore MLP interface.

    Experts remain DP replicated (EP=1). No auxiliary loss or MCore router
    replacement; MCore supplies DDP, distributed optimizer and checkpointing.
    """

    def __init__(self, config, moe_config):
        super().__init__(config)
        self.experts = DominoMoE(config.hidden_size, moe_config)

    def forward(self, hidden_states, padding_mask=None):
        return self.experts(hidden_states), None


class OriginalRMSNorm(Qwen3RMSNorm):
    """Keep HF's normalize -> input-dtype cast -> FP32 gamma ordering."""

    def __init__(self, hidden_size, eps, config=None, **kwargs):
        super().__init__(hidden_size, eps)


class OriginalRotarySelfAttention(SelfAttention):
    """Keep HF's BF16 trig tables and FP32 QK normalization intermediates.

    All projections and the attention kernel remain TE/MCore implementations.
    This token-only path deliberately does not implement inference KV caches.
    """

    def forward(self, hidden_states, attention_mask=None, *, rotary_pos_emb=None, **kwargs):
        if (
            kwargs.get('inference_context') is not None
            or kwargs.get('packed_seq_params') is not None
        ):
            raise ValueError('This adapter is for independent pretraining windows')
        if rotary_pos_emb is None:
            raise ValueError('Explicit RoPE is required')
        self._original_freqs = rotary_pos_emb
        try:
            return super().forward(hidden_states, attention_mask, rotary_pos_emb=None, **kwargs)
        finally:
            self._original_freqs = None

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True
    ):
        if output_gate or not split_qkv:
            raise ValueError('Unsupported QKV mode')
        q, k, v = super().get_query_key_value_tensors(hidden_states, key_value_states)
        freq = self._original_freqs
        cos = freq.cos().to(v.dtype)
        sin = freq.sin().to(v.dtype)

        def rotate(x):
            first, second = x.chunk(2, dim=-1)
            return (x * cos + torch.cat((-second, first), dim=-1) * sin).to(v.dtype)

        return rotate(q), rotate(k), v


def make_config(spec):
    return TransformerConfig(
        num_layers=spec.num_layers,
        hidden_size=spec.hidden_size,
        num_attention_heads=spec.num_heads,
        num_query_groups=spec.num_key_value_heads or spec.num_heads,
        kv_channels=spec.hidden_size // spec.num_heads,
        ffn_hidden_size=spec.intermediate_size,
        normalization='RMSNorm',
        layernorm_epsilon=spec.rms_norm_eps,
        qk_layernorm=True,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        init_method_std=spec.initializer_range,
        params_dtype=torch.float32,
        bf16=False,
        fp16=False,
        fp32_residual_connection=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        apply_rope_fusion=False,
        rotary_interleaved=False,
        calculate_per_token_loss=False,
    )


class MCoreLocalLM(MegatronModule):
    def __init__(self, spec, embedding, head, *, window_tokens):
        if not spec.share_lm_head:
            raise ValueError('This migration requires the shared frozen head')
        config = make_config(spec)
        super().__init__(config)
        self.spec = spec
        self.window_tokens = window_tokens
        self.tp_group = parallel_state.get_tensor_model_parallel_group()
        self.token_embedding_weight = embedding.detach()
        self.lm_head_weight = head.detach()
        if (
            tuple(embedding.shape) != (spec.vocab_size, spec.input_size)
            or head.shape != embedding.shape
        ):
            raise ValueError('Target table shapes do not match the original LocalSpec')
        self.input_proj = nn.Linear(spec.input_size, spec.hidden_size, bias=False)
        layer_spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
        layer_spec.submodules.input_layernorm = OriginalRMSNorm
        layer_spec.submodules.self_attention.module = OriginalRotarySelfAttention
        layer_spec.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
        layer_spec.submodules.self_attention.submodules.q_layernorm = OriginalRMSNorm
        layer_spec.submodules.self_attention.submodules.k_layernorm = OriginalRMSNorm
        layer_spec.submodules.pre_mlp_layernorm = OriginalRMSNorm
        layer_spec.submodules.mlp.keywords['submodules'].linear_fc1 = TEColumnParallelLinear
        self.layers = nn.ModuleList(
            [
                build_module(layer_spec, config=config, layer_number=i + 1)
                for i in range(spec.num_layers)
            ]
        )
        if spec.moe_config:
            for layer in self.layers:
                layer.mlp = MCoreDominoMoE(config, spec.moe_config)
        self.final_norm = OriginalRMSNorm(
            config=config, hidden_size=spec.hidden_size, eps=spec.rms_norm_eps
        )
        self.bridge = nn.Linear(spec.hidden_size, spec.input_size, bias=False)
        self.rotary = RotaryEmbedding(
            config.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=False,
            rotary_base=spec.rope_theta,
            cp_group=parallel_state.get_context_parallel_group(),
        )
        self.chunk_size = 512
        self.num_sources = 3

    def head_logits(self, features):
        return F.linear(features.to(self.lm_head_weight.dtype), self.lm_head_weight)

    def features(self, ids):
        if ids.ndim != 2 or ids.shape[1] != self.window_tokens - 1:
            raise ValueError('Expected independent [batch, window_tokens-1] windows')
        x = self.input_proj(F.embedding(ids, self.token_embedding_weight))
        x = x.transpose(0, 1).contiguous()  # MCore uses [sequence,batch,hidden].
        start = self.spec.token_rope_start
        if start is None:
            start = self.spec.block_size
        rope = self.rotary(ids.shape[1], offset=start)
        for layer in self.layers:
            x, _ = layer(x, attention_mask=None, rotary_pos_emb=rope)
        x = self.final_norm(x).transpose(0, 1).contiguous()
        return self.bridge(x)

    def forward(self, ids, labels=None, *, normalizer=None, sources=None, return_metrics=False):
        for moe in self.moe_modules():
            moe._balance_mask = (
                labels.ne(-100).transpose(0, 1).contiguous() if labels is not None else None
            )
        try:
            features = self.features(ids)
        finally:
            for moe in self.moe_modules():
                moe._balance_mask = None
        if labels is None:
            return F.linear(features.to(self.lm_head_weight.dtype), self.lm_head_weight)
        if labels.shape != ids.shape or normalizer is None:
            raise ValueError('Training requires aligned labels and a global-label normalizer')
        from frozen_head_ce import normalized_frozen_head_ce

        loss = normalized_frozen_head_ce(
            features.flatten(0, 1),
            self.lm_head_weight,
            labels.flatten(),
            normalizer=normalizer,
            chunk_size=self.chunk_size,
        )
        if return_metrics:
            from pretraining.model import LocalLanguageModel

            return loss, LocalLanguageModel.metric_sums(self, features.detach(), labels, sources)
        return loss

    def moe_modules(self):
        return [layer.mlp.experts for layer in self.layers] if self.spec.moe_config else []

    def update_balance(self, *, collect_metrics=True):
        metrics = {}
        for i, moe in enumerate(self.moe_modules()):
            metrics.update(
                {
                    f'layer_{i}/{k}': v
                    for k, v in moe.update_balance(collect_metrics=collect_metrics).items()
                }
            )
        # A single device-to-host copy replaces one synchronizing scalar copy
        # per expert/metric. Preserve names, insertion order, and FP32 values.
        if not metrics:
            return {}
        return dict(zip(metrics, torch.stack(list(metrics.values())).cpu().tolist()))

    def _mapping(self):
        mapping = {
            'context_transformer.input_proj.weight': 'input_proj.weight',
            'context_transformer.final_norm.weight': 'final_norm.weight',
            'embed_proj.0.weight': 'bridge.weight',
        }
        for i in range(self.spec.num_layers):
            a = f'context_transformer.layers.{i}.'
            b = f'layers.{i}.'
            for old, new in {
                'input_layernorm.weight': 'input_layernorm.weight',
                'self_attn.o_proj.weight': 'self_attention.linear_proj.weight',
                'self_attn.q_norm.weight': 'self_attention.q_layernorm.weight',
                'self_attn.k_norm.weight': 'self_attention.k_layernorm.weight',
                'post_attention_layernorm.weight': 'pre_mlp_layernorm.weight',
            }.items():
                mapping[a + old] = b + new
            if self.spec.moe_config:
                for name in self.layers[i].mlp.experts.state_dict():
                    mapping[a + 'mlp.' + name] = b + 'mlp.experts.' + name
            else:
                mapping[a + 'mlp.down_proj.weight'] = b + 'mlp.linear_fc2.weight'
        return mapping

    def _pack_qkv(self, q, k, v):
        groups = self.spec.num_key_value_heads or self.spec.num_heads
        dim = self.spec.hidden_size // self.spec.num_heads
        h = self.spec.hidden_size
        return torch.cat(
            [
                q.reshape(groups, -1, dim, h),
                k.reshape(groups, 1, dim, h),
                v.reshape(groups, 1, dim, h),
            ],
            dim=1,
        ).reshape(-1, h)

    def _unpack_qkv(self, qkv):
        groups = self.spec.num_key_value_heads or self.spec.num_heads
        dim = self.spec.hidden_size // self.spec.num_heads
        n = self.spec.num_heads // groups
        h = self.spec.hidden_size
        q, k, v = torch.split(qkv.reshape(groups, n + 2, dim, h), [n, 1, 1], dim=1)
        return [x.reshape(-1, h) for x in (q, k, v)]

    @torch.no_grad()
    def load_hf_weights(self, weights):
        expected = self.hf_weights().keys()
        if weights.keys() != expected:
            raise ValueError(
                f'Export key mismatch: missing {expected-weights.keys()}, extra {weights.keys()-expected}'
            )
        target = dict(self.named_parameters())
        target.update(
            {
                n: b
                for n, b in self.named_buffers()
                if n.endswith(('routing_bias', 'balance_updates'))
            }
        )
        loaded = set()
        for old, new in self._mapping().items():
            if target[new].shape != weights[old].shape:
                raise ValueError(old)
            target[new].copy_(weights[old])
            loaded.add(new)
        for i in range(self.spec.num_layers):
            a = f'context_transformer.layers.{i}.'
            b = f'layers.{i}.'
            qkv = self._pack_qkv(
                *[weights[a + 'self_attn.' + name + '_proj.weight'] for name in ['q', 'k', 'v']]
            )
            packed = [(b + 'self_attention.linear_qkv.weight', qkv)]
            if not self.spec.moe_config:
                fc1 = torch.cat(
                    [weights[a + 'mlp.gate_proj.weight'], weights[a + 'mlp.up_proj.weight']], dim=0
                )
                packed.append((b + 'mlp.linear_fc1.weight', fc1))
            for new, value in packed:
                if target[new].shape != value.shape:
                    raise ValueError(new)
                target[new].copy_(value)
                loaded.add(new)
        if loaded != target.keys():
            raise ValueError(f'Unmapped trainable parameters {target.keys()-loaded}')

    def hf_weights(self, *, gradients=False):
        if gradients:
            state = {
                n: (p.main_grad if hasattr(p, 'main_grad') else p.grad)
                for n, p in self.named_parameters()
            }
        else:
            state = dict(self.named_parameters())
            state.update(
                {
                    n: b
                    for n, b in self.named_buffers()
                    if n.endswith(('routing_bias', 'balance_updates'))
                }
            )
        result = {old: state[new] for old, new in self._mapping().items() if new in state}
        for i in range(self.spec.num_layers):
            a = f'context_transformer.layers.{i}.'
            b = f'layers.{i}.'
            for name, value in zip(
                ['q', 'k', 'v'], self._unpack_qkv(state[b + 'self_attention.linear_qkv.weight'])
            ):
                result[a + 'self_attn.' + name + '_proj.weight'] = value
            if not self.spec.moe_config:
                gate, up = state[b + 'mlp.linear_fc1.weight'].chunk(2, dim=0)
                result[a + 'mlp.gate_proj.weight'] = gate
                result[a + 'mlp.up_proj.weight'] = up
        return result

    def export(self, root, identity, extra=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        weights = {n: v.detach().cpu().contiguous() for n, v in self.hf_weights().items()}
        save_file(weights, root / 'expert.safetensors')
        sha = hashlib.sha256((root / 'expert.safetensors').read_bytes()).hexdigest()
        metadata = dict(
            format='domino-local-pretrain-v1',
            spec=asdict(self.spec),
            target_identity=identity,
            weight_sha256=sha,
            keys=sorted(weights),
            extra=dict(extra or {}, pretrain_window_tokens=self.window_tokens),
        )
        (root / 'pretrain.json').write_text(json.dumps(metadata, indent=2) + '\n')
        return metadata
