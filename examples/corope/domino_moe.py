"""Shared/fine-grained MoE; task gradients and post-step Loss-Free routing.

Expert layout is [up, gate], matching the serving CUTLASS kernels. Routing
statistics are explicitly collected by the caller, never by autograd replay.
"""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class FP32Router(nn.Linear):
    def __init__(self, hidden, experts):
        super().__init__(hidden, experts, bias=False, dtype=torch.float32)

    def _apply(self, fn, recurse=True):
        # Preserve the original FP32 values when the rest of the model is cast.
        weight = self.weight.detach().clone()
        super()._apply(fn, recurse=recurse)
        self.weight.data = weight.to(device=self.weight.device, dtype=torch.float32)
        return self

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(x.float(), self.weight.float()).sigmoid()


class SharedMLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DominoMoE(nn.Module):
    def __init__(self, hidden, config):
        super().__init__()
        self.experts = int(config['routed_experts'])
        self.intermediate = int(config['expert_intermediate_size'])
        self.topk = int(config['top_k'])
        self.routed_scale = float(config['routed_scale'])
        self.bias_update_rate = float(config['bias_update_rate'])
        assert config['shared_experts'] == 1 and 0 < self.topk <= self.experts
        self.router = FP32Router(hidden, self.experts)
        self.shared = SharedMLP(hidden, self.intermediate)
        self.up_gate = nn.Parameter(torch.empty(self.experts, 2 * self.intermediate, hidden))
        self.down = nn.Parameter(torch.empty(self.experts, hidden, self.intermediate))
        nn.init.normal_(self.up_gate, std=0.02)
        nn.init.normal_(self.down, std=0.02)
        self.register_buffer('routing_bias', torch.zeros(self.experts, dtype=torch.float32))
        self.register_buffer('balance_updates', torch.zeros((), dtype=torch.int64))
        self.register_buffer(
            'pending_counts', torch.zeros(self.experts, dtype=torch.float32), persistent=False
        )
        self._balance_mask = None
        self.last_indices = None

    def _apply(self, fn, recurse=True):
        bias = self.routing_bias.clone()
        super()._apply(fn, recurse=recurse)
        self.routing_bias = bias.to(device=self.routing_bias.device, dtype=torch.float32)
        self.pending_counts = self.pending_counts.float()
        return self

    def forward(self, hidden_states):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        scores = self.router(x)
        # Stable ties match the inference kernel's lowest-expert-ID convention.
        indices = torch.argsort(scores + self.routing_bias, dim=-1, descending=True, stable=True)[
            :, : self.topk
        ]
        weights = scores.gather(1, indices)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20) * self.routed_scale
        if self.training and self._balance_mask is not None:
            with torch.no_grad():
                mask = self._balance_mask.reshape(-1, 1).expand_as(indices).float()
                self.pending_counts.scatter_add_(0, indices.reshape(-1), mask.reshape(-1))
        self.last_indices = indices.detach()
        # Sort token/expert assignments once. CUDA grouped GEMMs support ragged
        # offsets and autograd, including zero-token experts. No capacity drops.
        flat_ids = indices.reshape(-1)
        order = torch.argsort(flat_ids, stable=True)
        counts = torch.bincount(flat_ids, minlength=self.experts)
        offsets = counts.cumsum(0).to(torch.int32)
        dispatched = x.index_select(0, order // self.topk)
        up, gate = torch._grouped_mm(
            dispatched, self.up_gate.to(x.dtype).transpose(-1, -2), offs=offsets
        ).chunk(2, dim=-1)
        activation = (F.silu(gate) * up).contiguous()
        values = torch._grouped_mm(
            activation, self.down.to(x.dtype).transpose(-1, -2), offs=offsets
        )
        # Restore route slots before summing, with FP32 weights and reduction.
        restored = torch.empty_like(values).index_copy(0, order, values)
        routed = (
            (restored.reshape(x.shape[0], self.topk, -1).float() * weights.unsqueeze(-1))
            .sum(1)
            .to(x.dtype)
        )
        return (self.shared(x) + routed).reshape(shape)

    @torch.no_grad()
    def update_balance(self, process_group=None, *, collect_metrics=True):
        counts = self.pending_counts.clone()
        if dist.is_initialized():
            dist.all_reduce(counts, group=process_group)
        mean = counts.mean()
        self.routing_bias.add_(self.bias_update_rate * torch.sign(mean - counts))
        # Removing a common offset changes neither selection nor gate weights.
        self.routing_bias.sub_(self.routing_bias.mean())
        self.balance_updates.add_((mean > 0).to(torch.int64))
        self.pending_counts.zero_()
        # Updating the router is part of every optimizer step. Diagnostics are
        # read-only and only need to launch kernels on reporting steps.
        if not collect_metrics:
            return {}
        return {
            'moe/load_max_over_mean': counts.max() / mean.clamp_min(1),
            'moe/load_min_over_mean': counts.min() / mean.clamp_min(1),
            'moe/load_cv': counts.std(unbiased=False) / mean.clamp_min(1),
            'moe/valid_tokens': counts.sum() / self.topk,
            'moe/bias_abs_max': self.routing_bias.abs().max(),
            'moe/balance_updates': self.balance_updates.float(),
            **{
                f'moe/expert_{i}_fraction': counts[i] / counts.sum().clamp_min(1)
                for i in range(self.experts)
            },
        }
