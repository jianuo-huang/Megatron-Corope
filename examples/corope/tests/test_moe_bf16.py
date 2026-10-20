"""BF16 expert GEMMs must preserve routing and support empty experts in backward."""

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from domino_moe import DominoMoE


def _per_expert_reference(model, x):
    scores = model.router(x)
    indices = torch.argsort(scores + model.routing_bias, descending=True, stable=True, dim=-1)[
        :, : model.topk
    ]
    weights = scores.gather(1, indices)
    weights = weights / weights.sum(-1, keepdim=True) * model.routed_scale
    slots = []
    # Evaluate each expert independently for all tokens, then select route
    # slots. This oracle does not use grouped offsets or dispatch permutations.
    for expert in range(model.experts):
        up, gate = F.linear(x.bfloat16(), model.up_gate[expert].bfloat16()).chunk(2, -1)
        activation = (F.silu(gate.float()) * up.float()).bfloat16()
        value = F.linear(activation, model.down[expert].bfloat16()).float()
        slots.append(value[:, None, :] * (indices == expert)[:, :, None])
    routed = (sum(slots) * weights[:, :, None]).sum(1)
    return model.shared(x) + routed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA expert GEMM check")
@pytest.mark.parametrize("empty_experts", [False, True])
def test_bf16_experts_match_independent_reference(empty_experts):
    torch.manual_seed(42)
    config = dict(
        routed_experts=4,
        expert_intermediate_size=32,
        top_k=2,
        routed_scale=1.73,
        bias_update_rate=0.001,
        shared_experts=1,
    )
    model = DominoMoE(32, config).cuda()
    model.expert_compute_dtype = torch.bfloat16
    if empty_experts:
        model.routing_bias.copy_(torch.tensor([2.0, 1.0, -2.0, -3.0], device="cuda"))
    reference = copy.deepcopy(model)
    baseline = copy.deepcopy(model)
    baseline.expert_compute_dtype = None
    x = torch.randn(17, 32, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    mask = torch.arange(17, device="cuda") < 11
    model._balance_mask = mask
    baseline._balance_mask = mask
    upstream = torch.randn_like(x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = model(x)
        expected = _per_expert_reference(reference, xr)
        baseline(x.detach())
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=0.02)
    pairs = [(x.grad, xr.grad)] + [
        (p.grad, dict(reference.named_parameters())[n].grad) for n, p in model.named_parameters()
    ]
    for a, b in pairs:
        assert a.dtype == torch.float32 and torch.isfinite(a).all()
        relative_error = (a - b).norm() / b.norm().clamp_min(1e-20)
        assert relative_error < 0.02
    assert torch.equal(model.last_indices, baseline.last_indices)
    assert torch.equal(model.pending_counts, baseline.pending_counts)
    assert model.pending_counts.sum() == mask.sum() * model.topk
    model.update_balance(collect_metrics=False)
    baseline.update_balance(collect_metrics=False)
    assert torch.equal(model.routing_bias, baseline.routing_bias)
    assert model.router.weight.dtype == model.routing_bias.dtype == torch.float32
    if empty_experts:
        assert torch.count_nonzero(model.up_gate.grad[2:]) == 0
        assert torch.count_nonzero(model.down.grad[2:]) == 0
