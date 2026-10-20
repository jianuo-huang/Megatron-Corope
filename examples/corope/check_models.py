"""Dense/MoE forward, gradients, causal isolation, routing and export checks."""

import argparse
import json
import os
import types
from pathlib import Path

import torch
import torch.distributed as dist
from mcore_adapter import MCoreLocalLM
from pretraining.model import LocalLanguageModel, LocalSpec, transferable
from torch.nn import functional as F

from megatron.core import parallel_state, tensor_parallel


def error(a, b):
    d = a.detach().float() - b.detach().float()
    return float((d.square().mean() / b.detach().float().square().mean().clamp_min(1e-30)).sqrt())


def fixed_shape_expert_oracle(self, hidden_states):
    """Independent fixed-row GEMMs avoid count-dependent grouped-MM rounding."""
    shape = hidden_states.shape
    x = hidden_states.reshape(-1, shape[-1])
    scores = self.router(x)
    indices = torch.argsort(scores + self.routing_bias, dim=-1, descending=True, stable=True)[
        :, : self.topk
    ]
    weights = scores.gather(1, indices)
    weights = weights / weights.sum(-1, keepdim=True) * self.routed_scale
    slots = torch.zeros((len(x), self.topk, x.shape[-1]), device=x.device, dtype=torch.float32)
    for e in range(self.experts):
        up, gate = F.linear(x, self.up_gate[e]).chunk(2, -1)
        value = F.linear(F.silu(gate) * up, self.down[e]).float()
        slots += value[:, None, :] * (indices == e)[:, :, None]
    routed = (slots * weights[:, :, None]).sum(1).to(x.dtype)
    return (self.shared(x) + routed).reshape(shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--result', required=True)
    a = p.parse_args()
    rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    if 'RANK' in os.environ:
        dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    else:
        dist.init_process_group(
            'nccl',
            init_method=f'file:///tmp/mcore-model-check-{os.getpid()}',
            rank=0,
            world_size=1,
            device_id=torch.device('cuda', local),
        )
    parallel_state.initialize_model_parallel()
    tensor_parallel.model_parallel_cuda_manual_seed(42)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads(Path(a.config).read_text())
    cfg['vocab_size'] = 257
    spec = LocalSpec.from_draft(cfg)
    emb = torch.randn(257, spec.input_size, device='cuda', dtype=torch.bfloat16) * 0.02
    hf = LocalLanguageModel(spec, emb, window_tokens=64).cuda()
    hf.lm_head_weight = torch.randn_like(emb) * 0.02
    model = MCoreLocalLM(spec, emb, hf.lm_head_weight, window_tokens=64).cuda()
    weights = {n: v.detach().clone() for n, v in hf.state_dict().items() if transferable(n)}
    model.load_hf_weights(weights)
    assert weights.keys() == model.hf_weights().keys()
    assert all(torch.equal(v, model.hf_weights()[n]) for n, v in weights.items())
    ids = torch.randint(257, (3, 63), device='cuda')
    labels = ids.clone()
    labels[0, 7:] = -100
    labels[1, 41:] = -100
    mask = labels.ne(-100)
    if spec.moe_config:
        for l in hf.context_transformer.layers:
            l.mlp._balance_mask = mask
        for m in model.moe_modules():
            m._balance_mask = mask.T.contiguous()
    upstream = torch.randn(3, 63, spec.input_size, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        x = hf.features(ids)
        y = model.features(ids)
    (x.float() * upstream).mean().backward()
    (y.float() * upstream).mean().backward()
    original = dict(hf.named_parameters())
    grad_errors = {
        n: error(g, original[n].grad) for n, g in model.hf_weights(gradients=True).items()
    }
    record = {
        'feature_relative_rms': error(y, x),
        'grad_relative_rms': grad_errors,
        'trainable_parameters': sum(p.numel() for p in model.parameters()),
        'export_tensors': len(weights),
    }
    if spec.moe_config:
        for l, m in zip(hf.context_transformer.layers, model.moe_modules()):
            assert torch.equal(
                l.mlp.last_indices.reshape(3, 63, -1),
                m.last_indices.reshape(63, 3, -1).transpose(0, 1),
            )
            assert torch.equal(l.mlp.pending_counts, m.pending_counts)
            assert int(m.pending_counts.sum()) == int(mask.sum()) * spec.moe_config['top_k']
            l.mlp.update_balance()
            m.update_balance()
            assert torch.equal(l.mlp.routing_bias, m.routing_bias)
            assert torch.equal(l.mlp.balance_updates, m.balance_updates)
            assert m.routing_bias.dtype == torch.float32
            m._balance_mask = None
            l.mlp._balance_mask = None
        record['routing_counts_and_updates_exact'] = True
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        y = model.features(ids)
        changed = ids.clone()
        changed[0, -1] = (changed[0, -1] + 1) % 257
        z = model.features(changed)
        record['grouped_future_prefix_relative_rms'] = error(z[0, :-1], y[0, :-1])
        record['grouped_other_windows_relative_rms'] = error(z[1:], y[1:])
        if spec.moe_config:
            for m in model.moe_modules():
                m.forward = types.MethodType(fixed_shape_expert_oracle, m)
            oracle = model.features(ids)
            record['fixed_shape_oracle_relative_rms'] = error(y, oracle)
            assert record['fixed_shape_oracle_relative_rms'] < 0.02
            y = oracle
            z = model.features(changed)
        assert torch.equal(y[0, :-1], z[0, :-1])
        assert torch.equal(y[1:], z[1:])
        assert record['grouped_future_prefix_relative_rms'] < 0.02
        assert record['grouped_other_windows_relative_rms'] < 0.02
    record['causal_and_window_isolation'] = True
    record['passed'] = record['feature_relative_rms'] < 0.02 and max(grad_errors.values()) < 0.04
    if rank == 0:
        Path(a.result).write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record), flush=True)
    assert record['passed']
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
