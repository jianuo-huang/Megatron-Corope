"""Compare full native distributed saves and exercise the original joint loader."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from pretraining.model import initialize_draft_from_target, transferable
from safetensors.torch import load_file

from megatron.core import dist_checkpointing

ROOT = Path(__file__).resolve().parent


def equal(a, b, path='root'):
    if torch.is_tensor(a):
        assert (
            torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        ), path
        return 1
    if isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
        return 0
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        return sum(equal(v, b[k], path + '/' + str(k)) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        return sum(equal(v, w, path + '/' + str(i)) for i, (v, w) in enumerate(zip(a, b)))
    assert a == b, (path, a, b)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('continuous')
    p.add_argument('resumed')
    p.add_argument('--step', type=int, default=6)
    p.add_argument('--result', default=str(ROOT / 'resume_checks.json'))
    p.add_argument('--config', required=True)
    p.add_argument('--world-size', type=int, default=2)
    p.add_argument('--target', required=True)
    p.add_argument(
        '--joint-source', help='Optional external Dmax-DFlash checkout for the joint loader check'
    )
    a = p.parse_args()
    roots = [Path(a.continuous), Path(a.resumed)]
    checkpoints = [x / f'step_{a.step:06d}' for x in roots]
    result = {}
    dist.init_process_group(
        'gloo', init_method=f'file:///tmp/mcore-compare-{os.getpid()}', rank=0, world_size=1
    )
    result['export_tensors_bitwise_equal'] = equal(
        *[load_file(x / 'export/expert.safetensors') for x in checkpoints]
    )
    result['distributed_tensors_bitwise_equal'] = equal(
        *[dist_checkpointing.load_plain_tensors(str(x / 'distributed')) for x in checkpoints]
    )
    result['common_state_tensors_bitwise_equal'] = equal(
        *[dist_checkpointing.load_common_state_dict(str(x / 'distributed')) for x in checkpoints]
    )
    result['rng_tensors_bitwise_equal'] = sum(
        equal(*[torch.load(x / f'rng_rank{rank}.pt', weights_only=False) for x in checkpoints])
        for rank in range(a.world_size)
    )
    events = [
        [json.loads(line) for line in (x / 'metrics.jsonl').read_text().splitlines()] for x in roots
    ]
    fields = [
        'step',
        'tokens',
        'source_tokens',
        'valid_labels',
        'loss',
        'lr',
        'grad_norm',
        'diagnostics',
        'moe',
    ]
    equal(*[[{k: e[k] for k in fields if k in e} for e in es] for es in events])
    result['per_step_losses_grad_norms_counts_diagnostics_bitwise_equal'] = True
    result['progress'] = {k: events[0][-1][k] for k in ['step', 'tokens', 'source_tokens']}
    if a.joint_source:
        # Instantiate the actual joint class on meta, then allocate only the PT
        # transfer whitelist. This exercises strict loader/schema without allocating
        # unrelated target/draft weights or changing a joint experiment.
        source = Path(a.joint_source).resolve()
        sys.path.insert(0, str(source))
        modspec = importlib.util.spec_from_file_location(
            'joint_draft_for_transfer', source / 'specforge/modeling/draft/dflash.py'
        )
        mod = importlib.util.module_from_spec(modspec)
        sys.modules[modspec.name] = mod
        modspec.loader.exec_module(mod)
        from transformers import Qwen3Config

        cfg = Qwen3Config.from_json_file(a.config)
        with torch.device('meta'):
            draft = mod.DFlashDraftModel(cfg)
        for name, param in list(draft.named_parameters()):
            if transferable(name):
                parent, leaf = name.rsplit('.', 1)
                draft.get_submodule(parent).register_parameter(
                    leaf, torch.nn.Parameter(torch.zeros(param.shape, dtype=param.dtype))
                )
        before = {n: id(p) for n, p in draft.named_parameters() if not transferable(n)}
        for name, buffer in list(draft.named_buffers()):
            if transferable(name):
                parent, leaf = name.rsplit('.', 1)
                module = draft.get_submodule(parent)
                persistent = leaf not in module._non_persistent_buffers_set
                module.register_buffer(
                    leaf, torch.zeros(buffer.shape, dtype=buffer.dtype), persistent=persistent
                )
        meta = json.loads((checkpoints[1] / 'export/pretrain.json').read_text())
        loaded = initialize_draft_from_target(draft, checkpoints[1] / 'export', a.target)
        exported = load_file(checkpoints[1] / 'export/expert.safetensors')
        result['original_joint_loader_tensors_exact'] = equal(
            {k: v for k, v in draft.state_dict().items() if transferable(k)}, exported
        )
        assert before == {n: id(p) for n, p in draft.named_parameters() if not transferable(n)}
        result['joint_loader'] = loaded
        result['joint_loader_checked'] = True
    else:
        result['joint_loader_checked'] = False
    result['passed'] = True
    Path(a.result).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
