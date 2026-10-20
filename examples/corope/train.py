"""Equivalent-window pretraining with MCore DDP, distributed AdamW, and DCP.

HF modes are comparison controls. New output directories only; no remote job
management. Frozen embedding/head tensors never enter MCore checkpoints.
"""

import argparse
import fcntl
import importlib.util
import itertools
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import frozen_head_ce
import numpy as np
import pretraining.data as original_data
import pretraining.model as original_model
import pretraining.shuffled as shuffled_data
import torch
import torch.distributed as dist
import tracking
from mcore_adapter import MCoreLocalLM
from pretraining.data import WindowData
from pretraining.metrics import summarize_sums
from pretraining.model import (
    LocalLanguageModel,
    LocalSpec,
    file_sha,
    load_target_embedding,
    load_target_head,
    read_expert,
    transferable,
)
from pretraining.shuffled import GlobalShuffledWindowSampler
from safetensors.torch import load_file

from megatron.core import dist_checkpointing, parallel_state, tensor_parallel
from megatron.core.distributed import DistributedDataParallel as MCoreDDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'reference_train.py'
base_spec = importlib.util.spec_from_file_location('original_pretrain', BASE)
original = importlib.util.module_from_spec(base_spec)
base_spec.loader.exec_module(original)


def save_json(path, data):
    p = Path(path)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(p)


@torch.no_grad()
def checkpoint(model, optimizer, out, progress, run, *, loading=False):
    state = model.sharded_state_dict()
    # v0.19 removed flattened ShardedTensor ranges; use the supported fully
    # parallel DP-reshardable optimizer format, with TP/PP fixed by this recipe.
    meta = {'distrib_optim_sharding_type': 'dp_reshardable'}
    optim_state = optimizer.sharded_state_dict(state, is_loading=loading, metadata=meta)
    bundle = {'model': state, 'optimizer': optim_state, 'progress': progress, 'run': run}
    if loading:
        restored = dist_checkpointing.load(bundle, str(out), strict='raise_all')
        if restored['run'] != run:
            raise ValueError('Checkpoint recipe/environment/source identity differs')
        model.load_state_dict(restored['model'], strict=True)
        optimizer.load_state_dict(restored['optimizer'])
        # The full parameter tensors are loaded before optimizer shards. No
        # initialization step is needed; distributed optimizer owns its moments.
        return restored['progress']
    if dist.get_rank() == 0:
        Path(out).mkdir(parents=True, exist_ok=False)
    dist.barrier()
    dist_checkpointing.save(bundle, str(out), async_sharded_save=False)
    return progress


def main():
    init_start = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', choices=['mcore', 'hf-fused', 'hf-chunked'], default='mcore')
    p.add_argument('--draft-config', required=True)
    p.add_argument('--target', required=True, help='Local target model/tokenizer directory')
    p.add_argument('--data', required=True)
    p.add_argument('--sampler', choices=['replacement', 'global-shuffle'], default='replacement')
    p.add_argument('--shuffle-cache')
    p.add_argument('--output', required=True)
    p.add_argument('--window', type=int, choices=[8, 64], required=True)
    p.add_argument('--micro-batch', type=int, required=True)
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--max-tokens', type=int)
    p.add_argument('--lr', type=float, default=6e-4)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--chunk-size', type=int, default=512)
    p.add_argument(
        '--moe-expert-dtype',
        choices=['input', 'bfloat16'],
        default='input',
        help='Opt-in expert GEMM precision; FP32 master weights/router/reduction are retained',
    )
    p.add_argument('--log-interval', type=int, default=50)
    p.add_argument('--eval-interval', type=int, default=1000)
    p.add_argument('--save-interval', type=int, default=1000)
    p.add_argument('--validation-windows', type=int, default=1024)
    p.add_argument('--resume')
    p.add_argument('--stop-after', type=int)
    p.add_argument(
        '--init-expert', help='Diagnostic comparison only; omit for random initialization'
    )
    p.add_argument(
        '--skip-data-hash',
        action='store_true',
        help='Diagnostic benchmark with separately verified data',
    )
    p.add_argument('--no-position-metrics', action='store_true')
    p.add_argument(
        '--deterministic',
        action='store_true',
        help='Deterministic route-gather backward and CUDA math, including restart checks',
    )
    tracking.add_arguments(p)
    a = p.parse_args()
    if a.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        # Keep deterministic kernels without filling every empty allocation
        # with NaNs/sentinels. Head chunks and MoE permutation buffers overwrite
        # every output element before reading it; full-state regression checks
        # cover this path. The fills otherwise consume substantial HBM bandwidth.
        torch.utils.deterministic.fill_uninitialized_memory = False
    if a.max_tokens is not None and a.max_tokens < 1:
        raise ValueError('max_tokens must be positive')
    for key in [
        'window',
        'micro_batch',
        'accumulation',
        'steps',
        'chunk_size',
        'log_interval',
        'eval_interval',
        'save_interval',
        'validation_windows',
    ]:
        if getattr(a, key) <= 0:
            raise ValueError(key)
    rank = int(os.environ['RANK'])
    local = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', device_id=device)
    parallel_state.initialize_model_parallel()
    tensor_parallel.model_parallel_cuda_manual_seed(a.seed)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.output)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        lock = (out / 'run.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'run.json').exists() and not a.resume:
            raise ValueError('Refusing to overwrite an existing run')
    dist.barrier()
    data = WindowData(a.data, verify=(rank == 0 and not a.skip_data_hash))
    validation = WindowData(a.data, split='validation', verify=(rank == 0 and not a.skip_data_hash))
    dist.barrier()
    if data.window != a.window:
        raise ValueError('Data window mismatch')
    sampler = None
    if a.sampler == 'global-shuffle':
        if not a.shuffle_cache:
            raise ValueError('--shuffle-cache is required')
        sampler = GlobalShuffledWindowSampler(
            data,
            a.shuffle_cache,
            seed=a.seed,
            world_size=world,
            rank=rank,
            micro_batch=a.micro_batch,
            accumulation=a.accumulation,
        )
    spec = LocalSpec.from_draft(json.loads(Path(a.draft_config).read_text()))
    if a.moe_expert_dtype != 'input' and (a.engine != 'mcore' or not spec.moe_config):
        raise ValueError('--moe-expert-dtype requires an MCore MoE model')
    if spec.moe_config and a.engine != 'mcore':
        raise ValueError('MoE training is validated through the MCore engine only')
    embedding, identity = load_target_embedding(a.target)
    head, head_id = load_target_head(a.target)
    identity.update(head_id)
    embedding = embedding.to(device)
    head = head.to(device)
    # Build the original initialization first, so all controls begin with
    # identical tensors, including the original constructor's RNG ordering.
    hf = LocalLanguageModel(spec, embedding, window_tokens=a.window).to(device)
    hf.lm_head_weight = head
    hf.chunk_size = a.chunk_size
    if a.init_expert:
        weights, _ = read_expert(a.init_expert, spec, identity)
    else:
        weights = {k: v for k, v in hf.state_dict().items() if transferable(k)}
    if a.engine == 'mcore':
        model = MCoreLocalLM(spec, embedding, head, window_tokens=a.window).to(device)
        model.load_hf_weights(weights)
        del hf, weights
        model.chunk_size = a.chunk_size
        for moe in model.moe_modules():
            moe.expert_compute_dtype = torch.bfloat16 if a.moe_expert_dtype == 'bfloat16' else None
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            use_distributed_optimizer=True,
            average_in_collective=False,
            check_for_nan_in_grad=True,
        )
        wrapped = MCoreDDP(model.config, ddp_config, model)
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer='adam',
                lr=a.lr,
                min_lr=a.lr * 0.1,
                weight_decay=a.weight_decay,
                adam_beta1=0.9,
                adam_beta2=0.95,
                adam_eps=1e-8,
                clip_grad=1.0,
                params_dtype=torch.float32,
                use_distributed_optimizer=True,
            ),
            [wrapped],
        )
    else:
        model = hf
        model.load_state_dict(weights, strict=False)
        del weights
        model.loss_backend = 'frozen-fused' if a.engine == 'hf-fused' else 'chunked'
        model._ddp_params_and_buffers_to_ignore = {
            'token_embedding.weight',
            'context_transformer.anchor_init_proj.weight',
        }
        wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
        optimizer = original.optimizer_for(model, a.lr, a.weight_decay)
    import transformer_engine

    import megatron.core

    recipe = {
        k: v
        for k, v in vars(a).items()
        if k not in ['output', 'resume', 'stop_after'] and not k.startswith('wandb_')
    }
    run = {
        'recipe': recipe,
        'spec': asdict(spec),
        'world_size': world,
        'target_identity': identity,
        'data_sha256': data.identity,
        'torch': torch.__version__,
        'mcore': megatron.core.__version__,
        'transformer_engine': transformer_engine.__version__,
        'source_sha256': {
            **{
                name: file_sha(ROOT / name)
                for name in ['train.py', 'mcore_adapter.py', 'tracking.py']
            },
            'original_trainer': file_sha(BASE),
            'original_model': file_sha(original_model.__file__),
            'original_data': file_sha(original_data.__file__),
            'frozen_head_ce': file_sha(frozen_head_ce.__file__),
        },
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'normalization': 'loss divided by global_valid/world, then DDP averages once',
    }
    run['source_sha256']['shuffled_data'] = file_sha(shuffled_data.__file__)
    for name in ['domino_context.py', 'domino_moe.py']:
        run['source_sha256'][name] = file_sha(ROOT / name)
    progress = {'step': 0, 'tokens': 0, 'source_tokens': [0] * len(data.sources)}
    if sampler:
        progress['sampler'] = sampler.state_dict()
    resume_seconds = None
    if a.resume:
        resume_start = time.perf_counter()
        if a.engine != 'mcore':
            raise ValueError(
                'This entry supports MCore distributed resume; HF is only a benchmark control'
            )
        progress = checkpoint(
            model, optimizer, Path(a.resume) / 'distributed', progress, run, loading=True
        )
        if sampler:
            sampler.load_state_dict(progress['sampler'])
            if sampler.next_micro != progress['step'] * a.accumulation:
                raise ValueError('Checkpoint sampler step differs')
        rng = torch.load(
            Path(a.resume) / f'rng_rank{rank}.pt', weights_only=False, map_location='cpu'
        )
        torch.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state(rng['cuda'])
        tensor_parallel.get_cuda_rng_tracker().set_states(rng['mcore'])
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.cuda.synchronize()
        resume_seconds = time.perf_counter() - resume_start
    if rank == 0:
        save_json(out / 'run.json', run)
    tracker = tracking.init_tracker(a, run, out) if rank == 0 else None
    elapsed_samples = []
    t_start = time.perf_counter()
    tokens_start = progress['tokens']
    initialization_seconds = t_start - init_start
    stop = a.stop_after if a.max_tokens else min(a.steps, a.stop_after or a.steps)
    if (stop is not None and progress['step'] >= stop) or (
        a.max_tokens and progress['tokens'] >= a.max_tokens
    ):
        raise ValueError('Checkpoint is already at/past the requested stop')
    for step in itertools.count(progress['step'] + 1):
        started = time.perf_counter()
        batches = [
            (
                sampler.sample(step=step, micro=m)
                if sampler
                else data.sample(a.micro_batch, seed=a.seed, step=step, micro=m, rank=rank)
            )
            for m in range(a.accumulation)
        ]
        # Labels are already on the CPU. Count there, then transfer/reduce once
        # instead of launching a scalar CUDA add for every source/microbatch.
        local_counts = np.zeros(len(data.sources), dtype=np.int64)
        for _, labels, sources in batches:
            for i in range(len(data.sources)):
                local_counts[i] += int((labels[sources == i] != -100).sum())
        counts = torch.from_numpy(local_counts).to(device)
        dist.all_reduce(counts)
        source_counts = counts.cpu().tolist()
        valid = sum(source_counts)
        normalizer = valid / world
        if not valid:
            raise ValueError('Empty global optimizer batch')
        final = step == stop or (a.max_tokens and progress['tokens'] + valid >= a.max_tokens)
        report = step % a.log_interval == 0 or step % a.eval_interval == 0 or final
        lr = (
            original.learning_rate(
                min(progress['tokens'] + valid, a.max_tokens), a.max_tokens, a.lr
            )
            if a.max_tokens
            else original.learning_rate(step, a.steps, a.lr)
        )
        for group in optimizer.param_groups:
            group['lr'] = lr
            # In MCore this multiplication is normally performed by its
            # scheduler, not by FusedAdam. We retain the original token schedule.
            if a.engine == 'mcore':
                group['weight_decay'] = a.weight_decay * group['wd_mult']
        if a.engine == 'mcore':
            wrapped.zero_grad_buffer()
            optimizer.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)
        collect = not a.no_position_metrics and step % a.log_interval == 0
        loss_sum = torch.zeros((), device=device) if report else None
        metric_sums = (
            torch.zeros((3, len(data.sources), a.window - 1), device=device, dtype=torch.float64)
            if collect
            else None
        )
        for m, (ids, labels, sources) in enumerate(batches):
            sync = wrapped.no_sync() if m + 1 < len(batches) else nullcontext()
            with sync, torch.autocast('cuda', dtype=torch.bfloat16):
                ids = torch.from_numpy(ids).to(device, non_blocking=True)
                labels = torch.from_numpy(labels).to(device, non_blocking=True)
                kwargs = dict(
                    sources=torch.from_numpy(sources).to(device) if collect else None,
                    return_metrics=collect,
                )
                if a.engine == 'mcore':
                    kwargs['normalizer'] = normalizer
                elif a.engine == 'hf-fused':
                    kwargs['loss_normalizer'] = normalizer
                value = wrapped(ids, labels, **kwargs)
                if collect:
                    loss, metrics = value
                    metric_sums += metrics
                else:
                    loss = value
                backward_loss = loss / normalizer if a.engine == 'hf-chunked' else loss
            backward_loss.backward()
            if report:
                loss_sum += backward_loss.detach() * normalizer
        if a.engine == 'mcore':
            wrapped.finish_grad_sync()
            success, norm, _ = optimizer.step()
            if not success:
                raise RuntimeError('Optimizer skipped a step')
            balance_metrics = model.update_balance(collect_metrics=report)
        else:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if report:
            dist.all_reduce(loss_sum)
        if collect:
            dist.all_reduce(metric_sums)
        torch.cuda.synchronize()
        duration = torch.tensor(time.perf_counter() - started, device=device, dtype=torch.float64)
        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        step_seconds = duration.item()
        elapsed_samples.append(step_seconds)
        progress = {
            'step': step,
            'tokens': progress['tokens'] + valid,
            'source_tokens': [x + y for x, y in zip(progress['source_tokens'], source_counts)],
        }
        if sampler:
            progress['sampler'] = sampler.state_dict()
        event = None
        if report:
            event = {
                **progress,
                'loss': float(loss_sum / valid),
                'lr': lr,
                'grad_norm': float(norm),
                'valid_labels': valid,
                'step_seconds': step_seconds,
                'effective_labels_per_second': valid / step_seconds,
                'position_metrics': collect,
                'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            }
            if sampler:
                event['data_coverage'] = sampler.metrics(progress['source_tokens'])
            if a.engine == 'mcore' and balance_metrics:
                event['moe'] = balance_metrics
        if collect:
            diagnostics = summarize_sums(metric_sums.cpu(), data.sources)
            if abs(diagnostics['loss'] - event['loss']) > 2e-4:
                raise ValueError('Training and position losses disagree')
            event['diagnostics'] = diagnostics
        # Shared-filesystem writes must follow log_interval. Per-step atomic
        # status rewrites stall rank 0 and then every rank at the next collective.
        tracked_event = None
        if rank == 0 and (step % a.log_interval == 0 or final):
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(event) + '\n')
            save_json(
                out / 'status.json',
                {**event, 'tokens_per_second': event['effective_labels_per_second']},
            )
            print(json.dumps({k: v for k, v in event.items() if k != 'diagnostics'}), flush=True)
            if tracker is not None:
                tracked_event = tracking.training_metrics(event)
        if step % a.eval_interval == 0 or final:
            if rank == 0:
                eval_start = time.perf_counter()
                evaluation = original.evaluate(
                    model, validation, device, a.validation_windows, a.micro_batch
                )
                evaluation = dict(seconds=time.perf_counter() - eval_start, **evaluation)
                save_json(out / f'validation_step_{step}.json', evaluation)
                if tracker is not None:
                    tracked_event = {
                        **(tracked_event or tracking.progress_metrics(event)),
                        **tracking.validation_metrics(evaluation),
                    }
            dist.barrier()
        # Merge train and eval before committing this step, so validation is
        # never dropped by a second W&B write at an already committed step.
        if tracked_event is not None:
            tracker.log(tracked_event, step=step)
        if step % a.save_interval == 0 or final:
            ckpt = out / f'step_{step:06d}'
            temporary = out / f'.step_{step:06d}.incomplete'
            if rank == 0:
                if ckpt.exists() or temporary.exists():
                    raise ValueError('Refusing to overwrite checkpoint')
                temporary.mkdir()
            dist.barrier()
            save_start = time.perf_counter()
            if a.engine == 'mcore':
                checkpoint(model, optimizer, temporary / 'distributed', progress, run)
                torch.save(
                    dict(
                        torch=torch.get_rng_state(),
                        cuda=torch.cuda.get_rng_state(),
                        mcore=tensor_parallel.get_cuda_rng_tracker().get_states(),
                        python=random.getstate(),
                        numpy=np.random.get_state(),
                    ),
                    temporary / f'rng_rank{rank}.pt',
                )
            dist.barrier()
            if rank == 0:
                exported = temporary / 'export'
                model.export(exported, identity, dict(**progress, data_sha256=data.identity))
                save_json(
                    temporary / 'complete.json',
                    dict(passed=True, seconds=time.perf_counter() - save_start, **progress),
                )
                temporary.rename(ckpt)
            dist.barrier()
        if final:
            break
    if rank == 0:
        completion = dict(
            passed=True,
            **progress,
            elapsed_seconds=time.perf_counter() - t_start,
            initialization_seconds=initialization_seconds,
            resume_seconds=resume_seconds,
            labels_per_second_including_eval_save=(progress['tokens'] - tokens_start)
            / (time.perf_counter() - t_start),
            step_seconds=elapsed_samples,
        )
        save_json(out / 'complete.json', completion)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
    if tracker is not None:
        tracker.summary.update(
            {f'completion/{k}': v for k, v in completion.items() if not isinstance(v, list)}
        )
        tracker.finish()


if __name__ == '__main__':
    main()
