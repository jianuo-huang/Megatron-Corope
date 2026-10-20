#!/usr/bin/env python3
"""Standalone causal local-expert pretraining; no target/DFlash model is built."""

from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import torch.distributed as dist
from pretraining.data import SOURCES, WindowData
from pretraining.metrics import evaluation_summary, flatten_metrics, summarize_sums
from pretraining.model import (
    LocalLanguageModel,
    LocalSpec,
    file_sha,
    load_target_embedding,
    load_target_head,
)
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def learning_rate(step, total, peak, warmup_ratio=0.04, min_ratio=0.1):
    warm = max(1, int(total * warmup_ratio))
    if step <= warm:
        return peak * step / warm
    phase = (step - warm) / max(1, total - warm)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * phase)))


def resume_identity(run):
    # Logging-only options are allowed to change while objective/data stay strict.
    identity = dict(run)
    identity.pop("logging_schema_version", None)
    identity["training"] = {k: v for k, v in run["training"].items() if k != "log_position_metrics"}
    return identity


def validation_metrics(results, sources=SOURCES):
    metrics = {
        "validation/weighted_nll": results["weighted_nll"],
        **flatten_metrics("eval", results["overall"]),
    }
    for source in sources:
        for name, value in results[source].items():
            if isinstance(value, list):
                metrics.update({f"validation/{source}/{name}/{i}": v for i, v in enumerate(value)})
            else:
                metrics[f"validation/{source}/{name}"] = value
    return metrics


def keep_recent_optimizer_states(output, keep):
    """Retain transferable weights; prune only old, completed optimizer states."""
    if keep <= 0:
        return
    checkpoints = sorted(Path(output).glob("step_*/trainer.pt"))
    for path in checkpoints[:-keep]:
        path.unlink()


def optimizer_for(model, lr, weight_decay):
    decay = []
    other = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else other).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": other, "weight_decay": 0}],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def evaluate(model, data, device, limit, batch_size):
    model.eval()
    results = {}
    depth = model.window_tokens - 1
    for source in range(len(data.sources)):
        sums = torch.zeros(3, depth, device=device, dtype=torch.float64)
        for ids, labels in data.validation_batches(source, limit, batch_size):
            ids = torch.from_numpy(ids).to(device)
            labels = torch.from_numpy(labels).to(device)
            with amp(device):
                features = model.features(ids).flatten(0, 1)
            flat = labels.flatten()
            positions = torch.arange(flat.numel(), device=device) % depth
            for start in range(0, len(flat), 128):
                target = flat[start : start + 128]
                pos = positions[start : start + 128]
                valid = target != -100
                with amp(device):
                    logits = model.head_logits(features[start : start + 128])
                nll = F.cross_entropy(logits.float(), target, reduction="none", ignore_index=-100)
                sums[0].scatter_add_(0, pos, nll.double())
                sums[1].scatter_add_(0, pos, valid.double())
                sums[2].scatter_add_(0, pos, ((logits.argmax(-1) == target) & valid).double())
        values = sums.cpu()
        den = values[1].clamp_min(1)
        results[data.sources[source]] = dict(
            nll=float(values[0].sum() / values[1].sum().clamp_min(1)),
            accuracy=float(values[2].sum() / values[1].sum().clamp_min(1)),
            tokens=int(values[1].sum()),
            nll_by_depth=[float(n / c) if c else None for n, c in zip(values[0], values[1])],
            accuracy_by_depth=[float(n / c) if c else None for n, c in zip(values[2], values[1])],
            tokens_by_depth=values[1].long().tolist(),
        )
    results["weighted_nll"] = sum(w * results[n]["nll"] for w, n in zip(data.weights, data.sources))
    results["overall"] = evaluation_summary(results, weights=data.weights, sources=data.sources)
    model.train()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft-config", required=True)
    p.add_argument("--target-tokenizer-embedding", required=True)
    p.add_argument("--data", required=True)
    p.add_argument(
        "--pretrain-window-tokens",
        type=int,
        default=None,
        help="Independent training window length; draft block/layout stay unchanged.",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument(
        "--max-tokens",
        type=int,
        help="Stop at this global valid-label budget, rounded up by at most one optimizer batch; overrides --steps and schedules LR by tokens.",
    )
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--keep-optimizer-states", type=int, default=0)
    p.add_argument("--wandb-project")
    p.add_argument("--wandb-name")
    p.add_argument("--micro-batch", type=int, default=128)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--loss-backend", choices=["auto", "liger", "chunked", "frozen-fused"], default="auto"
    )
    p.add_argument("--save-interval", type=int, default=200)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--validation-windows", type=int, default=256)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--log-position-metrics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--init-target-layer",
        type=int,
        help="Initialize the context head from this target decoder layer (plus the target final norm; square projections start at identity).",
    )
    p.add_argument("--resume")
    p.add_argument("--stop-after", type=int)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = p.parse_args()
    for n in (
        "steps",
        "chunk_size",
        "micro_batch",
        "accumulation",
        "save_interval",
        "eval_interval",
        "validation_windows",
        "log_interval",
    ):
        if getattr(a, n) < 1:
            raise ValueError(f"{n} must be positive")
    if a.max_tokens is not None and a.max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if a.keep_optimizer_states < 0:
        raise ValueError("keep_optimizer_states must be nonnegative")
    spec = LocalSpec.from_draft(json.loads(Path(a.draft_config).read_text()))
    window_tokens = (
        spec.window_tokens if a.pretrain_window_tokens is None else a.pretrain_window_tokens
    )
    if window_tokens < 2:
        raise ValueError("pretrain_window_tokens must be >= 2")
    if a.loss_backend == "auto":
        a.loss_backend = "chunked" if spec.share_lm_head else "liger"
    if spec.share_lm_head and a.loss_backend not in ("chunked", "frozen-fused"):
        raise ValueError("Shared-head pretraining requires chunked or frozen-fused CE.")
    if a.loss_backend == "frozen-fused" and (not spec.share_lm_head or a.device != "cuda"):
        raise ValueError("Frozen fused CE requires shared head on CUDA")
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(a.device, local_rank) if a.device == "cuda" else torch.device("cpu")
    if a.device == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if a.device == "cuda" else "gloo")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    lock = None
    if rank == 0:
        lock = (out / ".train.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not a.resume and any(x.name != ".train.lock" for x in out.iterdir()):
            raise ValueError("Output is not empty; use an explicit resume checkpoint.")
    if world > 1:
        dist.barrier()
    data = WindowData(a.data)
    validation = WindowData(a.data, "validation")
    weight, identity = load_target_embedding(a.target_tokenizer_embedding)
    if (
        data.manifest["tokenizer_files"] != identity["tokenizer_files"]
        or data.window != window_tokens
    ):
        raise ValueError("Data/model tokenizer or window-length mismatch.")
    model = LocalLanguageModel(spec, weight, window_tokens=window_tokens).to(device)
    if spec.share_lm_head:
        head_weight, head_identity = load_target_head(a.target_tokenizer_embedding)
        identity = dict(identity, **head_identity)
        model.lm_head_weight = head_weight.to(device)
    initialization = "random_local_expert"
    if a.init_target_layer is not None:
        from context_init import init_context_from_target_layer

        extra = [model.embed_proj[0]] if spec.share_lm_head else []
        info = init_context_from_target_layer(
            model.context_transformer,
            a.target_tokenizer_embedding,
            a.init_target_layer,
            extra_square_projections=extra,
        )
        initialization = dict(target_layer_init=info)
        print(json.dumps(dict(event="target_layer_init", **info)), flush=True)
    # Only the loaded embedding is BF16; trainable FP32 parameters are master weights.
    if device.type == "cpu":
        model.float()
    model.loss_backend = a.loss_backend
    model.chunk_size = a.chunk_size
    model.num_sources = len(data.sources)
    trainable = sum(x.numel() for x in model.parameters() if x.requires_grad)
    forbidden_imports = [
        n for n in ("sglang", "specforge.core", "specforge.modeling.target") if n in sys.modules
    ]
    if forbidden_imports:
        raise RuntimeError(f"Standalone pretraining imported target runtime: {forbidden_imports}")
    run = dict(
        logging_schema_version=2,
        spec=asdict(spec),
        target_identity=identity,
        data_sha256=data.identity,
        trainable_parameters=trainable,
        world_size=world,
        config_sha256=file_sha(a.draft_config),
        training={k: v for k, v in vars(a).items() if k not in ("resume", "stop_after", "output")},
        target_runtime_constructed=False,
        forbidden_runtime_imports=forbidden_imports,
        initialization=initialization,
    )
    optim = optimizer_for(model, a.lr, a.weight_decay)
    start = 0
    total_tokens = 0
    source_tokens = [0, 0, 0]
    if a.resume:
        resume = Path(a.resume)
        model.load_expert(resume, identity)
        state = torch.load(resume / "trainer.pt", map_location="cpu", weights_only=True)
        if resume_identity(state["run"]) != resume_identity(run):
            raise ValueError("Resume config/data/world size differs.")
        optim.load_state_dict(state["optimizer"])
        start = state["step"]
        total_tokens = state["tokens"]
        source_tokens = state["source_tokens"]
        torch.set_rng_state(state["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
    if world > 1:
        model._ddp_params_and_buffers_to_ignore = {
            "token_embedding.weight",
            "context_transformer.anchor_init_proj.weight",
        }
        wrapped = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    else:
        wrapped = model
    tracker = None
    if rank == 0 and a.wandb_project:
        import wandb

        wandb_id = out / "wandb_id.json"
        saved_id = (
            json.loads(wandb_id.read_text())["id"]
            if wandb_id.exists()
            else wandb.util.generate_id()
        )
        save_json(wandb_id, dict(id=saved_id))
        tracker = wandb.init(
            project=a.wandb_project,
            name=a.wandb_name,
            id=saved_id,
            resume="allow",
            config=run,
            dir=str(out),
        )
        tracker.define_metric("progress/step")
        tracker.define_metric("progress/tokens")
        tracker.define_metric("train/*", step_metric="progress/step")
        tracker.define_metric("eval/*", step_metric="progress/step")
        tracker.define_metric("validation/*", step_metric="progress/step")
    if rank == 0:
        save_json(out / "run.json", run)
        if not a.resume:
            initial = evaluate(model, validation, device, a.validation_windows, a.micro_batch)
            save_json(out / "validation_step_0.json", initial)
            with (out / "eval_metrics.jsonl").open("a") as f:
                f.write(json.dumps(dict(step=0, tokens=0, **initial)) + "\n")
            if tracker:
                tracker.log(
                    {
                        **validation_metrics(initial, data.sources),
                        "tokens": 0,
                        "progress/step": 0,
                        "progress/tokens": 0,
                    },
                    step=0,
                )
        print(
            json.dumps(
                dict(event="started", rank=rank, world=world, trainable=trainable, step=start)
            ),
            flush=True,
        )
    if world > 1:
        dist.barrier()
    stop = a.stop_after if a.max_tokens else min(a.steps, a.stop_after or a.steps)
    if (stop is not None and start >= stop) or (a.max_tokens and total_tokens >= a.max_tokens):
        raise ValueError("Resume checkpoint is already at/past the requested stop.")
    t0 = time.monotonic()
    start_tokens = total_tokens
    for step in itertools.count(start + 1):
        batches = [
            data.sample(a.micro_batch, seed=a.seed, step=step, micro=m, rank=rank)
            for m in range(a.accumulation)
        ]
        local_counts = np.zeros(len(data.sources), dtype=np.int64)
        for _, labels, sources in batches:
            for i in range(len(data.sources)):
                local_counts[i] += int((labels[sources == i] != -100).sum())
        counts = torch.tensor(local_counts, device=device)
        if world > 1:
            dist.all_reduce(counts)
        valid = int(counts.sum())
        scale = world / valid
        optim.zero_grad(set_to_none=True)
        lr = (
            learning_rate(min(total_tokens + valid, a.max_tokens), a.max_tokens, a.lr)
            if a.max_tokens
            else learning_rate(step, a.steps, a.lr)
        )
        for group in optim.param_groups:
            group["lr"] = lr
        loss_sum = torch.zeros((), device=device)
        will_log = (
            step % a.log_interval == 0
            or (a.max_tokens and total_tokens + valid >= a.max_tokens)
            or (stop is not None and step >= stop)
        )
        collect_metrics = a.log_position_metrics and will_log
        metric_sums = (
            torch.zeros(
                (3, len(data.sources), model.window_tokens - 1), dtype=torch.float64, device=device
            )
            if collect_metrics
            else None
        )
        for m, (ids, labels, sources) in enumerate(batches):
            sync = wrapped.no_sync() if world > 1 and m + 1 < len(batches) else nullcontext()
            with sync:
                with amp(device):
                    output = wrapped(
                        torch.from_numpy(ids).to(device),
                        torch.from_numpy(labels).to(device),
                        sources=(torch.from_numpy(sources).to(device) if collect_metrics else None),
                        return_metrics=collect_metrics,
                        loss_normalizer=(
                            valid / world if a.loss_backend == "frozen-fused" else None
                        ),
                    )
                    if collect_metrics:
                        loss, observed = output
                        metric_sums += observed
                    else:
                        loss = output
                (loss if a.loss_backend == "frozen-fused" else loss * scale).backward()
            # Restore the unnormalized sum for the existing global logging contract.
            loss_sum += loss.detach().float() * (
                1 / scale if a.loss_backend == "frozen-fused" else 1
            )
        if world > 1:
            dist.all_reduce(loss_sum)
            if collect_metrics:
                dist.all_reduce(metric_sums)
        norm = torch.nn.utils.clip_grad_norm_(
            [x for x in model.parameters() if x.requires_grad], 1.0
        )
        if not torch.isfinite(loss_sum) or not torch.isfinite(norm):
            raise RuntimeError("Non-finite loss/gradient")
        optim.step()
        total_tokens += valid
        budget_finished = total_tokens >= a.max_tokens if a.max_tokens else step >= a.steps
        final_step = budget_finished or (stop is not None and step >= stop)
        source_tokens = [x + y for x, y in zip(source_tokens, counts.cpu().tolist())]
        if rank == 0 and (step % a.log_interval == 0 or final_step):
            status = dict(
                step=step,
                total_steps=None if a.max_tokens else a.steps,
                max_tokens=a.max_tokens,
                tokens=total_tokens,
                source_tokens=source_tokens,
                loss=float(loss_sum / valid),
                lr=lr,
                grad_norm=float(norm),
                tokens_per_second=(total_tokens - start_tokens) / (time.monotonic() - t0),
                resumed=bool(a.resume),
            )
            if collect_metrics:
                diagnostics = summarize_sums(metric_sums.cpu().numpy(), data.sources)
                assert diagnostics["valid_tokens"] == valid
                if not math.isclose(
                    diagnostics["loss"], status["loss"], rel_tol=2e-5, abs_tol=2e-5
                ):
                    raise RuntimeError("Position metrics do not sum to the training objective")
                status.update({k: v for k, v in diagnostics.items() if k != "loss"})
                status["metric_scope"] = "current_pre_update_optimizer_batch"
            save_json(out / "status.json", status)
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(status) + "\n")
            print(json.dumps(status), flush=True)
            if tracker:
                tracker.log(
                    {
                        **{
                            f"train/{k}": v
                            for k, v in status.items()
                            if isinstance(v, (int, float))
                        },
                        **(flatten_metrics("train", diagnostics) if collect_metrics else {}),
                        "train/loss": status["loss"],
                        "tokens": total_tokens,
                        "progress/step": step,
                        "progress/tokens": total_tokens,
                    },
                    step=step,
                )
        if step % a.eval_interval == 0 or final_step:
            if rank == 0:
                result = evaluate(model, validation, device, a.validation_windows, a.micro_batch)
                save_json(out / f"validation_step_{step}.json", result)
                with (out / "eval_metrics.jsonl").open("a") as f:
                    f.write(json.dumps(dict(step=step, tokens=total_tokens, **result)) + "\n")
                if tracker:
                    tracker.log(
                        {
                            **validation_metrics(result, data.sources),
                            "tokens": total_tokens,
                            "progress/step": step,
                            "progress/tokens": total_tokens,
                        },
                        step=step,
                    )
            if world > 1:
                dist.barrier()
        if step % a.save_interval == 0 or final_step:
            if rank == 0:
                ckpt = out / f"step_{step:06d}"
                temporary = out / f".step_{step:06d}.incomplete"
                model.export(
                    temporary,
                    identity,
                    dict(step=step, tokens=total_tokens, data_sha256=data.identity),
                )
                state = dict(
                    run=run,
                    step=step,
                    tokens=total_tokens,
                    source_tokens=source_tokens,
                    optimizer=optim.state_dict(),
                    rng=torch.get_rng_state(),
                    cuda_rng=(
                        torch.cuda.get_rng_state(device)
                        if device.type == "cuda"
                        else torch.empty(0, dtype=torch.uint8)
                    ),
                )
                torch.save(state, temporary / "trainer.pt")
                temporary.rename(ckpt)
                save_json(
                    out / "latest.json", dict(checkpoint=str(ckpt), step=step, tokens=total_tokens)
                )
                keep_recent_optimizer_states(out, a.keep_optimizer_states)
            if world > 1:
                dist.barrier()
        if final_step:
            break
    if rank == 0:
        save_json(
            out / "completion.json",
            dict(
                completed_steps=step,
                planned_steps=None if a.max_tokens else a.steps,
                planned_tokens=a.max_tokens,
                tokens=total_tokens,
                finished=budget_finished,
                checkpoint=str(out / f"step_{step:06d}"),
            ),
        )
    if tracker:
        tracker.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
