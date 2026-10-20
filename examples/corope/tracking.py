"""Pretrain-only W&B metrics, shared by live training and historical imports."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

DEFAULT_PROJECT = 'corope-pretrain'
DEFAULT_ENTITY = None
DEFAULT_BASE_URL = 'https://api.wandb.ai'
SOURCES = ('fineweb', 'stackedu', 'finemath')


def flatten(prefix, value):
    """Keep numeric scalars and position indices; omit missing observations."""
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = prefix if key == 'domains' else f'{prefix}/{key}'
            result.update(flatten(child, item))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(flatten(f'{prefix}/d{index}', item))
    elif isinstance(value, (int, float)):
        result[prefix] = value
    return result


def progress_metrics(event):
    return {
        'progress/step': event['step'],
        'progress/tokens': event['tokens'],
        'progress/billion_tokens': event['tokens'] / 1e9,
    }


def training_metrics(event):
    values = {
        f'train/{key}': value
        for key, value in event.items()
        if isinstance(value, (int, float)) and key not in ('step', 'tokens')
    }
    values.update(flatten('train', event.get('diagnostics', {})))
    # Keep the actual optimizer-batch loss, rather than its diagnostic reduction.
    values['train/loss'] = event['loss']
    values.update(flatten('train', event.get('moe', {})))
    values.update(flatten('data', event.get('data_coverage', {})))
    for source, tokens in zip(SOURCES, event.get('source_tokens', [])):
        values[f'progress/{source}_tokens'] = tokens
    return {**values, **progress_metrics(event)}


def validation_metrics(evaluation):
    values = {'validation/weighted_nll': evaluation['weighted_nll']}
    values.update(flatten('eval', evaluation.get('overall', {})))
    for source in SOURCES:
        values.update(flatten(f'validation/{source}', evaluation[source]))
    if 'seconds' in evaluation:
        values['validation/seconds'] = evaluation['seconds']
    return values


def define_metrics(tracker):
    tracker.define_metric('progress/step')
    tracker.define_metric('progress/tokens')
    tracker.define_metric('progress/billion_tokens')
    for prefix in ('train', 'eval', 'validation', 'timing', 'data'):
        tracker.define_metric(f'{prefix}/*', step_metric='progress/tokens')


def add_arguments(parser):
    parser.add_argument('--wandb-project', default=os.getenv('WANDB_PROJECT', DEFAULT_PROJECT))
    parser.add_argument('--wandb-entity', default=os.getenv('WANDB_ENTITY', DEFAULT_ENTITY))
    parser.add_argument('--wandb-base-url', default=os.getenv('WANDB_BASE_URL', DEFAULT_BASE_URL))
    parser.add_argument('--wandb-name')
    parser.add_argument(
        '--wandb-mode',
        choices=['online', 'offline', 'disabled'],
        default=os.getenv('WANDB_MODE', 'disabled'),
    )
    parser.add_argument('--wandb-dir', help='Node-local SDK spool directory; defaults to /tmp')


def init_tracker(args, config, output):
    """Call on rank 0 only. SDK work is queued to its background service."""
    if args.wandb_mode == 'disabled':
        return None
    import wandb

    output = Path(output).resolve()
    identity_path = output / 'wandb_id.json'
    identity = {
        'id': hashlib.sha256(str(output).encode()).hexdigest()[:16],
        'project': args.wandb_project,
        'entity': args.wandb_entity,
        'base_url': args.wandb_base_url,
    }
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('W&B destination differs from the saved run identity')
    spool = Path(args.wandb_dir or tempfile.gettempdir()) / 'corope-pretrain-wandb' / identity['id']
    spool.mkdir(parents=True, exist_ok=True)
    name = args.wandb_name or (output.parent.name if output.name == 'pretrain' else output.name)
    tracker = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=identity['id'],
        name=name,
        job_type='pretrain',
        tags=['pretrain', config['recipe']['engine']],
        config=config,
        mode=args.wandb_mode,
        resume='allow',
        dir=str(spool),
        settings=wandb.Settings(
            base_url=args.wandb_base_url,
            console='off',
            disable_code=True,
            disable_git=True,
            init_timeout=60,
        ),
    )
    temporary = identity_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(identity, indent=2) + '\n')
    temporary.replace(identity_path)
    define_metrics(tracker)
    print(
        json.dumps({'event': 'wandb_initialized', 'url': tracker.url, 'mode': args.wandb_mode}),
        flush=True,
    )
    return tracker
