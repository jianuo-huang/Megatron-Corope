"""Run two-GPU resume gates in a fresh directory using caller-supplied fixtures."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Check continuous/resumed training, precision rejection, and optional transfer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--joint-source')
    parser.add_argument('--reference-export', help='Optional pre-relocation expert.safetensors')
    args = parser.parse_args()
    app = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(app.parents[1]), str(app), env.get('PYTHONPATH', '')])
    env['NVTE_FLASH_ATTN'] = '0'
    env['NVIDIA_TF32_OVERRIDE'] = '0'
    # Only six steps execute; the larger training token budget fixes the LR schedule.
    sys.path.insert(0, str(app))
    from pretraining.data import WindowData

    data = WindowData(args.data)
    global_windows = 6 * 2 * 2 * 2
    smallest_domain = min(len(array) for array in data.arrays)
    minimum_epochs = (global_windows + smallest_domain - 1) // smallest_domain + 1
    prepare = [
        sys.executable,
        str(app / 'prepare_shuffle.py'),
        '--data',
        args.data,
        '--cache',
        str(output / 'permutations'),
        '--seed',
        '42',
        '--max-tokens',
        str(global_windows * 63),
        '--minimum-epochs',
        str(minimum_epochs),
    ]
    with (output / 'prepare_shuffle.log').open('x') as log:
        subprocess.run(prepare, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    common = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--standalone',
        '--nproc-per-node=2',
        str(app / 'train.py'),
        '--target',
        str(Path(args.target).resolve()),
        '--draft-config',
        str(Path(args.config).resolve()),
        '--data',
        str(Path(args.data).resolve()),
        '--sampler',
        'global-shuffle',
        '--shuffle-cache',
        str(output / 'permutations'),
        '--window',
        '64',
        '--micro-batch',
        '2',
        '--accumulation',
        '2',
        '--steps',
        '1',
        '--max-tokens',
        '1000000',
        '--log-interval',
        '2',
        '--eval-interval',
        '2',
        '--save-interval',
        '2',
        '--validation-windows',
        '4',
        '--deterministic',
        '--wandb-mode',
        'disabled',
        '--moe-expert-dtype',
        'bfloat16',
    ]
    for name, stop, resume in [
        ('continuous', 6, False),
        ('resumed', 2, False),
        ('resumed', 6, True),
    ]:
        command = common + ['--output', str(output / name), '--stop-after', str(stop)]
        if resume:
            command += ['--resume', str(output / name / 'step_000002')]
        tag = f'{name}_{stop}'
        (output / f'{tag}.argv.json').write_text(json.dumps(command, indent=2) + '\n')
        print(f'Running {tag}', flush=True)
        with (output / f'{tag}.log').open('x') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    check = [
        sys.executable,
        str(app / 'check_resume.py'),
        str(output / 'continuous'),
        str(output / 'resumed'),
        '--step',
        '6',
        '--world-size',
        '2',
        '--config',
        args.config,
        '--target',
        args.target,
        '--result',
        str(output / 'resume_check.json'),
    ]
    if args.joint_source:
        check += ['--joint-source', str(Path(args.joint_source).resolve())]
    with (output / 'resume_check.log').open('x') as log:
        subprocess.run(check, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    command[command.index('--moe-expert-dtype') + 1] = 'input'
    with (output / 'precision_rejection.log').open('x') as log:
        rejected = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    expected = 'Checkpoint recipe/environment/source identity differs'
    if rejected.returncode == 0 or expected not in (output / 'precision_rejection.log').read_text():
        raise AssertionError('Changing precision must fail the strict checkpoint recipe check')
    result = {'resume_passed': True, 'precision_change_rejected': True}
    if args.reference_export:
        actual = output / 'continuous/step_000006/export/expert.safetensors'

        def digest(path: str | Path) -> str:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        result['relocated_export_sha256'] = digest(actual)
        result['reference_export_sha256'] = digest(args.reference_export)
        assert result['relocated_export_sha256'] == result['reference_export_sha256']
        result['pre_relocation_export_identical'] = True
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
