import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pretraining.data import WindowData, sha_file
from pretraining.shuffled import GlobalShuffledWindowSampler, prepare_permutations


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    files = {}
    for source, size in zip(('fineweb', 'stackedu', 'finemath'), (23, 17, 11)):
        array = np.arange(size * 8, dtype='<i4').reshape(size, 8)
        array[::3, -2:] = -1
        path = root / f'{source}.train.bin'
        array.tofile(path)
        files[f'{source}.train'] = dict(
            file=path.name,
            windows=size,
            tokens=int((array[:, 1:] >= 0).sum()),
            sha256=sha_file(path),
        )
    (root / 'manifest.json').write_text(
        json.dumps(
            dict(
                format='domino-local-windows-v1',
                block_size=8,
                pad_id=0,
                sources=['fineweb', 'stackedu', 'finemath'],
                files=files,
            )
        )
    )
    data = WindowData(root)
    cache = tmp_path / 'permutations'
    prepare_permutations(data, cache, seed=42, max_tokens=20000)
    return data, cache


def sampler(dataset, rank, world=4):
    data, cache = dataset
    return GlobalShuffledWindowSampler(
        data, cache, seed=42, world_size=world, rank=rank, micro_batch=3, accumulation=2
    )


def test_all_ranks_cover_each_epoch_once_and_resume_exactly(dataset):
    data, _ = dataset
    ranks = [sampler(dataset, r) for r in range(4)]
    seen = [[] for _ in data.sources]
    source_tokens = np.zeros(3, dtype=np.int64)
    for step in range(1, 101):
        for micro in range(2):
            outputs = [s.sample(step=step, micro=micro, return_indices=True) for s in ranks]
            for _, labels, sources, indices in outputs:
                for i in range(3):
                    seen[i].extend(indices[sources == i].tolist())
                    source_tokens[i] += int((labels[sources == i] != -100).sum())
            assert all(s.state_dict() == ranks[0].state_dict() for s in ranks)
        if step == 37:
            restored = [sampler(dataset, r) for r in range(4)]
            for a, b in zip(ranks, restored):
                b.load_state_dict(json.loads(json.dumps(a.state_dict())))
            for micro in range(2):
                for a, b in zip(ranks, restored):
                    left = a.sample(step=38, micro=micro, return_indices=True)
                    right = b.sample(step=38, micro=micro, return_indices=True)
                    for x, y in zip(left, right):
                        np.testing.assert_array_equal(x, y)
            # Return to the saved state to keep the traversal check contiguous.
            for s in ranks:
                state = s.state_dict()
                # Restore independently captured step-37 cursor from the seen stream.
                state['source_draws'] = [len(x) for x in seen]
                state['next_micro'] = 37 * 2
                s.load_state_dict(state)
    for values, array in zip(seen, data.arrays):
        n = len(array)
        for start in range(0, len(values), n):
            chunk = values[start : start + n]
            assert len(set(chunk)) == len(chunk)
            if len(chunk) == n:
                assert sorted(chunk) == list(range(n))
    coverage = ranks[0].metrics(source_tokens.tolist())
    assert sum(v['distinct_label_positions'] for v in coverage.values()) == sum(
        data.manifest['files'][s + '.train']['tokens'] for s in data.sources
    )
    shares = source_tokens / source_tokens.sum()
    np.testing.assert_allclose(shares, data.weights, atol=0.035, rtol=0)


def test_reject_inconsistent_cursor_and_rank_layout(dataset):
    a = sampler(dataset, 0)
    a.sample(step=1)
    with pytest.raises(ValueError, match='Out-of-order'):
        a.sample(step=1)
    state = a.state_dict()
    state['source_draws'][0] += 1
    with pytest.raises(ValueError, match='cursor'):
        sampler(dataset, 1).load_state_dict(state)
    with pytest.raises(ValueError, match='world_size'):
        sampler(dataset, 0, world=2).load_state_dict(a.state_dict())


def test_cache_integrity_and_identity(dataset):
    data, cache = dataset
    with pytest.raises(ValueError, match='Wrong permutation'):
        GlobalShuffledWindowSampler(
            data, cache, seed=43, world_size=1, rank=0, micro_batch=1, accumulation=1
        )
    path = cache / 'fineweb.epoch0000.uint32'
    values = np.fromfile(path, dtype='<u4')
    values[:2] = values[1::-1]
    values.tofile(path)
    with pytest.raises(ValueError, match='Corrupt'):
        prepare_permutations(data, cache, seed=42, max_tokens=10000)
