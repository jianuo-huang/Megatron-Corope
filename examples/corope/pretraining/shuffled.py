"""Global, per-domain shuffled epochs with an explicit resumable draw cursor.

Every rank reconstructs the same global domain sequence, then reads a disjoint
slice of each domain's permutation. A domain repeats only after its epoch ends.
Permutation files are prepared and verified before distributed training starts.
"""

import fcntl
import json
import math
from pathlib import Path

import numpy as np

from .data import sha_file

ALGORITHM = 'pcg64-global-domain-permutation-v1'


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare_permutations(data, root, *, seed, max_tokens, minimum_epochs=0):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(
        algorithm=ALGORITHM,
        data_sha256=data.identity,
        seed=seed,
        sources=list(data.sources),
        sizes=[len(a) for a in data.arrays],
    )
    with (root / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / 'manifest.json'
        manifest = (
            json.loads(path.read_text()) if path.exists() else dict(identity=identity, files={})
        )
        if manifest['identity'] != identity:
            raise ValueError('Permutation cache identity differs')
        for source, weight, array, mean in zip(data.sources, data.weights, data.arrays, data.means):
            n = len(array)
            if n >= 2**32:
                raise ValueError('uint32 permutation limit exceeded')
            epochs = max(minimum_epochs, math.ceil(max_tokens * weight / (n * mean)) + 1)
            source_index = data.sources.index(source)
            for epoch in range(epochs):
                name = f'{source}.epoch{epoch:04d}.uint32'
                target = root / name
                if name in manifest['files']:
                    item = manifest['files'][name]
                    if target.stat().st_size != n * 4 or sha_file(target) != item['sha256']:
                        raise ValueError('Corrupt permutation cache')
                    continue
                if target.exists():
                    raise ValueError('Unrecorded permutation cache file')
                permutation = np.arange(n, dtype='<u4')
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, source_index, epoch, 99173])
                )
                rng.shuffle(permutation)
                temporary = target.with_suffix('.building')
                permutation.tofile(temporary)
                del permutation
                digest = sha_file(temporary)
                temporary.replace(target)
                manifest['files'][name] = dict(size=n, sha256=digest)
                save_json(path, manifest)
        return manifest


class GlobalShuffledWindowSampler:
    def __init__(self, data, cache, *, seed, world_size, rank, micro_batch, accumulation):
        if not (world_size > 0 and 0 <= rank < world_size and micro_batch > 0 and accumulation > 0):
            raise ValueError('Invalid distributed sampler dimensions')
        self.data, self.cache = data, Path(cache)
        self.seed, self.world, self.rank = seed, world_size, rank
        self.micro_batch, self.accumulation = micro_batch, accumulation
        self.manifest = json.loads((self.cache / 'manifest.json').read_text())
        self.identity = dict(
            algorithm=ALGORITHM,
            data_sha256=data.identity,
            seed=seed,
            sources=list(data.sources),
            sizes=[len(a) for a in data.arrays],
        )
        if self.manifest['identity'] != self.identity:
            raise ValueError('Wrong permutation cache')
        self.probabilities = np.asarray(data.weights) / np.asarray(data.means)
        self.probabilities /= self.probabilities.sum()
        self.draws = [0] * len(data.sources)
        self.next_micro = 0
        self.maps = {}

    def state_dict(self):
        return dict(
            algorithm=ALGORITHM,
            data_sha256=self.data.identity,
            seed=self.seed,
            world_size=self.world,
            micro_batch=self.micro_batch,
            accumulation=self.accumulation,
            next_micro=self.next_micro,
            source_draws=list(self.draws),
        )

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in expected.keys() - {'next_micro', 'source_draws'}:
            if state[key] != expected[key]:
                raise ValueError(f'Sampler resume identity differs: {key}')
        draws = state['source_draws']
        if len(draws) != len(self.draws) or any(type(v) is not int or v < 0 for v in draws):
            raise ValueError('Invalid source draw cursor')
        if type(state['next_micro']) is not int or state['next_micro'] < 0:
            raise ValueError('Invalid micro cursor')
        if sum(draws) != state['next_micro'] * self.world * self.micro_batch:
            raise ValueError('Global draw cursor disagrees with micro cursor')
        self.draws, self.next_micro = list(draws), state['next_micro']

    def _indices(self, source, begin, count):
        n = len(self.data.arrays[source])
        output = np.empty(count, dtype=np.int64)
        offset = 0
        while offset < count:
            epoch, position = divmod(begin + offset, n)
            key = source, epoch
            if key not in self.maps:
                name = f'{self.data.sources[source]}.epoch{epoch:04d}.uint32'
                if name not in self.manifest['files']:
                    raise ValueError(f'Unprepared epoch {name}')
                path = self.cache / name
                if path.stat().st_size != n * 4:
                    raise ValueError('Permutation file size differs')
                self.maps[key] = np.memmap(path, dtype='<u4', mode='r', shape=(n,))
            take = min(count - offset, n - position)
            output[offset : offset + take] = self.maps[key][position : position + take]
            offset += take
        return output

    def sample(self, *, step, micro=0, return_indices=False):
        expected = (step - 1) * self.accumulation + micro
        if not 0 <= micro < self.accumulation or expected != self.next_micro:
            raise ValueError('Out-of-order sampler call or missing checkpoint cursor')
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.next_micro, 55109]))
        global_sources = rng.choice(
            len(self.data.sources), self.world * self.micro_batch, p=self.probabilities
        )
        first = self.rank * self.micro_batch
        sources = global_sources[first : first + self.micro_batch]
        indices = np.empty(self.micro_batch, dtype=np.int64)
        batch = np.empty((self.micro_batch, self.data.window), dtype=np.int32)
        for i, array in enumerate(self.data.arrays):
            selected = np.flatnonzero(sources == i)
            earlier = int(np.count_nonzero(global_sources[:first] == i))
            indices[selected] = self._indices(i, self.draws[i] + earlier, len(selected))
            batch[selected] = array[indices[selected]]
            self.draws[i] += int(np.count_nonzero(global_sources == i))
        self.next_micro += 1
        value = (*self.data.unpack(batch), sources)
        return (*value, indices) if return_indices else value

    def metrics(self, source_tokens):
        result = {}
        for source, array, mean, draws, tokens in zip(
            self.data.sources, self.data.arrays, self.data.means, self.draws, source_tokens
        ):
            n = len(array)
            result[source] = dict(
                window_draws=draws,
                completed_epochs=draws // n,
                epoch_fraction=(draws % n) / n,
                distinct_windows=min(draws, n),
                distinct_label_positions=min(tokens, round(n * mean)),
            )
        return result
