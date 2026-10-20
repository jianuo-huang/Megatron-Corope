"""Prepare and verify the permutation cache before starting distributed training."""

import argparse
import json

from pretraining.data import WindowData
from pretraining.shuffled import prepare_permutations


def main() -> None:
    """Build immutable per-domain epoch permutations for the requested token budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--max-tokens', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--minimum-epochs', type=int, default=0)
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.minimum_epochs < 0:
        parser.error('max-tokens must be positive and minimum-epochs must be nonnegative')
    manifest = prepare_permutations(
        WindowData(args.data),
        args.cache,
        seed=args.seed,
        max_tokens=args.max_tokens,
        minimum_epochs=args.minimum_epochs,
    )
    print(
        json.dumps(
            {'identity': manifest['identity'], 'permutation_files': len(manifest['files'])},
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
