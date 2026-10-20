#!/usr/bin/env python3
"""Tokenize provenance-bearing public documents without importing target code."""

import argparse
import json
import sys
from pathlib import Path

# Import the small standalone package, not specforge's eager runtime __init__.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pretraining.data import build_dataset, sha_file
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--documents", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-percent", type=int, default=5)
    a = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    names = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    ]
    identity = {
        n: sha_file(Path(a.tokenizer) / n) for n in names if (Path(a.tokenizer) / n).is_file()
    }
    with open(a.documents) as f:
        result = build_dataset(
            (json.loads(x) for x in f),
            a.output,
            tokenizer,
            identity,
            a.block_size,
            a.seed,
            a.validation_percent,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
