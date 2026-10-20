#!/usr/bin/env python3
"""Prepare a resumable, bounded public corpus with the validated window schema."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag

import numpy as np
import pyarrow.parquet as pq
import requests
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pretraining.data import SOURCES, document_split, sha_file

TOK = None
TLS = threading.local()


def init_tokenizer(path):
    global TOK
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    TOK = Tokenizer.from_file(path)


def encode(row):
    text = row.pop("text")
    row["content_sha256"] = hashlib.sha256(text.strip().encode()).hexdigest()
    row["ids"] = np.asarray(TOK.encode(text, add_special_tokens=False).ids, dtype=np.int32)
    return row


def fetch_code(args):
    row, tokenizer, proxy = args
    if not hasattr(TLS, "session"):
        TLS.session = requests.Session()
        TLS.session.proxies = {"https": proxy}
        TLS.tokenizer = Tokenizer.from_file(tokenizer)
    error = ""
    for attempt in range(3):
        try:
            response = TLS.session.get(
                "https://softwareheritage.s3.amazonaws.com/content/" + row["doc_id"],
                timeout=(10, 25),
            )
            response.raise_for_status()
            raw = gzip.decompress(response.content)
            if hashlib.sha1(raw).hexdigest() != row["doc_id"]:
                raise ValueError("SWH blob SHA1 mismatch")
            text = raw.decode(row.get("src_encoding") or "utf-8", errors="replace")
            row["content_sha256"] = hashlib.sha256(text.strip().encode()).hexdigest()
            row["ids"] = np.asarray(
                TLS.tokenizer.encode(text, add_special_tokens=False).ids, dtype=np.int32
            )
            row["content_verified"] = True
            return row
        except Exception as exc:
            error = str(exc)
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    return {**row, "error": error}


def bounded_map(executor, fn, values, capacity):
    pending = deque()
    values = iter(values)
    for _ in range(capacity):
        try:
            pending.append(executor.submit(fn, next(values)))
        except StopIteration:
            break
    while pending:
        yield pending.popleft().result()
        try:
            pending.append(executor.submit(fn, next(values)))
        except StopIteration:
            pass


def pack(tokens, block_size=8):
    n = (len(tokens) - 1 + block_size - 2) // (block_size - 1)
    positions = np.arange(n)[:, None] * (block_size - 1) + np.arange(block_size)[None, :]
    valid = positions < len(tokens)
    out = np.full((n, block_size), -1, dtype="<i4")
    out[valid] = tokens[positions[valid]]
    return out


def raw_rows(item, after, seed, code_fraction):
    origin = {k: item[k] for k in ("repo", "revision", "path", "sha256")}
    offset = 0
    for batch in pq.ParquetFile(item["local"]).iter_batches(batch_size=2048):
        if offset + len(batch) <= after + 1:
            offset += len(batch)
            continue
        for j, row in enumerate(batch.to_pylist()):
            index = offset + j
            if index <= after:
                continue
            if item["kind"] == "stackedu":
                blob = row["blob_id"]
                if not 128 <= row["length_bytes"] <= 1_000_000:
                    continue
                # Hash subsampling prevents taking only adjacent rows/repos.
                value = (
                    int.from_bytes(hashlib.sha256(f"{seed}:{blob}".encode()).digest()[:8], "big")
                    / 2**64
                )
                if value >= code_fraction:
                    continue
                yield dict(
                    source="stackedu",
                    doc_id=blob,
                    group_id="repository:" + row["repo_name"],
                    language=row["language"],
                    src_encoding=row.get("src_encoding"),
                    row_index=index,
                    origin=origin,
                    detected_licenses=row.get("detected_licenses"),
                )
            else:
                text = row["text"]
                if not text or len(text) > 1_000_000:
                    continue
                url = urldefrag(row.get("url", ""))[0]
                content_hash = hashlib.sha256(text.strip().encode()).hexdigest()
                yield dict(
                    source=item["kind"],
                    doc_id=row.get("id") or content_hash,
                    group_id="url:" + url if url else content_hash,
                    text=text,
                    row_index=index,
                    origin=origin,
                )
        offset += len(batch)


def save_json(path, value):
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(p)


def document_identity(row):
    """Avoid fetching already verified blobs or retokenizing the same raw row."""
    if row["source"] == "stackedu":
        return ("stackedu", str(row["doc_id"]))
    origin = row.get("origin", {})
    return (
        row["source"],
        origin.get("repo"),
        origin.get("revision"),
        origin.get("path"),
        str(row["doc_id"]),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-manifest", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--web-tokens", type=int, default=660_000_000)
    p.add_argument("--math-tokens", type=int, default=220_000_000)
    p.add_argument("--code-tokens-per-language", type=int, default=110_000_000)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--download-workers", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--code-fraction", type=float, default=0.125)
    p.add_argument("--proxy", default=os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"))
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--base-dataset", help="Extend a sealed dataset while preserving its exact validation files"
    )
    a = p.parse_args()
    for name in (
        "web_tokens",
        "math_tokens",
        "code_tokens_per_language",
        "workers",
        "download_workers",
    ):
        if getattr(a, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 < a.code_fraction <= 1:
        raise ValueError("code_fraction must be in (0, 1]")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (out / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_dataset(a, out)


def prepare_dataset(a, out):
    if (out / "manifest.json").exists():
        raise ValueError("Completed dataset is immutable.")
    items = json.loads(Path(a.raw_manifest).read_text())
    tokenizer = Path(a.tokenizer) / "tokenizer.json"
    eos = json.loads((Path(a.tokenizer) / "config.json").read_text())["eos_token_id"]
    identity = {
        n: sha_file(Path(a.tokenizer) / n)
        for n in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
        )
        if (Path(a.tokenizer) / n).is_file()
    }
    recipe = {
        k: v for k, v in vars(a).items() if k not in ("resume", "workers", "download_workers")
    }
    recipe.update(raw_manifest_sha256=sha_file(a.raw_manifest), tokenizer_files=identity)
    base_root = Path(a.base_dataset) if a.base_dataset else None
    base = None
    if base_root:
        base = json.loads((base_root / "manifest.json").read_text())
        if (
            base["format"] != "domino-local-windows-v1"
            or base["block_size"] != 8
            or base["seed"] != a.seed
            or base["validation_percent"] != 5
            or base["tokenizer_files"] != identity
            or base["pad_id"] != eos
        ):
            raise ValueError("Base dataset tokenizer, window, or split recipe differs")
        recipe["base_manifest_sha256"] = sha_file(base_root / "manifest.json")
    stages = []
    for source in ("fineweb", "finemath", "stackedu"):
        stages.extend(sorted([x for x in items if x["kind"] == source], key=lambda x: x["path"]))
    stats = {
        f"{s}.{split}": dict(windows=0, tokens=0, documents=0)
        for s in SOURCES
        for split in ("train", "validation")
    }
    state = dict(
        recipe=recipe,
        stats=stats,
        stage=0,
        cursor=-1,
        language_tokens={},
        duplicates=0,
        errors=0,
        processed=0,
        provenance_bytes=0,
        errors_bytes=0,
        known_documents_skipped=0,
        heldout_groups_skipped=0,
        heldout_content_skipped=0,
    )
    checkpoint = out / "prepare_state.json"
    if checkpoint.exists():
        if not a.resume:
            raise ValueError("Existing preparation state; use --resume")
        state = json.loads(checkpoint.read_text())
        if state["recipe"] != recipe:
            raise ValueError("Preparation recipe changed")
        stats = state["stats"]
    elif a.resume:
        raise ValueError("No preparation checkpoint")
    else:
        others = [x for x in out.iterdir() if x.name != ".prepare.lock"]
        if others:
            raise ValueError("Output is not empty")
        if base:
            for key, item in base["files"].items():
                source_path = base_root / item["file"]
                if sha_file(source_path) != item["sha256"]:
                    raise ValueError("Base dataset file hash mismatch")
                shutil.copyfile(source_path, out / f"{key}.bin")
                stats[key] = dict(item)
            if sha_file(base_root / "documents.jsonl") != base["documents_sha256"]:
                raise ValueError("Base provenance hash mismatch")
            shutil.copyfile(base_root / "documents.jsonl", out / "documents.jsonl")
            state["provenance_bytes"] = (out / "documents.jsonl").stat().st_size
            state["duplicates"] = base["duplicates_removed"]
    files = {}
    for key, item in stats.items():
        f = (out / f"{key}.bin").open("a+b")
        f.truncate(item["windows"] * 8 * 4)
        f.seek(0, 2)
        files[key] = f
    provenance = (out / "documents.jsonl").open("a+b")
    provenance.truncate(state["provenance_bytes"])
    provenance.seek(0)
    seen, known_ids, protected_hashes = set(), set(), set()
    language_counts = {}
    for line in provenance:
        row = json.loads(line)
        seen.add(row["content_sha256"])
        known_ids.add(document_identity(row))
        if row["split"] == "validation":
            protected_hashes.add(row["content_sha256"])
        elif row["source"] == "stackedu":
            language = row["language"]
            language_counts[language] = language_counts.get(language, 0) + row["tokens"]
    state["language_tokens"] = language_counts
    provenance.seek(0, 2)
    errors = (out / "fetch_errors.jsonl").open("a+b")
    errors.truncate(state["errors_bytes"])
    errors.seek(0, 2)
    start = time.monotonic()
    initial = sum(v["tokens"] for k, v in stats.items() if k.endswith(".train"))

    def persist():
        for f in files.values():
            f.flush()
            os.fsync(f.fileno())
        provenance.flush()
        os.fsync(provenance.fileno())
        errors.flush()
        os.fsync(errors.fileno())
        state.update(provenance_bytes=provenance.tell(), errors_bytes=errors.tell())
        save_json(checkpoint, state)
        total = sum(v["tokens"] for k, v in stats.items() if k.endswith(".train"))
        status = dict(
            stage=state["stage"],
            processed=state["processed"],
            train_tokens=total,
            domains={s: stats[s + ".train"]["tokens"] for s in SOURCES},
            language_tokens=state["language_tokens"],
            errors=state["errors"],
            duplicates=state["duplicates"],
            known_documents_skipped=state["known_documents_skipped"],
            heldout_groups_skipped=state["heldout_groups_skipped"],
            heldout_content_skipped=state["heldout_content_skipped"],
            tokens_per_second=(total - initial) / max(1, time.monotonic() - start),
        )
        save_json(out / "prepare_status.json", status)
        print(json.dumps(status), flush=True)

    def target_met(item):
        if item["kind"] == "fineweb":
            return stats["fineweb.train"]["tokens"] >= a.web_tokens
        if item["kind"] == "finemath":
            return stats["finemath.train"]["tokens"] >= a.math_tokens
        language = item["path"].split("/")[0]
        return state["language_tokens"].get(language, 0) >= a.code_tokens_per_language

    def candidates(item):
        for row in raw_rows(item, state["cursor"], a.seed, a.code_fraction):
            if base and document_identity(row) in known_ids:
                state["known_documents_skipped"] += 1
                continue
            if base and document_split(row["group_id"], a.seed, 5) == "validation":
                state["heldout_groups_skipped"] += 1
                continue
            yield row

    persist()
    try:
        for stage_index in range(state["stage"], len(stages)):
            item = stages[stage_index]
            if sha_file(item["local"]) != item["sha256"]:
                raise ValueError("Raw shard hash mismatch")
            if not target_met(item):
                source_rows = candidates(item)
                if item["kind"] == "stackedu":
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=a.download_workers)
                    iterator = bounded_map(
                        executor,
                        fetch_code,
                        ((row, str(tokenizer), a.proxy) for row in source_rows),
                        a.download_workers * 2,
                    )
                else:
                    executor = concurrent.futures.ProcessPoolExecutor(
                        max_workers=a.workers,
                        mp_context=mp.get_context("spawn"),
                        initializer=init_tokenizer,
                        initargs=(str(tokenizer),),
                    )
                    iterator = bounded_map(executor, encode, source_rows, a.workers * 4)
                try:
                    for row in iterator:
                        state["cursor"] = row.pop("row_index")
                        state["processed"] += 1
                        if "error" in row:
                            state["errors"] += 1
                            errors.write((json.dumps(row) + "\n").encode())
                        elif row["content_sha256"] in seen:
                            state["duplicates"] += 1
                            if row["content_sha256"] in protected_hashes:
                                state["heldout_content_skipped"] += 1
                        else:
                            seen.add(row["content_sha256"])
                            tokens = row.pop("ids")
                            if len(tokens):
                                if tokens[-1] != eos:
                                    tokens = np.append(tokens, np.int32(eos))
                                split = document_split(row["group_id"], a.seed, 5)
                                if base and split != "train":
                                    raise AssertionError(
                                        "Extension must not write new validation windows"
                                    )
                                key = row["source"] + "." + split
                                packed = pack(tokens)
                                files[key].write(packed.tobytes())
                                n = len(tokens) - 1
                                stats[key]["windows"] += len(packed)
                                stats[key]["tokens"] += n
                                stats[key]["documents"] += 1
                                if split == "train" and row["source"] == "stackedu":
                                    language = row["language"]
                                    state["language_tokens"][language] = (
                                        state["language_tokens"].get(language, 0) + n
                                    )
                                row.update(split=split, tokens=n)
                                known_ids.add(document_identity(row))
                                provenance.write(
                                    (json.dumps(row, ensure_ascii=False) + "\n").encode()
                                )
                        if state["processed"] % 2000 == 0:
                            persist()
                        if state["errors"] > max(100, state["processed"] * 0.02):
                            raise RuntimeError("Too many fetch failures; see errors log")
                        if target_met(item):
                            break
                finally:
                    executor.shutdown(wait=True, cancel_futures=True)
            state["stage"] = stage_index + 1
            state["cursor"] = -1
            persist()
        if (
            stats["fineweb.train"]["tokens"] < a.web_tokens
            or stats["finemath.train"]["tokens"] < a.math_tokens
        ):
            raise RuntimeError("Need more pinned raw shards to meet text token budgets")
        if any(
            state["language_tokens"].get(l, 0) < a.code_tokens_per_language
            for l in ("Python", "JavaScript")
        ):
            raise RuntimeError("Need more pinned Stack-Edu rows to meet code token budget")
    finally:
        try:
            persist()
        finally:
            for f in files.values():
                f.close()
            provenance.close()
            errors.close()
    for key, item in stats.items():
        if not item["windows"]:
            raise ValueError("Empty domain/split")
        item.update(file=f"{key}.bin", sha256=sha_file(out / f"{key}.bin"))
        if base and key.endswith(".validation") and item != base["files"][key]:
            raise ValueError("Fixed validation dataset changed during extension")
    manifest = dict(
        format="domino-local-windows-v1",
        block_size=8,
        seed=a.seed,
        validation_percent=5,
        pad_id=eos,
        tokenizer_files=identity,
        sources=SOURCES,
        files=stats,
        duplicates_removed=state["duplicates"],
        documents_sha256=sha_file(out / "documents.jsonl"),
        preparation_recipe=recipe,
        code_languages=["Python", "JavaScript"],
    )
    if base:
        manifest["fixed_validation"] = dict(
            base_dataset=str(base_root),
            base_manifest_sha256=recipe["base_manifest_sha256"],
            known_documents_skipped=state["known_documents_skipped"],
            heldout_groups_skipped=state["heldout_groups_skipped"],
            heldout_content_skipped=state["heldout_content_skipped"],
        )
    save_json(out / "manifest.json", manifest)
    print("DATASET_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
