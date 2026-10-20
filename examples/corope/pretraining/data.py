"""Document-isolated local windows and reproducible token-mixture sampling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SOURCES = ("fineweb", "stackedu", "finemath")


def sha_file(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def document_split(group, seed=42, validation_percent=5):
    key = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return "validation" if int.from_bytes(key[:8], "big") % 100 < validation_percent else "train"


def windows(tokens, block_size=8):
    """Each label is predicted once; only the last token overlaps as anchor."""
    if block_size < 2:
        raise ValueError("block_size must be >= 2")
    for start in range(0, len(tokens) - 1, block_size - 1):
        ids = tokens[start : start + block_size]
        if len(ids) > 1:
            yield ids + [-1] * (block_size - len(ids))


# High bit marking "this element is a valid input token but its label is not
# supervised" (outside an assistant span). Distinct from -1 (tail padding).
UNSUPERVISED_FLAG = 1 << 20


def chat_windows(tokens, supervised, window=16, stride=8):
    """Sliding chat windows: stride matches the serving block size so one
    window's trailing tokens are the next window's verified prefix. Windows
    whose labels are all outside assistant spans are dropped."""
    if window < 2 or not 1 <= stride < window:
        raise ValueError("Need window >= 2 and 1 <= stride < window.")
    if len(tokens) != len(supervised):
        raise ValueError("tokens and supervised mask lengths differ.")
    if any(t < 0 or t >= UNSUPERVISED_FLAG for t in tokens):
        raise ValueError("Token id collides with the supervision flag bit.")
    for start in range(0, len(tokens) - 1, stride):
        ids = tokens[start : start + window]
        mask = supervised[start : start + window]
        if len(ids) < 2 or not any(mask[1:]):
            continue
        values = [t if m else t | UNSUPERVISED_FLAG for t, m in zip(ids, mask)]
        yield values + [-1] * (window - len(values))


def chat_window_array(tokens, supervised, window=16, stride=8):
    """Vectorized chat_windows with identical output, for full-corpus builds."""
    tokens = np.asarray(tokens, dtype=np.int64)
    supervised = np.asarray(supervised, dtype=bool)
    if window < 2 or not 1 <= stride < window:
        raise ValueError("Need window >= 2 and 1 <= stride < window.")
    if tokens.shape != supervised.shape or tokens.ndim != 1:
        raise ValueError("tokens and supervised mask lengths differ.")
    if tokens.size and (tokens.min() < 0 or tokens.max() >= UNSUPERVISED_FLAG):
        raise ValueError("Token id collides with the supervision flag bit.")
    if tokens.size < 2:
        return np.empty((0, window), dtype=np.int32)
    starts = np.arange(0, tokens.size - 1, stride)
    index = starts[:, None] + np.arange(window)[None, :]
    inside = index < tokens.size
    index = np.minimum(index, tokens.size - 1)
    mask = supervised[index] & inside
    values = np.where(mask, tokens[index], tokens[index] | UNSUPERVISED_FLAG)
    values = np.where(inside, values, -1)
    return values[mask[:, 1:].any(axis=1)].astype(np.int32)


def build_window_dataset(
    records,
    output,
    tokenizer_files,
    pad_id,
    sources=("perfectblend",),
    weights=(1.0,),
    window=16,
    stride=8,
    seed=42,
    validation_percent=2,
    extra=None,
):
    """records: iterables of (source, group_id, token_ids, supervised_mask);
    token ids already produced by the target tokenizer (for chat data, exactly
    like joint training: same template and max length)."""
    if len(sources) != len(weights) or min(weights) <= 0:
        raise ValueError("Need one positive weight per source.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    streams = {}
    summaries = {}
    for split in ("train", "validation"):
        for source in sources:
            key = f"{source}.{split}"
            streams[key] = (output / f"{key}.bin").open("wb")
            summaries[key] = dict(windows=0, tokens=0, documents=0)
    try:
        for source, group, tokens, supervised in records:
            split = document_split(str(group), seed, validation_percent)
            key = f"{source}.{split}"
            array = chat_window_array(tokens, supervised, window, stride)
            if len(array):
                streams[key].write(array.astype("<i4").tobytes())
                labels = array[:, 1:]
                summaries[key]["tokens"] += int(
                    ((labels >= 0) & ((labels & UNSUPERVISED_FLAG) == 0)).sum()
                )
                summaries[key]["windows"] += len(array)
                summaries[key]["documents"] += 1
    finally:
        for f in streams.values():
            f.close()
    if any(not x["windows"] for x in summaries.values()):
        raise ValueError("Both splits must contain supervised windows.")
    for key in summaries:
        summaries[key]["file"] = f"{key}.bin"
        summaries[key]["sha256"] = sha_file(output / f"{key}.bin")
    manifest = dict(
        format="domino-local-windows-v2",
        window_length=window,
        stride=stride,
        seed=seed,
        validation_percent=validation_percent,
        pad_id=pad_id,
        unsupervised_flag=UNSUPERVISED_FLAG,
        tokenizer_files=tokenizer_files,
        sources=list(sources),
        weights=list(weights),
        files=summaries,
        extra=extra or {},
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_dataset(
    documents, output, tokenizer, tokenizer_files, block_size=8, seed=42, validation_percent=5
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("An existing tokenizer EOS token is required.")
    seen = set()
    summaries = {}
    streams = {}
    for split in ("train", "validation"):
        for source in SOURCES:
            key = f"{source}.{split}"
            streams[key] = (output / f"{key}.bin").open("wb")
            summaries[key] = dict(windows=0, tokens=0, documents=0)
    provenance = (output / "documents.jsonl").open("w")
    duplicates = 0
    try:
        for row in documents:
            source, text = row["source"], row["text"]
            if source not in SOURCES or not isinstance(text, str):
                raise ValueError("Invalid source/text.")
            content_hash = hashlib.sha256(text.strip().encode()).hexdigest()
            if content_hash in seen:
                duplicates += 1
                continue
            seen.add(content_hash)
            # Code: repository; web: canonical URL. Exact duplicates are removed
            # across all sources before a second copy can enter another split.
            group = str(row.get("group_id") or content_hash)
            split = document_split(group, seed, validation_percent)
            tokens = tokenizer.encode(text, add_special_tokens=False, verbose=False)
            if not tokens:
                continue
            if tokens[-1] != tokenizer.eos_token_id:
                tokens.append(tokenizer.eos_token_id)
            key = f"{source}.{split}"
            count = 0
            for win in windows(tokens, block_size):
                array = np.asarray(win, dtype="<i4")
                streams[key].write(array.tobytes())
                count += int((array[1:] >= 0).sum())
                summaries[key]["windows"] += 1
            summaries[key]["tokens"] += count
            summaries[key]["documents"] += 1
            record = {k: v for k, v in row.items() if k != "text"}
            record.update(content_sha256=content_hash, split=split, group_id=group, tokens=count)
            provenance.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        for f in streams.values():
            f.close()
        provenance.close()
    if any(not x["windows"] for x in summaries.values()):
        raise ValueError("Each source must have nonempty train and validation sets.")
    for key in summaries:
        summaries[key]["file"] = f"{key}.bin"
        summaries[key]["sha256"] = sha_file(output / f"{key}.bin")
    manifest = dict(
        format="domino-local-windows-v1",
        block_size=block_size,
        seed=seed,
        validation_percent=validation_percent,
        pad_id=tokenizer.eos_token_id,
        tokenizer_files=tokenizer_files,
        sources=SOURCES,
        files=summaries,
        duplicates_removed=duplicates,
        documents_sha256=sha_file(output / "documents.jsonl"),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class WindowData:
    def __init__(self, root, split="train", verify=True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["format"] not in ("domino-local-windows-v1", "domino-local-windows-v2"):
            raise ValueError("Unknown dataset format.")
        self.identity = sha_file(self.root / "manifest.json")
        self.sources = tuple(self.manifest.get("sources", SOURCES))
        self.weights = tuple(self.manifest.get("weights", (0.6, 0.2, 0.2)))
        self.window = int(self.manifest.get("window_length", self.manifest.get("block_size", 0)))
        self.flag = int(self.manifest.get("unsupervised_flag", 0))
        if len(self.weights) != len(self.sources) or min(self.weights) <= 0:
            raise ValueError("Manifest sources/weights mismatch.")
        if self.window < 2:
            raise ValueError("Manifest window length missing or too small.")
        self.arrays = []
        self.means = []
        for source in self.sources:
            item = self.manifest["files"][f"{source}.{split}"]
            path = self.root / item["file"]
            if verify and sha_file(path) != item["sha256"]:
                raise ValueError(f"Dataset checksum mismatch: {path}")
            expected_bytes = item["windows"] * self.window * 4
            if path.stat().st_size != expected_bytes:
                raise ValueError("Dataset shape/size mismatch.")
            self.arrays.append(
                np.memmap(path, dtype="<i4", mode="r", shape=(item["windows"], self.window))
            )
            self.means.append(item["tokens"] / item["windows"])

    def unpack(self, array):
        raw = np.array(array, dtype=np.int64)
        pad = raw < 0
        if self.flag:
            unsupervised = ((raw & self.flag) != 0) & ~pad
            raw = np.where(pad, raw, raw & (self.flag - 1))
        ids = raw[:, :-1].copy()
        labels = raw[:, 1:].copy()
        ids[pad[:, :-1]] = self.manifest["pad_id"]
        labels[pad[:, 1:]] = -100
        if self.flag:
            labels[unsupervised[:, 1:]] = -100
        return ids, labels

    def sample(self, batch_size, *, seed, step, micro=0, rank=0, weights=None):
        # Correct window probabilities for different tail-padding rates, so the
        # expected *valid label* mixture matches the requested token mixture.
        weights = self.weights if weights is None else weights
        probabilities = np.asarray(weights, dtype=float) / np.asarray(self.means)
        if len(probabilities) != len(self.sources) or np.any(probabilities <= 0):
            raise ValueError("Mixture weights must be positive, one per source.")
        probabilities /= probabilities.sum()
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, micro, rank]))
        sources = rng.choice(len(self.sources), size=batch_size, p=probabilities)
        batch = np.empty((batch_size, self.window), dtype=np.int32)
        for i, array in enumerate(self.arrays):
            selected = np.flatnonzero(sources == i)
            batch[selected] = array[rng.integers(len(array), size=len(selected))]
        return (*self.unpack(batch), sources)

    def validation_batches(self, source, limit, batch_size):
        array = self.arrays[source]
        count = min(len(array), limit)
        # Uniform, deterministic coverage over the fixed validation subset.
        indices = np.linspace(0, len(array) - 1, count, dtype=np.int64)
        for start in range(0, count, batch_size):
            yield self.unpack(array[indices[start : start + batch_size]])
