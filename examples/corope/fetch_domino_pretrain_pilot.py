#!/usr/bin/env python3
"""Fetch a bounded, provenance-preserving sample from pinned official shards.

Raw manifest rows contain repo, revision, path, local, sha256, and kind.
Stack-Edu content comes from the official Software Heritage content bucket.
"""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from urllib.parse import urldefrag

import pyarrow.parquet as pq


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def records(item, limit):
    n = 0
    for batch in pq.ParquetFile(item["local"]).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            if item["kind"] == "stackedu" and not 128 <= row["length_bytes"] <= 1_000_000:
                continue
            yield row
            n += 1
            if n >= limit:
                return


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--documents-per-shard", type=int, default=600)
    p.add_argument("--proxy", default=os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"))
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        raise ValueError("Output exists; use a new immutable sample path.")
    manifest = json.loads(Path(a.manifest).read_text())
    success = []
    failed = []
    seen = set()

    def fetch(pair):
        item, row = pair
        origin = {k: item[k] for k in ("repo", "revision", "path", "sha256")}
        if item["kind"] == "stackedu":
            blob = row["blob_id"]
            url = f"https://softwareheritage.s3.amazonaws.com/content/{blob}"
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"https": a.proxy} if a.proxy else {})
            )
            last = None
            for attempt in range(3):
                try:
                    with opener.open(url, timeout=30) as response:
                        payload = response.read(2_000_000)
                    content = gzip.decompress(payload)
                    if hashlib.sha1(content).hexdigest() != blob:
                        raise ValueError("Software Heritage content hash mismatch")
                    text = content.decode(row.get("src_encoding") or "utf-8", errors="replace")
                    return dict(
                        source="stackedu",
                        text=text,
                        doc_id=blob,
                        group_id="repository:" + row["repo_name"],
                        language=row["language"],
                        origin=origin,
                        content_url=url,
                        detected_licenses=row.get("detected_licenses"),
                        content_verified=True,
                    )
                except Exception as e:
                    last = str(e)
                    if attempt < 2:
                        time.sleep(1 + attempt)
            return dict(failed=True, source="stackedu", blob_id=blob, error=last, origin=origin)
        text = row["text"]
        url = urldefrag(row.get("url", ""))[0]
        return dict(
            source=item["kind"],
            text=text,
            doc_id=row.get("id") or hashlib.sha256(text.encode()).hexdigest(),
            group_id="url:" + url if url else hashlib.sha256(text.encode()).hexdigest(),
            origin=origin,
        )

    pairs = []
    for item in manifest:
        if file_sha(item["local"]) != item["sha256"]:
            raise ValueError("Raw shard checksum mismatch")
        for row in records(item, a.documents_per_shard):
            if item["kind"] == "stackedu":
                if row["blob_id"] in seen:
                    continue
                seen.add(row["blob_id"])
            pairs.append((item, row))
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, row in enumerate(pool.map(fetch, pairs)):
            (failed if row.get("failed") else success).append(row)
            if (i + 1) % 200 == 0:
                print(
                    json.dumps(dict(processed=i + 1, successful=len(success), failed=len(failed))),
                    flush=True,
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        for row in success:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = dict(
        raw_manifest_sha256=file_sha(a.manifest),
        documents_sha256=file_sha(output),
        source_counts={
            s: sum(r["source"] == s for r in success) for s in ("fineweb", "stackedu", "finemath")
        },
        code_languages=sorted({r["language"] for r in success if r["source"] == "stackedu"}),
        requested=len(pairs),
        successful=len(success),
        failures=failed,
        scope="Pipeline pilot: first bounded rows per pinned shard, not a representative corpus sample",
    )
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if len(failed) > len(pairs) * 0.05 or any(v < 100 for v in audit["source_counts"].values()):
        raise RuntimeError("Too few successful documents; inspect audit before proceeding.")
    print(json.dumps({k: v for k, v in audit.items() if k != "failures"}, indent=2))


if __name__ == "__main__":
    main()
