"""Token-count-correct summaries shared by online and replayed diagnostics."""

from __future__ import annotations

import numpy as np

from .data import SOURCES


def summarize_sums(sums, sources=SOURCES):
    """sums: [NLL sum / valid count / correct count, source, position]."""
    values = np.asarray(sums, dtype=np.float64)
    if values.ndim != 3 or values.shape[0:2] != (3, len(sources)):
        raise ValueError("Expected metric sums [3, sources, positions]")

    def summarize(nll, count, correct):
        total = float(count.sum())
        return dict(
            loss=float(nll.sum() / total) if total else None,
            accuracy=float(correct.sum() / total) if total else None,
            valid_tokens=int(total),
            loss_by_position=[float(n / c) if c else None for n, c in zip(nll, count)],
            accuracy_by_position=[float(n / c) if c else None for n, c in zip(correct, count)],
            valid_tokens_by_position=count.astype(np.int64).tolist(),
        )

    result = summarize(*values.sum(axis=1))
    result["domains"] = {source: summarize(*values[:, i]) for i, source in enumerate(sources)}
    return result


def flatten_metrics(prefix, result):
    """Use explicit d0..d6 names; don't publish fabricated zeros for empty cells."""
    metrics = {}
    for key in ("loss", "accuracy", "valid_tokens"):
        if result.get(key) is not None:
            metrics[f"{prefix}/{key}"] = result[key]
    for key in ("loss_by_position", "accuracy_by_position", "valid_tokens_by_position"):
        for i, value in enumerate(result.get(key, [])):
            if value is not None:
                metrics[f"{prefix}/{key}/d{i}"] = value
    for source, value in result.get("domains", {}).items():
        metrics.update(flatten_metrics(f"{prefix}/{source}", value))
    return metrics


def evaluation_summary(results, weights=(0.6, 0.2, 0.2), sources=SOURCES):
    domains = {
        source: dict(
            loss=results[source]["nll"],
            accuracy=results[source]["accuracy"],
            valid_tokens=results[source]["tokens"],
            loss_by_position=results[source]["nll_by_depth"],
            accuracy_by_position=results[source]["accuracy_by_depth"],
        )
        for source in sources
    }
    for source in sources:
        if "tokens_by_depth" in results[source]:
            domains[source]["valid_tokens_by_position"] = results[source]["tokens_by_depth"]
    positions = len(domains[sources[0]]["loss_by_position"])

    def mixed_position(key, i):
        values = [domains[s][key][i] for s in sources]
        return (
            None if any(v is None for v in values) else sum(w * v for w, v in zip(weights, values))
        )

    result = dict(
        loss=sum(w * domains[s]["loss"] for s, w in zip(sources, weights)),
        accuracy=sum(w * domains[s]["accuracy"] for s, w in zip(sources, weights)),
        loss_by_position=[mixed_position("loss_by_position", i) for i in range(positions)],
        accuracy_by_position=[mixed_position("accuracy_by_position", i) for i in range(positions)],
        valid_tokens=sum(domains[s]["valid_tokens"] for s in sources),
        domains=domains,
        aggregation="fixed_domain_weights_" + "_".join(str(w) for w in weights),
    )
    if all("valid_tokens_by_position" in domains[s] for s in sources):
        result["valid_tokens_by_position"] = [
            sum(domains[s]["valid_tokens_by_position"][i] for s in sources)
            for i in range(positions)
        ]
    return result
