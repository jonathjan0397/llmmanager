"""Benchmark metric calculations."""

from __future__ import annotations

import statistics
from llmmanager.models.benchmark import LatencyStats


def compute_latency_stats(samples_ms: list[float]) -> LatencyStats:
    if not samples_ms:
        return LatencyStats(0, 0, 0, 0, 0, 0)
    s = sorted(samples_ms)
    n = len(s)

    def percentile(p: float) -> float:
        idx = (p / 100) * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    return LatencyStats(
        min_ms=s[0],
        max_ms=s[-1],
        mean_ms=statistics.mean(s),
        p50_ms=percentile(50),
        p95_ms=percentile(95),
        p99_ms=percentile(99),
    )


def compute_tokens_per_sec(token_count: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0:
        return 0.0
    return token_count / (elapsed_ms / 1000.0)
