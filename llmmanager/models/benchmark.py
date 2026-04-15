"""Benchmark domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class BenchmarkCategory(str, Enum):
    THROUGHPUT   = "throughput"
    LATENCY      = "latency"
    MEMORY       = "memory"
    CONCURRENCY  = "concurrency"
    CONTEXT      = "context_scaling"
    QUALITY      = "quality"


class BenchmarkProfile(str, Enum):
    QUICK    = "quick"
    """~2 min: throughput + latency only."""
    STANDARD = "standard"
    """~10 min: full suite, single model."""
    COMPARE  = "compare"
    """Run standard suite across multiple models."""
    STRESS   = "stress"
    """Sustained load + concurrency ramp to 128."""


@dataclass
class BenchmarkConfig:
    server_type: str
    model_id: str
    profile: BenchmarkProfile = BenchmarkProfile.STANDARD
    categories: list[BenchmarkCategory] = field(
        default_factory=lambda: list(BenchmarkCategory)
    )
    prompt: str = "Explain what a transformer model is in one sentence."
    n_tokens: int = 200
    n_runs: int = 3
    warm_up: bool = True
    concurrency_levels: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128]
    )
    context_lengths: list[int] = field(
        default_factory=lambda: [1_024, 8_192, 32_768, 131_072]
    )
    quality_probe_sets: list[str] = field(
        default_factory=lambda: ["coding", "reasoning", "instruction", "chat"]
    )
    safety_max_p99_ms: int = 30_000
    safety_max_error_rate_pct: float = 10.0
    sustained_duration_s: int = 60


@dataclass
class LatencyStats:
    min_ms: float
    max_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


@dataclass
class ConcurrencyResult:
    """Results at a single concurrency level."""
    concurrency: int
    total_requests: int
    successful: int
    failed: int
    aggregate_tokens_per_sec: float
    per_request_latency: LatencyStats
    ttft_ms: LatencyStats
    vram_mb: float
    aborted: bool = False
    abort_reason: str | None = None


@dataclass
class ContextScalingResult:
    context_length: int
    tokens_per_sec: float
    ttft_ms: float
    vram_mb: float
    error: str | None = None


@dataclass
class QualityProbeResult:
    probe_set: str
    prompt: str
    response: str
    response_tokens: int
    latency_ms: float
    notes: str = ""


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    hardware_snapshot: dict[str, Any]
    """GPU/CPU/RAM state captured at benchmark start."""

    # Core metrics
    tokens_per_sec: float = 0.0
    ttft_ms: float = 0.0
    total_duration_ms: float = 0.0
    vram_baseline_mb: float = 0.0
    vram_peak_mb: float = 0.0
    vram_delta_mb: float = 0.0

    # Per-category detailed results
    concurrency_results: list[ConcurrencyResult] = field(default_factory=list)
    context_results: list[ContextScalingResult] = field(default_factory=list)
    quality_results: list[QualityProbeResult] = field(default_factory=list)

    # Computed recommendation
    recommended_max_concurrency: int | None = None
    hardware_tier: str | None = None
    """'comfortable' | 'limited' | 'too_large'"""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: str | None = None
