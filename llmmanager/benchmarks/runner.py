"""BenchmarkRunner — orchestrates all benchmark categories."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncIterator, TYPE_CHECKING

from llmmanager.benchmarks.metrics import compute_latency_stats, compute_tokens_per_sec
from llmmanager.benchmarks.probes import PROBE_SETS
from llmmanager.constants import BENCHMARK_DIR
from llmmanager.exceptions import BenchmarkAbortedError
from llmmanager.models.benchmark import (
    BenchmarkCategory,
    BenchmarkConfig,
    BenchmarkResult,
    ConcurrencyResult,
    ContextScalingResult,
    LatencyStats,
    QualityProbeResult,
)

if TYPE_CHECKING:
    from llmmanager.gpu.base import AbstractGPUProvider
    from llmmanager.servers.base import AbstractServer


class BenchmarkRunner:
    """
    Runs all benchmark categories for a single model + server combination.
    Yields progress strings and populates a BenchmarkResult.
    """

    def __init__(
        self,
        server: "AbstractServer",
        gpu_provider: "AbstractGPUProvider",
    ) -> None:
        self._server = server
        self._gpu = gpu_provider

    async def run(
        self, config: BenchmarkConfig
    ) -> AsyncIterator[tuple[str, BenchmarkResult | None]]:
        """
        Async generator that yields (progress_message, None) during the run,
        then (final_message, BenchmarkResult) when complete.
        """
        result = BenchmarkResult(
            config=config,
            hardware_snapshot=await self._snapshot_hardware(),
        )

        try:
            yield ("Capturing VRAM baseline...", None)
            result.vram_baseline_mb = await self._current_vram_mb()

            if config.warm_up:
                yield ("Warming up model...", None)
                await self._warmup(config)

            if BenchmarkCategory.THROUGHPUT in config.categories:
                yield ("Running throughput benchmark...", None)
                tps, ttft, total = await self._throughput(config)
                result.tokens_per_sec = tps
                result.ttft_ms = ttft
                result.total_duration_ms = total

            if BenchmarkCategory.MEMORY in config.categories:
                yield ("Measuring peak VRAM...", None)
                result.vram_peak_mb = await self._current_vram_mb()
                result.vram_delta_mb = result.vram_peak_mb - result.vram_baseline_mb

            if BenchmarkCategory.CONCURRENCY in config.categories:
                async for msg, partial in self._concurrency(config, result):
                    yield (msg, None)

            if BenchmarkCategory.CONTEXT in config.categories:
                async for msg, partial in self._context_scaling(config, result):
                    yield (msg, None)

            if BenchmarkCategory.QUALITY in config.categories:
                async for msg, partial in self._quality_probes(config, result):
                    yield (msg, None)

            # Determine recommended max concurrency from concurrency results
            result.recommended_max_concurrency = self._find_concurrency_knee(
                result.concurrency_results
            )
            result.hardware_tier = self._classify_hardware(result)

            # Persist to disk
            await self._save(result)

        except BenchmarkAbortedError as exc:
            result.error = str(exc)
            yield (f"Aborted: {exc.reason}", result)
            return
        except asyncio.CancelledError:
            result.error = "Cancelled by user"
            yield ("Benchmark cancelled.", result)
            return

        yield ("Benchmark complete.", result)

    # ------------------------------------------------------------------
    # Throughput
    # ------------------------------------------------------------------

    async def _throughput(
        self, config: BenchmarkConfig
    ) -> tuple[float, float, float]:
        """Returns (tokens_per_sec, ttft_ms, total_ms)."""
        tps_samples: list[float] = []
        ttft_samples: list[float] = []
        total_samples: list[float] = []

        for run_i in range(config.n_runs):
            start = time.monotonic()
            first_token_time: float | None = None
            token_count = 0

            async for token in self._server.quick_infer(config.model_id, config.prompt):
                if first_token_time is None:
                    first_token_time = time.monotonic()
                token_count += len(token.split())  # rough token count

            elapsed_ms = (time.monotonic() - start) * 1000
            ttft_ms = ((first_token_time - start) * 1000) if first_token_time else elapsed_ms

            tps_samples.append(compute_tokens_per_sec(token_count, elapsed_ms))
            ttft_samples.append(ttft_ms)
            total_samples.append(elapsed_ms)

        import statistics
        return (
            statistics.mean(tps_samples) if tps_samples else 0.0,
            statistics.mean(ttft_samples) if ttft_samples else 0.0,
            statistics.mean(total_samples) if total_samples else 0.0,
        )

    # ------------------------------------------------------------------
    # Concurrency ramp (1 → 128)
    # ------------------------------------------------------------------

    async def _concurrency(
        self, config: BenchmarkConfig, result: BenchmarkResult
    ) -> AsyncIterator[tuple[str, None]]:
        for level in config.concurrency_levels:
            yield (f"Concurrency test: {level} parallel requests...", None)

            conc_result = await self._run_concurrency_level(config, level)
            result.concurrency_results.append(conc_result)

            if conc_result.aborted:
                yield (
                    f"Safety cutoff hit at concurrency={level}: {conc_result.abort_reason}",
                    None,
                )
                break

    async def _run_concurrency_level(
        self, config: BenchmarkConfig, level: int
    ) -> ConcurrencyResult:
        """Fire `level` requests simultaneously and collect metrics."""
        start = time.monotonic()
        vram_before = await self._current_vram_mb()

        tasks = [
            asyncio.create_task(self._timed_infer(config))
            for _ in range(level)
        ]

        # Use sustained mode — maintain N concurrent for the duration
        if level > 8:
            # For high concurrency, run sustained test
            deadline = start + config.safety_max_p99_latency_ms / 1000.0
            results_raw = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=config.sustained_duration_s,
            )
        else:
            results_raw = await asyncio.gather(*tasks, return_exceptions=True)

        total_ms = (time.monotonic() - start) * 1000
        vram_after = await self._current_vram_mb()

        latencies_ms: list[float] = []
        ttft_ms_list: list[float] = []
        total_tokens = 0
        failures = 0

        for r in results_raw:
            if isinstance(r, Exception):
                failures += 1
            else:
                latency, ttft, tokens = r
                latencies_ms.append(latency)
                ttft_ms_list.append(ttft)
                total_tokens += tokens

        success = len(latencies_ms)
        error_rate = (failures / level * 100) if level > 0 else 0.0

        # Safety cutoff checks
        abort_reason: str | None = None
        lat_stats = compute_latency_stats(latencies_ms) if latencies_ms else LatencyStats(0,0,0,0,0,0)

        if lat_stats.p99_ms > config.safety_max_p99_ms:
            abort_reason = f"p99 latency {lat_stats.p99_ms:.0f}ms > {config.safety_max_p99_ms}ms"
        elif error_rate > config.safety_max_error_rate_pct:
            abort_reason = f"error rate {error_rate:.1f}% > {config.safety_max_error_rate_pct}%"

        agg_tps = compute_tokens_per_sec(total_tokens, total_ms)

        return ConcurrencyResult(
            concurrency=level,
            total_requests=level,
            successful=success,
            failed=failures,
            aggregate_tokens_per_sec=agg_tps,
            per_request_latency=lat_stats,
            ttft_ms=compute_latency_stats(ttft_ms_list) if ttft_ms_list else LatencyStats(0,0,0,0,0,0),
            vram_mb=vram_after,
            aborted=abort_reason is not None,
            abort_reason=abort_reason,
        )

    async def _timed_infer(
        self, config: BenchmarkConfig
    ) -> tuple[float, float, int]:
        """Returns (total_ms, ttft_ms, token_count)."""
        start = time.monotonic()
        first_token_time: float | None = None
        token_count = 0
        try:
            async for token in self._server.quick_infer(config.model_id, config.prompt):
                if first_token_time is None:
                    first_token_time = time.monotonic()
                token_count += len(token.split())
        except Exception as exc:
            raise exc
        elapsed_ms = (time.monotonic() - start) * 1000
        ttft_ms = ((first_token_time - start) * 1000) if first_token_time else elapsed_ms
        return elapsed_ms, ttft_ms, token_count

    # ------------------------------------------------------------------
    # Context scaling
    # ------------------------------------------------------------------

    async def _context_scaling(
        self, config: BenchmarkConfig, result: BenchmarkResult
    ) -> AsyncIterator[tuple[str, None]]:
        for ctx_len in config.context_lengths:
            yield (f"Context scaling test: {ctx_len} tokens...", None)
            # Build a prompt that fills the desired context length (rough)
            padded_prompt = (config.prompt + " ") * max(1, ctx_len // max(len(config.prompt.split()), 1))

            try:
                start = time.monotonic()
                first_token_time: float | None = None
                token_count = 0
                vram_before = await self._current_vram_mb()

                async for token in self._server.quick_infer(config.model_id, padded_prompt):
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                    token_count += len(token.split())

                elapsed_ms = (time.monotonic() - start) * 1000
                ttft_ms = ((first_token_time - start) * 1000) if first_token_time else elapsed_ms
                vram_after = await self._current_vram_mb()

                result.context_results.append(ContextScalingResult(
                    context_length=ctx_len,
                    tokens_per_sec=compute_tokens_per_sec(token_count, elapsed_ms),
                    ttft_ms=ttft_ms,
                    vram_mb=vram_after,
                ))
            except Exception as exc:
                result.context_results.append(ContextScalingResult(
                    context_length=ctx_len,
                    tokens_per_sec=0.0,
                    ttft_ms=0.0,
                    vram_mb=0.0,
                    error=str(exc),
                ))

    # ------------------------------------------------------------------
    # Quality probes
    # ------------------------------------------------------------------

    async def _quality_probes(
        self, config: BenchmarkConfig, result: BenchmarkResult
    ) -> AsyncIterator[tuple[str, None]]:
        for probe_set in config.quality_probe_sets:
            probes = PROBE_SETS.get(probe_set, [])
            for probe in probes:
                yield (f"Quality probe [{probe_set}]: {probe['prompt'][:50]}...", None)
                start = time.monotonic()
                response = ""
                try:
                    async for token in self._server.quick_infer(
                        config.model_id, probe["prompt"]
                    ):
                        response += token
                    latency_ms = (time.monotonic() - start) * 1000
                    result.quality_results.append(QualityProbeResult(
                        probe_set=probe_set,
                        prompt=probe["prompt"],
                        response=response,
                        response_tokens=len(response.split()),
                        latency_ms=latency_ms,
                        notes=probe.get("notes", ""),
                    ))
                except Exception as exc:
                    result.quality_results.append(QualityProbeResult(
                        probe_set=probe_set,
                        prompt=probe["prompt"],
                        response=f"ERROR: {exc}",
                        response_tokens=0,
                        latency_ms=0.0,
                        notes=probe.get("notes", ""),
                    ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _warmup(self, config: BenchmarkConfig) -> None:
        try:
            async for _ in self._server.quick_infer(config.model_id, "Hello"):
                pass
        except Exception:
            pass

    async def _current_vram_mb(self) -> float:
        try:
            gpus = await self._gpu.get_all_gpus()
            return sum(g.vram.used_mb for g in gpus)
        except Exception:
            return 0.0

    async def _snapshot_hardware(self) -> dict:
        import psutil
        try:
            gpus = await self._gpu.get_all_gpus()
            gpu_data = [
                {
                    "name": g.name,
                    "vendor": g.vendor.value,
                    "vram_total_mb": g.vram.total_mb,
                    "vram_free_mb": g.vram.free_mb,
                }
                for g in gpus
            ]
        except Exception:
            gpu_data = []

        mem = psutil.virtual_memory()
        return {
            "gpus": gpu_data,
            "cpu_count": psutil.cpu_count(),
            "ram_total_gb": mem.total / 1024**3,
            "ram_available_gb": mem.available / 1024**3,
        }

    def _find_concurrency_knee(
        self, results: list[ConcurrencyResult]
    ) -> int | None:
        """
        Find the concurrency level just before performance degrades or safety
        cutoffs trigger. Returns the recommended max concurrency.
        """
        best: int | None = None
        for r in results:
            if r.aborted:
                break
            if r.failed == 0:
                best = r.concurrency
        return best

    def _classify_hardware(self, result: BenchmarkResult) -> str:
        if result.tokens_per_sec >= 30:
            return "comfortable"
        elif result.tokens_per_sec >= 5:
            return "limited"
        else:
            return "too_large"

    async def _save(self, result: BenchmarkResult) -> None:
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        ts = result.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{result.config.server_type}_{result.config.model_id.replace('/', '_')}_{ts}.json"
        path = BENCHMARK_DIR / filename
        try:
            import dataclasses
            data = dataclasses.asdict(result)
            path.write_text(json.dumps(data, default=str, indent=2))
        except Exception:
            pass  # Saving is best-effort; don't abort on serialization failure
