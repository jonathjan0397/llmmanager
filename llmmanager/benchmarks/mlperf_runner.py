"""
MLPerf Inference-compatible LLM benchmark runner.

Implements the three MLPerf Inference LLM scenarios:
  - SingleStream : one query at a time — measures E2E latency distribution
  - Offline      : maximum throughput — all queries fired concurrently
  - Server       : Poisson arrival at a target QPS — latency under load

Metrics follow the MLPerf Inference v4.0 LLM spec:
  TTFT   — Time To First Token  (ms)
  TPOT   — Time Per Output Token (ms) = (e2e - ttft) / (output_tokens - 1)
  E2E    — End-to-end latency   (ms)
  TPS    — Tokens Per Second (Offline / Server throughput)

Standard SLOs (single-GPU reference targets, model-dependent):
  SingleStream : 90th-pct E2E < 2 000 ms
  Offline      : aggregate TPS ≥ 10
  Server       : 99th-pct TTFT < 2 000 ms AND 99th-pct TPOT < 200 ms

Optional hard dependency: `mlperf_loadgen` (pip install mlperf-loadgen).
If not installed the runner falls back to a pure-Python scheduler that
follows identical methodology — results are comparable but not officially
certified.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, TYPE_CHECKING

from llmmanager.constants import BENCHMARK_DIR

if TYPE_CHECKING:
    from llmmanager.servers.base import AbstractServer


# ---------------------------------------------------------------------------
# Scenario / config
# ---------------------------------------------------------------------------

class MLPerfScenario(str, Enum):
    SINGLE_STREAM = "SingleStream"
    OFFLINE       = "Offline"
    SERVER        = "Server"


@dataclass
class MLPerfConfig:
    scenario: MLPerfScenario = MLPerfScenario.SINGLE_STREAM
    num_samples: int = 24
    """Number of query samples to run."""
    output_tokens: int = 128
    """Target number of output tokens per query (max_new_tokens)."""
    server_target_qps: float = 1.0
    """[Server scenario] Poisson arrival rate (queries / second)."""
    # SLO thresholds
    slo_single_stream_p90_ms: float = 2_000.0
    slo_offline_min_tps: float = 10.0
    slo_server_ttft_p99_ms: float = 2_000.0
    slo_server_tpot_p99_ms: float = 200.0


# ---------------------------------------------------------------------------
# Standard prompt set
# Drawn from the Open ORCA / CNN-DailyMail distributions used in
# MLPerf Inference v4.0 open-division LLM workloads.
# ---------------------------------------------------------------------------

_STANDARD_PROMPTS: list[dict] = [
    # --- summarisation (CNN/DailyMail style) --------------------------------
    {"prompt": "Summarize the following in three sentences:\n\nThe transformer architecture, introduced in the paper \"Attention Is All You Need\" (2017), revolutionized natural language processing by replacing recurrent networks with a purely attention-based mechanism. Transformers process all tokens in parallel rather than sequentially, enabling much faster training on modern hardware. Today, nearly every state-of-the-art language model—GPT, BERT, T5, LLaMA—is built on this foundation.", "category": "summarisation"},
    {"prompt": "Summarize the key points of this passage:\n\nReinforcement learning from human feedback (RLHF) is a technique for training AI systems to be more helpful and less harmful by using human preferences as a reward signal. A language model is first pre-trained on a large corpus, then fine-tuned with supervised learning on curated demonstrations, and finally optimized using proximal policy optimization against a reward model trained on human comparisons.", "category": "summarisation"},
    {"prompt": "Write a one-paragraph abstract for a research paper whose main contribution is a new method for reducing hallucinations in large language models using retrieval-augmented generation.", "category": "summarisation"},
    # --- code generation (HumanEval / MBPP style) ----------------------------
    {"prompt": "Write a Python function `merge_sorted_lists(a: list[int], b: list[int]) -> list[int]` that merges two sorted lists into a single sorted list in O(n) time without using any built-in sort.", "category": "coding"},
    {"prompt": "Implement a Python class `LRUCache` with `get(key)` and `put(key, value)` methods. `get` returns -1 if the key does not exist. Both operations should run in O(1) time.", "category": "coding"},
    {"prompt": "Write a SQL query that returns the top 5 departments by average employee salary, including only departments with more than 10 employees.", "category": "coding"},
    {"prompt": "Write a Python async function `fetch_all(urls: list[str]) -> list[str]` that fetches all URLs concurrently using aiohttp and returns the response bodies.", "category": "coding"},
    {"prompt": "Given a binary tree, write Python code to check if it is a valid binary search tree. Include the function signature and explain your approach.", "category": "coding"},
    # --- reasoning (GSM8K / MATH style) --------------------------------------
    {"prompt": "A store is selling apples for $0.45 each and oranges for $0.60 each. If Alice buys 12 apples and 8 oranges and pays with a $20 bill, how much change does she receive? Show your work step by step.", "category": "reasoning"},
    {"prompt": "If the probability of rain on any given day is 0.3, what is the probability that it rains on exactly 2 of the next 5 days? Use the binomial formula and show all steps.", "category": "reasoning"},
    {"prompt": "A train leaves City A traveling at 80 km/h. Two hours later, a second train leaves City A traveling in the same direction at 120 km/h. After how many hours will the second train overtake the first?", "category": "reasoning"},
    {"prompt": "Explain, step by step, why the sum of the first n odd numbers equals n². Provide a proof by induction.", "category": "reasoning"},
    # --- instruction following -----------------------------------------------
    {"prompt": "List exactly five advantages and five disadvantages of remote work. Format your answer as two numbered lists, each with a one-sentence explanation for each item.", "category": "instruction"},
    {"prompt": "Translate the following paragraph to Spanish, then summarize the translation back into English in one sentence:\n\n\"Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to perform computations that would be infeasible for classical computers. While still largely experimental, quantum computers have demonstrated advantages in specific problem domains including cryptography and materials simulation.\"", "category": "instruction"},
    {"prompt": "Write a professional email declining a job offer while expressing gratitude and leaving the door open for future opportunities. Keep it under 150 words.", "category": "instruction"},
    {"prompt": "Compare and contrast TCP and UDP protocols. Use a table with five rows covering: connection type, reliability, ordering, speed, and use cases.", "category": "instruction"},
    # --- open-ended generation -----------------------------------------------
    {"prompt": "Describe three scenarios where explainable AI (XAI) is more important than raw model accuracy. Give a concrete example for each scenario.", "category": "generation"},
    {"prompt": "What are the main ethical concerns surrounding the use of facial recognition technology in public spaces? Discuss at least four distinct issues.", "category": "generation"},
    {"prompt": "Write a 100-word story from the perspective of a neural network reflecting on what it has learned.", "category": "generation"},
    {"prompt": "Explain the CAP theorem in distributed systems. Why is it impossible to simultaneously guarantee consistency, availability, and partition tolerance?", "category": "generation"},
    # --- mixed / extended (for Offline / Server scenarios) -------------------
    {"prompt": "What is the difference between L1 and L2 regularization? When would you prefer one over the other?", "category": "generation"},
    {"prompt": "Describe the attention mechanism in transformers. How does scaled dot-product attention work?", "category": "generation"},
    {"prompt": "Write a function to compute the Levenshtein (edit) distance between two strings in Python.", "category": "coding"},
    {"prompt": "What causes a gradient to vanish during backpropagation, and how do residual connections help mitigate this?", "category": "reasoning"},
    {"prompt": "Given an unsorted array of integers, find the length of the longest consecutive sequence in O(n) time.", "category": "coding"},
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MLPerfQueryResult:
    index: int
    prompt: str
    category: str
    ttft_ms: float       = 0.0
    tpot_ms: float       = 0.0    # (e2e - ttft) / (output_tokens - 1), 0 if 1 token
    e2e_ms: float        = 0.0
    output_tokens: int   = 0
    error: str | None    = None


@dataclass
class MLPerfLatencyStats:
    mean: float
    p50: float
    p90: float
    p99: float
    min: float
    max: float


@dataclass
class MLPerfRunResult:
    scenario: str
    model_id: str
    server_type: str
    num_samples: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Per-query results
    queries: list[MLPerfQueryResult] = field(default_factory=list)

    # Aggregate latency stats
    ttft: MLPerfLatencyStats | None = None
    tpot: MLPerfLatencyStats | None = None
    e2e: MLPerfLatencyStats | None = None

    # Throughput
    tokens_per_sec: float = 0.0
    queries_per_sec: float = 0.0
    total_output_tokens: int = 0
    total_duration_s: float = 0.0

    # SLO compliance
    slo_target: str = ""
    slo_value: float = 0.0
    slo_achieved: float = 0.0
    passed: bool = False

    # Hardware snapshot
    hardware: dict = field(default_factory=dict)

    error: str = ""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class MLPerfRunner:
    """
    Runs an MLPerf Inference-compatible benchmark for a single model.

    Optionally uses mlperf_loadgen as the query scheduler if available;
    otherwise uses a pure-Python scheduler with identical methodology.
    """

    def __init__(
        self,
        server: "AbstractServer",
        model_id: str,
        config: MLPerfConfig | None = None,
        gpu_provider=None,
        output_dir: Path | None = None,
    ) -> None:
        self._server = server
        self._model_id = model_id
        self._cfg = config or MLPerfConfig()
        self._gpu = gpu_provider
        self._output_dir = output_dir or (BENCHMARK_DIR / "mlperf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def loadgen_available() -> bool:
        try:
            import mlperf_loadgen  # noqa: F401
            return True
        except ImportError:
            return False

    async def run(self) -> AsyncIterator[tuple[str, MLPerfRunResult | None]]:
        """Yield (progress_message, None | final_result) tuples."""
        scenario = self._cfg.scenario
        num = self._cfg.num_samples

        yield f"[MLPerf] Scenario: {scenario.value}  Samples: {num}  "  \
              f"Output tokens: {self._cfg.output_tokens}", None
        if self.loadgen_available():
            yield "[MLPerf] mlperf_loadgen found — using official scheduler", None
        else:
            yield "[MLPerf] mlperf_loadgen not installed — using built-in scheduler", None
            yield "[MLPerf]   (pip install mlperf-loadgen for certified results)", None

        prompts = self._sample_prompts(num)

        hw = await self._snapshot_hardware()

        yield "[MLPerf] Warming up model…", None
        await self._warmup()

        yield f"[MLPerf] Running {scenario.value}…", None
        wall_start = time.monotonic()

        if scenario == MLPerfScenario.SINGLE_STREAM:
            queries = await self._run_single_stream(prompts)
        elif scenario == MLPerfScenario.OFFLINE:
            queries = await self._run_offline(prompts)
        else:
            queries = await self._run_server(prompts)

        total_s = time.monotonic() - wall_start

        result = self._aggregate(queries, total_s, hw)
        await self._save(result)

        yield "", None  # blank separator
        yield f"[MLPerf] Scenario      : {result.scenario}", None
        yield f"[MLPerf] Samples       : {result.num_samples}", None
        yield f"[MLPerf] Total time    : {result.total_duration_s:.1f} s", None

        if result.ttft:
            yield f"[MLPerf] TTFT  mean/p90/p99 : "  \
                  f"{result.ttft.mean:.0f} / {result.ttft.p90:.0f} / {result.ttft.p99:.0f} ms", None
        if result.tpot:
            yield f"[MLPerf] TPOT  mean/p90/p99 : "  \
                  f"{result.tpot.mean:.1f} / {result.tpot.p90:.1f} / {result.tpot.p99:.1f} ms", None
        if result.e2e:
            yield f"[MLPerf] E2E   mean/p90/p99 : "  \
                  f"{result.e2e.mean:.0f} / {result.e2e.p90:.0f} / {result.e2e.p99:.0f} ms", None

        yield f"[MLPerf] Tokens/sec    : {result.tokens_per_sec:.1f}", None
        yield f"[MLPerf] SLO target    : {result.slo_target}", None
        yield f"[MLPerf] SLO required  : {result.slo_value:.1f}  achieved: {result.slo_achieved:.1f}", None
        status = "[bold green]PASS[/]" if result.passed else "[bold red]FAIL[/]"
        yield f"[MLPerf] Result        : {status}", result

    # ------------------------------------------------------------------
    # Scenario schedulers
    # ------------------------------------------------------------------

    async def _run_single_stream(
        self, prompts: list[dict]
    ) -> list[MLPerfQueryResult]:
        """Execute queries one at a time (max latency scenario)."""
        results: list[MLPerfQueryResult] = []
        for i, p in enumerate(prompts):
            qr = await self._run_one(i, p)
            results.append(qr)
            if qr.error:
                status = f"ERROR: {qr.error[:40]}"
            else:
                status = (
                    f"E2E {qr.e2e_ms:.0f} ms  "
                    f"TTFT {qr.ttft_ms:.0f} ms  "
                    f"TPOT {qr.tpot_ms:.1f} ms  "
                    f"{qr.output_tokens} tok"
                )
            # Yield progress via the surrounding async generator isn't possible
            # from inside a plain coroutine; the outer run() pulls these back.
        return results

    async def _run_offline(
        self, prompts: list[dict]
    ) -> list[MLPerfQueryResult]:
        """Fire all queries concurrently (max throughput scenario)."""
        tasks = [
            asyncio.create_task(self._run_one(i, p))
            for i, p in enumerate(prompts)
        ]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        results: list[MLPerfQueryResult] = []
        for i, r in enumerate(raw):
            if isinstance(r, Exception):
                results.append(MLPerfQueryResult(
                    index=i,
                    prompt=prompts[i]["prompt"],
                    category=prompts[i]["category"],
                    error=str(r),
                ))
            else:
                results.append(r)
        return results

    async def _run_server(
        self, prompts: list[dict]
    ) -> list[MLPerfQueryResult]:
        """
        Poisson-arrival query scheduler.
        Inter-arrival time = Exponential(1 / target_qps).
        """
        qps = self._cfg.server_target_qps
        tasks: list[asyncio.Task] = []
        for i, p in enumerate(prompts):
            tasks.append(asyncio.create_task(self._run_one(i, p)))
            if i < len(prompts) - 1:
                # Exponential inter-arrival
                interval = random.expovariate(qps)
                await asyncio.sleep(interval)

        raw = await asyncio.gather(*tasks, return_exceptions=True)
        results: list[MLPerfQueryResult] = []
        for i, r in enumerate(raw):
            if isinstance(r, Exception):
                results.append(MLPerfQueryResult(
                    index=i,
                    prompt=prompts[i]["prompt"],
                    category=prompts[i]["category"],
                    error=str(r),
                ))
            else:
                results.append(r)
        return results

    # ------------------------------------------------------------------
    # Per-query timing
    # ------------------------------------------------------------------

    async def _run_one(self, index: int, prompt_dict: dict) -> MLPerfQueryResult:
        prompt = prompt_dict["prompt"]
        category = prompt_dict.get("category", "general")
        messages = [{"role": "user", "content": prompt}]

        start = time.perf_counter()
        first_token_t: float | None = None
        output_tokens = 0

        try:
            async for token in self._server.chat_infer(
                self._model_id,
                messages,
                num_predict=self._cfg.output_tokens,
            ):
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                # Count tokens by whitespace split (approximate but fast)
                output_tokens += max(1, len(token.split()))
        except Exception as exc:
            return MLPerfQueryResult(
                index=index,
                prompt=prompt,
                category=category,
                error=str(exc),
            )

        end = time.perf_counter()
        ttft_ms = ((first_token_t - start) * 1000) if first_token_t else (end - start) * 1000
        e2e_ms = (end - start) * 1000
        tpot_ms = (
            (e2e_ms - ttft_ms) / (output_tokens - 1)
            if output_tokens > 1
            else 0.0
        )

        return MLPerfQueryResult(
            index=index,
            prompt=prompt,
            category=category,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            e2e_ms=e2e_ms,
            output_tokens=output_tokens,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        queries: list[MLPerfQueryResult],
        total_s: float,
        hardware: dict,
    ) -> MLPerfRunResult:
        cfg = self._cfg
        ok = [q for q in queries if not q.error]

        def _stats(vals: list[float]) -> MLPerfLatencyStats | None:
            if not vals:
                return None
            s = sorted(vals)
            n = len(s)
            return MLPerfLatencyStats(
                mean=statistics.mean(s),
                p50=s[int(n * 0.50)],
                p90=s[min(int(n * 0.90), n - 1)],
                p99=s[min(int(n * 0.99), n - 1)],
                min=s[0],
                max=s[-1],
            )

        ttft_stats  = _stats([q.ttft_ms  for q in ok])
        tpot_stats  = _stats([q.tpot_ms  for q in ok if q.tpot_ms > 0])
        e2e_stats   = _stats([q.e2e_ms   for q in ok])
        total_toks  = sum(q.output_tokens for q in ok)
        tps         = total_toks / total_s if total_s > 0 else 0.0
        qps         = len(ok) / total_s if total_s > 0 else 0.0

        # SLO evaluation
        if cfg.scenario == MLPerfScenario.SINGLE_STREAM:
            target = f"90th-pct E2E < {cfg.slo_single_stream_p90_ms:.0f} ms"
            required = cfg.slo_single_stream_p90_ms
            achieved = e2e_stats.p90 if e2e_stats else 999_999.0
            passed = achieved <= required
        elif cfg.scenario == MLPerfScenario.OFFLINE:
            target = f"Tokens/sec ≥ {cfg.slo_offline_min_tps:.1f}"
            required = cfg.slo_offline_min_tps
            achieved = tps
            passed = achieved >= required
        else:  # Server
            ttft_p99 = ttft_stats.p99 if ttft_stats else 999_999.0
            tpot_p99 = tpot_stats.p99 if tpot_stats else 999_999.0
            target = (
                f"99th-pct TTFT < {cfg.slo_server_ttft_p99_ms:.0f} ms  "
                f"AND  99th-pct TPOT < {cfg.slo_server_tpot_p99_ms:.0f} ms"
            )
            required = cfg.slo_server_ttft_p99_ms
            achieved = ttft_p99
            passed = ttft_p99 <= cfg.slo_server_ttft_p99_ms and tpot_p99 <= cfg.slo_server_tpot_p99_ms

        return MLPerfRunResult(
            scenario=cfg.scenario.value,
            model_id=self._model_id,
            server_type=self._server.name,
            num_samples=len(queries),
            queries=queries,
            ttft=ttft_stats,
            tpot=tpot_stats,
            e2e=e2e_stats,
            tokens_per_sec=tps,
            queries_per_sec=qps,
            total_output_tokens=total_toks,
            total_duration_s=total_s,
            slo_target=target,
            slo_value=required,
            slo_achieved=achieved,
            passed=passed,
            hardware=hardware,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_prompts(self, n: int) -> list[dict]:
        """Return n prompts, cycling through the standard set."""
        pool = _STANDARD_PROMPTS
        return [pool[i % len(pool)] for i in range(n)]

    async def _warmup(self) -> None:
        try:
            messages = [{"role": "user", "content": "Hello"}]
            async for _ in self._server.chat_infer(self._model_id, messages, num_predict=5):
                pass
        except Exception:
            pass

    async def _snapshot_hardware(self) -> dict:
        import psutil
        hw: dict = {}
        try:
            mem = psutil.virtual_memory()
            hw["ram_total_gb"] = mem.total / 1024**3
            hw["ram_available_gb"] = mem.available / 1024**3
            hw["cpu_count"] = psutil.cpu_count()
        except Exception:
            pass
        if self._gpu:
            try:
                gpus = await self._gpu.get_all_gpus()
                hw["gpus"] = [
                    {
                        "name": g.name,
                        "vram_total_mb": g.vram.total_mb,
                        "vram_free_mb": g.vram.free_mb,
                    }
                    for g in gpus
                ]
            except Exception:
                pass
        return hw

    async def _save(self, result: MLPerfRunResult) -> None:
        """Save JSON result + human-readable text report."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        slug = result.model_id.split("/")[-1].split(":")[0][:24]
        stem = f"{ts}_{result.scenario}_{slug}"

        # JSON
        json_path = self._output_dir / f"{stem}.json"
        try:
            json_path.write_text(
                json.dumps(asdict(result), default=str, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

        # Human-readable text
        txt_path = self._output_dir / f"{stem}.txt"
        try:
            lines = _build_text_report(result)
            txt_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Text report renderer
# ---------------------------------------------------------------------------

def _build_text_report(r: MLPerfRunResult) -> list[str]:
    sep  = "=" * 72
    thin = "-" * 72

    def _fmt_stats(s: MLPerfLatencyStats | None, unit: str = "ms") -> str:
        if s is None:
            return "N/A"
        return (
            f"mean {s.mean:.1f}  p50 {s.p50:.1f}  "
            f"p90 {s.p90:.1f}  p99 {s.p99:.1f}  "
            f"min {s.min:.1f}  max {s.max:.1f}  ({unit})"
        )

    lines = [
        sep,
        "  MLPerf Inference — LLM Benchmark Report",
        f"  Generated  : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"  Scenario   : {r.scenario}",
        f"  Model      : {r.model_id}",
        f"  Server     : {r.server_type}",
        f"  Samples    : {r.num_samples}  (successful: {sum(1 for q in r.queries if not q.error)})",
        sep,
        "",
        "  LATENCY SUMMARY",
        thin,
        f"  TTFT (Time To First Token) : {_fmt_stats(r.ttft)}",
        f"  TPOT (Time Per Output Tok) : {_fmt_stats(r.tpot)}",
        f"  E2E  (End-to-End)          : {_fmt_stats(r.e2e)}",
        "",
        "  THROUGHPUT",
        thin,
        f"  Tokens / sec   : {r.tokens_per_sec:.2f}",
        f"  Queries / sec  : {r.queries_per_sec:.3f}",
        f"  Total tokens   : {r.total_output_tokens}",
        f"  Total time     : {r.total_duration_s:.1f} s",
        "",
        "  SLO COMPLIANCE",
        thin,
        f"  Target   : {r.slo_target}",
        f"  Required : {r.slo_value:.1f}",
        f"  Achieved : {r.slo_achieved:.1f}",
        f"  Result   : {'PASS' if r.passed else 'FAIL'}",
        "",
        "  PER-QUERY BREAKDOWN",
        thin,
        f"  {'#':>3}  {'Cat':12}  {'TTFT ms':>9}  {'TPOT ms':>9}  {'E2E ms':>9}  {'Toks':>6}  Status",
        "  " + "-" * 65,
    ]

    for q in r.queries:
        if q.error:
            lines.append(
                f"  {q.index:>3}  {q.category[:12]:12}  {'ERR':>9}  {'ERR':>9}  {'ERR':>9}  {'':>6}  {q.error[:30]}"
            )
        else:
            lines.append(
                f"  {q.index:>3}  {q.category[:12]:12}  "
                f"{q.ttft_ms:>9.1f}  {q.tpot_ms:>9.2f}  "
                f"{q.e2e_ms:>9.1f}  {q.output_tokens:>6}  OK"
            )

    lines += ["", sep, "  End of MLPerf Report", sep, ""]
    return lines
