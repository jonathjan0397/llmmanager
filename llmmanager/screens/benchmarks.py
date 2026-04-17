"""Benchmark Suite screen — multi-model benchmark runner with comparison charts."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Label,
    ProgressBar,
    Select,
    SelectionList,
    Static,
    TabbedContent,
    TabPane,
)

from llmmanager.benchmarks.runner import BenchmarkRunner
from llmmanager.benchmarks.mlperf_runner import MLPerfConfig, MLPerfRunner, MLPerfScenario
from llmmanager.constants import BENCHMARK_DIR, BENCHMARK_CONCURRENCY_LEVELS
from llmmanager.models.benchmark import (
    BenchmarkCategory,
    BenchmarkConfig,
    BenchmarkProfile,
    BenchmarkResult,
    ConcurrencyResult,
    ContextScalingResult,
)
from llmmanager.widgets.log_view import LogView

if TYPE_CHECKING:
    from llmmanager.app import LLMManagerApp


# ── chart constants ─────────────────────────────────────────────────────────

_BAR_WIDTH   = 32
_BAR_FILL    = "█"
_BAR_EMPTY   = "░"
_MODEL_COLORS = ["green", "cyan", "yellow", "magenta", "blue", "red"]

# Symbols for line/table charts — one per model
_SYMBOLS = ["*", "+", "o", "#", "@", "~"]


class BenchmarksScreen(Widget):
    """Screen 5 — run, compare, and review benchmarks."""

    DEFAULT_CSS = """
    BenchmarksScreen { width: 1fr; height: 1fr; }

    #bench-model-list {
        height: 8;
        border: ascii $primary-darken-2;
        margin-bottom: 1;
    }

    #bench-model-controls {
        height: auto;
        margin-bottom: 1;
    }
    #bench-model-controls Button { width: auto; margin-right: 1; }

    #bench-live-stats {
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        border: ascii $primary-darken-2;
        margin-top: 1;
        color: $text;
    }

    #run-controls Button { width: 1fr; }

    #compare-scroll {
        width: 1fr;
        height: 1fr;
    }

    .chart-section {
        height: auto;
        margin-bottom: 2;
    }

    .chart-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 0;
    }

    .chart-body {
        padding: 0 1;
    }

    #mlperf-form {
        height: 1fr;
        padding: 0 1;
    }

    #mlperf-config-scroll {
        height: auto;
        max-height: 18;
    }

    #mlperf-model-row {
        height: auto;
        margin-bottom: 1;
    }

    #mlperf-run-controls Button { width: 1fr; }

    #mlperf-hints {
        color: $text-muted;
        margin-bottom: 1;
    }

    #mlperf-loadgen-status {
        margin-bottom: 1;
    }

    #mlperf-live-status {
        height: auto;
        color: $accent;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("r", "run_benchmark",    "Run"),
        ("c", "cancel_benchmark", "Cancel"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._current_run: asyncio.Task | None = None
        self._last_results: list[BenchmarkResult] = []
        self._mlperf_run: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        with TabbedContent(id="bench-tabs"):
            # ── Run ──────────────────────────────────────────────────────
            with TabPane("Run", id="tab-run"):
                with Vertical(id="run-form"):
                    yield Label("Server:", classes="form-label")
                    yield Select(
                        options=[
                            ("Ollama",    "ollama"),
                            ("vLLM",      "vllm"),
                            ("LM Studio", "lmstudio"),
                            ("llama.cpp", "llamacpp"),
                        ],
                        value="ollama",
                        id="bench-server-select",
                    )

                    yield Label("Models  (check one or more):", classes="form-label")
                    yield SelectionList(id="bench-model-list")
                    with Horizontal(id="bench-model-controls"):
                        yield Button("↻ Refresh",      id="btn-refresh-models",  variant="default")
                        yield Button("Select All",     id="btn-select-all",      variant="default")
                        yield Button("Deselect All",   id="btn-deselect-all",    variant="default")

                    yield Checkbox(
                        "Auto-swap: unload current model and load selected before each run",
                        value=True,
                        id="bench-auto-swap",
                    )

                    yield Label("Profile:", classes="form-label")
                    yield Select(
                        options=[
                            ("Quick  (~1 min each)",   "quick"),
                            ("Standard (~5 min each)", "standard"),
                            ("Stress (full ramp)",      "stress"),
                        ],
                        value="standard",
                        id="bench-profile-select",
                    )

                    yield Label("Categories:", classes="form-label")
                    for cat in BenchmarkCategory:
                        yield Checkbox(
                            cat.value.replace("_", " ").title(),
                            value=True,
                            id=f"cat-{cat.value}",
                        )

                    yield Label(
                        f"Concurrency levels: {', '.join(str(x) for x in BENCHMARK_CONCURRENCY_LEVELS)}",
                        classes="form-hint",
                    )

                    with Horizontal(id="run-controls"):
                        yield Button("Run Benchmark", id="btn-run-bench",    variant="primary")
                        yield Button("Cancel",        id="btn-cancel-bench", variant="error", disabled=True)

                    yield Static("", id="bench-live-stats")
                    yield Label("", id="bench-progress-label")
                    yield ProgressBar(total=100, id="bench-progress-bar", show_eta=False)
                    yield LogView(max_lines=500, id="bench-log")

            # ── Compare ──────────────────────────────────────────────────
            with TabPane("Compare", id="tab-compare"):
                yield VerticalScroll(id="compare-scroll")

            # ── Results ──────────────────────────────────────────────────
            with TabPane("Results", id="tab-results"):
                yield Static(id="results-tab-content")

            # ── History ──────────────────────────────────────────────────
            with TabPane("History", id="tab-history"):
                yield DataTable(id="history-table", cursor_type="row")
                yield Button("View Details", id="btn-view-history", variant="default")

            # ── MLPerf ───────────────────────────────────────────────────
            with TabPane("MLPerf", id="tab-mlperf"):
                with Vertical(id="mlperf-form"):
                    # Config in a scrollable section so the Run button is always visible
                    with VerticalScroll(id="mlperf-config-scroll"):
                        yield Label("Server:", classes="form-label")
                        yield Select(
                            options=[
                                ("Ollama",    "ollama"),
                                ("vLLM",      "vllm"),
                                ("LM Studio", "lmstudio"),
                                ("llama.cpp", "llamacpp"),
                            ],
                            value="ollama",
                            id="mlperf-server-select",
                        )
                        yield Label("Model:", classes="form-label")
                        yield Select(
                            options=[("—", "__none__")],
                            value="__none__",
                            id="mlperf-model-select",
                        )
                        with Horizontal(id="mlperf-model-row"):
                            yield Button("↻ Refresh", id="btn-mlperf-refresh-models", variant="default")
                        yield Label("Scenario:", classes="form-label")
                        yield Select(
                            options=[
                                ("SingleStream  (latency — sequential queries)",          "SingleStream"),
                                ("Offline       (throughput — all queries at once)",      "Offline"),
                                ("Server        (Poisson arrival — latency under load)",  "Server"),
                            ],
                            value="SingleStream",
                            id="mlperf-scenario-select",
                        )
                        yield Label("Samples (# queries):", classes="form-label")
                        yield Select(
                            options=[
                                ("8   (quick test)", "8"),
                                ("24  (standard)",   "24"),
                                ("50  (extended)",   "50"),
                                ("100 (full)",       "100"),
                            ],
                            value="24",
                            id="mlperf-samples-select",
                        )
                        yield Label("Max output tokens per query:", classes="form-label")
                        yield Select(
                            options=[
                                ("64",  "64"),
                                ("128", "128"),
                                ("256", "256"),
                                ("512", "512"),
                            ],
                            value="128",
                            id="mlperf-tokens-select",
                        )
                        yield Label("Server target QPS  [Server scenario only]:", classes="form-label")
                        yield Select(
                            options=[
                                ("0.5 QPS", "0.5"),
                                ("1.0 QPS", "1.0"),
                                ("2.0 QPS", "2.0"),
                                ("4.0 QPS", "4.0"),
                            ],
                            value="1.0",
                            id="mlperf-qps-select",
                        )
                        yield Static(
                            "[dim]SLOs — SingleStream: 90th-pct E2E < 2000 ms  |  "
                            "Offline: TPS ≥ 10  |  "
                            "Server: 99th-pct TTFT < 2000 ms AND TPOT < 200 ms\n"
                            "Prompts: 25-item Open ORCA / HumanEval / GSM8K set "
                            "(MLPerf Inference v4.0 open-division LLM workloads)[/]",
                            id="mlperf-hints",
                        )
                        if MLPerfRunner.loadgen_available():
                            yield Static(
                                "[green]mlperf_loadgen detected — official scheduler active.[/]",
                                id="mlperf-loadgen-status",
                            )
                        else:
                            yield Static(
                                "[yellow]mlperf_loadgen not installed[/]  "
                                "[dim]— pip install llmmanager[benchmarks] for certified results[/]",
                                id="mlperf-loadgen-status",
                            )
                    # Run controls always visible outside the scroll area
                    with Horizontal(id="mlperf-run-controls"):
                        yield Button("Run MLPerf", id="btn-mlperf-run",    variant="primary")
                        yield Button("Cancel",     id="btn-mlperf-cancel", variant="error",
                                     disabled=True)
                    yield Static("", id="mlperf-live-status")
                    yield LogView(max_lines=400, id="mlperf-log")

    # ── Mount ────────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        table = self.query_one("#history-table", DataTable)
        table.add_columns(
            "Timestamp", "Server", "Model", "TPS", "TTFT ms",
            "Max Concurrency", "Tier",
        )
        self.run_worker(self._load_history(table))
        self.run_worker(self._populate_model_list())
        self.run_worker(self._populate_mlperf_model_list())

    # ── Model list ───────────────────────────────────────────────────────────

    async def _populate_model_list(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        server_type = str(self.query_one("#bench-server-select", Select).value)
        server = app.registry.get(server_type)
        ml = self.query_one("#bench-model-list", SelectionList)
        ml.clear_options()
        if server is None:
            return
        try:
            models = await server.list_loaded_models()
        except Exception:
            models = []
        for m in models:
            ml.add_option((m.display_name, m.model_id))

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "bench-server-select":
            self.run_worker(self._populate_model_list())
        elif event.select.id == "mlperf-server-select":
            self.run_worker(self._populate_mlperf_model_list())

    # ── Button handler ───────────────────────────────────────────────────────

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "btn-run-bench":
                await self._start_benchmark()
            case "btn-cancel-bench":
                self._cancel_benchmark()
            case "btn-refresh-models":
                self.run_worker(self._populate_model_list())
            case "btn-select-all":
                ml = self.query_one("#bench-model-list", SelectionList)
                for i in range(len(ml._options)):  # type: ignore[attr-defined]
                    ml.select(ml._options[i].value)  # type: ignore[attr-defined]
            case "btn-deselect-all":
                self.query_one("#bench-model-list", SelectionList).deselect_all()
            case "btn-view-history":
                pass  # TODO: expand selected history row
            case "btn-mlperf-run":
                self.run_worker(self._run_mlperf())
            case "btn-mlperf-cancel":
                self._cancel_mlperf()
            case "btn-mlperf-refresh-models":
                self.run_worker(self._populate_mlperf_model_list())

    # ── Benchmark orchestration ──────────────────────────────────────────────

    async def _start_benchmark(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]

        server_type  = str(self.query_one("#bench-server-select",  Select).value)
        profile_val  = str(self.query_one("#bench-profile-select", Select).value)
        auto_swap    = self.query_one("#bench-auto-swap", Checkbox).value
        ml           = self.query_one("#bench-model-list", SelectionList)
        selected_ids: list[str] = list(ml.selected)  # type: ignore[arg-type]

        if not selected_ids:
            self.notify("Select at least one model.", severity="warning")
            return

        server = app.registry.get(server_type)
        if server is None:
            self.notify(f"Server '{server_type}' not found.", severity="error")
            return

        categories = [cat for cat in BenchmarkCategory if self._cat_enabled(cat)]
        profile    = BenchmarkProfile(profile_val)

        concurrency = BENCHMARK_CONCURRENCY_LEVELS
        if profile == BenchmarkProfile.QUICK:
            categories  = [BenchmarkCategory.THROUGHPUT, BenchmarkCategory.LATENCY]
            concurrency = [1, 2, 4]

        self.query_one("#btn-run-bench",    Button).disabled = True
        self.query_one("#btn-cancel-bench", Button).disabled = False

        self._last_results = []
        self._current_run = asyncio.create_task(
            self._run_all_models(
                server, server_type, selected_ids, categories, concurrency,
                profile, auto_swap,
            )
        )

    async def _run_all_models(
        self,
        server,
        server_type: str,
        model_ids: list[str],
        categories: list[BenchmarkCategory],
        concurrency: list[int],
        profile: BenchmarkProfile,
        auto_swap: bool,
    ) -> None:
        import time as _time
        log          = self.query_one("#bench-log",          LogView)
        progress_bar = self.query_one("#bench-progress-bar", ProgressBar)
        progress_lbl = self.query_one("#bench-progress-label", Label)
        live_stats   = self.query_one("#bench-live-stats",   Static)

        log.clear_log()
        progress_bar.progress = 0

        total   = len(model_ids)
        results = []

        try:
            for model_idx, model_id in enumerate(model_ids):
                model_pct_start = model_idx / total * 100
                model_pct_end   = (model_idx + 1) / total * 100

                log.append_line(
                    f"\n{'='*60}\n"
                    f"  Model {model_idx+1}/{total}: {model_id}\n"
                    f"{'='*60}"
                )

                config = BenchmarkConfig(
                    server_type=server_type,
                    model_id=model_id,
                    profile=profile,
                    categories=categories,
                    concurrency_levels=concurrency,
                )

                runner = BenchmarkRunner(server, self.app.gpu_provider)  # type: ignore[attr-defined]
                run_start = _time.monotonic()

                async for msg, result in runner.run(config):
                    log.append_line(msg)
                    progress_lbl.update(f"[{model_idx+1}/{total}] {model_id}: {msg}")

                    # Progress within this model's slice
                    elapsed = _time.monotonic() - run_start
                    inner_pct = min(elapsed / 60 * 100, 99)
                    progress_bar.progress = model_pct_start + inner_pct * (model_pct_end - model_pct_start) / 100

                    # Live stats
                    live_stats.update(
                        f"[bold]Model:[/] {model_idx+1}/{total}  "
                        f"[bold]Current:[/] {model_id}  "
                        f"[bold]Elapsed:[/] {int(_time.monotonic()-run_start)//60:02d}:"
                        f"{int(_time.monotonic()-run_start)%60:02d}"
                    )

                    if result is not None:
                        results.append(result)
                        self._last_results = results[:]
                        progress_bar.progress = model_pct_end
                        log.append_line(
                            f"  >> {model_id}: "
                            f"{result.tokens_per_sec:.1f} TPS  "
                            f"TTFT {result.ttft_ms:.0f}ms  "
                            f"Tier: {result.hardware_tier or '?'}"
                        )
                        # Show single-model result in Results tab
                        self._show_single_result(result)

        except asyncio.CancelledError:
            log.append_line("\nBenchmark cancelled.")
        finally:
            self.query_one("#btn-run-bench",    Button).disabled = False
            self.query_one("#btn-cancel-bench", Button).disabled = True
            progress_bar.progress = 100
            if results:
                log.append_line(f"\nAll done. {len(results)}/{total} models completed.")
                report_path = self._save_report(results)
                if report_path:
                    log.append_line(f"[bold]Report saved:[/] {report_path}")
                self._render_comparison(results)
                # Switch to Compare tab
                self.query_one("#bench-tabs", TabbedContent).active = "tab-compare"

    def _save_report(self, results: list[BenchmarkResult]) -> Path | None:
        """Write a human-readable plain-text report and return its path."""
        try:
            reports_dir = BENCHMARK_DIR / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            now = datetime.now()
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")

            # Build filename slug from model IDs (sanitised, max 3 models shown)
            def _slug(model_id: str) -> str:
                # Strip registry prefix (e.g. "hf.co/org/") and tag suffix
                name = model_id.split("/")[-1].split(":")[0]
                return re.sub(r"[^a-zA-Z0-9_.-]", "-", name)[:20]

            model_slugs = [_slug(r.config.model_id) for r in results]
            if len(model_slugs) > 3:
                slug_part = "_".join(model_slugs[:3]) + f"_and_{len(model_slugs)-3}_more"
            else:
                slug_part = "_".join(model_slugs)

            filename = f"{timestamp_str}_{slug_part}.txt"
            report_path = reports_dir / filename

            lines: list[str] = []
            sep = "=" * 72
            thin = "-" * 72

            lines += [
                sep,
                "  LLMManager Benchmark Report",
                f"  Generated : {now.strftime('%Y-%m-%d %H:%M:%S')}",
                f"  Server    : {results[0].config.server_type}",
                f"  Profile   : {results[0].config.profile.value}",
                f"  Models    : {len(results)}",
                sep,
                "",
            ]

            # ── Per-model summaries ────────────────────────────────────────
            for i, r in enumerate(results):
                lines += [
                    f"  Model {i+1}/{len(results)}: {r.config.model_id}",
                    thin,
                    f"    Tokens / sec       : {r.tokens_per_sec:.2f}",
                    f"    Time to first token: {r.ttft_ms:.0f} ms",
                    f"    VRAM delta         : {r.vram_delta_mb:.0f} MB",
                    f"    Hardware tier      : {r.hardware_tier or '—'}",
                    f"    Max concurrency    : {r.recommended_max_concurrency or '—'}",
                ]
                if r.error:
                    lines.append(f"    ERROR              : {r.error}")

                if r.concurrency_results:
                    lines.append("    Concurrency scaling (req → TPS):")
                    for cr in sorted(r.concurrency_results, key=lambda x: x.concurrency):
                        status = "ABORTED" if cr.aborted else f"{cr.aggregate_tokens_per_sec:.1f} TPS"
                        lines.append(f"      {cr.concurrency:>3} req  {status}")

                if r.context_results:
                    lines.append("    Context scaling (ctx → TPS):")
                    for cr in sorted(r.context_results, key=lambda x: x.context_length):
                        def _fmt_ctx(c: int) -> str:
                            return f"{c//1024}K" if c >= 1024 else str(c)
                        status = f"ERROR ({cr.error})" if cr.error else f"{cr.tokens_per_sec:.1f} TPS"
                        lines.append(f"      {_fmt_ctx(cr.context_length):>6}  {status}")

                if r.quality_results:
                    lines.append("    Quality probe latency (avg ms per set):")
                    from itertools import groupby
                    for ps, probes in groupby(
                        sorted(r.quality_results, key=lambda x: x.probe_set),
                        key=lambda x: x.probe_set,
                    ):
                        probe_list = list(probes)
                        avg = sum(p.latency_ms for p in probe_list) / len(probe_list)
                        lines.append(f"      {ps:<20}  {avg:.0f} ms")

                lines.append("")

            # ── Side-by-side comparison table ─────────────────────────────
            if len(results) > 1:
                lines += [
                    sep,
                    "  COMPARISON SUMMARY",
                    sep,
                    "",
                ]
                col_w = 22
                header = "Metric".ljust(28) + "".join(r.config.model_id[:col_w].ljust(col_w) for r in results)
                lines.append("  " + header)
                lines.append("  " + thin)

                def _row(label: str, values: list[str]) -> str:
                    return "  " + label.ljust(28) + "".join(v.ljust(col_w) for v in values)

                lines.append(_row("Tokens/sec", [f"{r.tokens_per_sec:.2f}" for r in results]))
                lines.append(_row("TTFT (ms)", [f"{r.ttft_ms:.0f}" for r in results]))
                lines.append(_row("VRAM delta (MB)", [f"{r.vram_delta_mb:.0f}" for r in results]))
                lines.append(_row("Hardware tier", [r.hardware_tier or "—" for r in results]))
                lines.append(_row("Max concurrency", [str(r.recommended_max_concurrency or "—") for r in results]))
                lines.append("")

            lines += [sep, "  End of report", sep, ""]

            report_path.write_text("\n".join(lines), encoding="utf-8")
            return report_path

        except Exception as exc:
            self.log.error(f"Failed to save benchmark report: {exc}")
            return None

    def _cancel_benchmark(self) -> None:
        if self._current_run and not self._current_run.done():
            self._current_run.cancel()

    # ── MLPerf ───────────────────────────────────────────────────────────────

    async def _populate_mlperf_model_list(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        server_type = str(self.query_one("#mlperf-server-select", Select).value)
        server = app.registry.get(server_type)
        sel = self.query_one("#mlperf-model-select", Select)
        if server is None:
            sel.set_options([("—", "__none__")])
            return
        try:
            models = await server.list_loaded_models()
        except Exception:
            models = []
        if models:
            sel.set_options([(m.display_name, m.model_id) for m in models])
            sel.value = models[0].model_id
        else:
            sel.set_options([("No models available", "__none__")])

    async def _run_mlperf(self) -> None:
        import time as _time
        app: LLMManagerApp = self.app  # type: ignore[assignment]

        server_type  = str(self.query_one("#mlperf-server-select",  Select).value)
        model_id     = str(self.query_one("#mlperf-model-select",   Select).value)
        scenario_val = str(self.query_one("#mlperf-scenario-select", Select).value)
        num_samples  = int(self.query_one("#mlperf-samples-select", Select).value)
        out_tokens   = int(self.query_one("#mlperf-tokens-select",  Select).value)
        qps          = float(self.query_one("#mlperf-qps-select",   Select).value)

        log    = self.query_one("#mlperf-log",         LogView)
        status = self.query_one("#mlperf-live-status", Static)

        if model_id == "__none__":
            self.notify("Select a model first.", severity="warning")
            return

        server = app.registry.get(server_type)
        if server is None:
            self.notify(f"Server '{server_type}' not found.", severity="error")
            return

        self.query_one("#btn-mlperf-run",    Button).disabled = True
        self.query_one("#btn-mlperf-cancel", Button).disabled = False

        log.clear_log()
        cfg = MLPerfConfig(
            scenario=MLPerfScenario(scenario_val),
            num_samples=num_samples,
            output_tokens=out_tokens,
            server_target_qps=qps,
        )
        runner = MLPerfRunner(
            server=server,
            model_id=model_id,
            config=cfg,
            gpu_provider=app.gpu_provider,  # type: ignore[attr-defined]
        )

        run_start = _time.monotonic()
        self._mlperf_run = asyncio.current_task()

        try:
            async for msg, result in runner.run():
                log.append_line(msg)
                elapsed = int(_time.monotonic() - run_start)
                status.update(
                    f"[bold]Scenario:[/] {scenario_val}  "
                    f"[bold]Model:[/] {model_id}  "
                    f"[bold]Elapsed:[/] {elapsed//60:02d}:{elapsed%60:02d}"
                )
                if result is not None:
                    status.update(
                        f"[bold {'green' if result.passed else 'red'}]"
                        f"{'PASS' if result.passed else 'FAIL'}[/]  "
                        f"TPS {result.tokens_per_sec:.1f}  "
                        f"TTFT {result.ttft.mean:.0f} ms"
                        if result.ttft else ""
                    )
                    log.append_line(
                        f"\nReport saved to: "
                        f"{runner._output_dir / ''}"
                    )
        except asyncio.CancelledError:
            log.append_line("[MLPerf] Run cancelled.")
        finally:
            self.query_one("#btn-mlperf-run",    Button).disabled = False
            self.query_one("#btn-mlperf-cancel", Button).disabled = True

    def _cancel_mlperf(self) -> None:
        if self._mlperf_run and not self._mlperf_run.done():
            self._mlperf_run.cancel()

    def _cat_enabled(self, cat: BenchmarkCategory) -> bool:
        try:
            return self.query_one(f"#cat-{cat.value}", Checkbox).value
        except Exception:
            return True

    # ── Comparison charts ────────────────────────────────────────────────────

    def _render_comparison(self, results: list[BenchmarkResult]) -> None:
        scroll = self.query_one("#compare-scroll", VerticalScroll)
        scroll.remove_children()

        sections: list[Widget] = []

        # Header
        model_names = [r.config.model_id for r in results]
        legend = "  ".join(
            f"[{_MODEL_COLORS[i % len(_MODEL_COLORS)]}]{_SYMBOLS[i % len(_SYMBOLS)]} {name}[/]"
            for i, name in enumerate(model_names)
        )
        sections.append(Static(
            f"[bold white]BENCHMARK COMPARISON[/]\n{legend}\n",
            classes="chart-title",
        ))

        # ── Bar charts ──────────────────────────────────────────────────
        sections.append(self._bar_chart_widget(
            "THROUGHPUT  (tokens / sec)",
            [(r.config.model_id, r.tokens_per_sec) for r in results],
            "TPS", lower_is_better=False,
        ))

        sections.append(self._bar_chart_widget(
            "TIME TO FIRST TOKEN  (ms)",
            [(r.config.model_id, r.ttft_ms) for r in results],
            "ms", lower_is_better=True,
        ))

        sections.append(self._bar_chart_widget(
            "MAX SAFE CONCURRENCY",
            [(r.config.model_id, float(r.recommended_max_concurrency or 0)) for r in results],
            "req", lower_is_better=False,
        ))

        sections.append(self._bar_chart_widget(
            "VRAM DELTA  (MB loaded vs unloaded)",
            [(r.config.model_id, r.vram_delta_mb) for r in results],
            "MB", lower_is_better=True,
        ))

        # ── Concurrency scaling chart ───────────────────────────────────
        has_conc = any(r.concurrency_results for r in results)
        if has_conc:
            sections.append(self._concurrency_chart_widget(results))

        # ── Context scaling chart ───────────────────────────────────────
        has_ctx = any(r.context_results for r in results)
        if has_ctx:
            sections.append(self._context_chart_widget(results))

        # ── Quality summary ─────────────────────────────────────────────
        has_quality = any(r.quality_results for r in results)
        if has_quality:
            sections.append(self._quality_table_widget(results))

        self.call_after_refresh(scroll.mount, *sections)

    # ── Chart builders ───────────────────────────────────────────────────────

    def _bar_chart_widget(
        self,
        title: str,
        data: list[tuple[str, float]],
        unit: str,
        lower_is_better: bool = False,
    ) -> Widget:
        hint = "[dim](lower is better)[/]" if lower_is_better else "[dim](higher is better)[/]"
        max_val = max((v for _, v in data), default=1.0) or 1.0
        label_w = min(max((len(n) for n, _ in data), default=10), 24)

        lines = [f"[bold cyan]{title}[/]  {hint}"]
        lines.append("[dim]" + "-" * (label_w + _BAR_WIDTH + 14) + "[/]")

        for i, (name, val) in enumerate(data):
            color  = _MODEL_COLORS[i % len(_MODEL_COLORS)]
            filled = int(_BAR_WIDTH * val / max_val)
            bar    = _BAR_FILL * filled + _BAR_EMPTY * (_BAR_WIDTH - filled)
            padded = name[:label_w].ljust(label_w)
            lines.append(f"[white]{padded}[/] [{color}]{bar}[/]  [bold]{val:.1f}[/] {unit}")

        return Static("\n".join(lines) + "\n", classes="chart-section")

    def _concurrency_chart_widget(self, results: list[BenchmarkResult]) -> Widget:
        """Table showing TPS at each concurrency level for every model."""
        # Gather all concurrency levels tested
        all_levels: list[int] = sorted({
            cr.concurrency
            for r in results
            for cr in r.concurrency_results
        })
        if not all_levels:
            return Static("")

        label_w = min(max((len(r.config.model_id) for r in results), default=10), 24)
        header_levels = "".join(f"{str(lv):>7}" for lv in all_levels)

        lines = ["[bold cyan]CONCURRENCY SCALING  (aggregate TPS at each parallel-request level)[/]"]
        lines.append("[dim]" + "-" * (label_w + 3 + len(all_levels) * 7) + "[/]")
        lines.append(f"[dim]{'Model'.ljust(label_w)}   {header_levels}[/]")
        lines.append("[dim]" + "-" * (label_w + 3 + len(all_levels) * 7) + "[/]")

        for i, r in enumerate(results):
            color = _MODEL_COLORS[i % len(_MODEL_COLORS)]
            sym   = _SYMBOLS[i % len(_SYMBOLS)]
            conc_map = {cr.concurrency: cr for cr in r.concurrency_results}
            padded = r.config.model_id[:label_w].ljust(label_w)
            row = ""
            for lv in all_levels:
                if lv in conc_map:
                    cr = conc_map[lv]
                    if cr.aborted:
                        cell = f"[red]{'CUT':>7}[/]"
                    else:
                        cell = f"[{color}]{cr.aggregate_tokens_per_sec:>7.1f}[/]"
                else:
                    cell = f"[dim]{'---':>7}[/]"
                row += cell
            lines.append(f"[{color}]{sym}[/] [{color}]{padded}[/]{row}")

        # Mini ASCII sparkline per model
        lines.append("")
        lines.append("[dim]TPS sparkline (concurrency →)[/]")
        for i, r in enumerate(results):
            if not r.concurrency_results:
                continue
            color  = _MODEL_COLORS[i % len(_MODEL_COLORS)]
            sym    = _SYMBOLS[i % len(_SYMBOLS)]
            vals   = [cr.aggregate_tokens_per_sec for cr in r.concurrency_results if not cr.aborted]
            spark  = self._sparkline(vals, width=40)
            label  = r.config.model_id[:20].ljust(20)
            lines.append(f"[{color}]{sym} {label}[/] [{color}]{spark}[/]")

        return Static("\n".join(lines) + "\n", classes="chart-section")

    def _context_chart_widget(self, results: list[BenchmarkResult]) -> Widget:
        """Table showing TPS at each context length for every model."""
        all_ctxs: list[int] = sorted({
            cr.context_length
            for r in results
            for cr in r.context_results
        })
        if not all_ctxs:
            return Static("")

        def fmt_ctx(c: int) -> str:
            return f"{c//1024}K" if c >= 1024 else str(c)

        label_w = min(max((len(r.config.model_id) for r in results), default=10), 24)
        header  = "".join(f"{fmt_ctx(c):>8}" for c in all_ctxs)

        lines = ["[bold cyan]CONTEXT SCALING  (TPS at each context length)[/]"]
        lines.append("[dim]" + "-" * (label_w + 3 + len(all_ctxs) * 8) + "[/]")
        lines.append(f"[dim]{'Model'.ljust(label_w)}   {header}[/]")
        lines.append("[dim]" + "-" * (label_w + 3 + len(all_ctxs) * 8) + "[/]")

        for i, r in enumerate(results):
            color   = _MODEL_COLORS[i % len(_MODEL_COLORS)]
            sym     = _SYMBOLS[i % len(_SYMBOLS)]
            ctx_map = {cr.context_length: cr for cr in r.context_results}
            padded  = r.config.model_id[:label_w].ljust(label_w)
            row     = ""
            for c in all_ctxs:
                if c in ctx_map:
                    cr = ctx_map[c]
                    if cr.error:
                        cell = f"[red]{'ERR':>8}[/]"
                    else:
                        cell = f"[{color}]{cr.tokens_per_sec:>8.1f}[/]"
                else:
                    cell = f"[dim]{'---':>8}[/]"
                row += cell
            lines.append(f"[{color}]{sym}[/] [{color}]{padded}[/]{row}")

        # Sparklines
        lines.append("")
        lines.append("[dim]TPS sparkline (context length →)[/]")
        for i, r in enumerate(results):
            if not r.context_results:
                continue
            color  = _MODEL_COLORS[i % len(_MODEL_COLORS)]
            sym    = _SYMBOLS[i % len(_SYMBOLS)]
            vals   = [cr.tokens_per_sec for cr in r.context_results if not cr.error]
            spark  = self._sparkline(vals, width=40)
            label  = r.config.model_id[:20].ljust(20)
            lines.append(f"[{color}]{sym} {label}[/] [{color}]{spark}[/]")

        return Static("\n".join(lines) + "\n", classes="chart-section")

    def _quality_table_widget(self, results: list[BenchmarkResult]) -> Widget:
        """Average latency per quality probe set for each model."""
        probe_sets: list[str] = sorted({
            qr.probe_set
            for r in results
            for qr in r.quality_results
        })
        if not probe_sets:
            return Static("")

        label_w = min(max((len(r.config.model_id) for r in results), default=10), 24)
        header  = "".join(f"{ps:>12}" for ps in probe_sets)

        lines = ["[bold cyan]QUALITY PROBE LATENCY  (avg ms per probe set)[/]"]
        lines.append("[dim]" + "-" * (label_w + 3 + len(probe_sets) * 12) + "[/]")
        lines.append(f"[dim]{'Model'.ljust(label_w)}   {header}[/]")
        lines.append("[dim]" + "-" * (label_w + 3 + len(probe_sets) * 12) + "[/]")

        for i, r in enumerate(results):
            color  = _MODEL_COLORS[i % len(_MODEL_COLORS)]
            sym    = _SYMBOLS[i % len(_SYMBOLS)]
            padded = r.config.model_id[:label_w].ljust(label_w)
            row    = ""
            for ps in probe_sets:
                probes = [qr for qr in r.quality_results if qr.probe_set == ps]
                if probes:
                    avg = sum(qr.latency_ms for qr in probes) / len(probes)
                    row += f"[{color}]{avg:>12.0f}[/]"
                else:
                    row += f"[dim]{'---':>12}[/]"
            lines.append(f"[{color}]{sym}[/] [{color}]{padded}[/]{row}")

        return Static("\n".join(lines) + "\n", classes="chart-section")

    @staticmethod
    def _sparkline(values: list[float], width: int = 30) -> str:
        """Render a mini terminal sparkline using block characters."""
        if not values:
            return ""
        _BLOCKS = " ▁▂▃▄▅▆▇█"
        mx = max(values) or 1.0
        # Resample to `width` points
        step = len(values) / width
        resampled = [values[min(int(i * step), len(values) - 1)] for i in range(width)]
        return "".join(_BLOCKS[int(v / mx * (len(_BLOCKS) - 1))] for v in resampled)

    # ── Single-model results (Results tab) ───────────────────────────────────

    def _show_single_result(self, result: BenchmarkResult) -> None:
        results_content = self.query_one("#results-tab-content", Static)
        lines = [
            f"[bold cyan]{result.config.model_id}[/]",
            f"Server: {result.config.server_type}",
            f"Tokens/sec: [bold]{result.tokens_per_sec:.2f}[/]",
            f"TTFT: [bold]{result.ttft_ms:.0f} ms[/]",
            f"VRAM delta: {result.vram_delta_mb:.0f} MB",
            f"Tier: {result.hardware_tier or '—'}",
        ]
        if result.recommended_max_concurrency:
            lines.append(f"Recommended max concurrency: {result.recommended_max_concurrency}")
        if result.error:
            lines.append(f"[red]Error: {result.error}[/]")
        results_content.update("\n".join(lines))

    # ── History ───────────────────────────────────────────────────────────────

    async def _load_history(self, table: DataTable) -> None:
        if not BENCHMARK_DIR.exists():
            return
        for f in sorted(BENCHMARK_DIR.glob("*.json"), reverse=True)[:50]:
            try:
                data = json.loads(f.read_text())
                cfg  = data.get("config", {})
                table.add_row(
                    data.get("timestamp", "")[:19],
                    cfg.get("server_type", ""),
                    cfg.get("model_id", ""),
                    f"{data.get('tokens_per_sec', 0):.1f}",
                    f"{data.get('ttft_ms', 0):.0f}",
                    str(data.get("recommended_max_concurrency") or "—"),
                    data.get("hardware_tier", "—"),
                    key=str(f),
                )
            except Exception:
                pass

    # ── Keyboard actions ──────────────────────────────────────────────────────

    def action_run_benchmark(self) -> None:
        self.run_worker(self._start_benchmark())

    def action_cancel_benchmark(self) -> None:
        self._cancel_benchmark()
