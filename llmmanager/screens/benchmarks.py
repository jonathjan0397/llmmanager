"""Benchmark Suite screen — full hardware-aware benchmark runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Input,
    Label,
    ProgressBar,
    Select,
    Static,
    Tab,
    TabbedContent,
    TabPane,
)

from llmmanager.benchmarks.runner import BenchmarkRunner
from llmmanager.constants import BENCHMARK_DIR, BENCHMARK_CONCURRENCY_LEVELS
from llmmanager.models.benchmark import (
    BenchmarkCategory,
    BenchmarkConfig,
    BenchmarkProfile,
    BenchmarkResult,
)
from llmmanager.widgets.log_view import LogView

if TYPE_CHECKING:
    from llmmanager.app import LLMManagerApp


class BenchmarksScreen(Screen):
    """Screen 5 — run, compare, and review benchmarks."""

    BINDINGS = [
        ("r", "run_benchmark",    "Run"),
        ("c", "cancel_benchmark", "Cancel"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._current_run: asyncio.Task | None = None
        self._last_result: BenchmarkResult | None = None

    def compose(self) -> ComposeResult:
        with TabbedContent(id="bench-tabs"):
            with TabPane("Run", id="tab-run"):
                yield self._compose_run_tab()

            with TabPane("Results", id="tab-results"):
                yield self._compose_results_tab()

            with TabPane("History", id="tab-history"):
                yield self._compose_history_tab()

            with TabPane("Compare", id="tab-compare"):
                yield Label("Select two results in History to compare.", id="compare-placeholder")

    def _compose_run_tab(self) -> Static:
        return Static(id="run-tab-content")

    def _compose_results_tab(self) -> Static:
        return Static(id="results-tab-content")

    def _compose_history_tab(self) -> Static:
        return Static(id="history-tab-content")

    def on_mount(self) -> None:
        self._build_run_tab()
        self._build_history_tab()

    def _build_run_tab(self) -> None:
        run_content = self.query_one("#run-tab-content", Static)
        run_content.remove_children()
        run_content.mount(self._run_tab_widgets())

    def _run_tab_widgets(self) -> Vertical:
        v = Vertical(id="run-form")

        # Server + model selection
        v.mount(Label("Server:", classes="form-label"))
        v.mount(Select(
            options=[("Ollama", "ollama"), ("vLLM", "vllm")],
            value="ollama",
            id="bench-server-select",
        ))
        v.mount(Label("Model ID:", classes="form-label"))
        v.mount(Input(placeholder="e.g. llama3.2:3b", id="bench-model-input"))

        # Profile selection
        v.mount(Label("Profile:", classes="form-label"))
        v.mount(Select(
            options=[
                ("Quick (~2 min)",    "quick"),
                ("Standard (~10 min)","standard"),
                ("Stress (full ramp)","stress"),
            ],
            value="standard",
            id="bench-profile-select",
        ))

        # Category toggles
        v.mount(Label("Categories:", classes="form-label"))
        for cat in BenchmarkCategory:
            v.mount(Checkbox(cat.value.replace("_", " ").title(), value=True, id=f"cat-{cat.value}"))

        v.mount(Label(
            f"Concurrency levels: {', '.join(str(x) for x in BENCHMARK_CONCURRENCY_LEVELS)}",
            classes="form-hint",
        ))

        # Hardware snapshot
        v.mount(Label("", id="hw-snapshot-label", classes="form-hint"))

        # Run controls
        with Horizontal(id="run-controls"):
            v.mount(Button("Run Benchmark", id="btn-run-bench", variant="primary"))
            v.mount(Button("Cancel",        id="btn-cancel-bench", variant="error", disabled=True))

        # Progress
        v.mount(Label("", id="bench-progress-label"))
        v.mount(ProgressBar(total=100, id="bench-progress-bar", show_eta=False))
        v.mount(LogView(max_lines=200, id="bench-log"))

        return v

    def _build_history_tab(self) -> None:
        hist = self.query_one("#history-tab-content", Static)
        hist.remove_children()

        table = DataTable(id="history-table", cursor_type="row")
        table.add_columns("Timestamp", "Server", "Model", "TPS", "TTFT ms", "Max Concurrency", "Tier")
        hist.mount(table)

        # Load saved results
        self.run_worker(self._load_history(table))

        hist.mount(Button("View Details", id="btn-view-history", variant="default"))

    async def _load_history(self, table: DataTable) -> None:
        if not BENCHMARK_DIR.exists():
            return
        for f in sorted(BENCHMARK_DIR.glob("*.json"), reverse=True)[:50]:
            try:
                data = json.loads(f.read_text())
                cfg = data.get("config", {})
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

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "btn-run-bench":
                await self._start_benchmark()
            case "btn-cancel-bench":
                self._cancel_benchmark()

    async def _start_benchmark(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]

        server_type = str(self.query_one("#bench-server-select", Select).value)
        model_id = self.query_one("#bench-model-input", Input).value.strip()
        profile_val = str(self.query_one("#bench-profile-select", Select).value)

        if not model_id:
            self.notify("Enter a model ID.", severity="warning")
            return

        server = app.registry.get(server_type)
        if server is None:
            self.notify(f"Server '{server_type}' not found.", severity="error")
            return

        categories = [
            cat for cat in BenchmarkCategory
            if self._cat_enabled(cat)
        ]

        profile = BenchmarkProfile(profile_val)
        concurrency = BENCHMARK_CONCURRENCY_LEVELS

        if profile == BenchmarkProfile.QUICK:
            categories = [BenchmarkCategory.THROUGHPUT, BenchmarkCategory.LATENCY]
            concurrency = [1, 2, 4]

        config = BenchmarkConfig(
            server_type=server_type,
            model_id=model_id,
            profile=profile,
            categories=categories,
            concurrency_levels=concurrency,
        )

        runner = BenchmarkRunner(server, app.gpu_provider)

        self.query_one("#btn-run-bench", Button).disabled = True
        self.query_one("#btn-cancel-bench", Button).disabled = False

        self._current_run = asyncio.create_task(
            self._run_benchmark_task(runner, config)
        )

    async def _run_benchmark_task(
        self, runner: BenchmarkRunner, config: BenchmarkConfig
    ) -> None:
        log = self.query_one("#bench-log", LogView)
        log.clear_log()
        progress_label = self.query_one("#bench-progress-label", Label)

        try:
            async for msg, result in runner.run(config):
                log.append_line(msg)
                progress_label.update(msg)
                if result is not None:
                    self._last_result = result
                    self._show_results(result)
        except asyncio.CancelledError:
            log.append_line("Benchmark cancelled.")
        finally:
            self.query_one("#btn-run-bench", Button).disabled = False
            self.query_one("#btn-cancel-bench", Button).disabled = True

    def _cancel_benchmark(self) -> None:
        if self._current_run and not self._current_run.done():
            self._current_run.cancel()

    def _cat_enabled(self, cat: BenchmarkCategory) -> bool:
        try:
            cb = self.query_one(f"#cat-{cat.value}", Checkbox)
            return cb.value
        except Exception:
            return True

    def _show_results(self, result: BenchmarkResult) -> None:
        results_content = self.query_one("#results-tab-content", Static)
        results_content.remove_children()

        v = Vertical()
        v.mount(Label(f"Model: {result.config.model_id}", classes="section-heading"))
        v.mount(Label(f"Server: {result.config.server_type}"))
        v.mount(Label(f"Tokens/sec: {result.tokens_per_sec:.2f}"))
        v.mount(Label(f"TTFT: {result.ttft_ms:.0f} ms"))
        v.mount(Label(f"VRAM delta: {result.vram_delta_mb:.0f} MB"))
        v.mount(Label(f"Hardware tier: {result.hardware_tier or '—'}"))

        if result.recommended_max_concurrency:
            v.mount(Label(
                f"Recommended max concurrency: {result.recommended_max_concurrency}",
                classes="section-heading",
            ))

        if result.concurrency_results:
            v.mount(Label("Concurrency Results:", classes="section-heading"))
            table = DataTable(id="conc-results-table")
            table.add_columns(
                "Concurrency", "Req/s", "TPS", "Latency p50", "Latency p99",
                "TTFT p50", "Errors", "Status"
            )
            for cr in result.concurrency_results:
                table.add_row(
                    str(cr.concurrency),
                    f"{cr.successful}/{cr.total_requests}",
                    f"{cr.aggregate_tokens_per_sec:.1f}",
                    f"{cr.per_request_latency.p50_ms:.0f} ms",
                    f"{cr.per_request_latency.p99_ms:.0f} ms",
                    f"{cr.ttft_ms.p50_ms:.0f} ms",
                    str(cr.failed),
                    "[red]ABORTED[/]" if cr.aborted else "[green]OK[/]",
                )
            v.mount(table)

        if result.context_results:
            v.mount(Label("Context Scaling:", classes="section-heading"))
            ctx_table = DataTable(id="ctx-results-table")
            ctx_table.add_columns("Context Len", "TPS", "TTFT ms", "VRAM MB", "Error")
            for cr in result.context_results:
                ctx_table.add_row(
                    f"{cr.context_length:,}",
                    f"{cr.tokens_per_sec:.1f}",
                    f"{cr.ttft_ms:.0f}",
                    f"{cr.vram_mb:.0f}",
                    cr.error or "",
                )
            v.mount(ctx_table)

        results_content.mount(v)
