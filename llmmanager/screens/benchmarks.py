"""Benchmark Suite screen — full hardware-aware benchmark runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
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


class BenchmarksScreen(Widget):
    """Screen 5 — run, compare, and review benchmarks."""

    DEFAULT_CSS = "BenchmarksScreen { width: 1fr; height: 1fr; }"

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
                with Vertical(id="run-form"):
                    yield Label("Server:", classes="form-label")
                    yield Select(
                        options=[("Ollama", "ollama"), ("vLLM", "vllm")],
                        value="ollama",
                        id="bench-server-select",
                    )
                    yield Label("Model ID:", classes="form-label")
                    yield Input(placeholder="e.g. llama3.2:3b", id="bench-model-input")

                    yield Label("Profile:", classes="form-label")
                    yield Select(
                        options=[
                            ("Quick (~2 min)",     "quick"),
                            ("Standard (~10 min)", "standard"),
                            ("Stress (full ramp)", "stress"),
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
                    yield Label("", id="hw-snapshot-label", classes="form-hint")

                    with Horizontal(id="run-controls"):
                        yield Button("Run Benchmark", id="btn-run-bench", variant="primary")
                        yield Button("Cancel", id="btn-cancel-bench", variant="error", disabled=True)

                    yield Label("", id="bench-progress-label")
                    yield ProgressBar(total=100, id="bench-progress-bar", show_eta=False)
                    yield LogView(max_lines=200, id="bench-log")

            with TabPane("Results", id="tab-results"):
                yield Static(id="results-tab-content")

            with TabPane("History", id="tab-history"):
                yield DataTable(id="history-table", cursor_type="row")
                yield Button("View Details", id="btn-view-history", variant="default")

            with TabPane("Compare", id="tab-compare"):
                yield Label("Select two results in History to compare.", id="compare-placeholder")

    def on_mount(self) -> None:
        table = self.query_one("#history-table", DataTable)
        table.add_columns("Timestamp", "Server", "Model", "TPS", "TTFT ms", "Max Concurrency", "Tier")
        self.run_worker(self._load_history(table))

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

        categories = [cat for cat in BenchmarkCategory if self._cat_enabled(cat)]
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

        # Build children list and pass to Vertical constructor — avoids mounting
        # to an unattached widget (not supported in Textual 8.2.3+).
        children: list[Widget] = [
            Label(f"Model: {result.config.model_id}", classes="section-heading"),
            Label(f"Server: {result.config.server_type}"),
            Label(f"Tokens/sec: {result.tokens_per_sec:.2f}"),
            Label(f"TTFT: {result.ttft_ms:.0f} ms"),
            Label(f"VRAM delta: {result.vram_delta_mb:.0f} MB"),
            Label(f"Hardware tier: {result.hardware_tier or '—'}"),
        ]

        if result.recommended_max_concurrency:
            children.append(Label(
                f"Recommended max concurrency: {result.recommended_max_concurrency}",
                classes="section-heading",
            ))

        if result.concurrency_results:
            children.append(Label("Concurrency Results:", classes="section-heading"))
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
            children.append(table)

        if result.context_results:
            children.append(Label("Context Scaling:", classes="section-heading"))
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
            children.append(ctx_table)

        results_content.mount(Vertical(*children))
