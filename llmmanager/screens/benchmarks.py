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

    DEFAULT_CSS = """
    BenchmarksScreen { width: 1fr; height: 1fr; }
    #bench-model-row { height: auto; }
    #bench-model-row Select { width: 1fr; }
    #btn-refresh-models { width: 5; margin-left: 1; }
    #bench-live-stats {
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary-darken-2;
        margin-top: 1;
        color: $text;
    }
    #run-controls Button { width: 1fr; }
    """

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
                    yield Label("Model:", classes="form-label")
                    with Horizontal(id="bench-model-row"):
                        yield Select(
                            options=[("—", "__none__")],
                            value="__none__",
                            id="bench-model-select",
                        )
                        yield Button("↻", id="btn-refresh-models", variant="default")
                    yield Checkbox(
                        "Auto-swap: unload current model and load selected before benchmark",
                        value=True,
                        id="bench-auto-swap",
                    )

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
                        yield Button("Cancel",        id="btn-cancel-bench", variant="error", disabled=True)

                    yield Static("", id="bench-live-stats")
                    yield Label("", id="bench-progress-label")
                    yield ProgressBar(total=100, id="bench-progress-bar", show_eta=False)
                    yield LogView(max_lines=300, id="bench-log")

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
        self.run_worker(self._populate_model_select())

    async def _populate_model_select(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        server_type = str(self.query_one("#bench-server-select", Select).value)
        server = app.registry.get(server_type)
        select = self.query_one("#bench-model-select", Select)
        if server is None:
            return
        try:
            models = await server.list_loaded_models()
        except Exception:
            models = []
        if models:
            select.set_options([(m.display_name, m.model_id) for m in models])
        else:
            select.set_options([("No models loaded — check Ollama is running", "__none__")])

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "bench-server-select":
            self.run_worker(self._populate_model_select())

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
            case "btn-refresh-models":
                self.run_worker(self._populate_model_select())

    async def _start_benchmark(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]

        server_type = str(self.query_one("#bench-server-select", Select).value)
        model_id = str(self.query_one("#bench-model-select", Select).value)
        profile_val = str(self.query_one("#bench-profile-select", Select).value)
        auto_swap = self.query_one("#bench-auto-swap", Checkbox).value

        if not model_id or model_id == "__none__":
            self.notify("Select a model first.", severity="warning")
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
            self._run_benchmark_task(runner, config, auto_swap=auto_swap)
        )

    async def _run_benchmark_task(
        self, runner: BenchmarkRunner, config: BenchmarkConfig, auto_swap: bool = False
    ) -> None:
        import time as _time
        log          = self.query_one("#bench-log", LogView)
        progress_bar = self.query_one("#bench-progress-bar", ProgressBar)
        progress_lbl = self.query_one("#bench-progress-label", Label)
        live_stats   = self.query_one("#bench-live-stats", Static)

        log.clear_log()
        progress_bar.progress = 0
        live_stats.update("")

        run_start = _time.monotonic()

        # Phase tracking for progress bar (rough % weights)
        _PHASE_PCT = {
            "warmup": 5, "throughput": 30, "memory": 5,
            "concurrency": 40, "context": 15, "quality": 5,
        }
        _phase_done = 0.0

        def _elapsed() -> str:
            s = int(_time.monotonic() - run_start)
            m, s = divmod(s, 60)
            return f"{m:02d}:{s:02d}"

        # Accumulated live stats for the stats panel
        _stats: dict = {}

        def _render_stats() -> str:
            parts = [f"[bold]Elapsed:[/] {_elapsed()}"]
            if "tps" in _stats:
                parts.append(f"[bold]TPS:[/] {_stats['tps']:.1f}")
            if "ttft" in _stats:
                parts.append(f"[bold]TTFT:[/] {_stats['ttft']:.0f} ms")
            if "phase" in _stats:
                parts.append(f"[bold]Phase:[/] {_stats['phase']}")
            if "conc" in _stats:
                parts.append(f"[bold]Concurrency:[/] {_stats['conc']}")
            return "   ".join(parts)

        # Auto-swap model
        if auto_swap:
            app: LLMManagerApp = self.app  # type: ignore[assignment]
            server = app.registry.get(config.server_type)
            if server:
                try:
                    loaded = await server.list_loaded_models()
                    for m in loaded:
                        if m.model_id != config.model_id:
                            log.append_line(f"Unloading {m.model_id}…")
                            await server.unload_model(m.model_id)
                    if not any(m.model_id == config.model_id for m in loaded):
                        log.append_line(f"Loading {config.model_id}…")
                        await server.load_model(config.model_id)
                        log.append_line("Model ready.")
                except Exception as exc:
                    log.append_line(f"[yellow]Warning: model swap failed: {exc}[/]")

        # Start a timer task that refreshes the stats panel every second
        async def _tick() -> None:
            while True:
                live_stats.update(_render_stats())
                await asyncio.sleep(1.0)

        ticker = asyncio.create_task(_tick())

        try:
            async for msg, result in runner.run(config):
                log.append_line(msg)
                progress_lbl.update(msg)

                # Parse phase transitions for progress bar
                ml = msg.lower()
                if "warmup" in ml:
                    _stats["phase"] = "Warm-up"
                    progress_bar.progress = _phase_done
                elif "throughput" in ml:
                    _stats["phase"] = "Throughput"
                    progress_bar.progress = _phase_done
                elif "vram" in ml or "memory" in ml:
                    _stats["phase"] = "Memory"
                    _phase_done += _PHASE_PCT["memory"]
                    progress_bar.progress = _phase_done
                elif "concurrency test" in ml:
                    m_conc = msg.split(":")[-1].strip().split()[0] if ":" in msg else ""
                    _stats["phase"] = "Concurrency"
                    _stats["conc"] = m_conc
                    progress_bar.progress = min(_phase_done + _PHASE_PCT["concurrency"] * 0.5, 95)
                elif "complete" in ml or "avg" in ml:
                    _phase_done = min(_phase_done + 30, 95)
                    progress_bar.progress = _phase_done

                # Extract TPS / TTFT from per-run lines
                if "tps" in ml and "ttft" in ml:
                    try:
                        parts = msg.split()
                        tps_i = next(i for i, p in enumerate(parts) if p == "TPS")
                        ttft_i = next(i for i, p in enumerate(parts) if p == "ms")
                        _stats["tps"] = float(parts[tps_i - 1])
                        _stats["ttft"] = float(parts[ttft_i - 1])
                    except Exception:
                        pass

                live_stats.update(_render_stats())

                if result is not None:
                    self._last_result = result
                    progress_bar.progress = 100
                    _stats["tps"] = result.tokens_per_sec
                    _stats["ttft"] = result.ttft_ms
                    _stats["phase"] = "Complete"
                    live_stats.update(_render_stats())
                    self._show_results(result)

        except asyncio.CancelledError:
            log.append_line("Benchmark cancelled.")
        finally:
            ticker.cancel()
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
