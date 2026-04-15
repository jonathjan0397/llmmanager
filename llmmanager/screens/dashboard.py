"""Dashboard screen — live status overview."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import psutil
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widget import Widget
from textual.widgets import Button, Label, ProgressBar, Select, Static

from llmmanager.widgets.gpu_meter import GPUMeter
from llmmanager.widgets.server_card import ServerCard

if TYPE_CHECKING:
    from llmmanager.app import LLMManagerApp


class DashboardScreen(Widget):
    """Screen 1 — live metrics grid."""

    DEFAULT_CSS = """
    DashboardScreen { width: 1fr; height: 1fr; }

    #quick-load-box {
        height: auto;
        border: ascii $primary-darken-2;
        background: $surface-darken-1;
        padding: 1;
        margin-top: 1;
    }

    #quick-load-row {
        height: auto;
    }

    #quick-load-row Label  { margin: 1 1 0 0; }
    #ql-server-select      { width: 14; margin-right: 1; }
    #ql-model-select       { width: 1fr; }
    #btn-ql-refresh        { width: 3; margin-left: 1; }
    #btn-ql-load           { width: 10; margin-left: 1; }
    #btn-ql-unload         { width: 10; margin-left: 1; }
    #ql-status             { color: $text-muted; margin-top: 0; }
    """

    BINDINGS = [
        ("r", "restart_selected", "Restart server"),
        ("f5", "force_refresh",   "Refresh now"),
    ]

    def compose(self) -> ComposeResult:
        yield Label("Server Status", id="dash-servers-heading", classes="section-heading")
        with Horizontal(id="server-cards-row"):
            yield ServerCard("ollama")
            yield ServerCard("vllm")
            yield ServerCard("lmstudio")
            yield ServerCard("llamacpp")

        yield Label("GPU", id="dash-gpu-heading", classes="section-heading")
        with ScrollableContainer(id="gpu-meters-container"):
            pass

        yield Label("Quick Load Model", classes="section-heading")
        with Vertical(id="quick-load-box"):
            with Horizontal(id="quick-load-row"):
                yield Label("Server:")
                yield Select(
                    options=[
                        ("Ollama",    "ollama"),
                        ("vLLM",      "vllm"),
                        ("LM Studio", "lmstudio"),
                        ("llama.cpp", "llamacpp"),
                    ],
                    value="ollama",
                    id="ql-server-select",
                )
                yield Label("Model:")
                yield Select(
                    options=[("—", "__none__")],
                    value="__none__",
                    id="ql-model-select",
                )
                yield Button("↻",      id="btn-ql-refresh", variant="default",
                             tooltip="Refresh model list")
                yield Button("Load",   id="btn-ql-load",    variant="success")
                yield Button("Unload", id="btn-ql-unload",  variant="warning")
            yield Label("", id="ql-status")

        yield Label("System", id="dash-system-heading", classes="section-heading")
        with Vertical(id="system-meters"):
            yield Label("CPU", id="cpu-label")
            yield ProgressBar(total=100, id="cpu-bar", show_eta=False)
            yield Label("RAM", id="ram-label")
            yield ProgressBar(total=100, id="ram-bar", show_eta=False)

    def on_mount(self) -> None:
        self._update_task = self.set_interval(2.0, self._refresh)
        self.run_worker(self._populate_model_select())

    # ------------------------------------------------------------------
    # Model select population
    # ------------------------------------------------------------------

    async def _populate_model_select(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        server_type = str(self.query_one("#ql-server-select", Select).value)
        server = app.registry.get(server_type)
        select = self.query_one("#ql-model-select", Select)
        if server is None:
            select.set_options([("—", "__none__")])
            return
        try:
            models = await server.list_loaded_models()
        except Exception:
            models = []
        if models:
            select.set_options([(m.display_name, m.model_id) for m in models])
        else:
            select.set_options([("No models available", "__none__")])

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "ql-server-select":
            self.run_worker(self._populate_model_select())

    # ------------------------------------------------------------------
    # Dashboard refresh
    # ------------------------------------------------------------------

    async def _refresh(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        try:
            snapshot = app.poller.queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        for info in snapshot.servers:
            try:
                card = self.query_one(f"#card-{info.server_type}", ServerCard)
                card.update_info(info)
            except Exception:
                pass

        container = self.query_one("#gpu-meters-container")
        for gpu in snapshot.gpus:
            meter_id = f"gpu-meter-{gpu.index}"
            try:
                meter = self.query_one(f"#{meter_id}", GPUMeter)
            except Exception:
                meter = GPUMeter(gpu.index)
                container.mount(meter)
            meter.update_gpu(gpu)

        self.query_one("#cpu-label", Label).update(
            f"CPU   {snapshot.cpu_pct:.1f}%"
        )
        cpu_bar = self.query_one("#cpu-bar", ProgressBar)
        cpu_bar.advance(snapshot.cpu_pct - (cpu_bar.progress or 0))

        ram_pct = (snapshot.ram_used_mb / snapshot.ram_total_mb * 100) if snapshot.ram_total_mb else 0
        self.query_one("#ram-label", Label).update(
            f"RAM   {snapshot.ram_used_mb / 1024:.1f} / {snapshot.ram_total_mb / 1024:.1f} GB  "
            f"({ram_pct:.1f}%)"
        )
        ram_bar = self.query_one("#ram-bar", ProgressBar)
        ram_bar.advance(ram_pct - (ram_bar.progress or 0))

    # ------------------------------------------------------------------
    # Button handler
    # ------------------------------------------------------------------

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "btn-ql-load":
                await self._quick_load()
            case "btn-ql-unload":
                await self._quick_unload()
            case "btn-ql-refresh":
                self.run_worker(self._populate_model_select())

    async def _quick_load(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        status = self.query_one("#ql-status", Label)
        server_type = str(self.query_one("#ql-server-select", Select).value)
        model_id = str(self.query_one("#ql-model-select", Select).value)
        if not model_id or model_id == "__none__":
            status.update("[red]Select a model first.[/]")
            return
        server = app.registry.get(server_type)
        if server is None:
            status.update(f"[red]Server '{server_type}' not found.[/]")
            return
        status.update(f"Loading {model_id}…")
        try:
            await server.load_model(model_id)
            status.update(f"[green]+ {model_id} loaded.[/]")
        except Exception as exc:
            status.update(f"[red]Error: {exc}[/]")

    async def _quick_unload(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        status = self.query_one("#ql-status", Label)
        server_type = str(self.query_one("#ql-server-select", Select).value)
        model_id = str(self.query_one("#ql-model-select", Select).value)
        if not model_id or model_id == "__none__":
            status.update("[red]Select a model first.[/]")
            return
        server = app.registry.get(server_type)
        if server is None:
            status.update(f"[red]Server '{server_type}' not found.[/]")
            return
        status.update(f"Unloading {model_id}…")
        try:
            await server.unload_model(model_id)
            status.update(f"[yellow]- {model_id} unloaded.[/]")
        except Exception as exc:
            status.update(f"[red]Error: {exc}[/]")

    def action_force_refresh(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        self.run_worker(app.poller.force_poll())
