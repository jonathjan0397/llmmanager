"""Dashboard screen — live status overview."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import psutil
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widget import Widget
from textual.widgets import Label, ProgressBar, Static

from textual.widgets import Button, Input, Select
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
        border: solid $primary-darken-2;
        background: $surface-darken-1;
        padding: 1;
        margin-top: 1;
    }

    #quick-load-row {
        height: auto;
    }

    #quick-load-row Label   { margin: 1 1 0 0; }
    #quick-load-row Select  { width: 16; margin-right: 1; }
    #ql-model-input         { width: 1fr; }
    #btn-ql-load            { width: 10; margin-left: 1; }
    #btn-ql-unload          { width: 10; margin-left: 1; }
    #ql-status              { color: $text-muted; margin-top: 0; }
    """

    BINDINGS = [
        ("r", "restart_selected", "Restart server"),
        ("f5", "force_refresh", "Refresh now"),
    ]

    def compose(self) -> ComposeResult:
        yield Label("Server Status", id="dash-servers-heading", classes="section-heading")
        with Horizontal(id="server-cards-row"):
            yield ServerCard("ollama")
            yield ServerCard("vllm")
            yield ServerCard("lmstudio")

        yield Label("GPU", id="dash-gpu-heading", classes="section-heading")
        with ScrollableContainer(id="gpu-meters-container"):
            # GPU meters are added dynamically in on_mount
            pass

        yield Label("Quick Load Model", classes="section-heading")
        with Vertical(id="quick-load-box"):
            with Horizontal(id="quick-load-row"):
                yield Label("Server:")
                yield Select(
                    options=[("Ollama", "ollama"), ("vLLM", "vllm"), ("LM Studio", "lmstudio")],
                    value="ollama",
                    id="ql-server-select",
                )
                yield Input(placeholder="model name, e.g. llama3.2:3b", id="ql-model-input")
                yield Button("Load",   id="btn-ql-load",   variant="success")
                yield Button("Unload", id="btn-ql-unload", variant="warning")
            yield Label("", id="ql-status")

        yield Label("System", id="dash-system-heading", classes="section-heading")
        with Vertical(id="system-meters"):
            yield Label("CPU", id="cpu-label")
            yield ProgressBar(total=100, id="cpu-bar", show_eta=False)
            yield Label("RAM", id="ram-label")
            yield ProgressBar(total=100, id="ram-bar", show_eta=False)

    def on_mount(self) -> None:
        self._update_task = self.set_interval(2.0, self._refresh)

    async def _refresh(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        try:
            snapshot = app.poller.queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        # Update server cards
        for info in snapshot.servers:
            try:
                card = self.query_one(f"#card-{info.server_type}", ServerCard)
                card.update_info(info)
            except Exception:
                pass

        # Update GPU meters — add new ones if we see new GPU indices
        container = self.query_one("#gpu-meters-container")
        for gpu in snapshot.gpus:
            meter_id = f"gpu-meter-{gpu.index}"
            try:
                meter = self.query_one(f"#{meter_id}", GPUMeter)
            except Exception:
                meter = GPUMeter(gpu.index)
                container.mount(meter)
            meter.update_gpu(gpu)

        # Update system meters
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

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "btn-ql-load":
                await self._quick_load()
            case "btn-ql-unload":
                await self._quick_unload()

    async def _quick_load(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        status = self.query_one("#ql-status", Label)
        server_type = str(self.query_one("#ql-server-select", Select).value)
        model_id = self.query_one("#ql-model-input", Input).value.strip()
        if not model_id:
            status.update("[red]Enter a model name.[/]")
            return
        server = app.registry.get(server_type)
        if server is None:
            status.update(f"[red]Server '{server_type}' not found.[/]")
            return
        status.update(f"Loading {model_id}…")
        try:
            await server.load_model(model_id)
            status.update(f"[green]✓ {model_id} loaded.[/]")
        except Exception as exc:
            status.update(f"[red]Error: {exc}[/]")

    async def _quick_unload(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        status = self.query_one("#ql-status", Label)
        server_type = str(self.query_one("#ql-server-select", Select).value)
        model_id = self.query_one("#ql-model-input", Input).value.strip()
        if not model_id:
            status.update("[red]Enter a model name.[/]")
            return
        server = app.registry.get(server_type)
        if server is None:
            status.update(f"[red]Server '{server_type}' not found.[/]")
            return
        status.update(f"Unloading {model_id}…")
        try:
            await server.unload_model(model_id)
            status.update(f"[yellow]✓ {model_id} unloaded.[/]")
        except Exception as exc:
            status.update(f"[red]Error: {exc}[/]")

    def action_force_refresh(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        self.run_worker(app.poller.force_poll())
