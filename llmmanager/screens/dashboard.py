"""Dashboard screen — live status overview."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import psutil
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Label, ProgressBar, Static

from llmmanager.widgets.gpu_meter import GPUMeter
from llmmanager.widgets.server_card import ServerCard

if TYPE_CHECKING:
    from llmmanager.app import LLMManagerApp


class DashboardScreen(Screen):
    """Screen 1 — live metrics grid."""

    BINDINGS = [
        ("r", "restart_selected", "Restart server"),
        ("f5", "force_refresh", "Refresh now"),
    ]

    def compose(self) -> ComposeResult:
        yield Label("Server Status", id="dash-servers-heading", classes="section-heading")
        with Horizontal(id="server-cards-row"):
            yield ServerCard("ollama", id="card-ollama")
            yield ServerCard("vllm",   id="card-vllm")
            yield ServerCard("lmstudio", id="card-lmstudio")

        yield Label("GPU", id="dash-gpu-heading", classes="section-heading")
        with ScrollableContainer(id="gpu-meters-container"):
            # GPU meters are added dynamically in on_mount
            pass

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

    def action_force_refresh(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        self.run_worker(app.poller.force_poll())
