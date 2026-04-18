"""GPU utilization screen — per-GPU stats and running process table."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import ScrollableContainer
from textual.widget import Widget
from textual.widgets import DataTable, Label, ProgressBar

if TYPE_CHECKING:
    from llmmanager.app import LLMManagerApp


class GPUScreen(Widget):
    """Screen 9 — live GPU utilization with per-GPU stats and process table."""

    DEFAULT_CSS = """
    GPUScreen { width: 1fr; height: 1fr; }

    /* Scrollable row of cards — horizontal scroll when > 3 GPUs */
    #gpu-cards-scroll {
        height: auto;
        max-height: 18;
        overflow-x: auto;
        overflow-y: hidden;
    }

    #gpu-cards-container {
        layout: grid;
        grid-size: 3;
        grid-columns: 1fr 1fr 1fr;
        grid-gutter: 1;
        width: 1fr;
        height: auto;
    }

    #gpu-procs-heading { margin-top: 1; }
    #gpu-procs-table   { height: 1fr; }
    """

    def compose(self) -> ComposeResult:
        yield Label("GPU Utilization", classes="section-heading")
        with ScrollableContainer(id="gpu-cards-scroll"):
            yield Widget(id="gpu-cards-container")
        yield Label("Running GPU Processes", id="gpu-procs-heading", classes="section-heading")
        yield DataTable(id="gpu-procs-table", cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one("#gpu-procs-table", DataTable)
        table.add_columns("GPU", "PID", "Process", "VRAM (MB)")
        self.set_interval(1.0, self._refresh)

    async def _refresh(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        if app.poller is None:
            return
        snapshot = app.poller.latest
        if snapshot is None:
            return

        container = self.query_one("#gpu-cards-container", Widget)

        for gpu in snapshot.gpus:
            card_id = f"gpu-card-{gpu.index}"
            try:
                card = self.query_one(f"#{card_id}", _GPUCard)
            except Exception:
                card = _GPUCard(gpu.index)
                container.mount(card)
            card.update_gpu(gpu)

        # Refresh process table
        table = self.query_one("#gpu-procs-table", DataTable)
        table.clear()
        for proc in sorted(snapshot.gpu_processes, key=lambda p: (p.gpu_index, p.pid)):
            table.add_row(
                str(proc.gpu_index),
                str(proc.pid),
                proc.name,
                f"{proc.vram_mb:.0f}",
            )
        if not snapshot.gpu_processes:
            table.add_row("—", "—", "No GPU processes detected", "—")


class _GPUCard(Widget):
    """Single-GPU stat card — util bar, VRAM bar, temp, power, fan."""

    DEFAULT_CSS = """
    _GPUCard {
        border: ascii $primary-darken-2;
        background: $surface-darken-1;
        padding: 1;
        height: auto;
        min-width: 30;
    }
    _GPUCard Label { height: 1; }
    """

    def __init__(self, index: int) -> None:
        super().__init__(id=f"gpu-card-{index}")
        self._index = index

    def compose(self) -> ComposeResult:
        yield Label("", id=f"gc-title-{self._index}")
        yield Label("Utilization", id=f"gc-util-lbl-{self._index}")
        yield ProgressBar(total=100, id=f"gc-util-bar-{self._index}", show_eta=False)
        yield Label("VRAM", id=f"gc-vram-lbl-{self._index}")
        yield ProgressBar(total=100, id=f"gc-vram-bar-{self._index}", show_eta=False)
        yield Label("", id=f"gc-extra-{self._index}")

    def update_gpu(self, gpu) -> None:  # type: ignore[no-untyped-def]
        i = self._index

        self.query_one(f"#gc-title-{i}", Label).update(
            f"[bold]GPU {i} — {gpu.name}[/bold]"
        )

        util = gpu.utilization_pct
        self.query_one(f"#gc-util-lbl-{i}", Label).update(
            f"Utilization   {util:.1f}%"
        )
        self.query_one(f"#gc-util-bar-{i}", ProgressBar).update(progress=util)

        vram_pct = gpu.vram.used_pct
        self.query_one(f"#gc-vram-lbl-{i}", Label).update(
            f"VRAM          {gpu.vram.used_mb:.0f} / {gpu.vram.total_mb:.0f} MB  ({vram_pct:.1f}%)"
        )
        self.query_one(f"#gc-vram-bar-{i}", ProgressBar).update(progress=vram_pct)

        parts: list[str] = []
        if gpu.temperature_c is not None:
            parts.append(f"Temp: {gpu.temperature_c:.0f}°C")
        if gpu.power_watts is not None:
            pw = f"{gpu.power_watts:.0f}W"
            if gpu.power_limit_watts:
                pw += f" / {gpu.power_limit_watts:.0f}W"
            parts.append(f"Power: {pw}")
        if gpu.fan_speed_pct is not None:
            parts.append(f"Fan: {gpu.fan_speed_pct:.0f}%")
        if gpu.driver_version:
            parts.append(f"Driver: {gpu.driver_version}")
        self.query_one(f"#gc-extra-{i}", Label).update("   ".join(parts))
