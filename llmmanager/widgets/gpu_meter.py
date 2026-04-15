"""GPUMeter widget — animated VRAM and utilization bars."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, ProgressBar
from textual import on
from rich.text import Text

from llmmanager.models.gpu import GPUInfo


class GPUMeter(Widget):
    """Displays stats for a single GPU: name, VRAM bar, utilization, temp, power."""

    DEFAULT_CSS = """
    GPUMeter {
        height: auto;
        border: round $surface;
        padding: 0 1;
        margin-bottom: 1;
    }
    GPUMeter .gpu-name {
        color: $accent;
        text-style: bold;
    }
    GPUMeter .gpu-stat-label {
        color: $text-muted;
        width: 8;
    }
    GPUMeter .gpu-warning {
        color: $warning;
    }
    GPUMeter .gpu-danger {
        color: $error;
    }
    """

    gpu_info: reactive[GPUInfo | None] = reactive(None)

    def __init__(self, gpu_index: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._gpu_index = gpu_index
        self.id = f"gpu-meter-{gpu_index}"

    def compose(self) -> ComposeResult:
        yield Label("", id=f"gpu-{self._gpu_index}-name", classes="gpu-name")
        yield Label("", id=f"gpu-{self._gpu_index}-vram-label")
        yield ProgressBar(total=100, id=f"gpu-{self._gpu_index}-vram-bar", show_eta=False)
        yield Label("", id=f"gpu-{self._gpu_index}-util-label")
        yield ProgressBar(total=100, id=f"gpu-{self._gpu_index}-util-bar", show_eta=False)
        yield Label("", id=f"gpu-{self._gpu_index}-extra")

    def watch_gpu_info(self, info: GPUInfo | None) -> None:
        if info is None:
            return
        i = self._gpu_index

        name_label = self.query_one(f"#gpu-{i}-name", Label)
        name_label.update(f"GPU {info.index}: {info.name}  [{info.vendor.value.upper()}]")

        vram = info.vram
        vram_pct = vram.used_pct
        vram_label = self.query_one(f"#gpu-{i}-vram-label", Label)
        vram_label.update(
            f"VRAM   {vram.used_mb:.0f} / {vram.total_mb:.0f} MB  ({vram_pct:.1f}%)"
        )
        vram_bar = self.query_one(f"#gpu-{i}-vram-bar", ProgressBar)
        vram_bar.advance(vram_pct - (vram_bar.progress or 0))

        util_label = self.query_one(f"#gpu-{i}-util-label", Label)
        util_label.update(f"Util   {info.utilization_pct:.1f}%")
        util_bar = self.query_one(f"#gpu-{i}-util-bar", ProgressBar)
        util_bar.advance(info.utilization_pct - (util_bar.progress or 0))

        extras: list[str] = []
        if info.temperature_c is not None:
            extras.append(f"Temp: {info.temperature_c:.0f}°C")
        if info.power_watts is not None:
            pw = f"{info.power_watts:.0f}W"
            if info.power_limit_watts:
                pw += f"/{info.power_limit_watts:.0f}W"
            extras.append(pw)
        if info.fan_speed_pct is not None:
            extras.append(f"Fan: {info.fan_speed_pct:.0f}%")
        if info.cuda_version:
            extras.append(f"CUDA {info.cuda_version}")

        extra_label = self.query_one(f"#gpu-{i}-extra", Label)
        extra_label.update("  ".join(extras))

    def update_gpu(self, info: GPUInfo) -> None:
        self.gpu_info = info
