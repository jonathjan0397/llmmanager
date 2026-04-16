"""GPUMeter widget — animated VRAM and utilization bars with fan controls."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Input, Label, ProgressBar

from llmmanager.models.gpu import GPUInfo


def _is_permission_error(msg: str) -> bool:
    m = msg.lower()
    return any(w in m for w in ("permission denied", "requires root", "insufficient permissions"))


class GPUMeter(Widget):
    """Displays stats for a single GPU: name, VRAM bar, utilization, temp, power, fan control."""

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
    GPUMeter .fan-control-row {
        height: auto;
        margin-top: 1;
    }
    GPUMeter .fan-control-row Label {
        width: auto;
        margin: 1 1 0 0;
    }
    GPUMeter .fan-speed-input {
        width: 8;
        margin-right: 1;
    }
    """

    gpu_info: reactive[GPUInfo | None] = reactive(None)

    def __init__(self, gpu_index: int, **kwargs) -> None:
        kwargs.setdefault("id", f"gpu-meter-{gpu_index}")
        super().__init__(**kwargs)
        self._gpu_index = gpu_index

    def compose(self) -> ComposeResult:
        i = self._gpu_index
        yield Label("", id=f"gpu-{i}-name", classes="gpu-name")
        yield Label("", id=f"gpu-{i}-vram-label")
        yield ProgressBar(total=100, id=f"gpu-{i}-vram-bar", show_eta=False)
        yield Label("", id=f"gpu-{i}-util-label")
        yield ProgressBar(total=100, id=f"gpu-{i}-util-bar", show_eta=False)
        yield Label("", id=f"gpu-{i}-extra")
        # Fan controls — hidden until a GPU with fan data is mounted
        with Horizontal(id=f"gpu-{i}-fan-row", classes="fan-control-row"):
            yield Label("Fan:", id=f"gpu-{i}-fan-label")
            yield Input(
                placeholder="0-100",
                id=f"gpu-{i}-fan-input",
                restrict=r"[0-9]*",
                max_length=3,
                classes="fan-speed-input",
            )
            yield Label("%", id=f"gpu-{i}-fan-pct-label")
            yield Button("Set",  id=f"gpu-{i}-btn-fan-set",  variant="primary")
            yield Button("Auto", id=f"gpu-{i}-btn-fan-auto", variant="default")
    def on_mount(self) -> None:
        self.query_one(f"#gpu-{self._gpu_index}-fan-row").display = False

    def watch_gpu_info(self, info: GPUInfo | None) -> None:
        if info is None:
            return
        i = self._gpu_index

        self.query_one(f"#gpu-{i}-name", Label).update(
            f"GPU {info.index}: {info.name}  [{info.vendor.value.upper()}]"
        )

        vram = info.vram
        vram_pct = vram.used_pct
        self.query_one(f"#gpu-{i}-vram-label", Label).update(
            f"VRAM   {vram.used_mb:.0f} / {vram.total_mb:.0f} MB  ({vram_pct:.1f}%)"
        )
        vram_bar = self.query_one(f"#gpu-{i}-vram-bar", ProgressBar)
        vram_bar.advance(vram_pct - (vram_bar.progress or 0))

        self.query_one(f"#gpu-{i}-util-label", Label).update(
            f"Util   {info.utilization_pct:.1f}%"
        )
        util_bar = self.query_one(f"#gpu-{i}-util-bar", ProgressBar)
        util_bar.advance(info.utilization_pct - (util_bar.progress or 0))

        extras: list[str] = []
        if info.temperature_c is not None:
            extras.append(f"Temp: {info.temperature_c:.0f}C")
        if info.power_watts is not None:
            pw = f"{info.power_watts:.0f}W"
            if info.power_limit_watts:
                pw += f"/{info.power_limit_watts:.0f}W"
            extras.append(pw)
        if info.cuda_version:
            extras.append(f"CUDA {info.cuda_version}")

        self.query_one(f"#gpu-{i}-extra", Label).update("  ".join(extras))

        # Show fan control row only when fan data is available
        fan_row = self.query_one(f"#gpu-{i}-fan-row")
        if info.fan_speed_pct is not None:
            fan_row.display = True
            self.query_one(f"#gpu-{i}-fan-label", Label).update(
                f"Fan: {info.fan_speed_pct:.0f}%  Target:"
            )
        else:
            fan_row.display = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        i = self._gpu_index
        if event.button.id == f"gpu-{i}-btn-fan-set":
            self.run_worker(self._set_fan())
        elif event.button.id == f"gpu-{i}-btn-fan-auto":
            self.run_worker(self._set_fan_auto())

    async def _set_fan(self) -> None:
        i = self._gpu_index
        raw = self.query_one(f"#gpu-{i}-fan-input", Input).value.strip()
        if not raw.isdigit():
            self.notify("Enter a speed between 0 and 100.", severity="warning")
            return
        speed = max(0, min(100, int(raw)))
        ok, msg = await self.app.gpu_provider.set_fan_speed(i, speed)  # type: ignore[attr-defined]
        if not ok and _is_permission_error(msg):
            ok, msg = await self._retry_with_sudo("set_fan_speed", i, speed)
        if ok:
            self.notify(
                f"{msg}  —  Remember to click Auto when done to restore automatic fan control.",
                severity="information",
            )
        else:
            self.notify(msg, severity="error")

    async def _set_fan_auto(self) -> None:
        i = self._gpu_index
        ok, msg = await self.app.gpu_provider.set_fan_auto(i)  # type: ignore[attr-defined]
        if not ok and _is_permission_error(msg):
            ok, msg = await self._retry_with_sudo("set_fan_auto", i)
        self.notify(msg, severity="information" if ok else "error")

    async def _retry_with_sudo(self, operation: str, gpu_index: int, speed: int | None = None) -> tuple[bool, str]:
        from llmmanager.widgets.sudo_dialog import SudoDialog
        sudo_pw = await self.app.push_screen_wait(
            SudoDialog("Fan control requires root. Enter sudo password.")
        )
        if sudo_pw is None:
            return False, "Cancelled."
        provider = self.app.gpu_provider  # type: ignore[attr-defined]
        if operation == "set_fan_speed" and speed is not None:
            return await provider.set_fan_speed_sudo(gpu_index, speed, sudo_pw)
        return await provider.set_fan_auto_sudo(gpu_index, sudo_pw)

    def update_gpu(self, info: GPUInfo) -> None:
        self.gpu_info = info
