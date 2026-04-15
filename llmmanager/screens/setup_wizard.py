"""Setup Wizard screen — first-run guided installation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Input, Label, Static

from llmmanager.widgets.log_view import LogView

if TYPE_CHECKING:
    from llmmanager.app import LLMManagerApp


class SetupWizardScreen(Screen):
    """First-run wizard: hardware detection, pre-flight checks, one-click install."""

    BINDINGS = [("escape", "skip_wizard", "Skip")]

    def compose(self) -> ComposeResult:
        yield Label("LLM Manager — Setup Wizard", id="wizard-title", classes="section-heading")
        yield Label("Detected hardware will appear below. Select servers to install.", id="wizard-subtitle")

        with Horizontal(id="wizard-layout"):
            # Left: hardware + server selection
            with Vertical(id="wizard-left"):
                yield Label("Hardware", classes="section-heading")
                yield Static("", id="hw-summary")

                yield Label("Servers to Install", classes="section-heading")
                yield Checkbox("Ollama",    value=True,  id="install-ollama")
                yield Checkbox("vLLM",      value=False, id="install-vllm")
                yield Checkbox("LM Studio", value=False, id="install-lmstudio")

                yield Label("Sudo password:", classes="form-label")
                yield Input(
                    placeholder="required for system install",
                    password=True,
                    id="sudo-password-input",
                )
                yield Button("Run Pre-flight Checks", id="btn-preflight", variant="default")
                yield Button("Install Selected",      id="btn-install-all", variant="primary", disabled=True)
                yield Button("Skip Wizard",           id="btn-skip", variant="default")

            # Right: pre-flight results + install log
            with VerticalScroll(id="wizard-right"):
                yield Label("Pre-flight Checks", classes="section-heading")
                yield Static("", id="preflight-results")
                yield Label("Install Log", classes="section-heading")
                yield LogView(max_lines=300, id="wizard-log")

    def on_mount(self) -> None:
        self.run_worker(self._detect_hardware())

    async def _detect_hardware(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        hw = self.query_one("#hw-summary", Static)
        try:
            gpus = await app.gpu_provider.get_all_gpus()
            lines: list[str] = []
            for g in gpus:
                lines.append(
                    f"GPU {g.index}: {g.name}  "
                    f"VRAM {g.vram.total_mb / 1024:.1f} GB  "
                    f"({g.vendor.value.upper()})"
                )
            import psutil
            mem = psutil.virtual_memory()
            lines.append(f"CPU: {psutil.cpu_count()} cores")
            lines.append(f"RAM: {mem.total / 1024**3:.1f} GB")
            hw.update("\n".join(lines))
        except Exception as exc:
            hw.update(f"Hardware detection failed: {exc}")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "btn-preflight":
                await self._run_preflight()
            case "btn-install-all":
                self.run_worker(self._install_selected())
            case "btn-skip":
                self.action_skip_wizard()

    async def _run_preflight(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        results_widget = self.query_one("#preflight-results", Static)
        lines: list[str] = []
        all_ok = True

        servers_to_check = []
        if self.query_one("#install-ollama", Checkbox).value:
            s = app.registry.get("ollama")
            if s:
                servers_to_check.append(s)
        if self.query_one("#install-vllm", Checkbox).value:
            s = app.registry.get("vllm")
            if s:
                servers_to_check.append(s)

        for server in servers_to_check:
            lines.append(f"\n{server.display_name}:")
            checks = await server.preflight_checks()
            for check_name, passed, detail in checks:
                icon = "[green]✓[/]" if passed else "[red]✗[/]"
                line = f"  {icon} {check_name}"
                if detail:
                    line += f"  ({detail})"
                lines.append(line)
                if not passed:
                    all_ok = False

        results_widget.update("\n".join(lines))
        self.query_one("#btn-install-all", Button).disabled = False

    async def _install_selected(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        log = self.query_one("#wizard-log", LogView)
        log.clear_log()
        sudo_pw = self.query_one("#sudo-password-input", Input).value

        for server_type, checkbox_id in [
            ("ollama", "#install-ollama"),
            ("vllm",   "#install-vllm"),
        ]:
            if not self.query_one(checkbox_id, Checkbox).value:
                continue
            server = app.registry.get(server_type)
            if server is None:
                continue
            log.append_line(f"=== Installing {server.display_name} ===")
            try:
                async for line in server.install(sudo_password=sudo_pw):
                    log.append_line(line)
            except Exception as exc:
                log.append_line(f"ERROR: {exc}")

        log.append_line("=== Setup complete. Press Escape or click Skip to continue. ===")

    def action_skip_wizard(self) -> None:
        self.app.pop_screen()
