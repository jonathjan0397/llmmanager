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

    DEFAULT_CSS = """
    SetupWizardScreen {
        background: $surface;
    }

    #wizard-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        padding: 1 2;
        width: 100%;
    }

    #wizard-subtitle {
        text-align: center;
        color: $text-muted;
        padding: 0 2 1 2;
        width: 100%;
    }

    #wizard-layout {
        width: 100%;
        height: 1fr;
    }

    #wizard-left {
        width: 36;
        min-width: 34;
        padding: 0 1;
        border-right: solid $primary-darken-2;
    }

    #wizard-right {
        width: 1fr;
        padding: 0 1;
    }

    /* ---- Section headings ---- */
    .wizard-section-label {
        text-style: bold;
        color: $primary;
        padding: 1 0 0 0;
        border-bottom: solid $primary-darken-2;
        width: 100%;
    }

    /* ---- Hardware summary ---- */
    #hw-summary {
        color: $text;
        padding: 1 1;
        margin-bottom: 1;
        background: $surface-darken-1;
        border: solid $primary-darken-3;
        width: 100%;
    }

    /* ---- Server checkboxes ---- */
    .server-checkbox {
        margin: 0 0 0 1;
    }

    /* ---- Password section ---- */
    #sudo-password-label {
        color: $text-muted;
        margin-top: 1;
    }

    #sudo-password-input {
        width: 100%;
        margin-bottom: 1;
    }

    /* ---- Action buttons ---- */
    #btn-preflight {
        width: 100%;
        margin-top: 0;
        margin-bottom: 1;
    }

    #btn-install-all {
        width: 100%;
        margin-bottom: 1;
    }

    #btn-skip {
        width: 100%;
        color: $text-muted;
    }

    /* ---- Right panel ---- */
    #preflight-results {
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary-darken-3;
        width: 100%;
        min-height: 5;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("LLM Manager — Setup Wizard", id="wizard-title")
        yield Label(
            "Detect hardware, run pre-flight checks, then install selected servers.",
            id="wizard-subtitle",
        )

        with Horizontal(id="wizard-layout"):
            # Left: hardware + server selection + actions
            with Vertical(id="wizard-left"):
                yield Label("Hardware", classes="wizard-section-label")
                yield Static("Detecting...", id="hw-summary")

                yield Label("Servers to Install", classes="wizard-section-label")
                yield Checkbox("Ollama",    value=True,  id="install-ollama",   classes="server-checkbox")
                yield Checkbox("vLLM",      value=False, id="install-vllm",     classes="server-checkbox")
                yield Checkbox("LM Studio", value=False, id="install-lmstudio", classes="server-checkbox")

                yield Label("Sudo password", id="sudo-password-label")
                yield Input(
                    placeholder="required for system install",
                    password=True,
                    id="sudo-password-input",
                )

                yield Button("Run Pre-flight Checks", id="btn-preflight",   variant="default")
                yield Button("Install Selected",       id="btn-install-all", variant="primary")
                yield Button("Skip Wizard",            id="btn-skip",        variant="default")

            # Right: pre-flight results + install log
            with VerticalScroll(id="wizard-right"):
                yield Label("Pre-flight Checks", classes="wizard-section-label")
                yield Static("Run pre-flight checks to see results here.", id="preflight-results")
                yield Label("Install Log", classes="wizard-section-label")
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
