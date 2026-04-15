"""API Panel screen — live server status, endpoints, and quick inference."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.markup import escape as rich_escape

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Select, Static

from llmmanager.models.server import ServerState

if TYPE_CHECKING:
    from llmmanager.app import LLMManagerApp


_STATE_BADGE = {
    ServerState.RUNNING:  "[bold green]● RUNNING[/]",
    ServerState.STOPPED:  "[bold red]○ STOPPED[/]",
    ServerState.STARTING: "[bold yellow]◌ STARTING[/]",
    ServerState.STOPPING: "[bold yellow]◌ STOPPING[/]",
    ServerState.ERROR:    "[bold red]✗ ERROR[/]",
    ServerState.UNKNOWN:  "[dim]? UNKNOWN[/]",
}


class APIPanelScreen(Widget):
    """Screen 7 — live server status, active endpoints, and quick inference."""

    DEFAULT_CSS = """
    APIPanelScreen { width: 1fr; height: 1fr; }

    #api-layout { width: 1fr; height: 1fr; }

    /* ---- Left: status + endpoints ---- */
    #api-left {
        width: 40;
        border-right: solid $primary-darken-2;
        padding: 0 1;
    }

    .server-status-box {
        height: auto;
        border: solid $primary-darken-3;
        background: $surface-darken-1;
        padding: 1;
        margin-bottom: 1;
    }

    .server-status-name {
        text-style: bold;
        color: $primary;
    }

    .endpoint-url {
        color: $accent;
        text-style: italic;
        margin: 0 0 0 2;
    }

    .endpoint-proto {
        color: $text-muted;
        margin: 0 0 0 2;
    }

    /* ---- Right: inference ---- */
    #api-right {
        width: 1fr;
        padding: 0 1;
    }

    #infer-controls {
        height: auto;
        margin-bottom: 1;
    }

    #infer-controls Label { margin: 1 1 0 0; }
    #infer-controls Select { width: 20; margin-right: 1; }
    #infer-model-select { width: 26; }
    #btn-refresh-models { width: 3; margin-left: 1; }

    #infer-prompt-input { width: 1fr; margin-bottom: 1; }

    #infer-actions { height: auto; margin-bottom: 1; }
    #btn-send-infer { width: 12; }
    #infer-latency { margin-left: 2; color: $text-muted; }

    #btn-refresh-api { width: 12; margin-left: 1; }

    #infer-response-scroll {
        height: 1fr;
        border: solid $primary-darken-2;
        background: $surface-darken-1;
        padding: 1;
    }

    #infer-response {
        width: 1fr;
    }
    """

    BINDINGS = [("r", "refresh_endpoints", "Refresh")]

    def compose(self) -> ComposeResult:
        with Horizontal(id="api-layout"):
            with VerticalScroll(id="api-left"):
                yield Label("Server Status & Endpoints", classes="section-heading")
                yield Static("Loading…", id="server-status-area")

            with Vertical(id="api-right"):
                yield Label("Quick Inference", classes="section-heading")
                with Horizontal(id="infer-controls"):
                    yield Label("Server:")
                    yield Select(
                        options=[("Ollama", "ollama"), ("vLLM", "vllm"), ("LM Studio", "lmstudio"), ("llama.cpp", "llamacpp")],
                        value="ollama",
                        id="infer-server-select",
                    )
                    yield Label("Model:")
                    yield Select(
                        options=[("—", "__none__")],
                        value="__none__",
                        id="infer-model-select",
                    )
                    yield Button("↻", id="btn-refresh-models", variant="default", tooltip="Refresh model list")

                yield Input(
                    placeholder="Enter your prompt here…",
                    id="infer-prompt-input",
                )
                with Horizontal(id="infer-actions"):
                    yield Button("Send", id="btn-send-infer", variant="primary")
                    yield Button("Refresh", id="btn-refresh-api", variant="default")
                    yield Label("", id="infer-latency")

                with VerticalScroll(id="infer-response-scroll"):
                    yield Static("", id="infer-response")

    def on_mount(self) -> None:
        self.run_worker(self._refresh_all())

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    async def _refresh_all(self) -> None:
        await self._load_server_status()
        await self._populate_model_select()

    async def _load_server_status(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        area = self.query_one("#server-status-area", Static)

        lines: list[str] = []
        for server in app.registry.all_enabled():
            try:
                status = await server.get_status()
                badge = _STATE_BADGE.get(status.state, "[dim]?[/]")
                lines.append(f"[bold]{server.display_name}[/]  {badge}")

                if status.state == ServerState.RUNNING:
                    if status.loaded_models:
                        models = ", ".join(status.loaded_models[:3])
                        lines.append(f"  [dim]Models:[/] {models}")
                    if status.pid:
                        lines.append(f"  [dim]PID:[/] {status.pid}")
                    if status.uptime_seconds is not None:
                        s = int(status.uptime_seconds)
                        h, m = divmod(s, 3600)
                        m, sec = divmod(m, 60)
                        lines.append(f"  [dim]Up:[/] {h:02d}:{m:02d}:{sec:02d}")

                    try:
                        endpoints = await server.get_endpoints()
                        for ep in endpoints:
                            lines.append(f"  [cyan]{ep.url}[/]")
                            lines.append(f"  [dim]{ep.protocol}[/]")
                    except Exception:
                        pass
                lines.append("")
            except Exception as exc:
                lines.append(f"[bold]{server.display_name}[/]  [red]Error: {exc}[/]")
                lines.append("")

        area.update("\n".join(lines) if lines else "No servers configured.")

    async def _populate_model_select(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        server_type = str(self.query_one("#infer-server-select", Select).value)
        server = app.registry.get(server_type)
        select = self.query_one("#infer-model-select", Select)
        if server is None:
            return
        try:
            models = await server.list_loaded_models()
        except Exception:
            models = []
        if models:
            select.set_options([(m.display_name, m.model_id) for m in models])
        else:
            select.set_options([("No models loaded", "__none__")])

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "infer-server-select":
            self.run_worker(self._populate_model_select())

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "btn-send-infer":
                await self._run_inference()
            case "btn-refresh-models":
                self.run_worker(self._populate_model_select())
            case "btn-refresh-api":
                self.run_worker(self._refresh_all())

    async def _run_inference(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        server_type = str(self.query_one("#infer-server-select", Select).value)
        model_id    = str(self.query_one("#infer-model-select",  Select).value)
        prompt      = self.query_one("#infer-prompt-input", Input).value.strip()

        if model_id == "__none__" or not model_id:
            self.notify("Select a model first.", severity="warning")
            return
        if not prompt:
            self.notify("Enter a prompt.", severity="warning")
            return

        server = app.registry.get(server_type)
        if server is None:
            self.notify(f"Server '{server_type}' not configured.", severity="error")
            return

        response = self.query_one("#infer-response", Static)
        scroll = self.query_one("#infer-response-scroll", VerticalScroll)
        response.update("")
        latency = self.query_one("#infer-latency", Label)
        latency.update("Running…")

        start = time.monotonic()
        full = ""
        try:
            async for token in server.quick_infer(model_id, prompt):
                full += token
                response.update(rich_escape(full))
                scroll.scroll_end(animate=False)
        except Exception as exc:
            response.update(f"[red]ERROR: {rich_escape(str(exc))}[/]")

        elapsed_ms = (time.monotonic() - start) * 1000
        latency.update(f"{elapsed_ms:.0f} ms")

    def action_refresh_endpoints(self) -> None:
        self.run_worker(self._refresh_all())
