"""Server Management screen — install, configure, start/stop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Label, ListItem, ListView, Static

from llmmanager.widgets.confirm_dialog import ConfirmDialog
from llmmanager.widgets.flag_form import FlagForm

if TYPE_CHECKING:
    from llmmanager.app import LLMManagerApp


_SERVER_TYPES = ["ollama", "vllm", "lmstudio"]


class ServerManagementScreen(Screen):
    """Screen 2 — per-server install, config, and lifecycle controls."""

    BINDINGS = [
        ("s", "start_server",   "Start"),
        ("S", "stop_server",    "Stop"),
        ("r", "restart_server", "Restart"),
        ("i", "install_server", "Install"),
        ("e", "edit_flags",     "Edit flags"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._selected: str = "ollama"

    def compose(self) -> ComposeResult:
        with Horizontal():
            # Left panel — server list
            with Vertical(id="server-list-panel"):
                yield Label("Servers", classes="section-heading")
                yield ListView(
                    *[ListItem(Label(t.capitalize()), id=f"server-item-{t}") for t in _SERVER_TYPES],
                    id="server-list",
                )
                yield Static()
                yield Button("Install",  id="btn-install",  variant="primary")
                yield Button("Start",    id="btn-start",    variant="success")
                yield Button("Stop",     id="btn-stop",     variant="error")
                yield Button("Restart",  id="btn-restart",  variant="warning")
                yield Button("Uninstall",id="btn-uninstall",variant="default")

            # Right panel — flag form
            with Vertical(id="flag-form-panel"):
                yield Label("", id="flag-form-title", classes="section-heading")
                yield Label("", id="server-status-badge")
                with VerticalScroll(id="flag-form-scroll"):
                    yield Label("Select a server to configure.", id="flag-form-placeholder")
                yield Button("Apply & Restart", id="btn-apply", variant="primary")
                yield Button("Reset Defaults",  id="btn-reset", variant="default")

    def on_mount(self) -> None:
        self._load_server("ollama")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        server_type = item_id.removeprefix("server-item-")
        if server_type in _SERVER_TYPES:
            self._selected = server_type
            self._load_server(server_type)

    def _load_server(self, server_type: str) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        server = app.registry.get(server_type)
        if server is None:
            return

        title = self.query_one("#flag-form-title", Label)
        title.update(f"{server.display_name} — Configuration")

        scroll = self.query_one("#flag-form-scroll")
        scroll.remove_children()

        flags = server.get_flag_definitions()
        if flags:
            form = FlagForm(
                flag_definitions=flags,
                current_values=server.config.flags,
                id="current-flag-form",
            )
            scroll.mount(form)
        else:
            scroll.mount(Label(
                f"{server.display_name} has no configurable flags.",
                id="flag-form-placeholder",
            ))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        server = app.registry.get(self._selected)
        if server is None:
            return

        match event.button.id:
            case "btn-start":
                self.run_worker(self._lifecycle(server.start, "Starting..."))
            case "btn-stop":
                self.run_worker(self._lifecycle(server.stop, "Stopping..."))
            case "btn-restart":
                self.run_worker(self._lifecycle(server.restart, "Restarting..."))
            case "btn-install":
                self.run_worker(self._stream_install(server, "latest"))
            case "btn-uninstall":
                confirmed = await self.app.push_screen_wait(
                    ConfirmDialog(
                        f"Uninstall {server.display_name}?",
                        "This will remove the server binary/venv.",
                    )
                )
                if confirmed:
                    self.run_worker(self._stream_install_output(server.uninstall()))
            case "btn-apply":
                self._save_flags(server)
                self.run_worker(self._lifecycle(server.restart, "Applying config and restarting..."))
            case "btn-reset":
                from llmmanager.config.defaults import SERVER_DEFAULTS
                defaults = SERVER_DEFAULTS.get(self._selected)
                if defaults:
                    server.config.flags = {}
                    self._load_server(self._selected)

    async def _lifecycle(self, fn, msg: str) -> None:
        self.notify(msg)
        try:
            await fn()
        except Exception as exc:
            self.notify(str(exc), severity="error")

    async def _stream_install(self, server, version: str) -> None:
        scroll = self.query_one("#flag-form-scroll")
        scroll.remove_children()
        log = __import__("llmmanager.widgets.log_view", fromlist=["LogView"]).LogView(id="install-log")
        scroll.mount(log)
        try:
            async for line in server.install(version):
                log.append_line(line)
        except Exception as exc:
            log.append_line(f"ERROR: {exc}")

    async def _stream_install_output(self, agen) -> None:
        scroll = self.query_one("#flag-form-scroll")
        scroll.remove_children()
        log = __import__("llmmanager.widgets.log_view", fromlist=["LogView"]).LogView(id="install-log")
        scroll.mount(log)
        try:
            async for line in agen:
                log.append_line(line)
        except Exception as exc:
            log.append_line(f"ERROR: {exc}")

    def _save_flags(self, server) -> None:
        try:
            form = self.query_one("#current-flag-form", FlagForm)
            values = form.get_values()
            server.config.flags = {k: v for k, v in values.items() if v is not None}
            app: LLMManagerApp = self.app  # type: ignore[assignment]
            app.config_manager.config.servers[self._selected] = server.config
            app.config_manager.save()
        except Exception as exc:
            self.notify(f"Failed to save config: {exc}", severity="error")
