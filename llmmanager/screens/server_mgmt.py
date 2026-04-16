"""Server Management screen — install, configure, start/stop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Input, Label, ListItem, ListView, Select, Static

from llmmanager.widgets.confirm_dialog import ConfirmDialog
from llmmanager.widgets.flag_form import FlagForm
from llmmanager.widgets.sudo_dialog import SudoDialog

if TYPE_CHECKING:
    from llmmanager.app import LLMManagerApp
    from llmmanager.servers.base import AbstractServer


_SERVER_TYPES = ["ollama", "vllm", "lmstudio", "llamacpp"]

# Servers that use a discrete installed-model list vs. a free-text model path
_MODEL_LIST_SERVERS = {"ollama", "lmstudio"}


class ServerManagementScreen(Widget):
    """Screen 2 — per-server install, config, and lifecycle controls."""

    DEFAULT_CSS = """
    ServerManagementScreen { width: 1fr; height: 1fr; }

    #model-picker-row {
        height: auto;
        margin-bottom: 1;
    }
    #model-picker-row Label { margin: 1 1 0 0; width: auto; }
    #model-picker-select { width: 1fr; }
    #model-picker-input  { width: 1fr; }
    #btn-refresh-model-list { width: 3; margin-left: 1; }

    .model-picker-hint {
        color: $text-muted;
        margin: 0 0 1 0;
    }
    """

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
                yield Button("Install",   id="btn-install",   variant="primary")
                yield Button("Start",     id="btn-start",     variant="success")
                yield Button("Stop",      id="btn-stop",      variant="error")
                yield Button("Restart",   id="btn-restart",   variant="warning")
                yield Button("Uninstall", id="btn-uninstall", variant="default")

            # Right panel — model picker + flag form
            with Vertical(id="flag-form-panel"):
                yield Label("", id="flag-form-title", classes="section-heading")
                yield Label("", id="server-status-badge")

                # Model picker row — shown above the flag form for every server
                with Horizontal(id="model-picker-row"):
                    yield Label("Model:")
                    # Populated dynamically — see _load_server
                    yield Select(
                        options=[("—", "__none__")],
                        value="__none__",
                        id="model-picker-select",
                    )
                    yield Input(
                        placeholder="e.g. meta-llama/Llama-3.1-8B-Instruct",
                        id="model-picker-input",
                    )
                    yield Button("↻", id="btn-refresh-model-list", variant="default",
                                 tooltip="Refresh installed model list")

                yield Label(
                    "Select the model to load when this server starts.",
                    classes="model-picker-hint",
                    id="model-picker-hint",
                )

                with VerticalScroll(id="flag-form-scroll"):
                    yield Label("Select a server to configure.", id="flag-form-placeholder")

                with Horizontal():
                    yield Button("Apply & Restart", id="btn-apply", variant="primary")
                    yield Button("Reset Defaults",  id="btn-reset", variant="default")

    def on_mount(self) -> None:
        self.run_worker(self._load_server("ollama"))

    # ------------------------------------------------------------------
    # Server selection
    # ------------------------------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        server_type = item_id.removeprefix("server-item-")
        if server_type in _SERVER_TYPES:
            self._selected = server_type
            self.run_worker(self._load_server(server_type))

    async def _load_server(self, server_type: str) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        server = app.registry.get(server_type)
        if server is None:
            return

        title = self.query_one("#flag-form-title", Label)
        title.update(f"{server.display_name} — Configuration")

        # Show/hide the right picker widget based on server type
        select_widget = self.query_one("#model-picker-select", Select)
        input_widget  = self.query_one("#model-picker-input",  Input)
        hint          = self.query_one("#model-picker-hint",   Label)

        if server_type in _MODEL_LIST_SERVERS:
            # Dropdown: will be populated async; show saved default immediately
            select_widget.display = True
            input_widget.display  = False
            saved = server.config.default_model or ""
            opts = [(saved, saved)] if saved else [("—", "__none__")]
            select_widget.set_options(opts)
            if saved:
                select_widget.value = saved
            hint.update(
                f"Select from installed {server.display_name} models. "
                f"The server will auto-load this model on startup."
            )
            # Kick off async model list population
            self.run_worker(self._populate_model_dropdown(server))
        else:
            # Free-text: maps directly to the server's --model flag
            select_widget.display = False
            input_widget.display  = True
            current = (
                server.config.default_model
                or server.config.flags.get("model", "")
                or ""
            )
            input_widget.value = current
            hint.update(
                f"Enter a HuggingFace model ID or local path for {server.display_name}. "
                f"Saved here and applied as --model at startup."
            )

        # Rebuild the flag form — await removal so no duplicate-ID crash
        scroll = self.query_one("#flag-form-scroll")
        await scroll.remove_children()
        flags = server.get_flag_definitions()
        if flags:
            form = FlagForm(
                flag_definitions=flags,
                current_values=server.config.flags,
                id="current-flag-form",
            )
            await scroll.mount(form)
        else:
            await scroll.mount(Label(
                f"{server.display_name} has no configurable flags.",
                id="flag-form-placeholder",
            ))

        # Adjust lifecycle buttons for GUI-only servers (e.g. LM Studio)
        is_gui_only = server_type == "lmstudio"
        self.query_one("#btn-apply",   Button).label = "Save & Poll" if is_gui_only else "Apply & Restart"
        self.query_one("#btn-start",   Button).disabled = is_gui_only
        self.query_one("#btn-stop",    Button).disabled = is_gui_only
        self.query_one("#btn-restart", Button).disabled = is_gui_only
        self.query_one("#btn-install", Button).disabled = is_gui_only
        self.query_one("#btn-uninstall", Button).disabled = is_gui_only

    async def _populate_model_dropdown(self, server: "AbstractServer") -> None:
        """Async: fetch installed models and fill the Select dropdown."""
        try:
            models = await server.list_loaded_models()
        except Exception:
            models = []

        select = self.query_one("#model-picker-select", Select)
        saved  = server.config.default_model or ""

        if models:
            options = [(m.display_name, m.model_id) for m in models]
            select.set_options(options)
            # Restore saved selection if it's still in the list
            if saved and any(m.model_id == saved for m in models):
                select.value = saved
            elif models:
                select.value = models[0].model_id
        else:
            if saved:
                select.set_options([(saved, saved)])
                select.value = saved
            else:
                select.set_options([("No models found — download first", "__none__")])

    # ------------------------------------------------------------------
    # Button handler
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        server = app.registry.get(self._selected)
        if server is None:
            return

        match event.button.id:
            case "btn-start":
                self._save_model_selection(server)
                self.run_worker(self._start_with_model(server))
            case "btn-stop":
                self.run_worker(self._lifecycle(server.stop, "Stopping…"))
            case "btn-restart":
                self._save_model_selection(server)
                self.run_worker(self._start_with_model(server, restart=True))
            case "btn-install":
                self.run_worker(self._do_install(server))
            case "btn-uninstall":
                self.run_worker(self._do_uninstall(server))
            case "btn-apply":
                self._save_flags(server)
                self._save_model_selection(server)
                if server.name == "lmstudio":
                    self.run_worker(self._poll_lmstudio(server))
                else:
                    self.run_worker(self._start_with_model(server, restart=True))
            case "btn-reset":
                from llmmanager.config.defaults import SERVER_DEFAULTS
                defaults = SERVER_DEFAULTS.get(self._selected)
                if defaults:
                    server.config.flags = {}
                    self.run_worker(self._load_server(self._selected))
            case "btn-refresh-model-list":
                self.run_worker(self._populate_model_dropdown(server))

    # ------------------------------------------------------------------
    # Model selection persistence
    # ------------------------------------------------------------------

    def _save_model_selection(self, server: "AbstractServer") -> None:
        """Read the picker widget and persist to config."""
        if self._selected in _MODEL_LIST_SERVERS:
            val = str(self.query_one("#model-picker-select", Select).value)
            model = "" if val == "__none__" else val
        else:
            model = self.query_one("#model-picker-input", Input).value.strip()
            # Also push into the --model flag so the flag form reflects it
            if model:
                server.config.flags["model"] = model

        server.config.default_model = model or None

        app: LLMManagerApp = self.app  # type: ignore[assignment]
        app.config_manager.config.servers[self._selected] = server.config
        app.config_manager.save()

    # ------------------------------------------------------------------
    # Server lifecycle with model pre-warm
    # ------------------------------------------------------------------

    async def _start_with_model(
        self, server: "AbstractServer", restart: bool = False
    ) -> None:
        """Start (or restart) the server, then pre-load the default model."""
        if restart:
            await self._lifecycle(server.restart, "Restarting…")
        else:
            await self._lifecycle(server.start, "Starting…")

        model = server.config.default_model
        if not model:
            return

        # For servers that support pre-warming, trigger a lightweight generate
        if self._selected in ("ollama", "llamacpp"):
            self.notify(f"Pre-loading {model}…")
            try:
                async for _ in server.quick_infer(model, "Hi", num_predict=1):
                    break  # First token confirms load succeeded
                self.notify(f"{model} loaded and ready.")
            except Exception as exc:
                msg = str(exc)
                if "unable to load model" in msg:
                    self.notify(
                        f"{model} failed to load — it may be corrupted. "
                        "Re-download it from Model Management.",
                        severity="error",
                    )
                else:
                    self.notify(f"Model pre-load: {msg}", severity="warning")

    # ------------------------------------------------------------------
    # Lifecycle / install helpers
    # ------------------------------------------------------------------

    async def _poll_lmstudio(self, server: "AbstractServer") -> None:
        """Save settings and verify LM Studio is reachable at the configured host:port."""
        host = server.config.host
        port = server.config.port
        self.notify(f"Settings saved. Checking LM Studio at {host}:{port}…")
        try:
            status = await server.get_status()
            from llmmanager.models.server import ServerState
            if status.state == ServerState.RUNNING:
                loaded = ", ".join(status.loaded_models) if status.loaded_models else "none loaded"
                self.notify(
                    f"LM Studio is running at {host}:{port}  •  models: {loaded}",
                    severity="information",
                )
                # Trigger a dashboard refresh so the server card updates immediately
                app: LLMManagerApp = self.app  # type: ignore[assignment]
                if app.poller:
                    self.run_worker(app.poller.force_poll())
            else:
                self.notify(
                    f"LM Studio not detected at {host}:{port}. "
                    "Open LM Studio and enable Local Server, then click Save & Poll again.",
                    severity="warning",
                )
        except Exception as exc:
            self.notify(f"Poll failed: {exc}", severity="error")

    async def _lifecycle(self, fn, msg: str) -> None:
        self.notify(msg)
        try:
            await fn()
        except Exception as exc:
            self.notify(str(exc), severity="error")

    async def _do_install(self, server: "AbstractServer") -> None:
        sudo_pw = await self._maybe_sudo(server)
        if sudo_pw is False:
            return
        await self._stream_install(server, "latest", sudo_password=sudo_pw or "")

    async def _do_uninstall(self, server: "AbstractServer") -> None:
        confirmed = await self.app.push_screen_wait(
            ConfirmDialog(
                f"Uninstall {server.display_name}?",
                "This will remove the server binary/venv.",
            )
        )
        if not confirmed:
            return
        sudo_pw = await self._maybe_sudo(server)
        if sudo_pw is False:
            return
        scroll = self.query_one("#flag-form-scroll")
        await scroll.remove_children()
        LogView = __import__("llmmanager.widgets.log_view", fromlist=["LogView"]).LogView
        log = LogView(id="install-log")
        await scroll.mount(log)
        try:
            async for line in server.uninstall(sudo_password=sudo_pw or ""):
                log.append_line(line)
        except Exception as exc:
            log.append_line(f"ERROR: {exc}")

    async def _maybe_sudo(self, server: "AbstractServer") -> str | bool:
        """
        For servers that need sudo (Ollama), show a password dialog.
        Returns the password string (may be empty if user left it blank),
        or False if the user cancelled.
        For servers that don't use sudo, returns "" immediately.
        """
        if server.name != "ollama":
            return ""
        result = await self.app.push_screen_wait(
            SudoDialog(f"{server.display_name} install/uninstall requires sudo.")
        )
        if result is None:
            return False  # cancelled
        return result

    async def _stream_install(self, server: "AbstractServer", version: str, sudo_password: str = "") -> None:
        scroll = self.query_one("#flag-form-scroll")
        scroll.remove_children()
        log = __import__("llmmanager.widgets.log_view", fromlist=["LogView"]).LogView(id="install-log")
        scroll.mount(log)
        try:
            async for line in server.install(version, sudo_password=sudo_password):
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

    def _save_flags(self, server: "AbstractServer") -> None:
        try:
            form = self.query_one("#current-flag-form", FlagForm)
            values = form.get_values()
            server.config.flags = {k: v for k, v in values.items() if v is not None}
            app: LLMManagerApp = self.app  # type: ignore[assignment]
            app.config_manager.config.servers[self._selected] = server.config
            app.config_manager.save()
        except Exception as exc:
            self.notify(f"Failed to save config: {exc}", severity="error")

    # ------------------------------------------------------------------
    # Keyboard action bindings
    # ------------------------------------------------------------------

    def action_start_server(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        server = app.registry.get(self._selected)
        if server:
            self._save_model_selection(server)
            self.run_worker(self._start_with_model(server))

    def action_stop_server(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        server = app.registry.get(self._selected)
        if server:
            self.run_worker(self._lifecycle(server.stop, "Stopping…"))

    def action_restart_server(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        server = app.registry.get(self._selected)
        if server:
            self._save_model_selection(server)
            self.run_worker(self._start_with_model(server, restart=True))

    def action_install_server(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        server = app.registry.get(self._selected)
        if server:
            self.run_worker(self._stream_install(server, "latest"))
