"""ServerCard widget — compact status card for one server."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Label, Static

from llmmanager.models.server import ServerInfo, ServerState


_STATE_COLORS = {
    ServerState.RUNNING:  "green",
    ServerState.STOPPED:  "red",
    ServerState.STARTING: "yellow",
    ServerState.STOPPING: "yellow",
    ServerState.ERROR:    "red",
    ServerState.UNKNOWN:  "white",
}

_STATE_ICONS = {
    ServerState.RUNNING:  "●",
    ServerState.STOPPED:  "○",
    ServerState.STARTING: "◌",
    ServerState.STOPPING: "◌",
    ServerState.ERROR:    "✗",
    ServerState.UNKNOWN:  "?",
}


class ServerCard(Widget):
    """One card per server — shows state, model, port, and action buttons."""

    DEFAULT_CSS = """
    ServerCard {
        height: auto;
        border: round $surface;
        padding: 1 2;
        margin: 0 1 1 0;
        min-width: 28;
    }
    ServerCard .card-header {
        text-style: bold;
    }
    ServerCard .card-state-running  { color: green; }
    ServerCard .card-state-stopped  { color: red; }
    ServerCard .card-state-starting { color: yellow; }
    ServerCard .card-state-stopping { color: yellow; }
    ServerCard .card-state-error    { color: red; }
    ServerCard .card-state-unknown  { color: white; }
    ServerCard .card-detail         { color: $text-muted; }
    ServerCard .card-buttons        { margin-top: 1; }
    """

    server_info: reactive[ServerInfo | None] = reactive(None)

    def __init__(self, server_type: str, **kwargs) -> None:
        kwargs.setdefault("id", f"card-{server_type}")
        super().__init__(**kwargs)
        self._server_type = server_type

    def compose(self) -> ComposeResult:
        yield Label("", id=f"{self._server_type}-card-header", classes="card-header")
        yield Label("", id=f"{self._server_type}-card-state")
        yield Label("", id=f"{self._server_type}-card-model", classes="card-detail")
        yield Label("", id=f"{self._server_type}-card-port", classes="card-detail")
        yield Label("", id=f"{self._server_type}-card-uptime", classes="card-detail")
        with Widget(classes="card-buttons"):
            yield Button("Start",   id=f"{self._server_type}-btn-start",   variant="success")
            yield Button("Stop",    id=f"{self._server_type}-btn-stop",    variant="error")
            yield Button("Restart", id=f"{self._server_type}-btn-restart", variant="warning")

    def watch_server_info(self, info: ServerInfo | None) -> None:
        if info is None:
            return
        t = self._server_type
        state = info.status.state
        color = _STATE_COLORS.get(state, "white")
        icon = _STATE_ICONS.get(state, "?")

        self.query_one(f"#{t}-card-header", Label).update(info.display_name)
        self.query_one(f"#{t}-card-state", Label).update(
            f"[{color}]{icon} {state.value.capitalize()}[/]"
        )

        model = info.status.active_model or (
            ", ".join(info.status.loaded_models[:2]) if info.status.loaded_models else "none"
        )
        self.query_one(f"#{t}-card-model", Label).update(f"Model: {model}")
        self.query_one(f"#{t}-card-port", Label).update(f"Port:  {info.port}")

        uptime = ""
        if info.status.uptime_seconds is not None:
            s = int(info.status.uptime_seconds)
            h, m = divmod(s, 3600)
            m, sec = divmod(m, 60)
            uptime = f"Up:    {h:02d}:{m:02d}:{sec:02d}"
        self.query_one(f"#{t}-card-uptime", Label).update(uptime)

        # Enable/disable buttons based on state
        running = state == ServerState.RUNNING
        self.query_one(f"#{t}-btn-start",   Button).disabled = running
        self.query_one(f"#{t}-btn-stop",    Button).disabled = not running
        self.query_one(f"#{t}-btn-restart", Button).disabled = not running

    def update_info(self, info: ServerInfo) -> None:
        self.server_info = info
