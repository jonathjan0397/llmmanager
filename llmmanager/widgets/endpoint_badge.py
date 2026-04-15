"""EndpointBadge — URL display with copy-to-clipboard button."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Button, Label

from llmmanager.models.server import EndpointInfo


class EndpointBadge(Widget):
    DEFAULT_CSS = """
    EndpointBadge {
        height: auto;
        margin-bottom: 1;
    }
    EndpointBadge Horizontal {
        height: auto;
        align: left middle;
    }
    EndpointBadge .endpoint-url {
        color: $accent;
        width: 1fr;
    }
    EndpointBadge .endpoint-proto {
        color: $text-muted;
        width: 16;
    }
    """

    def __init__(self, endpoint: EndpointInfo, **kwargs) -> None:
        super().__init__(**kwargs)
        self._endpoint = endpoint

    def compose(self) -> ComposeResult:
        proto_color = "green" if "openai" in self._endpoint.protocol else "cyan"
        with Horizontal():
            yield Label(f"[{proto_color}]●[/]", classes="endpoint-proto")
            yield Label(self._endpoint.url, classes="endpoint-url")
            yield Button("Copy", id=f"copy-{id(self)}", variant="default")
            yield Label(self._endpoint.description, classes="endpoint-proto")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        try:
            import pyperclip
            pyperclip.copy(self._endpoint.url)
            self.notify(f"Copied: {self._endpoint.url}")
        except Exception:
            self.notify(self._endpoint.url, title="URL (copy manually)")
