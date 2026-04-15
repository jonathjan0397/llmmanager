"""ConfirmDialog — modal yes/no prompt."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button, Label
from textual.containers import Horizontal, Vertical


class ConfirmDialog(ModalScreen[bool]):
    """
    Push this screen to ask the user to confirm a destructive action.
    Returns True if confirmed, False if cancelled.

    Usage:
        confirmed = await self.app.push_screen_wait(
            ConfirmDialog("Delete model?", "This cannot be undone.")
        )
    """

    DEFAULT_CSS = """
    ConfirmDialog {
        align: center middle;
    }
    ConfirmDialog > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 2 4;
        width: 50;
        height: auto;
    }
    ConfirmDialog .dialog-title {
        text-style: bold;
        margin-bottom: 1;
    }
    ConfirmDialog .dialog-body {
        color: $text-muted;
        margin-bottom: 2;
    }
    ConfirmDialog Horizontal {
        align: right middle;
        height: auto;
    }
    ConfirmDialog Button {
        margin-left: 1;
    }
    """

    def __init__(self, title: str, body: str = "") -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title, classes="dialog-title")
            if self._body:
                yield Label(self._body, classes="dialog-body")
            with Horizontal():
                yield Button("Cancel", id="btn-cancel", variant="default")
                yield Button("Confirm", id="btn-confirm", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-confirm")
