"""PasteDialog — modal input for pasting text in SSH/headless sessions."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label


class PasteDialog(ModalScreen[str | None]):
    """
    A modal text input for environments where clipboard access isn't available
    (e.g. PuTTY / SSH sessions). Press Shift+Insert inside the input to paste
    from PuTTY's clipboard, then confirm.

    Returns the entered string, or None if cancelled.
    """

    DEFAULT_CSS = """
    PasteDialog {
        align: center middle;
    }
    PasteDialog > Vertical {
        background: $surface;
        border: ascii $primary;
        padding: 2 4;
        width: 64;
        height: auto;
    }
    PasteDialog .dialog-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    PasteDialog .dialog-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    PasteDialog Input {
        margin-bottom: 1;
    }
    PasteDialog Horizontal {
        align: right middle;
        height: auto;
    }
    PasteDialog Button {
        margin-left: 1;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, field_name: str = "value") -> None:
        super().__init__()
        self._field_name = field_name

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"Enter {self._field_name}", classes="dialog-title")
            yield Label(
                "Paste with Shift+Insert (PuTTY) or Ctrl+Shift+V, then press Enter or Confirm.",
                classes="dialog-hint",
            )
            yield Input(placeholder=f"Paste {self._field_name} here…", id="paste-input")
            with Horizontal():
                yield Button("Cancel",  id="btn-cancel",  variant="default")
                yield Button("Confirm", id="btn-confirm", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#paste-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-confirm":
            self.dismiss(self.query_one("#paste-input", Input).value.strip() or None)
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.input.value.strip() or None)

    def action_cancel(self) -> None:
        self.dismiss(None)
