"""SudoDialog — modal that collects a sudo password before a privileged action."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label
from textual.containers import Horizontal, Vertical


class SudoDialog(ModalScreen[str | None]):
    """
    Push this screen when a privileged operation (sudo) is required.
    Returns the entered password string, or None if the user cancelled.

    Usage:
        password = await self.app.push_screen_wait(
            SudoDialog("Uninstall Ollama requires sudo access.")
        )
        if password is None:
            return  # cancelled
    """

    DEFAULT_CSS = """
    SudoDialog {
        align: center middle;
    }
    SudoDialog > Vertical {
        background: $surface;
        border: ascii $primary;
        padding: 2 4;
        width: 56;
        height: auto;
    }
    SudoDialog .dialog-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    SudoDialog .dialog-body {
        color: $text-muted;
        margin-bottom: 1;
    }
    SudoDialog Input {
        margin-bottom: 1;
    }
    SudoDialog Horizontal {
        align: right middle;
        height: auto;
    }
    SudoDialog Button {
        margin-left: 1;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, body: str = "") -> None:
        super().__init__()
        self._body = body

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Sudo Password Required", classes="dialog-title")
            if self._body:
                yield Label(self._body, classes="dialog-body")
            yield Input(
                placeholder="Enter sudo password…",
                password=True,
                id="sudo-password-input",
            )
            with Horizontal():
                yield Button("Cancel",  id="btn-cancel",  variant="default")
                yield Button("Confirm", id="btn-confirm", variant="warning")

    def on_mount(self) -> None:
        self.query_one("#sudo-password-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-confirm":
            pw = self.query_one("#sudo-password-input", Input).value
            self.dismiss(pw)
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "sudo-password-input":
            pw = event.input.value
            self.dismiss(pw)

    def action_cancel(self) -> None:
        self.dismiss(None)
