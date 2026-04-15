"""VersionPickerDialog — choose a model tag before downloading."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Select


class VersionPickerDialog(ModalScreen[str | None]):
    """
    Modal that lists available tags for a model and returns the chosen
    tag string, or None if cancelled.

    Usage:
        tag = await self.app.push_screen_wait(
            VersionPickerDialog("llama3.2", ["1b", "3b", "latest"])
        )
        if tag:
            enqueue(f"{model_name}:{tag}")
    """

    DEFAULT_CSS = """
    VersionPickerDialog {
        align: center middle;
    }
    VersionPickerDialog > Vertical {
        background: $surface;
        border: ascii $accent;
        padding: 2 4;
        width: 52;
        height: auto;
    }
    VersionPickerDialog .dialog-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    VersionPickerDialog .dialog-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    VersionPickerDialog Select {
        width: 1fr;
        margin-bottom: 2;
    }
    VersionPickerDialog Horizontal {
        align: right middle;
        height: auto;
    }
    VersionPickerDialog Button {
        margin-left: 1;
    }
    """

    def __init__(self, model_name: str, tags: list[str]) -> None:
        super().__init__()
        self._model_name = model_name
        self._tags = tags if tags else ["latest"]

    def compose(self) -> ComposeResult:
        options = [(tag, tag) for tag in self._tags]
        with Vertical():
            yield Label(f"Download: {self._model_name}", classes="dialog-title")
            yield Label("Select a version / size:", classes="dialog-hint")
            yield Select(options=options, value=self._tags[0], id="tag-select")
            with Horizontal():
                yield Button("Cancel",   id="btn-cancel",   variant="default")
                yield Button("Download", id="btn-download", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-download":
            tag = str(self.query_one("#tag-select", Select).value)
            self.dismiss(tag)
        else:
            self.dismiss(None)
