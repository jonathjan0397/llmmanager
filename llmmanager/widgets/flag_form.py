"""FlagForm — dynamic configuration form built from FlagDefinition metadata."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Checkbox, Input, Label, Select

from llmmanager.config.schema import FlagDefinition


class FlagForm(Widget):
    """
    Renders a scrollable form from a list of FlagDefinitions.
    Groups flags by category into collapsible sections.
    """

    DEFAULT_CSS = """
    FlagForm {
        height: 1fr;
    }
    FlagForm .form-category {
        color: $accent;
        text-style: bold;
        margin-top: 1;
        border-bottom: solid $surface;
    }
    FlagForm .form-field-label {
        color: $text;
        margin-top: 1;
    }
    FlagForm .form-field-desc {
        color: $text-muted;
        margin-bottom: 0;
    }
    FlagForm .restart-banner {
        background: $warning 20%;
        color: $warning;
        padding: 0 1;
        display: none;
    }
    FlagForm .restart-banner.visible {
        display: block;
    }
    """

    def __init__(
        self,
        flag_definitions: list[FlagDefinition],
        current_values: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._flags = flag_definitions
        self._values: dict[str, Any] = current_values or {}
        self._dirty_requires_restart = False

    def compose(self) -> ComposeResult:
        yield Label("", id="restart-banner", classes="restart-banner")
        with VerticalScroll():
            # Group by category
            categories: dict[str, list[FlagDefinition]] = {}
            for f in self._flags:
                categories.setdefault(f.category, []).append(f)

            for category, flags in categories.items():
                yield Label(category, classes="form-category")
                for flag in flags:
                    key = flag.name.lstrip("-")
                    current = self._values.get(key, flag.default)
                    yield Label(flag.name, classes="form-field-label")
                    if flag.description:
                        yield Label(flag.description, classes="form-field-desc")

                    if flag.type == "bool":
                        yield Checkbox(
                            label="",
                            value=bool(current),
                            id=f"flag-{key}",
                        )
                    elif flag.type == "choice" and flag.choices:
                        options = [(c, c) for c in flag.choices]
                        yield Select(
                            options=options,
                            value=str(current) if current is not None else flag.choices[0],
                            id=f"flag-{key}",
                        )
                    else:
                        yield Input(
                            value=str(current) if current is not None else "",
                            placeholder=str(flag.default) if flag.default is not None else "",
                            id=f"flag-{key}",
                        )

    def get_values(self) -> dict[str, Any]:
        """Collect current form values. Call before saving config."""
        result: dict[str, Any] = {}
        for flag in self._flags:
            key = flag.name.lstrip("-")
            widget_id = f"flag-{key}"
            try:
                if flag.type == "bool":
                    w = self.query_one(f"#{widget_id}", Checkbox)
                    result[key] = w.value
                elif flag.type == "choice":
                    w = self.query_one(f"#{widget_id}", Select)
                    result[key] = w.value
                elif flag.type == "int":
                    w = self.query_one(f"#{widget_id}", Input)
                    try:
                        result[key] = int(w.value) if w.value else flag.default
                    except ValueError:
                        result[key] = flag.default
                elif flag.type == "float":
                    w = self.query_one(f"#{widget_id}", Input)
                    try:
                        result[key] = float(w.value) if w.value else flag.default
                    except ValueError:
                        result[key] = flag.default
                else:
                    w = self.query_one(f"#{widget_id}", Input)
                    result[key] = w.value if w.value else flag.default
            except Exception:
                result[key] = self._values.get(key, flag.default)
        return result

    def on_input_changed(self, event: Input.Changed) -> None:
        self._check_restart_banner(event.input.id or "")

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        self._check_restart_banner(event.checkbox.id or "")

    def on_select_changed(self, event: Select.Changed) -> None:
        self._check_restart_banner(event.select.id or "")

    def _check_restart_banner(self, widget_id: str) -> None:
        key = widget_id.removeprefix("flag-")
        for flag in self._flags:
            if flag.name.lstrip("-") == key and flag.requires_restart:
                banner = self.query_one("#restart-banner", Label)
                banner.update("Changes require a server restart to take effect.")
                banner.add_class("visible")
                return
