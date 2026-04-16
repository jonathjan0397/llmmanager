"""Logs screen — per-server scrollable log tail."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Select

from llmmanager.services.log_tailer import LogLine
from llmmanager.widgets.log_view import LogView

if TYPE_CHECKING:
    from llmmanager.app import LLMManagerApp


class LogsScreen(Widget):
    """Screen 4 — live log viewer with per-server selection and filtering."""

    DEFAULT_CSS = "LogsScreen { width: 1fr; height: 1fr; }"

    BINDINGS = [
        ("ctrl+l", "clear_logs",  "Clear"),
        ("/",      "focus_filter","Filter"),
    ]

    def compose(self) -> ComposeResult:
        with Horizontal(id="logs-toolbar"):
            yield Label("Server:", classes="toolbar-label")
            yield Select(
                options=[
                    ("Ollama",    "ollama"),
                    ("vLLM",      "vllm"),
                    ("LM Studio", "lmstudio"),
                    ("All",       "all"),
                ],
                value="all",
                id="server-select",
            )
            yield Label("Filter:", classes="toolbar-label")
            yield Input(placeholder="keyword filter...", id="log-filter")
            yield Button("Clear", id="btn-clear-logs", variant="default")
            yield Label("", id="log-stats")

        yield LogView(id="main-log-view")

    def on_mount(self) -> None:
        self._selected_server = "all"
        self._log_task = asyncio.create_task(self._consume_logs())
        self._line_count = 0
        self._error_count = 0
        self._warn_count = 0

    def on_unmount(self) -> None:
        if hasattr(self, "_log_task"):
            self._log_task.cancel()

    async def _consume_logs(self) -> None:
        app: LLMManagerApp = self.app  # type: ignore[assignment]
        try:
            while True:
                try:
                    entry: LogLine = app.log_tailer.queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.05)
                    continue

                if self._selected_server != "all" and entry.server_type != self._selected_server:
                    continue

                log_view = self.query_one("#main-log-view", LogView)
                prefix = f"[{entry.server_type}] " if self._selected_server == "all" else ""
                log_view.append_line(prefix + entry.line)

                self._line_count += 1
                line_lower = entry.line.lower()
                if any(w in line_lower for w in ("error", "critical", "fatal")):
                    self._error_count += 1
                elif any(w in line_lower for w in ("warn", "warning")):
                    self._warn_count += 1

                self._update_stats()
        except asyncio.CancelledError:
            pass

    def _update_stats(self) -> None:
        stats = self.query_one("#log-stats", Label)
        stats.update(
            f"Lines: {self._line_count}  "
            f"[red]Errors: {self._error_count}[/]  "
            f"[yellow]Warnings: {self._warn_count}[/]"
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "server-select":
            self._selected_server = str(event.value)
            stats = self.query_one("#log-stats", Label)
            if self._selected_server == "lmstudio":
                stats.update("[yellow]LM Studio does not stream logs — check the LM Studio GUI.[/]")
            else:
                self._update_stats()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "log-filter":
            log_view = self.query_one("#main-log-view", LogView)
            log_view.set_filter(event.value or None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-clear-logs":
            self.action_clear_logs()

    def action_clear_logs(self) -> None:
        log_view = self.query_one("#main-log-view", LogView)
        log_view.clear_log()
        self._line_count = 0
        self._error_count = 0
        self._warn_count = 0
        self._update_stats()

    def action_focus_filter(self) -> None:
        self.query_one("#log-filter", Input).focus()
