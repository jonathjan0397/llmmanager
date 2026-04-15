"""LogView widget — scrollable log with error/warning color highlighting."""

from __future__ import annotations

import re

from rich.text import Text
from textual.widgets import RichLog


_LEVEL_PATTERNS = [
    (re.compile(r"\b(ERROR|CRITICAL|FATAL|error|critical|fatal)\b"),   "red"),
    (re.compile(r"\b(WARN|WARNING|warn|warning)\b"),                    "yellow"),
    (re.compile(r"\b(INFO|info)\b"),                                    "cyan"),
    (re.compile(r"\b(DEBUG|debug)\b"),                                  "dim"),
]


class LogView(RichLog):
    """
    Extends RichLog with automatic error/warning color highlighting
    and a configurable max-line buffer.
    """

    DEFAULT_CSS = """
    LogView {
        height: 1fr;
        border: round $surface;
        background: $background;
    }
    """

    def __init__(self, max_lines: int = 500, **kwargs) -> None:
        super().__init__(max_lines=max_lines, highlight=False, markup=False, **kwargs)
        self._filter: str | None = None

    def append_line(self, line: str) -> None:
        """Colorize and append a log line."""
        if self._filter and self._filter.lower() not in line.lower():
            return
        text = Text(line)
        for pattern, color in _LEVEL_PATTERNS:
            for match in pattern.finditer(line):
                text.stylize(color, match.start(), match.end())
        self.write(text)

    def set_filter(self, filter_text: str | None) -> None:
        self._filter = filter_text or None

    def clear_log(self) -> None:
        self.clear()
