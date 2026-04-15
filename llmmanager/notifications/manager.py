"""NotificationManager — central event-driven notification bus."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmmanager.services.poller import PollSnapshot


class Severity(str, Enum):
    INFO    = "info"
    WARNING = "warning"
    ERROR   = "error"


@dataclass
class Notification:
    title: str
    body: str
    severity: Severity
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    read: bool = False


class NotificationManager:
    """
    Subscribes to PollSnapshots from the PollerService and fires
    notifications based on configurable rules.

    Also accepts manual notifications from anywhere in the app.
    """

    def __init__(self, config) -> None:
        self._config = config
        self._notifications: list[Notification] = []
        self._queue: asyncio.Queue[Notification] = asyncio.Queue(maxsize=100)
        self._last_server_states: dict[str, str] = {}

    @property
    def queue(self) -> asyncio.Queue[Notification]:
        return self._queue

    @property
    def unread_count(self) -> int:
        return sum(1 for n in self._notifications if not n.read)

    def all(self) -> list[Notification]:
        return list(reversed(self._notifications))

    def mark_all_read(self) -> None:
        for n in self._notifications:
            n.read = True

    def add(self, title: str, body: str, severity: Severity, source: str = "") -> None:
        n = Notification(title=title, body=body, severity=severity, source=source)
        self._notifications.append(n)
        if len(self._notifications) > 500:
            self._notifications = self._notifications[-500:]
        try:
            self._queue.put_nowait(n)
        except asyncio.QueueFull:
            pass

    def process_snapshot(self, snapshot: "PollSnapshot") -> None:
        """Check rules against the latest poll snapshot."""
        self._check_low_vram(snapshot)
        self._check_server_crashes(snapshot)

    def _check_low_vram(self, snapshot: "PollSnapshot") -> None:
        threshold = self._config.notifications.low_vram_threshold_pct
        for gpu in snapshot.gpus:
            if gpu.vram.free_pct < threshold:
                self.add(
                    title=f"Low VRAM: GPU {gpu.index}",
                    body=(
                        f"{gpu.name}: only {gpu.vram.free_mb:.0f} MB free "
                        f"({gpu.vram.free_pct:.1f}%)"
                    ),
                    severity=Severity.WARNING,
                    source="gpu_monitor",
                )

    def _check_server_crashes(self, snapshot: "PollSnapshot") -> None:
        from llmmanager.models.server import ServerState
        for info in snapshot.servers:
            prev_state = self._last_server_states.get(info.server_type)
            curr_state = info.status.state.value

            if (
                prev_state == ServerState.RUNNING.value
                and curr_state == ServerState.ERROR.value
            ):
                self.add(
                    title=f"{info.display_name} crashed",
                    body=info.status.error_message or "Server process exited unexpectedly.",
                    severity=Severity.ERROR,
                    source=info.server_type,
                )

            self._last_server_states[info.server_type] = curr_state
