"""LogTailerService — async per-server log line streaming."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmmanager.servers.base import AbstractServer


@dataclass
class LogLine:
    server_type: str
    line: str


class LogTailerService:
    """
    Maintains one asyncio task per server that streams log lines
    into a shared queue consumed by the Logs screen.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[LogLine] = asyncio.Queue(maxsize=1000)
        self._tasks: dict[str, asyncio.Task] = {}

    @property
    def queue(self) -> asyncio.Queue[LogLine]:
        return self._queue

    def start_server(self, server: "AbstractServer") -> None:
        if server.name in self._tasks:
            return
        task = asyncio.create_task(
            self._tail(server),
            name=f"log_tailer_{server.name}",
        )
        self._tasks[server.name] = task

    def stop_server(self, server_name: str) -> None:
        task = self._tasks.pop(server_name, None)
        if task:
            task.cancel()

    async def stop_all(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

    async def _tail(self, server: "AbstractServer") -> None:
        while True:
            try:
                async for line in server.stream_logs():
                    await self._enqueue(LogLine(server_type=server.name, line=line))
            except asyncio.CancelledError:
                return
            except Exception:
                # Server not yet running or restarted — retry after a delay
                await asyncio.sleep(2.0)

    async def _enqueue(self, entry: LogLine) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            pass
