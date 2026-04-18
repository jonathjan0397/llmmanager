"""PollerService — drives all live UI updates via periodic async polling."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llmmanager.models.gpu import GPUInfo, GPUProcess
from llmmanager.models.server import ServerInfo

if TYPE_CHECKING:
    from llmmanager.gpu.base import AbstractGPUProvider
    from llmmanager.servers.registry import ServerRegistry


@dataclass
class PollSnapshot:
    """All current stats, posted to the TUI as a single message."""
    gpus: list[GPUInfo]
    servers: list[ServerInfo]
    cpu_pct: float
    ram_used_mb: float
    ram_total_mb: float
    gpu_processes: list[GPUProcess] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.gpu_processes is None:
            self.gpu_processes = []


class PollerService:
    """
    Runs a background asyncio task that polls GPU and server status
    at a configurable interval, posting PollSnapshot updates to a queue
    that the TUI consumes.

    Design: all state is read-only from the TUI's perspective.
    The TUI subscribes to the queue; the poller never writes to the UI directly.
    """

    def __init__(
        self,
        gpu_provider: "AbstractGPUProvider",
        server_registry: "ServerRegistry",
        interval_ms: int = 2000,
    ) -> None:
        self._gpu = gpu_provider
        self._registry = server_registry
        self._interval_ms = interval_ms
        self._queue: asyncio.Queue[PollSnapshot] = asyncio.Queue(maxsize=2)
        self._task: asyncio.Task | None = None
        self._running = False
        self._paused = False
        self._latest: PollSnapshot | None = None

    @property
    def queue(self) -> asyncio.Queue[PollSnapshot]:
        return self._queue

    @property
    def latest(self) -> PollSnapshot | None:
        """Most recent snapshot — readable without consuming the queue."""
        return self._latest

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def toggle_pause(self) -> bool:
        """Toggle pause state. Returns new paused state."""
        self._paused = not self._paused
        return self._paused

    def set_interval(self, ms: int) -> None:
        self._interval_ms = ms

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop(), name="poller")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def force_poll(self) -> None:
        """Trigger an immediate poll outside the normal interval."""
        snapshot = await self._collect()
        self._latest = snapshot
        self._enqueue(snapshot)

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                if not self._paused:
                    snapshot = await self._collect()
                    self._latest = snapshot
                    self._enqueue(snapshot)
            except Exception:
                pass  # Never crash the poller — log silently
            await asyncio.sleep(self._interval_ms / 1000.0)

    async def _collect(self) -> PollSnapshot:
        import psutil

        # Run GPU, process, and server queries concurrently
        gpu_task = asyncio.create_task(self._gpu.get_all_gpus())
        proc_task = asyncio.create_task(self._gpu.get_processes())
        server_task = asyncio.create_task(self._collect_servers())

        gpus, gpu_procs, servers = await asyncio.gather(
            gpu_task, proc_task, server_task, return_exceptions=True
        )

        if isinstance(gpus, Exception):
            gpus = []
        if isinstance(gpu_procs, Exception):
            gpu_procs = []
        if isinstance(servers, Exception):
            servers = []

        mem = psutil.virtual_memory()
        return PollSnapshot(
            gpus=gpus,  # type: ignore[arg-type]
            servers=servers,  # type: ignore[arg-type]
            cpu_pct=psutil.cpu_percent(interval=None),
            ram_used_mb=mem.used / 1024**2,
            ram_total_mb=mem.total / 1024**2,
            gpu_processes=gpu_procs,  # type: ignore[arg-type]
        )

    async def _collect_servers(self) -> list[ServerInfo]:
        tasks = [s.get_info() for s in self._registry.all_enabled()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    def _enqueue(self, snapshot: PollSnapshot) -> None:
        """Non-blocking enqueue — drop the oldest snapshot if the queue is full."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._queue.put_nowait(snapshot)
        except asyncio.QueueFull:
            pass
