"""DownloadManager — queued async model downloads with progress events."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from llmmanager.models.llm_model import DownloadProgress

if TYPE_CHECKING:
    from llmmanager.servers.base import AbstractServer


@dataclass
class DownloadRequest:
    server_type: str
    model_id: str
    server: "AbstractServer"


class DownloadManager:
    """
    Serializes model downloads (one at a time per server) and publishes
    DownloadProgress updates to a queue consumed by the Model Management screen.
    """

    def __init__(self) -> None:
        self._progress_queue: asyncio.Queue[DownloadProgress] = asyncio.Queue(maxsize=200)
        self._active: dict[str, asyncio.Task] = {}
        self._pending: asyncio.Queue[DownloadRequest] = asyncio.Queue()
        self._dispatcher_task: asyncio.Task | None = None

    @property
    def progress_queue(self) -> asyncio.Queue[DownloadProgress]:
        return self._progress_queue

    async def start(self) -> None:
        self._dispatcher_task = asyncio.create_task(
            self._dispatch_loop(), name="download_dispatcher"
        )

    async def stop(self) -> None:
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
        for task in self._active.values():
            task.cancel()
        await asyncio.gather(
            *self._active.values(),
            *([] if not self._dispatcher_task else [self._dispatcher_task]),
            return_exceptions=True,
        )

    def enqueue(self, server: "AbstractServer", model_id: str) -> None:
        """Queue a model download. Non-blocking."""
        req = DownloadRequest(
            server_type=server.name,
            model_id=model_id,
            server=server,
        )
        try:
            self._pending.put_nowait(req)
        except asyncio.QueueFull:
            pass

    def is_downloading(self, server_type: str, model_id: str) -> bool:
        return f"{server_type}:{model_id}" in self._active

    async def _dispatch_loop(self) -> None:
        while True:
            req = await self._pending.get()
            key = f"{req.server_type}:{req.model_id}"
            if key not in self._active:
                task = asyncio.create_task(
                    self._run_download(req, key),
                    name=f"download_{key}",
                )
                self._active[key] = task

    async def _run_download(self, req: DownloadRequest, key: str) -> None:
        try:
            start = time.monotonic()
            downloaded = 0
            last_time = start
            last_downloaded = 0

            # For Ollama — stream pull progress events
            async for event in req.server._api.pull_model(req.model_id):  # type: ignore[attr-defined]
                total = event.get("total")
                completed = event.get("completed", 0)
                status = event.get("status", "downloading")

                now = time.monotonic()
                delta_t = now - last_time
                delta_b = completed - last_downloaded

                speed = delta_b / delta_t if delta_t > 0 else 0.0
                eta = (
                    (total - completed) / speed
                    if speed > 0 and total
                    else None
                )

                progress = DownloadProgress(
                    model_id=req.model_id,
                    server_type=req.server_type,
                    total_bytes=total,
                    downloaded_bytes=completed,
                    speed_bps=speed,
                    eta_seconds=eta,
                    status=status,
                )
                last_time = now
                last_downloaded = completed

                try:
                    self._progress_queue.put_nowait(progress)
                except asyncio.QueueFull:
                    pass

            # Emit completion
            done = DownloadProgress(
                model_id=req.model_id,
                server_type=req.server_type,
                total_bytes=None,
                downloaded_bytes=0,
                speed_bps=0.0,
                eta_seconds=None,
                status="complete",
            )
            try:
                self._progress_queue.put_nowait(done)
            except asyncio.QueueFull:
                pass

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            error = DownloadProgress(
                model_id=req.model_id,
                server_type=req.server_type,
                total_bytes=None,
                downloaded_bytes=0,
                speed_bps=0.0,
                eta_seconds=None,
                status="error",
                error=str(exc),
            )
            try:
                self._progress_queue.put_nowait(error)
            except asyncio.QueueFull:
                pass
        finally:
            self._active.pop(key, None)
