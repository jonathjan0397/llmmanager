"""Subprocess lifecycle management for LLM server processes."""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from typing import AsyncIterator

import psutil

from llmmanager.exceptions import ServerStartError, ServerStopError


class ManagedProcess:
    """
    Wraps an asyncio subprocess for a single server instance.

    Design: LLMManager does NOT own the process exclusively — if the app
    exits, the server keeps running. On re-launch, re_attach() is used to
    reconnect to an existing PID.
    """

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._pid: int | None = None
        self._log_file: Path | None = None
        self._log_fh = None

    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def is_running(self) -> bool:
        if self._pid is None:
            return False
        try:
            proc = psutil.Process(self._pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    async def start(
        self,
        cmd: list[str],
        env: dict[str, str] | None = None,
        log_file: Path | None = None,
    ) -> None:
        """Launch the process. Raises ServerStartError if it fails immediately."""
        self._log_file = log_file
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(log_file, "ab")  # noqa: SIM115

        try:
            stdout = self._log_fh if self._log_fh else asyncio.subprocess.PIPE
            stderr = asyncio.subprocess.STDOUT

            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=stdout,
                stderr=stderr,
                env=env,
                start_new_session=True,  # detach from terminal session
            )
            self._pid = self._process.pid

            # Give it a moment then verify it didn't immediately die
            await asyncio.sleep(0.5)
            if self._process.returncode is not None:
                raise ServerStartError(
                    f"Process exited immediately with code {self._process.returncode}"
                )
        except ServerStartError:
            raise
        except Exception as exc:
            raise ServerStartError(f"Failed to launch {cmd[0]}: {exc}") from exc

    async def stop(self, timeout: float = 10.0) -> None:
        """Send SIGTERM, wait for graceful exit, then SIGKILL if needed."""
        if not self.is_running:
            return
        try:
            proc = psutil.Process(self._pid)
            proc.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(proc.wait),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await asyncio.sleep(0.5)
        except psutil.NoSuchProcess:
            pass
        except Exception as exc:
            raise ServerStopError(f"Failed to stop PID {self._pid}: {exc}") from exc
        finally:
            if self._log_fh:
                self._log_fh.close()
                self._log_fh = None
            self._pid = None
            self._process = None

    def re_attach(self, pid: int, expected_name_fragment: str) -> bool:
        """
        Attempt to re-attach to an already-running server by PID.
        Returns True if the process exists and name matches.
        """
        try:
            proc = psutil.Process(pid)
            if expected_name_fragment.lower() in proc.name().lower():
                self._pid = pid
                return True
        except psutil.NoSuchProcess:
            pass
        return False

    async def stream_stdout(self) -> AsyncIterator[str]:
        """Yield stdout lines from the process. Only works if not logging to file."""
        if self._process is None or self._process.stdout is None:
            return
        async for line in self._process.stdout:
            yield line.decode(errors="replace").rstrip()

    def get_uptime(self) -> float | None:
        """Return seconds since the process started, or None."""
        if not self._pid:
            return None
        try:
            import time
            proc = psutil.Process(self._pid)
            return time.time() - proc.create_time()
        except psutil.NoSuchProcess:
            return None
