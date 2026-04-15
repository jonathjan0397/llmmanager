"""Port conflict detection using psutil and socket."""

from __future__ import annotations

import asyncio
import socket

import psutil

from llmmanager.exceptions import PortConflictError


async def check_port_free(host: str, port: int) -> None:
    """
    Raise PortConflictError if the port is already bound.
    Checks both via socket bind attempt and psutil connections.
    """
    # Try to bind the socket — most reliable check
    loop = asyncio.get_running_loop()
    conflict_pid = await loop.run_in_executor(None, _socket_check, host, port)
    if conflict_pid is not None:
        raise PortConflictError(port, conflict_pid)


def _socket_check(host: str, port: int) -> int | None:
    """Returns the PID holding the port, or None if port is free."""
    # First try psutil connections for a PID
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr and conn.laddr.port == port:
            return conn.pid  # may be None for some systems

    # Fallback: try binding — if it fails the port is in use
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            s.bind((host, port))
        return None  # bind succeeded, port is free
    except OSError:
        return 0  # port in use, PID unknown
