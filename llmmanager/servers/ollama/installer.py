"""Ollama installation and version management."""

from __future__ import annotations

import asyncio
import shutil
from typing import AsyncIterator

import httpx

from llmmanager.constants import OLLAMA_INSTALL_SCRIPT
from llmmanager.exceptions import ServerInstallError


async def is_installed() -> bool:
    return shutil.which("ollama") is not None


async def get_installed_version() -> str | None:
    if not await is_installed():
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "ollama", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        text = stdout.decode().strip()
        # Output: "ollama version 0.3.12"
        parts = text.split()
        return parts[-1] if parts else None
    except Exception:
        return None


async def list_available_versions() -> list[str]:
    """Fetch available versions from GitHub releases API."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://api.github.com/repos/ollama/ollama/releases",
                headers={"Accept": "application/vnd.github+json"},
            )
            r.raise_for_status()
            return [rel["tag_name"].lstrip("v") for rel in r.json()]
    except Exception:
        return []


async def install(version: str = "latest") -> AsyncIterator[str]:
    """
    Stream installation progress lines.
    Uses the official install script from ollama.ai.
    """
    # Download the install script
    yield "Downloading Ollama install script..."
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            r = await client.get(OLLAMA_INSTALL_SCRIPT)
            r.raise_for_status()
            script = r.text
    except Exception as exc:
        raise ServerInstallError(f"Failed to download install script: {exc}") from exc

    env_extra: dict[str, str] = {}
    if version != "latest":
        env_extra["OLLAMA_VERSION"] = version

    yield f"Running install script{f' (version {version})' if version != 'latest' else ''}..."

    import os
    env = {**os.environ, **env_extra}

    proc = await asyncio.create_subprocess_exec(
        "sh", "-s", "--",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    assert proc.stdin is not None
    proc.stdin.write(script.encode())
    proc.stdin.close()

    assert proc.stdout is not None
    async for line in proc.stdout:
        yield line.decode(errors="replace").rstrip()

    await proc.wait()
    if proc.returncode != 0:
        raise ServerInstallError(f"Install script exited with code {proc.returncode}")

    yield "Ollama installed successfully."


async def uninstall() -> AsyncIterator[str]:
    """Remove the ollama binary and service files."""
    yield "Removing Ollama binary..."
    path = shutil.which("ollama")
    if path:
        proc = await asyncio.create_subprocess_exec(
            "sudo", "rm", "-f", path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line in proc.stdout:  # type: ignore[union-attr]
            yield line.decode(errors="replace").rstrip()
        await proc.wait()

    yield "Removing systemd service (if present)..."
    for path in [
        "/etc/systemd/system/ollama.service",
        "/usr/lib/systemd/system/ollama.service",
    ]:
        proc = await asyncio.create_subprocess_exec(
            "sudo", "rm", "-f", path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        await proc.wait()

    yield "Ollama removed."
