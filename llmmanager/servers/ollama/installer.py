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


async def install(version: str = "latest", sudo_password: str = "") -> AsyncIterator[str]:
    """
    Stream installation progress lines.
    Uses the official install script from ollama.ai.

    sudo_password: if provided, the script is written to a temp file and run
    via `sudo -S bash /tmp/script.sh` with the password piped to stdin.
    This is more reliable than SUDO_ASKPASS since it doesn't require the
    script to be patched or env vars to survive sudo's env_reset.
    """
    import os
    import tempfile

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

    env = {**os.environ, **env_extra}

    # Write script to a temp file so stdin is free for the sudo password
    fd, script_path = tempfile.mkstemp(suffix=".sh", prefix="llm_install_")
    try:
        os.write(fd, script.encode())
        os.close(fd)
    except Exception as exc:
        raise ServerInstallError(f"Failed to write install script: {exc}") from exc

    yield f"Running install script{f' (version {version})' if version != 'latest' else ''}..."

    try:
        if sudo_password:
            # sudo -S reads password from stdin; bash reads the script from the file
            proc = await asyncio.create_subprocess_exec(
                "sudo", "-S", "bash", script_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            assert proc.stdin is not None
            proc.stdin.write(f"{sudo_password}\n".encode())
            proc.stdin.close()
        else:
            proc = await asyncio.create_subprocess_exec(
                "bash", script_path,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )

        assert proc.stdout is not None
        async for line in proc.stdout:
            yield line.decode(errors="replace").rstrip()

        await proc.wait()
        if proc.returncode != 0:
            raise ServerInstallError(f"Install script exited with code {proc.returncode}")
    finally:
        try:
            os.unlink(script_path)
        except Exception:
            pass

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
