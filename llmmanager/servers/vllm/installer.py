"""vLLM installation into an isolated virtual environment."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import AsyncIterator

from llmmanager.constants import VENV_DIR
from llmmanager.exceptions import ServerInstallError

VLLM_VENV = VENV_DIR / "vllm"
VLLM_PYTHON = VLLM_VENV / "bin" / "python"
VLLM_PIP = VLLM_VENV / "bin" / "pip"


def venv_exists() -> bool:
    return VLLM_PYTHON.exists()


async def is_installed() -> bool:
    if not venv_exists():
        return False
    proc = await asyncio.create_subprocess_exec(
        str(VLLM_PYTHON), "-c", "import vllm",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    return proc.returncode == 0


async def get_installed_version() -> str | None:
    if not await is_installed():
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            str(VLLM_PYTHON), "-c", "import vllm; print(vllm.__version__)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode().strip() or None
    except Exception:
        return None


async def list_available_versions() -> list[str]:
    """Fetch available vLLM versions from PyPI."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get("https://pypi.org/pypi/vllm/json")
            r.raise_for_status()
            releases = list(r.json().get("releases", {}).keys())
            releases.sort(reverse=True)
            return releases[:20]  # return most recent 20
    except Exception:
        return []


async def install(version: str = "latest") -> AsyncIterator[str]:
    """Install vLLM into an isolated venv, streaming progress."""
    VENV_DIR.mkdir(parents=True, exist_ok=True)

    # Create venv if needed
    if not venv_exists():
        yield f"Creating virtual environment at {VLLM_VENV}..."
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "venv", str(VLLM_VENV),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line in proc.stdout:  # type: ignore[union-attr]
            yield line.decode(errors="replace").rstrip()
        await proc.wait()
        if proc.returncode != 0:
            raise ServerInstallError("Failed to create venv for vLLM.")

    yield "Upgrading pip..."
    proc = await asyncio.create_subprocess_exec(
        str(VLLM_PIP), "install", "--upgrade", "pip",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    async for line in proc.stdout:  # type: ignore[union-attr]
        yield line.decode(errors="replace").rstrip()
    await proc.wait()

    pkg = "vllm" if version == "latest" else f"vllm=={version}"
    yield f"Installing {pkg} (this may take several minutes)..."

    proc = await asyncio.create_subprocess_exec(
        str(VLLM_PIP), "install", pkg,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    async for line in proc.stdout:
        yield line.decode(errors="replace").rstrip()
    await proc.wait()
    if proc.returncode != 0:
        raise ServerInstallError(f"pip install {pkg} failed.")

    yield "vLLM installed successfully."


async def uninstall() -> AsyncIterator[str]:
    """Remove the vLLM venv entirely."""
    import shutil
    yield f"Removing vLLM venv at {VLLM_VENV}..."
    if VLLM_VENV.exists():
        await asyncio.to_thread(shutil.rmtree, str(VLLM_VENV), ignore_errors=True)
    yield "vLLM removed."
