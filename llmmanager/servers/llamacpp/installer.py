"""llama.cpp installation into an isolated virtual environment.

We install llama-cpp-python[server] which bundles the llama-server HTTP API
as a Python module runnable via `python -m llama_cpp.server`.

GPU support is enabled at install time via CMAKE_ARGS:
  - CUDA:   CMAKE_ARGS="-DGGML_CUDA=on"
  - ROCm:   CMAKE_ARGS="-DGGML_HIPBLAS=on"
  - Metal:  CMAKE_ARGS="-DGGML_METAL=on"
  - CPU:    (no CMAKE_ARGS — default)
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import AsyncIterator

from llmmanager.constants import VENV_DIR
from llmmanager.exceptions import ServerInstallError

LLAMACPP_VENV   = VENV_DIR / "llamacpp"
LLAMACPP_PYTHON = LLAMACPP_VENV / "bin" / "python"
LLAMACPP_PIP    = LLAMACPP_VENV / "bin" / "pip"

# Windows paths
if sys.platform == "win32":
    LLAMACPP_PYTHON = LLAMACPP_VENV / "Scripts" / "python.exe"
    LLAMACPP_PIP    = LLAMACPP_VENV / "Scripts" / "pip.exe"


def venv_exists() -> bool:
    return LLAMACPP_PYTHON.exists()


async def is_installed() -> bool:
    if not venv_exists():
        return False
    proc = await asyncio.create_subprocess_exec(
        str(LLAMACPP_PYTHON), "-c", "import llama_cpp",
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
            str(LLAMACPP_PYTHON), "-c",
            "import llama_cpp; print(llama_cpp.__version__)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode().strip() or None
    except Exception:
        return None


async def list_available_versions() -> list[str]:
    """Fetch available llama-cpp-python versions from PyPI."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get("https://pypi.org/pypi/llama-cpp-python/json")
            r.raise_for_status()
            releases = list(r.json().get("releases", {}).keys())
            releases.sort(reverse=True)
            return releases[:20]
    except Exception:
        return []


def _detect_cmake_args() -> str:
    """Auto-detect GPU backend and return appropriate CMAKE_ARGS."""
    # CUDA
    if shutil.which("nvcc") or Path("/usr/local/cuda").exists():
        return "-DGGML_CUDA=on"
    # ROCm
    if shutil.which("hipcc") or Path("/opt/rocm").exists():
        return "-DGGML_HIPBLAS=on"
    # Metal (macOS)
    if sys.platform == "darwin":
        return "-DGGML_METAL=on"
    # CPU fallback
    return ""


async def install(version: str = "latest", cmake_args: str = "") -> AsyncIterator[str]:
    """Install llama-cpp-python[server] into an isolated venv, streaming progress."""
    VENV_DIR.mkdir(parents=True, exist_ok=True)

    if not venv_exists():
        yield f"Creating virtual environment at {LLAMACPP_VENV}..."
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "venv", str(LLAMACPP_VENV),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line in proc.stdout:  # type: ignore[union-attr]
            yield line.decode(errors="replace").rstrip()
        await proc.wait()
        if proc.returncode != 0:
            raise ServerInstallError("Failed to create venv for llama.cpp.")

    yield "Upgrading pip..."
    proc = await asyncio.create_subprocess_exec(
        str(LLAMACPP_PIP), "install", "--upgrade", "pip",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    async for line in proc.stdout:  # type: ignore[union-attr]
        yield line.decode(errors="replace").rstrip()
    await proc.wait()

    # Resolve CMAKE_ARGS
    if not cmake_args:
        cmake_args = _detect_cmake_args()

    if cmake_args:
        yield f"Detected GPU build flags: CMAKE_ARGS=\"{cmake_args}\""
    else:
        yield "No GPU detected — installing CPU-only build."

    pkg = "llama-cpp-python[server]" if version == "latest" else f"llama-cpp-python[server]=={version}"
    yield f"Installing {pkg} (compiles from source — may take several minutes)..."

    env = dict(os.environ)
    if cmake_args:
        env["CMAKE_ARGS"] = cmake_args
    env["FORCE_CMAKE"] = "1"

    proc = await asyncio.create_subprocess_exec(
        str(LLAMACPP_PIP), "install", pkg, "--no-cache-dir",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    assert proc.stdout is not None
    async for line in proc.stdout:
        yield line.decode(errors="replace").rstrip()
    await proc.wait()
    if proc.returncode != 0:
        raise ServerInstallError(f"pip install {pkg} failed.")

    yield "llama.cpp installed successfully."


async def uninstall() -> AsyncIterator[str]:
    """Remove the llama.cpp venv entirely."""
    yield f"Removing llama.cpp venv at {LLAMACPP_VENV}..."
    if LLAMACPP_VENV.exists():
        await asyncio.to_thread(shutil.rmtree, str(LLAMACPP_VENV), ignore_errors=True)
    yield "llama.cpp removed."
