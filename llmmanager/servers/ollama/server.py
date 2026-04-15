"""Ollama server backend — full AbstractServer implementation."""

from __future__ import annotations

import asyncio
import os
import shutil
from typing import AsyncIterator

from llmmanager.config.schema import FlagDefinition, ServerConfig
from llmmanager.exceptions import (
    ServerAlreadyRunningError,
    ServerNotInstalledError,
    ServerNotRunningError,
)
from llmmanager.models.llm_model import LLMModel, ModelSource
from llmmanager.models.server import EndpointInfo, ServerInfo, ServerState, ServerStatus
from llmmanager.servers.base import AbstractServer
from llmmanager.servers.ollama.api_client import OllamaAPIClient
from llmmanager.servers.ollama import installer as _installer
from llmmanager.servers.ollama.flags import OLLAMA_FLAGS
from llmmanager.servers.port_checker import check_port_free
from llmmanager.servers.process_manager import ManagedProcess


class OllamaServer(AbstractServer):
    name = "ollama"
    display_name = "Ollama"

    def __init__(self, config: ServerConfig) -> None:
        super().__init__(config)
        self._proc = ManagedProcess()
        self._api = OllamaAPIClient(
            host=config.host,
            port=config.port if config.port else 11434,
        )
        self._pid_hint: int | None = None  # persisted PID for re-attach

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not await self.is_installed():
            raise ServerNotInstalledError("Ollama is not installed. Use the Setup Wizard.")
        if self._proc.is_running or await self._api.health():
            raise ServerAlreadyRunningError("Ollama is already running.")

        port = self.config.port or 11434
        await check_port_free(self.config.host, port)

        env = self._build_env()
        cmd = ["ollama", "serve"]

        await self._proc.start(cmd, env=env, log_file=self.config.log_file)

        # Wait up to 10s for the API to become healthy
        for _ in range(20):
            await asyncio.sleep(0.5)
            if await self._api.health():
                return
        raise ServerNotRunningError("Ollama started but API did not become healthy.")

    async def stop(self) -> None:
        await self._proc.stop()

    async def restart(self) -> None:
        await self.stop()
        await asyncio.sleep(0.5)
        await self.start()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_status(self) -> ServerStatus:
        # Try re-attaching to an existing process if we lost track
        if not self._proc.is_running and self._pid_hint:
            self._proc.re_attach(self._pid_hint, "ollama")

        healthy = await self._api.health()

        if healthy:
            models = await self._api.list_models()
            loaded = [m["name"] for m in models]
            version = await self._api.version()
            return ServerStatus(
                state=ServerState.RUNNING,
                pid=self._proc.pid,
                uptime_seconds=self._proc.get_uptime(),
                loaded_models=loaded,
                endpoints=await self.get_endpoints(),
            )

        if self._proc.is_running:
            return ServerStatus(state=ServerState.STARTING, pid=self._proc.pid)

        return ServerStatus(state=ServerState.STOPPED)

    async def get_info(self) -> ServerInfo:
        status = await self.get_status()
        version = await _installer.get_installed_version()
        return ServerInfo(
            server_type=self.name,
            display_name=self.display_name,
            version=version,
            host=self.config.host,
            port=self.config.port or 11434,
            status=status,
        )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    async def list_loaded_models(self) -> list[LLMModel]:
        raw = await self._api.list_models()
        models: list[LLMModel] = []
        for m in raw:
            details = m.get("details", {})
            models.append(LLMModel(
                model_id=m["name"],
                display_name=m["name"],
                source=ModelSource.OLLAMA_LIBRARY,
                size_gb=m.get("size", 0) / 1024**3,
                parameter_count_b=_parse_param_count(details.get("parameter_size", "")),
                quantization=details.get("quantization_level"),
                format=details.get("format"),
                is_downloaded=True,
                is_loaded=True,
                ollama_tag=m["name"],
            ))
        return models

    async def load_model(self, model_id: str) -> None:
        """Pull (download) a model via the Ollama API."""
        async for _ in self._api.pull_model(model_id):
            pass  # progress is surfaced via DownloadManager separately

    async def unload_model(self, model_id: str) -> None:
        """Evict a model from Ollama's memory (keep_alive=0)."""
        await self._api.unload_model(model_id)

    async def delete_model(self, model_id: str) -> None:
        await self._api.delete_model(model_id)

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    async def get_endpoints(self) -> list[EndpointInfo]:
        base = f"http://{self.config.host}:{self.config.port or 11434}"
        return [
            EndpointInfo(
                url=f"{base}/api/generate",
                protocol="ollama-native",
                description="Ollama native generate API",
            ),
            EndpointInfo(
                url=f"{base}/v1/chat/completions",
                protocol="openai-compat",
                description="OpenAI-compatible chat completions",
            ),
            EndpointInfo(
                url=f"{base}/v1/completions",
                protocol="openai-compat",
                description="OpenAI-compatible text completions",
            ),
        ]

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    async def stream_logs(self) -> AsyncIterator[str]:
        if self.config.log_file and self.config.log_file.exists():
            async for line in _tail_file(self.config.log_file):
                yield line
        else:
            async for line in self._proc.stream_stdout():
                yield line

    # ------------------------------------------------------------------
    # Quick inference
    # ------------------------------------------------------------------

    async def quick_infer(self, model_id: str, prompt: str, **kwargs) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": prompt}]
        async for token in self._api.chat_stream(model_id, messages, **kwargs):
            yield token

    async def chat_infer(
        self, model_id: str, messages: list[dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        """Stream a multi-turn chat response using /api/chat."""
        async for token in self._api.chat_stream(model_id, messages, **kwargs):
            yield token

    # ------------------------------------------------------------------
    # Installation
    # ------------------------------------------------------------------

    async def is_installed(self) -> bool:
        return await _installer.is_installed()

    async def get_installed_version(self) -> str | None:
        return await _installer.get_installed_version()

    async def list_available_versions(self) -> list[str]:
        return await _installer.list_available_versions()

    async def install(self, version: str = "latest", sudo_password: str = "") -> AsyncIterator[str]:
        async for line in _installer.install(version, sudo_password=sudo_password):
            yield line

    async def uninstall(self, sudo_password: str = "") -> AsyncIterator[str]:
        async for line in _installer.uninstall(sudo_password=sudo_password):
            yield line

    # ------------------------------------------------------------------
    # Flag definitions
    # ------------------------------------------------------------------

    @classmethod
    def get_flag_definitions(cls) -> list[FlagDefinition]:
        return OLLAMA_FLAGS

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------

    async def preflight_checks(self) -> list[tuple[str, bool, str]]:
        results: list[tuple[str, bool, str]] = []

        # Check if already installed
        installed = await self.is_installed()
        results.append(("ollama binary present", installed, shutil.which("ollama") or "not found"))

        # Check port availability
        port = self.config.port or 11434
        try:
            await check_port_free(self.config.host, port)
            results.append((f"port {port} available", True, ""))
        except Exception as exc:
            results.append((f"port {port} available", False, str(exc)))

        # Check disk space (need at least 5GB free for a small model)
        import shutil as _shutil
        disk = _shutil.disk_usage(os.path.expanduser("~"))
        free_gb = disk.free / 1024**3
        results.append((
            "disk space (≥5 GB free)",
            free_gb >= 5.0,
            f"{free_gb:.1f} GB free",
        ))

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        port = self.config.port or 11434
        env["OLLAMA_HOST"] = f"{self.config.host}:{port}"
        for flag_def in OLLAMA_FLAGS:
            flag_name = flag_def.name.lstrip("-").replace("-", "_").upper()
            cfg_key = flag_def.name.lstrip("-")
            if cfg_key in self.config.flags and flag_def.env_var:
                env[flag_def.env_var] = str(self.config.flags[cfg_key])
        env.update(self.config.extra_env)
        return env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_param_count(s: str) -> float | None:
    """Parse '7B', '70B', '3.2B' etc. into a float (billions)."""
    if not s:
        return None
    s = s.upper().replace("B", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


async def _tail_file(path, chunk_size: int = 4096) -> AsyncIterator[str]:
    """Async tail -f equivalent for a log file path."""
    import aiofiles
    async with aiofiles.open(path, mode="r", errors="replace") as f:
        await f.seek(0, 2)  # seek to end
        while True:
            line = await f.readline()
            if line:
                yield line.rstrip()
            else:
                await asyncio.sleep(0.1)
