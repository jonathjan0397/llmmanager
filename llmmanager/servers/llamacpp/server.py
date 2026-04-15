"""llama.cpp server backend — full AbstractServer implementation."""

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
from llmmanager.servers.llamacpp.api_client import LlamaCppAPIClient
from llmmanager.servers.llamacpp import installer as _installer
from llmmanager.servers.llamacpp.flags import LLAMACPP_FLAGS
from llmmanager.servers.port_checker import check_port_free
from llmmanager.servers.process_manager import ManagedProcess


class LlamaCppServer(AbstractServer):
    name = "llamacpp"
    display_name = "llama.cpp"

    def __init__(self, config: ServerConfig) -> None:
        super().__init__(config)
        self._proc = ManagedProcess()
        self._api = LlamaCppAPIClient(
            host=config.host,
            port=config.port if config.port else 8080,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not await self.is_installed():
            raise ServerNotInstalledError(
                "llama.cpp is not installed. Use the Setup Wizard."
            )
        if self._proc.is_running or await self._api.health():
            raise ServerAlreadyRunningError("llama.cpp server is already running.")

        model = self.config.default_model or self.config.flags.get("model", "")
        if not model:
            raise ServerNotRunningError(
                "No model configured for llama.cpp. Set --model in flags."
            )

        port = self.config.port or 8080
        await check_port_free(self.config.host, port)

        cmd = self._build_cmd(model)
        env = self._build_env()
        await self._proc.start(cmd, env=env, log_file=self.config.log_file)

        # Wait up to 60s for the server to load the model
        for _ in range(120):
            await asyncio.sleep(0.5)
            if await self._api.health():
                return
        raise ServerNotRunningError(
            "llama.cpp server started but /health endpoint never became ready."
        )

    async def stop(self) -> None:
        await self._proc.stop()

    async def restart(self) -> None:
        await self.stop()
        await asyncio.sleep(1.0)
        await self.start()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_status(self) -> ServerStatus:
        healthy = await self._api.health()

        if healthy:
            models = await self._api.list_models()
            loaded = [m.get("id", "") for m in models]
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
            port=self.config.port or 8080,
            status=status,
        )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    async def list_loaded_models(self) -> list[LLMModel]:
        try:
            raw = await self._api.list_models()
        except Exception:
            raw = []
        models = []
        for m in raw:
            model_id = m.get("id", "")
            models.append(LLMModel(
                model_id=model_id,
                display_name=model_id,
                source=ModelSource.LOCAL,
                is_downloaded=True,
                is_loaded=True,
            ))
        return models

    async def load_model(self, model_id: str) -> None:
        """llama.cpp loads one model at server start — restart with new --model flag."""
        raise NotImplementedError(
            "llama.cpp serves a single model per instance. "
            "Update the model path and restart the server."
        )

    async def unload_model(self, model_id: str) -> None:
        raise NotImplementedError("Stop the server to unload the model.")

    async def delete_model(self, model_id: str) -> None:
        raise NotImplementedError(
            "llama.cpp does not manage model storage. Delete the GGUF file manually."
        )

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    async def get_endpoints(self) -> list[EndpointInfo]:
        base = f"http://{self.config.host}:{self.config.port or 8080}"
        return [
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
            EndpointInfo(
                url=f"{base}/v1/models",
                protocol="openai-compat",
                description="List loaded model",
            ),
        ]

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    async def stream_logs(self) -> AsyncIterator[str]:
        async for line in self._proc.stream_stdout():
            yield line

    # ------------------------------------------------------------------
    # Quick inference
    # ------------------------------------------------------------------

    async def quick_infer(
        self, model_id: str, prompt: str, **kwargs
    ) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": prompt}]
        async for token in self._api.chat_stream(model_id, messages, **kwargs):
            yield token

    async def chat_infer(
        self, model_id: str, messages: list[dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
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
        cmake_args = self.config.flags.get("cmake-args", "")
        async for line in _installer.install(version, cmake_args=cmake_args):
            yield line

    async def uninstall(self, sudo_password: str = "") -> AsyncIterator[str]:
        async for line in _installer.uninstall():
            yield line

    # ------------------------------------------------------------------
    # Flag definitions
    # ------------------------------------------------------------------

    @classmethod
    def get_flag_definitions(cls) -> list[FlagDefinition]:
        return LLAMACPP_FLAGS

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------

    async def preflight_checks(self) -> list[tuple[str, bool, str]]:
        results: list[tuple[str, bool, str]] = []

        installed = await self.is_installed()
        results.append((
            "llama-cpp-python installed",
            installed,
            str(_installer.LLAMACPP_VENV),
        ))

        model = self.config.default_model or self.config.flags.get("model", "")
        model_exists = bool(model) and os.path.isfile(model)
        results.append((
            "GGUF model file exists",
            model_exists,
            model if model else "not configured",
        ))

        port = self.config.port or 8080
        try:
            await check_port_free(self.config.host, port)
            results.append((f"port {port} available", True, ""))
        except Exception as exc:
            results.append((f"port {port} available", False, str(exc)))

        disk = shutil.disk_usage(os.path.expanduser("~"))
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

    def _build_cmd(self, model_path: str) -> list[str]:
        cmd = [
            str(_installer.LLAMACPP_PYTHON),
            "-m", "llama_cpp.server",
            "--model", model_path,
        ]

        host = self.config.host or "127.0.0.1"
        port = self.config.port or 8080
        cmd += ["--host", host, "--port", str(port)]

        # Map flag definitions to CLI args
        _FLAG_KEY_MAP = {f.name.lstrip("-"): f for f in LLAMACPP_FLAGS}
        for flag_def in LLAMACPP_FLAGS:
            key = flag_def.name.lstrip("-")
            if key in ("model",):
                continue  # already added above
            if key not in self.config.flags:
                continue
            val = self.config.flags[key]
            if val is None or val == "":
                continue
            if flag_def.type == "bool":
                if val:
                    cmd.append(flag_def.name)
            else:
                cmd += [flag_def.name, str(val)]

        return cmd

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.config.extra_env)
        return env
