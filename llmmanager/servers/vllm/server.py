"""vLLM server backend — full AbstractServer implementation."""

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
from llmmanager.servers.vllm.api_client import VLLMAPIClient
from llmmanager.servers.vllm import installer as _installer
from llmmanager.servers.vllm.flags import VLLM_FLAGS
from llmmanager.servers.port_checker import check_port_free
from llmmanager.servers.process_manager import ManagedProcess


class VLLMServer(AbstractServer):
    name = "vllm"
    display_name = "vLLM"

    def __init__(self, config: ServerConfig) -> None:
        super().__init__(config)
        self._proc = ManagedProcess()
        self._api = VLLMAPIClient(
            host=config.host,
            port=config.port if config.port else 8000,
            api_key=config.flags.get("api-key", ""),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not await self.is_installed():
            raise ServerNotInstalledError("vLLM is not installed. Use the Setup Wizard.")
        if self._proc.is_running or await self._api.health():
            raise ServerAlreadyRunningError("vLLM is already running.")

        model = self.config.flags.get("model", "")
        if not model:
            raise ServerNotRunningError("No model configured for vLLM. Set --model in flags.")

        port = self.config.port or 8000
        await check_port_free(self.config.host, port)

        cmd = self._build_cmd()
        env = self._build_env()
        await self._proc.start(cmd, env=env, log_file=self.config.log_file)

        # Wait up to 60s for vLLM to load the model (can be slow)
        for _ in range(120):
            await asyncio.sleep(0.5)
            if await self._api.health():
                return
        raise ServerNotRunningError("vLLM started but /health endpoint never became ready.")

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
            port=self.config.port or 8000,
            status=status,
        )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    async def list_loaded_models(self) -> list[LLMModel]:
        raw = await self._api.list_models()
        return [
            LLMModel(
                model_id=m.get("id", ""),
                display_name=m.get("id", ""),
                source=ModelSource.HUGGINGFACE,
                is_downloaded=True,
                is_loaded=True,
            )
            for m in raw
        ]

    async def load_model(self, model_id: str) -> None:
        """vLLM loads one model at server start — restart with new --model flag."""
        raise NotImplementedError(
            "vLLM serves a single model per instance. "
            "Update the --model flag and restart the server."
        )

    async def unload_model(self, model_id: str) -> None:
        raise NotImplementedError("Use server stop to unload the model.")

    async def delete_model(self, model_id: str) -> None:
        raise NotImplementedError(
            "vLLM does not manage model storage. Delete files manually from the HuggingFace cache."
        )

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    async def get_endpoints(self) -> list[EndpointInfo]:
        base = f"http://{self.config.host}:{self.config.port or 8000}"
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
                description="List loaded models",
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

    async def quick_infer(self, model_id: str, prompt: str) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": prompt}]
        async for token in self._api.chat_stream(model_id, messages):
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
        async for line in _installer.install(version):
            yield line

    async def uninstall(self, sudo_password: str = "") -> AsyncIterator[str]:
        async for line in _installer.uninstall():
            yield line

    # ------------------------------------------------------------------
    # Flag definitions
    # ------------------------------------------------------------------

    @classmethod
    def get_flag_definitions(cls) -> list[FlagDefinition]:
        return VLLM_FLAGS

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------

    async def preflight_checks(self) -> list[tuple[str, bool, str]]:
        results: list[tuple[str, bool, str]] = []

        installed = await self.is_installed()
        results.append(("vLLM installed in venv", installed, str(_installer.VLLM_VENV)))

        port = self.config.port or 8000
        try:
            await check_port_free(self.config.host, port)
            results.append((f"port {port} available", True, ""))
        except Exception as exc:
            results.append((f"port {port} available", False, str(exc)))

        model = self.config.flags.get("model", "")
        results.append(("--model flag set", bool(model), model or "not configured"))

        # Check CUDA availability inside the venv
        if _installer.venv_exists():
            proc = await asyncio.create_subprocess_exec(
                str(_installer.VLLM_PYTHON),
                "-c",
                "import torch; print(torch.cuda.is_available())",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            cuda_ok = stdout.decode().strip().lower() == "true"
            results.append(("CUDA available in venv", cuda_ok, ""))

        disk = shutil.disk_usage(os.path.expanduser("~"))
        free_gb = disk.free / 1024**3
        results.append((
            "disk space (≥20 GB free for models)",
            free_gb >= 20.0,
            f"{free_gb:.1f} GB free",
        ))

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cmd(self) -> list[str]:
        cmd = [
            str(_installer.VLLM_PYTHON),
            "-m", "vllm.entrypoints.openai.api_server",
        ]
        port = self.config.port or 8000
        cmd += ["--host", self.config.host, "--port", str(port)]

        for flag_def in VLLM_FLAGS:
            key = flag_def.name.lstrip("-")
            if key in self.config.flags:
                val = self.config.flags[key]
                if flag_def.type == "bool":
                    if val:
                        cmd.append(flag_def.name)
                elif val not in (None, ""):
                    cmd += [flag_def.name, str(val)]
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.config.extra_env)
        return env
