"""LM Studio server stub — detects and connects to a running LM Studio instance."""

from __future__ import annotations

import shutil
from typing import AsyncIterator

import httpx

from llmmanager.config.schema import FlagDefinition, ServerConfig
from llmmanager.exceptions import ServerNotInstalledError
from llmmanager.models.llm_model import LLMModel, ModelSource
from llmmanager.models.server import EndpointInfo, ServerInfo, ServerState, ServerStatus
from llmmanager.servers.base import AbstractServer


class LMStudioServer(AbstractServer):
    """
    LM Studio stub.

    LM Studio is a GUI application without a proper CLI for headless management.
    This backend detects whether LM Studio's local server is running and exposes
    its OpenAI-compatible endpoints. Install/start/stop are not supported.
    """

    name = "lmstudio"
    display_name = "LM Studio"

    def __init__(self, config: ServerConfig) -> None:
        super().__init__(config)
        self._port = config.port or 1234

    # ------------------------------------------------------------------
    # Lifecycle — not supported
    # ------------------------------------------------------------------

    async def start(self) -> None:
        raise NotImplementedError(
            "LM Studio is a GUI application and cannot be started from LLMManager. "
            "Open LM Studio manually and enable the local server."
        )

    async def stop(self) -> None:
        raise NotImplementedError("LM Studio cannot be stopped from LLMManager.")

    async def restart(self) -> None:
        raise NotImplementedError("LM Studio cannot be restarted from LLMManager.")

    # ------------------------------------------------------------------
    # Status — detect via health check
    # ------------------------------------------------------------------

    async def get_status(self) -> ServerStatus:
        healthy = await self._health()
        if healthy:
            models = await self._list_models_raw()
            return ServerStatus(
                state=ServerState.RUNNING,
                loaded_models=[m.get("id", "") for m in models],
                endpoints=await self.get_endpoints(),
            )
        return ServerStatus(state=ServerState.STOPPED)

    async def get_info(self) -> ServerInfo:
        return ServerInfo(
            server_type=self.name,
            display_name=self.display_name,
            version=None,
            host=self.config.host,
            port=self._port,
            status=await self.get_status(),
        )

    # ------------------------------------------------------------------
    # Models — read-only via OpenAI-compat API
    # ------------------------------------------------------------------

    async def list_loaded_models(self) -> list[LLMModel]:
        raw = await self._list_models_raw()
        return [
            LLMModel(
                model_id=m.get("id", ""),
                display_name=m.get("id", ""),
                source=ModelSource.LOCAL,
                is_downloaded=True,
                is_loaded=True,
            )
            for m in raw
        ]

    async def load_model(self, model_id: str) -> None:
        raise NotImplementedError("Use the LM Studio GUI to load models.")

    async def unload_model(self, model_id: str) -> None:
        raise NotImplementedError("Use the LM Studio GUI to unload models.")

    async def delete_model(self, model_id: str) -> None:
        raise NotImplementedError("Use the LM Studio GUI to delete models.")

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    async def get_endpoints(self) -> list[EndpointInfo]:
        base = f"http://{self.config.host}:{self._port}"
        return [
            EndpointInfo(
                url=f"{base}/v1/chat/completions",
                protocol="openai-compat",
                description="LM Studio OpenAI-compatible chat completions",
            ),
            EndpointInfo(
                url=f"{base}/v1/completions",
                protocol="openai-compat",
                description="LM Studio OpenAI-compatible text completions",
            ),
        ]

    # ------------------------------------------------------------------
    # Logs — not available
    # ------------------------------------------------------------------

    async def stream_logs(self) -> AsyncIterator[str]:
        yield "LM Studio log streaming is not supported. Check the LM Studio GUI."

    # ------------------------------------------------------------------
    # Quick inference
    # ------------------------------------------------------------------

    async def quick_infer(self, model_id: str, prompt: str) -> AsyncIterator[str]:
        import json
        base = f"http://{self.config.host}:{self._port}"
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{base}/v1/chat/completions", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            content = (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

    # ------------------------------------------------------------------
    # Installation — not supported (point to download page)
    # ------------------------------------------------------------------

    async def is_installed(self) -> bool:
        return await self._health()

    async def get_installed_version(self) -> str | None:
        return None

    async def list_available_versions(self) -> list[str]:
        return []

    async def install(self, version: str = "latest", sudo_password: str = "") -> AsyncIterator[str]:
        yield "LM Studio cannot be installed automatically."
        yield "Download from: https://lmstudio.ai"
        yield "After installing, open LM Studio and enable 'Local Server' in settings."

    async def uninstall(self) -> AsyncIterator[str]:
        yield "Remove LM Studio manually via your package manager or the .AppImage file."

    # ------------------------------------------------------------------
    # Flag definitions — none (LM Studio is GUI-configured)
    # ------------------------------------------------------------------

    @classmethod
    def get_flag_definitions(cls) -> list[FlagDefinition]:
        return []

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------

    async def preflight_checks(self) -> list[tuple[str, bool, str]]:
        running = await self._health()
        return [
            (
                "LM Studio local server reachable",
                running,
                f"http://{self.config.host}:{self._port}" if running else "not running",
            )
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(
                    f"http://{self.config.host}:{self._port}/v1/models"
                )
                return r.status_code == 200
        except Exception:
            return False

    async def _list_models_raw(self) -> list[dict]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    f"http://{self.config.host}:{self._port}/v1/models"
                )
                r.raise_for_status()
                return r.json().get("data", [])
        except Exception:
            return []
