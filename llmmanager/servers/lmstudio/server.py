"""LM Studio server stub — detects and connects to a running LM Studio instance."""

from __future__ import annotations

from typing import AsyncIterator

import httpx

from llmmanager.config.schema import FlagDefinition, ServerConfig
from llmmanager.servers.lmstudio.flags import LMSTUDIO_FLAGS
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
        self._client: httpx.AsyncClient = self._build_client()

    def _build_client(self) -> httpx.AsyncClient:
        """Build (or rebuild) the httpx client from current config."""
        # Port: flags["port"] takes precedence over config.port so the flag
        # form value is always respected after a Save & Poll.
        port_flag = self.config.flags.get("port")
        self._port = int(port_flag) if port_flag else (self.config.port or 1234)
        # Keep config.port in sync so the rest of the app sees the right value.
        self.config.port = self._port

        api_key = self.config.flags.get("api-key", "").strip()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        return httpx.AsyncClient(
            base_url=f"http://{self.config.host}:{self._port}",
            headers=headers,
            timeout=httpx.Timeout(connect=2.0, read=5.0, write=10.0, pool=5.0),
        )

    async def refresh_client(self) -> None:
        """Rebuild the HTTP client after connection settings change."""
        old = self._client
        self._client = self._build_client()
        await old.aclose()

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
        # Try /api/v0/models first — returns all available models (LM Studio ≥ 0.3.x)
        all_models = await self._list_all_models_raw()
        if all_models:
            loaded_ids = {m.get("id", "") for m in await self._list_models_raw()}
            return [
                LLMModel(
                    model_id=m.get("path") or m.get("id", ""),
                    display_name=m.get("id") or m.get("path", ""),
                    source=ModelSource.LOCAL,
                    is_downloaded=True,
                    is_loaded=(m.get("path") or m.get("id", "")) in loaded_ids
                    or m.get("id", "") in loaded_ids,
                )
                for m in all_models
                if m.get("path") or m.get("id")
            ]
        # Fallback: only currently loaded models via OpenAI-compat /v1/models
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
        """Load a model via LM Studio's /api/v0/models/load endpoint."""
        try:
            r = await self._client.post(
                "/api/v0/models/load",
                json={"identifier": model_id},
                timeout=httpx.Timeout(connect=2.0, read=60.0, write=10.0, pool=5.0),
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"LM Studio rejected load request for '{model_id}': {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Could not load '{model_id}' — is LM Studio running with the local server enabled? ({exc})"
            ) from exc

    async def unload_model(self, model_id: str) -> None:
        """Unload a model via LM Studio's /api/v0/models/unload endpoint."""
        try:
            r = await self._client.post(
                "/api/v0/models/unload",
                json={"identifier": model_id},
                timeout=httpx.Timeout(connect=2.0, read=30.0, write=10.0, pool=5.0),
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"LM Studio rejected unload request for '{model_id}': {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Could not unload '{model_id}' — is LM Studio running? ({exc})"
            ) from exc

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

    async def quick_infer(self, model_id: str, prompt: str, **kwargs) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": prompt}]
        async for token in self.chat_infer(model_id, messages, **kwargs):
            yield token

    async def chat_infer(
        self, model_id: str, messages: list[dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        import json
        payload = {"model": model_id, "messages": messages, "stream": True, **kwargs}
        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
            timeout=httpx.Timeout(connect=2.0, read=120.0, write=30.0, pool=5.0),
        ) as r:
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

    async def uninstall(self, sudo_password: str = "") -> AsyncIterator[str]:
        yield "Remove LM Studio manually via your package manager or the .AppImage file."

    # ------------------------------------------------------------------
    # Flag definitions — none (LM Studio is GUI-configured)
    # ------------------------------------------------------------------

    @classmethod
    def get_flag_definitions(cls) -> list[FlagDefinition]:
        return LMSTUDIO_FLAGS

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
            r = await self._client.get("/v1/models")
            return r.status_code == 200
        except Exception:
            return False

    async def _list_models_raw(self) -> list[dict]:
        try:
            r = await self._client.get("/v1/models")
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception:
            return []

    async def _list_all_models_raw(self) -> list[dict]:
        """GET /api/v0/models — all models on disk (LM Studio ≥ 0.3.x). Returns [] on older builds."""
        try:
            r = await self._client.get("/api/v0/models")
            if r.status_code != 200:
                return []
            data = r.json()
            # Response is either {"data": [...]} or a plain list
            if isinstance(data, list):
                return data
            return data.get("data", [])
        except Exception:
            return []
