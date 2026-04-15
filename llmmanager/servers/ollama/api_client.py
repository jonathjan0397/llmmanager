"""Async HTTP client for the Ollama REST API."""

from __future__ import annotations

import json
from typing import AsyncIterator, Any

import httpx

from llmmanager.constants import HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT, HTTP_QUICK_INFER_TIMEOUT
from llmmanager.exceptions import ServerNotRunningError


class OllamaAPIClient:
    def __init__(self, host: str, port: int) -> None:
        self._base = f"http://{host}:{port}"
        self._client = httpx.AsyncClient(
            base_url=self._base,
            timeout=httpx.Timeout(
                connect=HTTP_CONNECT_TIMEOUT,
                read=HTTP_READ_TIMEOUT,
                write=30.0,
                pool=5.0,
            ),
        )

    async def health(self) -> bool:
        try:
            r = await self._client.get("/")
            return r.status_code == 200
        except httpx.ConnectError:
            return False

    async def version(self) -> str | None:
        try:
            r = await self._client.get("/api/version")
            r.raise_for_status()
            return r.json().get("version")
        except Exception:
            return None

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            r = await self._client.get("/api/tags")
            r.raise_for_status()
            return r.json().get("models", [])
        except httpx.ConnectError as exc:
            raise ServerNotRunningError("Ollama is not reachable") from exc

    async def show_model(self, model_name: str) -> dict[str, Any]:
        r = await self._client.post("/api/show", json={"name": model_name})
        r.raise_for_status()
        return r.json()

    async def pull_model(self, model_name: str) -> AsyncIterator[dict[str, Any]]:
        async with self._client.stream(
            "POST",
            "/api/pull",
            json={"name": model_name, "stream": True},
            timeout=httpx.Timeout(connect=HTTP_CONNECT_TIMEOUT, read=3600.0, write=30.0, pool=5.0),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    yield json.loads(line)

    async def delete_model(self, model_name: str) -> None:
        r = await self._client.request("DELETE", "/api/delete", json={"name": model_name})
        r.raise_for_status()

    async def generate_stream(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        payload = {"model": model, "prompt": prompt, "stream": True, **kwargs}
        async with self._client.stream(
            "POST",
            "/api/generate",
            json=payload,
            timeout=httpx.Timeout(
                connect=HTTP_CONNECT_TIMEOUT,
                read=HTTP_QUICK_INFER_TIMEOUT,
                write=30.0,
                pool=5.0,
            ),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    chunk = json.loads(line)
                    if token := chunk.get("response"):
                        yield token
                    if chunk.get("done"):
                        break

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        payload = {"model": model, "messages": messages, "stream": True, **kwargs}
        async with self._client.stream(
            "POST",
            "/api/chat",
            json=payload,
            timeout=httpx.Timeout(
                connect=HTTP_CONNECT_TIMEOUT,
                read=HTTP_QUICK_INFER_TIMEOUT,
                write=30.0,
                pool=5.0,
            ),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    chunk = json.loads(line)
                    if content := chunk.get("message", {}).get("content"):
                        yield content
                    if chunk.get("done"):
                        break

    async def close(self) -> None:
        await self._client.aclose()
