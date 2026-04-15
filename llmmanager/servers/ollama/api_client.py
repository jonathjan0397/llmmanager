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

    async def unload_model(self, model_name: str) -> None:
        """Signal Ollama to evict a model from memory immediately."""
        r = await self._client.post(
            "/api/generate",
            json={"model": model_name, "keep_alive": 0, "prompt": ""},
            timeout=httpx.Timeout(connect=HTTP_CONNECT_TIMEOUT, read=30.0, write=10.0, pool=5.0),
        )
        r.raise_for_status()
        data = r.json()
        if data.get("done_reason") not in ("unload", "stop") and not data.get("done"):
            raise RuntimeError(f"Unexpected unload response: {data}")

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
            if resp.status_code >= 400:
                body = await resp.aread()
                try:
                    err_msg = json.loads(body).get("error", body.decode())
                except Exception:
                    err_msg = body.decode() or f"HTTP {resp.status_code}"
                raise httpx.HTTPStatusError(
                    f"Ollama error {resp.status_code}: {err_msg}",
                    request=resp.request,
                    response=resp,
                )
            async for line in resp.aiter_lines():
                if line.strip():
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise RuntimeError(f"Ollama: {chunk['error']}")
                    if token := chunk.get("response"):
                        yield token
                    if chunk.get("done"):
                        break

    # Inference parameters that must go under the "options" key for /api/chat
    _CHAT_OPTIONS = frozenset({
        "num_predict", "temperature", "top_p", "top_k", "repeat_penalty",
        "seed", "num_ctx", "stop", "tfs_z", "mirostat", "mirostat_tau",
        "mirostat_eta", "penalize_newline", "num_keep",
    })

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        # Split kwargs: inference options go under "options", rest stay top-level
        options = {k: v for k, v in kwargs.items() if k in self._CHAT_OPTIONS}
        top_level = {k: v for k, v in kwargs.items() if k not in self._CHAT_OPTIONS}
        payload: dict[str, Any] = {
            "model": model, "messages": messages, "stream": True, **top_level
        }
        if options:
            payload["options"] = options
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
            if resp.status_code >= 400:
                body = await resp.aread()
                try:
                    err_msg = json.loads(body).get("error", body.decode())
                except Exception:
                    err_msg = body.decode() or f"HTTP {resp.status_code}"
                raise httpx.HTTPStatusError(
                    f"Ollama error {resp.status_code}: {err_msg}",
                    request=resp.request,
                    response=resp,
                )
            async for line in resp.aiter_lines():
                if line.strip():
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise RuntimeError(f"Ollama: {chunk['error']}")
                    if content := chunk.get("message", {}).get("content"):
                        yield content
                    if chunk.get("done"):
                        break

    async def close(self) -> None:
        await self._client.aclose()
