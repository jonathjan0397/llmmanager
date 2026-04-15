"""Async HTTP client for the vLLM OpenAI-compatible API."""

from __future__ import annotations

import json
from typing import AsyncIterator, Any

import httpx

from llmmanager.constants import HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT, HTTP_QUICK_INFER_TIMEOUT
from llmmanager.exceptions import ServerNotRunningError


class VLLMAPIClient:
    def __init__(self, host: str, port: int, api_key: str = "") -> None:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._base = f"http://{host}:{port}"
        self._client = httpx.AsyncClient(
            base_url=self._base,
            headers=headers,
            timeout=httpx.Timeout(
                connect=HTTP_CONNECT_TIMEOUT,
                read=HTTP_READ_TIMEOUT,
                write=30.0,
                pool=5.0,
            ),
        )

    async def health(self) -> bool:
        try:
            r = await self._client.get("/health")
            return r.status_code == 200
        except httpx.ConnectError:
            return False

    async def version(self) -> str | None:
        try:
            r = await self._client.get("/version")
            r.raise_for_status()
            return r.json().get("version")
        except Exception:
            return None

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            r = await self._client.get("/v1/models")
            r.raise_for_status()
            return r.json().get("data", [])
        except httpx.ConnectError as exc:
            raise ServerNotRunningError("vLLM is not reachable") from exc

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            **kwargs,
        }
        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
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

    async def completions_stream(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "max_tokens": max_tokens,
            **kwargs,
        }
        async with self._client.stream(
            "POST",
            "/v1/completions",
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
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        text = chunk.get("choices", [{}])[0].get("text", "")
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        continue

    async def close(self) -> None:
        await self._client.aclose()
